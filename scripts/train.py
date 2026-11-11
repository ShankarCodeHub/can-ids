"""Train detection models.

Stages:
    anomaly  (Phase 3) Isolation Forest + rule baseline
    hybrid   (Phase 4) RF/XGBoost classifier search + hybrid threshold tuning
             (needs the Phase 3 anomaly model)

Uses ONLY the train split (fitting) and the validation split (tuning). The
test split is never read here.

Usage:
    python scripts/train.py anomaly --dataset road
    python scripts/train.py hybrid --dataset road
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import UTC, datetime

import numpy as np

from canids.evaluation.metrics import alarm_metrics, multiclass_metrics
from canids.ingestion.schema import LABEL_NORMAL
from canids.models.anomaly import AnomalyDetector, tune_isolation_forest
from canids.models.baseline import BASELINE_FEATURES, tune_baseline
from canids.models.classifier import search_classifiers
from canids.models.hybrid import (
    UNKNOWN,
    HybridDetector,
    HybridThresholds,
    hybrid_decide,
    tune_hybrid,
)
from canids.preprocessing.datasets import count_labels, load_split
from canids.preprocessing.features import FEATURE_COLUMNS, PerIdStats
from canids.utils.config import load_config, resolve_path
from canids.utils.logger import get_logger
from canids.utils.provenance import git_commit
from canids.utils.seed import set_seed

log = get_logger("scripts.train")


def train_anomaly(dataset: str) -> int:
    """Phase 3: tune + save the Isolation Forest and the rule baseline."""
    mcfg, dcfg = load_config("model"), load_config("data")
    seed = int(mcfg["seed"])
    set_seed(seed)
    acfg, bcfg = mcfg["anomaly"], mcfg["baseline"]
    target_fpr = float(acfg["target_fpr"])
    art = resolve_path(dcfg["paths"]["artifacts"]) / dataset
    results = resolve_path(dcfg["paths"]["results"])
    features = [f for f in FEATURE_COLUMNS if f not in set(acfg.get("exclude_features", []))]
    feat_meta = json.loads((art / "feature_artifacts" / "feature_metadata.json").read_text("utf-8"))

    t0 = time.perf_counter()
    train_normal = load_split(
        dataset,
        "train",
        FEATURE_COLUMNS,
        max_normal=int(acfg["train_normal_sample"]),
        seed=seed,
        normal_only=True,
    )
    val = load_split(
        dataset,
        "val",
        [*FEATURE_COLUMNS, "attack_type", "source_file"],
        max_normal=int(acfg["val_normal_sample"]),
        seed=seed,
    )
    y_val = (val["label"] != LABEL_NORMAL).to_numpy()
    log.info(
        "loaded train-normal=%d, val=%d (attacks=%d) in %.1fs",
        len(train_normal),
        len(val),
        int(y_val.sum()),
        time.perf_counter() - t0,
    )
    if y_val.sum() == 0:
        log.error("validation split has no attack frames - cannot tune")
        return 1

    # ---------------- Isolation Forest
    detector, if_table = tune_isolation_forest(
        train_normal, val, y_val, features, acfg["search"], target_fpr, seed, int(acfg["n_jobs"])
    )
    provenance = {
        "dataset": dataset,
        "feature_version": feat_meta["feature_version"],
        "features_built_at": feat_meta["built_at"],
        "git_commit": git_commit(),
        "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "train_normal_sample": int(acfg["train_normal_sample"]),
        "val_normal_sample": int(acfg["val_normal_sample"]),
    }
    detector.meta.update(provenance)
    detector.save(art / "anomaly")
    if_table.to_csv(results / f"{dataset}_if_tuning.csv", index=False)

    # ---------------- rule baseline
    stats = PerIdStats.from_json(art / "feature_artifacts" / "per_id_stats.json")
    baseline, bl_table = tune_baseline(
        train_normal[BASELINE_FEATURES],
        val[BASELINE_FEATURES],
        y_val,
        stats.period_mean,
        bcfg["rate_quantiles"],
        bcfg["iat_ratios"],
        target_fpr,
        bool(bcfg["flag_unknown_ids"]),
    )
    baseline.meta.update(provenance)
    baseline.save(art / "baseline" / "rule_baseline.json")
    bl_table.to_csv(results / f"{dataset}_baseline_tuning.csv", index=False)

    vm, bm = detector.meta["val_metrics"], baseline.meta["val_metrics"]
    print(f"\n=== {dataset}: validation (tuning) results at target FPR {target_fpr} ===")
    print(f"IsolationForest {detector.meta['params']}  thr={detector.threshold:.5f}")
    print(
        f"  recall={vm['recall']:.4f} precision={vm['precision']:.4f} fpr={vm['fpr']:.4f} "
        f"roc_auc={vm.get('roc_auc', np.nan):.4f} pr_auc={vm.get('pr_auc', np.nan):.4f}"
    )
    print(
        f"Baseline rate_q={baseline.meta['rate_quantile']} iat_ratio={baseline.iat_ratio} "
        f"({baseline.meta['note']})"
    )
    print(
        f"  recall={bm['recall']:.4f} precision={bm['precision']:.4f} fpr={bm['fpr']:.4f} "
        f"roc_auc={bm.get('roc_auc', np.nan):.4f} pr_auc={bm.get('pr_auc', np.nan):.4f}"
    )
    print(f"Total {time.perf_counter() - t0:.0f}s. Models in {art}")
    return 0


def train_hybrid(dataset: str) -> int:
    """Phase 4: random-search the classifiers, then tune the hybrid thresholds on validation."""
    mcfg, dcfg = load_config("model"), load_config("data")
    seed = int(mcfg["seed"])
    set_seed(seed)
    ccfg, hcfg, acfg = mcfg["classifier"], mcfg["hybrid"], mcfg["anomaly"]
    art = resolve_path(dcfg["paths"]["artifacts"]) / dataset
    results = resolve_path(dcfg["paths"]["results"])
    feat_meta = json.loads((art / "feature_artifacts" / "feature_metadata.json").read_text("utf-8"))
    features = list(FEATURE_COLUMNS)

    t0 = time.perf_counter()
    train = load_split(
        dataset, "train", features, max_normal=int(ccfg["train_normal_sample"]), seed=seed
    )
    val = load_split(
        dataset,
        "val",
        [*features, "attack_type"],
        max_normal=int(acfg["val_normal_sample"]),
        seed=seed,
    )
    y_train = train["label"].astype(str).to_numpy()
    y_val = val["label"].astype(str).to_numpy()
    n_normal_total = int(count_labels(dataset, "val")[LABEL_NORMAL])
    is_n = y_val == LABEL_NORMAL
    w_val = np.where(is_n, n_normal_total / max(is_n.sum(), 1), 1.0)
    log.info(
        "train=%d %s | val=%d (normal weight %.2f)",
        len(train),
        dict(zip(*np.unique(y_train, return_counts=True), strict=True)),
        len(val),
        w_val.max(),
    )

    clf, table = search_classifiers(
        train,
        y_train,
        val,
        y_val,
        w_val,
        features,
        {k: ccfg[k] for k in ccfg["candidates"]},
        ccfg["candidates"],
        int(ccfg["search"]["n_iter"]),
        seed,
    )
    provenance = {
        "dataset": dataset,
        "feature_version": feat_meta["feature_version"],
        "features_built_at": feat_meta["built_at"],
        "git_commit": git_commit(),
        "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    clf.meta.update(provenance)
    clf.save(art / "classifier")
    table.to_csv(results / f"{dataset}_classifier_tuning.csv", index=False)

    # ---------------- hybrid thresholds (validation only)
    detector = AnomalyDetector.load(art / "anomaly")
    _, is_anom = detector.predict(val)
    proba = clf.predict_proba(val)
    th, htable = tune_hybrid(
        is_anom,
        proba,
        clf.classes,
        y_val,
        w_val,
        float(hcfg["t_normal"]),
        hcfg["search"]["t_attack"],
        hcfg["search"]["t_suppress"],
        float(hcfg["target_fpr"]),
    )
    htable.to_csv(results / f"{dataset}_hybrid_tuning.csv", index=False)
    literal = HybridThresholds(th.t_normal, th.t_attack, 1.01)
    val_summary = {}
    for name, t in (("tuned", th), ("no_overrule", literal)):
        labels, _ = hybrid_decide(is_anom, proba, clf.classes, t)
        val_summary[name] = {
            "thresholds": t.__dict__,
            **multiclass_metrics(y_val, labels, clf.classes, w_val, (UNKNOWN,)),
            **alarm_metrics(y_val, labels, LABEL_NORMAL, w_val),
        }
    HybridDetector(
        detector,
        clf,
        th,
        {
            **provenance,
            "selection": "max weighted val macro-F1 with alarm FPR <= target (Unknown = alarm)",
            "target_fpr": float(hcfg["target_fpr"]),
            "val": val_summary,
        },
    ).save(art / "hybrid")

    vm = clf.meta["val_metrics"]
    print(f"\n=== {dataset}: validation (tuning) results ===")
    print(f"Classifier: {clf.meta['algorithm']} {clf.meta['params']}")
    print(
        f"  macro-F1={vm['macro_f1']:.4f} weighted-F1={vm['weighted_f1']:.4f} "
        f"ECE={clf.meta['val_ece']:.4f}  "
        + "  ".join(f"F1[{c}]={vm['per_class'][c]['f1']:.4f}" for c in clf.classes)
    )
    for name, s in val_summary.items():
        print(
            f"Hybrid ({name}) {s['thresholds']}: macro-F1={s['macro_f1']:.4f} "
            f"detection={s['detection_recall']:.4f} alarm-FPR={s['alarm_fpr']:.4f}"
        )
    print(f"Total {time.perf_counter() - t0:.0f}s. Models in {art}")
    return 0


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("stage", choices=["anomaly", "hybrid"])
    ap.add_argument("--dataset", choices=["hcrl", "road"], required=True)
    args = ap.parse_args(argv)
    if args.stage == "anomaly":
        return train_anomaly(args.dataset)
    return train_hybrid(args.dataset)


if __name__ == "__main__":
    raise SystemExit(main())
