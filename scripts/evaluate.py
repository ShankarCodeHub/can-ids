"""Evaluate saved models on the held-out TEST split.

Stages: ``anomaly`` (Phase 3: baseline vs Isolation Forest) and ``hybrid``
(Phase 4: baseline vs IF-only vs classifier-only vs hybrid, plus per-class
metrics, confusion matrices and Unknown counts; outputs ``<ds>_phase4_*``).

The test split is read here and nowhere else. Every run is appended to
``reports/results/test_access_log.csv`` (time, dataset, stage, git commit), so
it is auditable that the test set was used once for final reporting rather
than for tuning.

Outputs:
    reports/results/<ds>_phase3_test_metrics.json   all metrics + curve points
    reports/results/<ds>_phase3_comparison.csv      Baseline vs Isolation Forest table
    reports/results/<ds>_phase3_detection_by_attack.csv
    reports/results/<ds>_phase3_fpr_by_context.csv
    reports/figures/<ds>_phase3_roc_pr.png

Usage:
    python scripts/evaluate.py anomaly --dataset road
"""

from __future__ import annotations

import argparse
import csv
import time
from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from canids.evaluation.metrics import (
    binary_metrics,
    curve_points,
    detection_rate_by_group,
    false_positive_rate_by_group,
    multiclass_metrics,
    plot_confusion_matrix,
    plot_roc_pr,
    save_json,
)
from canids.ingestion.schema import LABEL_NORMAL
from canids.models.anomaly import AnomalyDetector
from canids.models.baseline import RuleBaseline
from canids.models.hybrid import (
    RULE_NAMES,
    UNKNOWN,
    HybridDetector,
    HybridThresholds,
    hybrid_decide,
)
from canids.preprocessing.datasets import split_dir
from canids.preprocessing.features import FEATURE_COLUMNS
from canids.utils.config import load_config, resolve_path
from canids.utils.logger import get_logger
from canids.utils.provenance import git_commit

log = get_logger("scripts.evaluate")


def log_test_access(dataset: str, stage: str, results_dir) -> None:
    """Append one line to the test-access audit log."""
    path = results_dir / "test_access_log.csv"
    new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(["utc_time", "dataset", "stage", "git_commit"])
        w.writerow([datetime.now(UTC).isoformat(timespec="seconds"), dataset, stage, git_commit()])


def evaluate_anomaly(dataset: str) -> int:
    """Score the full test split with the Isolation Forest and the baseline."""
    dcfg = load_config("data")
    art = resolve_path(dcfg["paths"]["artifacts"]) / dataset
    results = resolve_path(dcfg["paths"]["results"])
    figures = resolve_path(dcfg["paths"]["figures"])
    detector = AnomalyDetector.load(art / "anomaly")
    baseline = RuleBaseline.load(art / "baseline" / "rule_baseline.json")
    log_test_access(dataset, "phase3_anomaly", results)

    files = sorted(split_dir(dataset, "test").glob("*.parquet"))
    cols = [*FEATURE_COLUMNS, "label", "attack_type", "source_file"]
    y, atype, src, s_if, s_bl = [], [], [], [], []
    t0 = time.perf_counter()
    for f in files:
        df = pq.read_table(f, columns=cols).to_pandas()
        y.append((df["label"] != LABEL_NORMAL).to_numpy())
        atype.append(df["attack_type"].astype(str).to_numpy())
        src.append(df["source_file"].astype(str).to_numpy())
        s_if.append(detector.score(df))
        s_bl.append(baseline.score(df))
        log.info("scored %s (%d rows)", f.name, len(df))
    y_true, attack_type, source = np.concatenate(y), np.concatenate(atype), np.concatenate(src)
    scores = {"Baseline (rules)": np.concatenate(s_bl), "Isolation Forest": np.concatenate(s_if)}
    thresholds = {"Baseline (rules)": 1.0, "Isolation Forest": detector.threshold}
    seconds = time.perf_counter() - t0
    context = np.where(
        pd.Series(source).str.startswith("ambient").to_numpy(),
        "ambient capture",
        "normal frames in attack capture",
    )

    out, rows, by_attack, by_ctx, curves = {}, [], [], [], {}
    for name, s in scores.items():
        pred = s > thresholds[name]
        m = binary_metrics(y_true, pred, s)
        out[name] = {"threshold": thresholds[name], "metrics": m}
        rows.append(
            {
                "model": name,
                **{
                    k: m[k]
                    for k in (
                        "precision",
                        "recall",
                        "f1",
                        "fpr",
                        "roc_auc",
                        "pr_auc",
                        "tp",
                        "fp",
                        "fn",
                        "tn",
                    )
                },
            }
        )
        d = detection_rate_by_group(y_true, pred, attack_type)
        d.insert(0, "model", name)
        by_attack.append(d)
        c = false_positive_rate_by_group(y_true, pred, context)
        c.insert(0, "model", name)
        by_ctx.append(c)
        curves[name] = curve_points(y_true, s)
        out[name]["curves"] = curves[name]

    comparison = pd.DataFrame(rows)
    detection = pd.concat(by_attack, ignore_index=True)
    fpr_ctx = pd.concat(by_ctx, ignore_index=True)
    comparison.to_csv(results / f"{dataset}_phase3_comparison.csv", index=False)
    detection.to_csv(results / f"{dataset}_phase3_detection_by_attack.csv", index=False)
    fpr_ctx.to_csv(results / f"{dataset}_phase3_fpr_by_context.csv", index=False)
    save_json(
        {
            "dataset": dataset,
            "split": "test",
            "test_rows": int(len(y_true)),
            "test_attack_rows": int(y_true.sum()),
            "scoring_seconds": round(seconds, 1),
            "evaluated_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "git_commit": git_commit(),
            "isolation_forest_params": detector.meta.get("params"),
            "models": out,
        },
        results / f"{dataset}_phase3_test_metrics.json",
    )
    plot_roc_pr(
        curves,
        figures / f"{dataset}_phase3_roc_pr.png",
        f"{dataset.upper()} test set: stage-1 anomaly detection "
        f"({len(y_true):,} frames, {int(y_true.sum()):,} attacks)",
        fpr_marker=float(detector.meta.get("target_fpr", 0.01)),
    )

    pd.set_option("display.width", 200)
    print(f"\n=== {dataset.upper()} TEST: Baseline vs Isolation Forest ===")
    print(f"{len(y_true):,} frames, {int(y_true.sum()):,} attack frames; scored in {seconds:.0f}s")
    print(comparison.round(4).to_string(index=False))
    print("\nDetection rate by attack type:")
    print(detection.round(4).to_string(index=False))
    print("\nFalse-positive rate by context:")
    print(fpr_ctx.round(5).to_string(index=False))
    return 0


def evaluate_hybrid(dataset: str) -> int:
    """Phase 4: Baseline vs IF-only vs Classifier-only vs Hybrid on the same test set."""
    dcfg = load_config("data")
    art = resolve_path(dcfg["paths"]["artifacts"]) / dataset
    results = resolve_path(dcfg["paths"]["results"])
    figures = resolve_path(dcfg["paths"]["figures"])
    hybrid = HybridDetector.load(art)
    baseline = RuleBaseline.load(art / "baseline" / "rule_baseline.json")
    clf, det, th = hybrid.classifier, hybrid.anomaly, hybrid.thresholds
    no_overrule = HybridThresholds(th.t_normal, th.t_attack, 1.01)
    log_test_access(dataset, "phase4_hybrid", results)

    names = {
        "bl": "Baseline (rules)",
        "if": "Isolation Forest only",
        "clf": f"Classifier only ({clf.meta['algorithm']})",
        "hy": "Hybrid (tuned)",
        "hy4": "Hybrid (no overrule, rules 1/2/4/5)",
    }
    acc: dict[str, list[np.ndarray]] = {
        k: [] for k in ("y", "atype", "s_bl", "s_if", "s_clf", "clf", "hy", "hy4", "rule")
    }
    t0 = time.perf_counter()
    cols = [*FEATURE_COLUMNS, "label", "attack_type"]
    for f in sorted(split_dir(dataset, "test").glob("*.parquet")):
        df = pq.read_table(f, columns=cols).to_pandas()
        s_if, is_anom = det.predict(df)
        proba = clf.predict_proba(df)
        hy, rule = hybrid_decide(is_anom, proba, clf.classes, th)
        hy4, _ = hybrid_decide(is_anom, proba, clf.classes, no_overrule)
        acc["y"].append(df["label"].astype(str).to_numpy())
        acc["atype"].append(df["attack_type"].astype(str).to_numpy())
        acc["s_bl"].append(baseline.score(df))
        acc["s_if"].append(s_if)
        acc["s_clf"].append(1.0 - proba[:, clf.classes.index(LABEL_NORMAL)])
        acc["clf"].append(np.asarray(clf.classes, dtype=object)[proba.argmax(axis=1)])
        acc["hy"].append(hy)
        acc["hy4"].append(hy4)
        acc["rule"].append(rule)
        log.info("scored %s (%d rows)", f.name, len(df))
    a = {k: np.concatenate(v) for k, v in acc.items()}
    seconds = time.perf_counter() - t0
    y = a["y"]
    attack = y != LABEL_NORMAL

    labels_of = {
        "bl": np.where(a["s_bl"] > 1.0, "Alarm", LABEL_NORMAL),
        "if": np.where(a["s_if"] > det.threshold, "Alarm", LABEL_NORMAL),
        "clf": a["clf"],
        "hy": a["hy"],
        "hy4": a["hy4"],
    }
    scores_of = {"bl": a["s_bl"], "if": a["s_if"], "clf": a["s_clf"]}
    rows, per_attack, report = [], [], {}
    for key, name in names.items():
        lab = labels_of[key].astype(str)
        alarm = lab != LABEL_NORMAL
        bm = binary_metrics(attack, alarm, scores_of.get(key))
        row = {
            "system": name,
            "alarm_precision": bm["precision"],
            "detection_recall": bm["recall"],
            "binary_f1": bm["f1"],
            "alarm_fpr": bm["fpr"],
            "roc_auc": bm.get("roc_auc"),
            "pr_auc": bm.get("pr_auc"),
            "macro_f1": None,
            "weighted_f1": None,
            "unknown_on_attacks": None,
            "unknown_on_normal": None,
        }
        entry: dict = {"binary": bm}
        if key in ("clf", "hy", "hy4"):
            mc = multiclass_metrics(y, lab, clf.classes, extra_pred_labels=(UNKNOWN,))
            row["macro_f1"], row["weighted_f1"] = mc["macro_f1"], mc["weighted_f1"]
            entry["multiclass"] = mc
            plot_confusion_matrix(
                mc["confusion_matrix"],
                figures / f"{dataset}_phase4_cm_{key}.png",
                f"{dataset.upper()} test: {name}",
            )
        if key in ("hy", "hy4"):
            row["unknown_on_attacks"] = int(((lab == UNKNOWN) & attack).sum())
            row["unknown_on_normal"] = int(((lab == UNKNOWN) & ~attack).sum())
        rows.append(row)
        report[name] = entry
        d = detection_rate_by_group(attack, alarm, a["atype"])
        d.insert(0, "system", name)
        if key in ("clf", "hy", "hy4"):  # how attack frames were labelled
            t = pd.crosstab(
                pd.Series(a["atype"][attack], name="group"), pd.Series(lab[attack], name="pred")
            )
            d = d.merge(t.add_prefix("pred_").reset_index(), on="group", how="left")
        per_attack.append(d)

    comparison = pd.DataFrame(rows)
    comparison.to_csv(results / f"{dataset}_phase4_comparison.csv", index=False)
    pd.concat(per_attack, ignore_index=True).to_csv(
        results / f"{dataset}_phase4_detection_by_attack.csv", index=False
    )
    rule_counts = pd.crosstab(
        pd.Series(a["rule"]).map(RULE_NAMES), np.where(attack, "attack", "normal")
    )
    rule_counts.to_csv(results / f"{dataset}_phase4_rule_usage.csv")
    save_json(
        {
            "dataset": dataset,
            "split": "test",
            "test_rows": int(len(y)),
            "test_attack_rows": int(attack.sum()),
            "scoring_seconds": round(seconds, 1),
            "evaluated_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "git_commit": git_commit(),
            "classes": clf.classes,
            "hybrid_thresholds": th.__dict__,
            "systems": report,
        },
        results / f"{dataset}_phase4_test_metrics.json",
    )

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print(
        f"\n=== {dataset.upper()} TEST: Phase 4 comparison ({len(y):,} frames, "
        f"{int(attack.sum()):,} attacks; {seconds:.0f}s) ==="
    )
    print(comparison.round(4).to_string(index=False))
    print("\nHybrid rule usage (frames):")
    print(rule_counts.to_string())
    for key in ("clf", "hy"):
        mc = report[names[key]]["multiclass"]
        print(f"\n{names[key]} per-class:")
        print(pd.DataFrame(mc["per_class"]).T.round(4).to_string())
        cm = mc["confusion_matrix"]
        print(
            pd.DataFrame(cm["values"], index=cm["labels_true"], columns=cm["labels_pred"])
            .astype(int)
            .to_string()
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("stage", choices=["anomaly", "hybrid"])
    ap.add_argument("--dataset", choices=["hcrl", "road"], required=True)
    args = ap.parse_args(argv)
    if args.stage == "anomaly":
        return evaluate_anomaly(args.dataset)
    return evaluate_hybrid(args.dataset)


if __name__ == "__main__":
    raise SystemExit(main())
