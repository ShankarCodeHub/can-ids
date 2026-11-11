"""Phase 7 ablation study (see canids/evaluation/ablation.py for the protocol).

Fits every variant on train, tunes it on validation, then scores ALL variants
in one pass over the test split (logged in test_access_log.csv).

Outputs: reports/results/<ds>_ablation.csv, <ds>_ablation_by_attack.csv,
         <ds>_ablation_val.json, reports/figures/<ds>_ablation.png

Usage:
    python scripts/ablation.py --dataset road
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from canids.evaluation.ablation import ABLATIONS, fit_variant, summarise
from canids.evaluation.metrics import save_json
from canids.ingestion.schema import LABEL_NORMAL
from canids.preprocessing.datasets import count_labels, load_split, split_dir
from canids.preprocessing.features import FEATURE_COLUMNS
from canids.utils.config import load_config, resolve_path
from canids.utils.logger import get_logger
from canids.utils.provenance import git_commit
from canids.utils.seed import set_seed

log = get_logger("scripts.ablation")


def log_test_access(results, dataset: str, stage: str) -> None:
    """Append to the test-access audit log."""
    path = results / "test_access_log.csv"
    new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(["utc_time", "dataset", "stage", "git_commit"])
        w.writerow([datetime.now(UTC).isoformat(timespec="seconds"), dataset, stage, git_commit()])


def plot(table: pd.DataFrame, path, title: str) -> None:
    """Grouped bars of macro-F1 / detection / FPR per variant."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, col, label in zip(
        axes,
        ["macro_f1", "detection_recall", "alarm_fpr"],
        ["Macro-F1", "Detection recall", "Alarm FPR"],
        strict=True,
    ):
        ax.bar(table["variant"], table[col], color=["#1f77b4"] + ["#ff7f0e"] * (len(table) - 1))
        ax.set_title(label)
        ax.tick_params(axis="x", rotation=30)
        for i, v in enumerate(table[col]):
            ax.text(
                i,
                v,
                f"{v:.3f}" if col != "alarm_fpr" else f"{v:.4f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--dataset", choices=["hcrl", "road"], required=True)
    args = ap.parse_args(argv)
    ds = args.dataset
    mcfg, dcfg = load_config("model"), load_config("data")
    seed = int(mcfg["seed"])
    set_seed(seed)
    art = resolve_path(dcfg["paths"]["artifacts"]) / ds
    results = resolve_path(dcfg["paths"]["results"])
    figures = resolve_path(dcfg["paths"]["figures"])
    acfg, ccfg, hcfg = mcfg["anomaly"], mcfg["classifier"], mcfg["hybrid"]
    if_meta = json.loads((art / "anomaly" / "isolation_forest_metadata.json").read_text("utf-8"))
    clf_meta = json.loads((art / "classifier" / "classifier_metadata.json").read_text("utf-8"))

    t0 = time.perf_counter()
    # identical seeded samples to Phases 3-4, so 'full' reproduces the frozen models
    train_normal = load_split(
        ds,
        "train",
        FEATURE_COLUMNS,
        max_normal=int(acfg["train_normal_sample"]),
        seed=seed,
        normal_only=True,
    )
    train = load_split(
        ds, "train", FEATURE_COLUMNS, max_normal=int(ccfg["train_normal_sample"]), seed=seed
    )
    val = load_split(
        ds, "val", FEATURE_COLUMNS, max_normal=int(acfg["val_normal_sample"]), seed=seed
    )
    y_train = train["label"].astype(str).to_numpy()
    y_val = val["label"].astype(str).to_numpy()
    n_norm = int(count_labels(ds, "val")[LABEL_NORMAL])
    w_val = np.where(y_val == LABEL_NORMAL, n_norm / max((y_val == LABEL_NORMAL).sum(), 1), 1.0)
    log.info("data loaded in %.0fs", time.perf_counter() - t0)

    variants = []
    for name, spec in ABLATIONS.items():
        variants.append(
            fit_variant(
                name,
                spec["drop"],
                spec["use_iforest"],
                train_normal,
                train,
                y_train,
                val,
                y_val,
                w_val,
                if_meta["params"],
                clf_meta["algorithm"],
                clf_meta["params"],
                float(hcfg["t_normal"]),
                hcfg["search"]["t_attack"],
                hcfg["search"]["t_suppress"],
                float(hcfg["target_fpr"]),
                seed,
            )
        )
    del train_normal, train
    classes = variants[0].classifier.classes

    log_test_access(results, ds, "phase7_ablation")
    ys, atypes, labs = [], [], {v.name: [] for v in variants}
    for f in sorted(split_dir(ds, "test").glob("*.parquet")):
        df = pq.read_table(f, columns=[*FEATURE_COLUMNS, "label", "attack_type"]).to_pandas()
        ys.append(df["label"].astype(str).to_numpy())
        atypes.append(df["attack_type"].astype(str).to_numpy())
        for v in variants:
            labs[v.name].append(v.decide(df))
        log.info("scored %s", f.name)
    y, at = np.concatenate(ys), np.concatenate(atypes)
    rows, dets = [], []
    for v in variants:
        row, det = summarise(v.name, y, np.concatenate(labs[v.name]).astype(str), at, classes)
        row["n_features"] = len(v.features)
        row["t_attack"], row["t_suppress"] = v.thresholds.t_attack, v.thresholds.t_suppress
        rows.append(row)
        dets.append(det)
    table = pd.DataFrame(rows)
    table.to_csv(results / f"{ds}_ablation.csv", index=False)
    pd.concat(dets).to_csv(results / f"{ds}_ablation_by_attack.csv", index=False)
    save_json(
        {
            "dataset": ds,
            "protocol": "fixed Phase 3/4 hyperparameters, thresholds re-tuned on val",
            "if_params": if_meta["params"],
            "classifier": clf_meta["algorithm"],
            "classifier_params": clf_meta["params"],
            "git_commit": git_commit(),
            "val": {v.name: v.val for v in variants},
        },
        results / f"{ds}_ablation_val.json",
    )
    plot(
        table,
        figures / f"{ds}_ablation.png",
        f"{ds.upper()} ablation (test set, {len(y):,} frames)",
    )

    pd.set_option("display.width", 250)
    print(
        f"\n=== {ds.upper()} ablation on TEST ({len(y):,} frames) - "
        f"{time.perf_counter() - t0:.0f}s ==="
    )
    print(table.round(4).to_string(index=False))
    det = pd.concat(dets).pivot(index="attack_type", columns="variant", values="detection_rate")
    print("\nDetection rate by attack type:")
    print(det[[v.name for v in variants]].round(4).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
