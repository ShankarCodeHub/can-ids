"""Phase 7 cross-dataset evaluation: HCRL -> ROAD and ROAD -> HCRL.

For each direction and each mode (multiclass on {Normal, Fuzzy, Spoofing};
binary Normal vs Attack), a classifier with the SOURCE dataset's selected
hyperparameters is trained on the source TRAIN split, using only the common,
vehicle-agnostic features. It is then tested on (a) the source TEST split, the
within-dataset reference with the same feature subset, and (b) the target TEST
split. The drop between (a) and (b) is the transfer gap.

Requires processed features for BOTH datasets (build_dataset.py +
build_features.py for hcrl and road) and a trained classifier for each source.

Outputs: reports/results/cross_dataset_results.csv, cross_dataset_results.json

Usage:
    python scripts/cross_dataset.py
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import UTC, datetime

import pandas as pd

from canids.evaluation.cross_dataset import common_features, train_and_test
from canids.evaluation.metrics import save_json
from canids.preprocessing.datasets import load_split, split_dir
from canids.preprocessing.features import FEATURE_COLUMNS
from canids.utils.config import load_config, resolve_path
from canids.utils.logger import get_logger
from canids.utils.provenance import git_commit
from canids.utils.seed import set_seed

log = get_logger("scripts.cross_dataset")
DATASETS = ("hcrl", "road")


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument(
        "--normal-sample",
        type=int,
        default=300000,
        help="Normal rows sampled per split (all attack rows are kept)",
    )
    args = ap.parse_args(argv)
    mcfg, dcfg = load_config("model"), load_config("data")
    seed = int(mcfg["seed"])
    set_seed(seed)
    art = resolve_path(dcfg["paths"]["artifacts"])
    results = resolve_path(dcfg["paths"]["results"])

    missing = [d for d in DATASETS if not any(split_dir(d, "test").glob("*.parquet"))]
    missing += [
        d
        for d in DATASETS
        if not (art / d / "classifier" / "classifier_metadata.json").exists() and d not in missing
    ]
    if missing:
        msg = (
            f"Cross-dataset evaluation NOT run: missing processed data / trained models for "
            f"{missing}. Run build_dataset.py, build_features.py and train.py for both datasets."
        )
        log.error(msg)
        print(msg)
        return 1

    features, dropped = common_features(FEATURE_COLUMNS)
    print(f"Common features ({len(features)}): {features}")
    print("Dropped:")
    for f, why in dropped.items():
        print(f"  {f:<12} {why}")

    data = {
        d: {
            s: load_split(d, s, features, max_normal=args.normal_sample, seed=seed)
            for s in ("train", "test")
        }
        for d in DATASETS
    }
    with (results / "test_access_log.csv").open("a", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerow(
            [
                datetime.now(UTC).isoformat(timespec="seconds"),
                "hcrl+road",
                "phase7_cross_dataset",
                git_commit(),
            ]
        )

    rows, full = [], {}
    for src in DATASETS:
        tgt = "road" if src == "hcrl" else "hcrl"
        meta = json.loads(
            (art / src / "classifier" / "classifier_metadata.json").read_text("utf-8")
        )
        for mode in ("multiclass", "binary"):
            for name, test_ds in (("within", src), ("cross", tgt)):
                r = train_and_test(
                    data[src]["train"],
                    data[test_ds]["test"],
                    features,
                    mode,
                    meta["algorithm"],
                    meta["params"],
                    seed,
                )
                key = f"{src}->{test_ds}:{mode}"
                full[key] = r
                rows.append(
                    {
                        "train": src,
                        "test": test_ds,
                        "setting": name,
                        "mode": mode,
                        "macro_f1": r["macro_f1"],
                        "weighted_f1": r["weighted_f1"],
                        "detection_recall": r["detection_recall"],
                        "alarm_fpr": r["alarm_fpr"],
                        "alarm_precision": r["alarm_precision"],
                        "roc_auc": r.get("roc_auc"),
                        "pr_auc": r.get("pr_auc"),
                        "test_rows": r["test_rows"],
                    }
                )
                log.info("%s macro-F1=%.4f", key, r["macro_f1"])
    table = pd.DataFrame(rows)
    table.to_csv(results / "cross_dataset_results.csv", index=False)
    save_json(
        {
            "features": features,
            "dropped": dropped,
            "normal_sample": args.normal_sample,
            "note": "Normal rows subsampled for compute; metrics unweighted on the sample",
            "results": full,
            "git_commit": git_commit(),
        },
        results / "cross_dataset_results.json",
    )
    print(table.round(4).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
