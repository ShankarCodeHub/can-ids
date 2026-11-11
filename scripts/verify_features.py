"""Independent post-build checks of data/processed/<dataset> (Phase 2 acceptance).

Re-reads the processed parquet files, file by file (memory-bounded), and checks:
1. every file has exactly FEATURE_COLUMNS + META_COLUMNS, and each file's ``split``
   column matches the folder it is stored in;
2. no feature is NaN/inf; sentinel -1 appears only in features that allow it;
   bytes are in -1..255, shares are in [0, 1];
3. leakage re-check from the processed data alone (chronological files: train
   rows end before val rows start, val before test; capture files: one split each);
4. row counts per split/label match ``<dataset>_split_balance.csv``.

It also writes ``reports/results/<dataset>_feature_means_by_class.csv``: the mean of
every feature per (split, label). This is descriptive only and fits nothing.

Usage: python scripts/verify_features.py --dataset road
"""

from __future__ import annotations

import argparse
from collections import defaultdict

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from canids.preprocessing.features import FEATURE_COLUMNS, NA_VALUE
from canids.preprocessing.splits import SPLITS
from canids.utils.config import load_config, resolve_path

META_COLUMNS = ["timestamp", "label", "attack_type", "source_file", "split"]
SENTINEL_OK = {"hamming_prev_id", "iat_global", "iat_id", "win_iat_mean", "win_iat_std"}
SHARE_COLS = ["win_top_id_share", "win_id0_share", "win_cur_id_share"]


def main(argv: list[str] | None = None) -> int:
    """Run all checks; return 0 if everything passes."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", choices=["hcrl", "road"], required=True)
    args = ap.parse_args(argv)
    cfg = load_config("data")
    root = resolve_path(cfg["paths"]["processed"]) / args.dataset
    res = resolve_path(cfg["paths"]["results"])

    problems: list[str] = []
    counts: dict[tuple[str, str], int] = defaultdict(int)
    sums: dict[tuple[str, str], np.ndarray] = {}
    ranges: dict[tuple[str, str], tuple[float, float]] = {}

    for split in SPLITS:
        for path in sorted((root / split).glob("*.parquet")):
            df = pq.read_table(path).to_pandas()
            if list(df.columns) != FEATURE_COLUMNS + META_COLUMNS:
                problems.append(f"{path.name}: unexpected columns")
                continue
            if set(df["split"].unique()) != {split}:
                problems.append(f"{split}/{path.name}: split column disagrees with folder")
            feats = df[FEATURE_COLUMNS].to_numpy(dtype=np.float64)
            if not np.isfinite(feats).all():
                problems.append(f"{split}/{path.name}: NaN/inf in features")
            for col in FEATURE_COLUMNS:
                lo = float(df[col].min())
                if (
                    lo < 0
                    and col not in SENTINEL_OK
                    and not col.startswith("b")
                    and col != "iat_id_z"
                ):
                    problems.append(f"{split}/{path.name}: {col} has negative min {lo}")
                if col in SENTINEL_OK and lo < NA_VALUE:
                    problems.append(f"{split}/{path.name}: {col} below sentinel ({lo})")
            byte_vals = df[[f"b{i}" for i in range(8)]].to_numpy()
            if byte_vals.min() < -1 or byte_vals.max() > 255:
                problems.append(f"{split}/{path.name}: byte out of range")
            if ((df[SHARE_COLS] < 0) | (df[SHARE_COLS] > 1)).to_numpy().any():
                problems.append(f"{split}/{path.name}: share outside [0, 1]")
            src = str(df["source_file"].iloc[0])
            ranges[(src, split)] = (float(df["timestamp"].min()), float(df["timestamp"].max()))
            for label, g in df.groupby("label"):
                key = (split, str(label))
                counts[key] += len(g)
                s = g[FEATURE_COLUMNS].to_numpy(dtype=np.float64).sum(axis=0)
                sums[key] = sums.get(key, 0) + s

    # leakage re-check from processed data only
    by_file: dict[str, dict[str, tuple[float, float]]] = defaultdict(dict)
    for (src, split), rng in ranges.items():
        by_file[src][split] = rng
    for src, splits in by_file.items():
        present = [s for s in SPLITS if s in splits]
        if len(present) > 1:  # chronological file: blocks must be ordered and disjoint
            for a, b in zip(present[:-1], present[1:], strict=True):
                if splits[a][1] >= splits[b][0]:
                    problems.append(f"{src}: {a} overlaps {b}")

    # counts vs the build's balance table
    bal = pd.read_csv(res / f"{args.dataset}_split_balance.csv", index_col="split")
    for (split, label), n in counts.items():
        expected = int(bal.loc[split, label]) if label in bal.columns else -1
        if n != expected:
            problems.append(f"count mismatch {split}/{label}: processed={n} balance={expected}")

    means = pd.DataFrame(
        [
            {
                "split": s,
                "label": lab,
                "frames": counts[(s, lab)],
                **dict(zip(FEATURE_COLUMNS, sums[(s, lab)] / counts[(s, lab)], strict=True)),
            }
            for (s, lab) in sorted(counts)
        ]
    )
    means.to_csv(res / f"{args.dataset}_feature_means_by_class.csv", index=False)

    multi = sum(1 for v in by_file.values() if len(v) > 1)
    print(f"files: {len(by_file)} ({multi} chronological, {len(by_file) - multi} single-split)")
    print("rows per split/label:")
    print(pd.Series(counts).unstack(fill_value=0).to_string())
    if problems:
        print("\nPROBLEMS:\n  " + "\n  ".join(problems))
        return 1
    print("\nAll processed-data checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
