"""Phase 2: clean -> split -> fit per-ID stats on TRAIN -> streaming features -> processed parquet.

Pipeline (per dataset)
----------------------
Pass 1, for each interim file: clean, assign splits, accumulate per-ID normal
        statistics from TRAIN frames only, collect split/balance reports.
        -> leakage check; the build ABORTS if any problem is found.
Pass 2, for each file: clean + split again (deterministic), run the streaming
        feature extractor over the whole capture (causal, so a val/test frame
        only sees its own past), write rows to
        ``data/processed/<dataset>/<split>/<capture>.parquet``.
        Purged boundary frames are not written.

Artifacts: ``models/<dataset>/feature_artifacts/``
    per_id_stats.json, class_weights.json, feature_scaler.joblib (fit on TRAIN,
    for the optional LSTM; tree models use raw features), feature_metadata.json
Reports:   ``reports/results/<dataset>_*.csv|json``

Usage:
    python scripts/build_features.py --dataset road
    python scripts/build_features.py --dataset road --max-files 3   # smoke test
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.preprocessing import StandardScaler

from canids.ingestion.schema import enforce_dtypes
from canids.preprocessing.cleaning import clean_frames
from canids.preprocessing.features import (
    FEATURE_COLUMNS,
    FEATURE_GROUPS,
    FEATURE_VERSION,
    PerIdStatsAccumulator,
    extract_features,
)
from canids.preprocessing.splits import (
    PURGED,
    SPLITS,
    TRAIN,
    assign_splits,
    capture_name,
    class_balance_report,
    compute_class_weights,
    cross_split_duplicates,
    leakage_check,
    recording_group,
    split_index,
)
from canids.utils.config import load_config, resolve_path
from canids.utils.logger import get_logger
from canids.utils.provenance import git_commit
from canids.utils.seed import set_seed

log = get_logger("scripts.build_features")

META_COLUMNS = ["timestamp", "label", "attack_type", "source_file", "split"]


def discover_files(dataset: str, interim: Path, cfg: dict[str, Any]) -> list[Path]:
    """Interim parquet files of a dataset, after the masquerade pair policy."""
    files = sorted((interim / dataset).glob("*.parquet"))
    if dataset == "road":
        policy = cfg["splits"]["road"].get("pair_policy", "both")
        if policy == "original_only":
            files = [f for f in files if not f.stem.endswith("_masquerade")]
        elif policy == "masquerade_only":
            groups_with_masq = {
                recording_group(f.stem) for f in files if f.stem.endswith("_masquerade")
            }
            files = [
                f
                for f in files
                if f.stem.endswith("_masquerade") or recording_group(f.stem) not in groups_with_masq
            ]
        elif policy != "both":
            raise ValueError(f"unknown pair_policy {policy!r}")
    return files


def strategy_for(dataset: str, source_file: str, cfg: dict[str, Any]) -> tuple[str, list | None]:
    """(strategy, capture rules) for one source file."""
    if dataset == "hcrl":
        return cfg["splits"]["hcrl"]["strategy"], None
    road = cfg["splits"]["road"]
    if str(source_file).startswith(cfg["road"]["ambient_dir"]):
        return road["ambient_strategy"], None
    return road["attack_strategy"], road["capture_assignment"]


def load_clean_split(path: Path, dataset: str, cfg: dict[str, Any]):
    """Load one interim file, clean it, and assign splits."""
    df = enforce_dtypes(pd.read_parquet(path))
    df, rep = clean_frames(df)
    s = cfg["splits"]
    strategy, rules = strategy_for(dataset, str(df["source_file"].iloc[0]), cfg)
    split = assign_splits(
        df, strategy, (s["train"], s["val"], s["test"]), int(s["purge_frames"]), rules
    )
    return df, split, rep, strategy


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--dataset", choices=["hcrl", "road"], required=True)
    ap.add_argument("--max-files", type=int, default=None, help="debug: only the first N files")
    args = ap.parse_args(argv)

    cfg = load_config("data")
    set_seed(int(cfg["seed"]))
    ds = args.dataset
    interim = resolve_path(cfg["paths"]["interim"])
    out_dir = resolve_path(cfg["paths"]["processed"]) / ds
    art_dir = resolve_path(cfg["paths"]["artifacts"]) / ds / "feature_artifacts"
    res_dir = resolve_path(cfg["paths"]["results"])
    res_dir.mkdir(parents=True, exist_ok=True)
    win_n = int(cfg["windows"]["frame_count"])
    win_ms = float(cfg["windows"]["time_ms"])
    chunk_rows = int(cfg["features"]["chunk_rows"])

    files = discover_files(ds, interim, cfg)
    if args.max_files:
        files = files[: args.max_files]
    if not files:
        log.error("No interim files for %s - run scripts/build_dataset.py first.", ds)
        return 1
    log.info("%s: %d interim files", ds, len(files))

    # ------------------------------------------------------------------ pass 1
    acc = PerIdStatsAccumulator()
    index_parts, balance_parts, cleaning_rows = [], [], []
    dup_total = 0
    for path in files:
        df, split, rep, strategy = load_clean_split(path, ds, cfg)
        index_parts.append(split_index(df, split, strategy))
        b = class_balance_report(df["label"], split).reset_index()
        b.insert(0, "source_file", rep.source_file)
        balance_parts.append(b)
        cleaning_rows.append(rep.__dict__ | {"notes": " | ".join(rep.notes)})
        dup_total += cross_split_duplicates(df, split)
        acc.add_file(df[split == TRAIN])
        del df

    index = pd.concat(index_parts, ignore_index=True)
    problems = leakage_check(index)
    if dup_total:
        problems.append(f"{dup_total} identical frames of one recording found in >1 split")
    leak = {
        "dataset": ds,
        "checked_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "files": len(files),
        "problems": problems,
        "passed": not problems,
    }
    (res_dir / f"{ds}_leakage_check.json").write_text(json.dumps(leak, indent=2), "utf-8")
    index.to_csv(res_dir / f"{ds}_split_index.csv", index=False)
    pd.DataFrame(cleaning_rows).to_csv(res_dir / f"{ds}_cleaning_report.csv", index=False)
    by_file = pd.concat(balance_parts, ignore_index=True).fillna(0)
    by_file.to_csv(res_dir / f"{ds}_split_balance_by_file.csv", index=False)
    label_cols = [
        c for c in by_file.columns if c not in ("source_file", "split", "total", "attack_share_pct")
    ]
    balance = by_file.groupby("split")[[*label_cols, "total"]].sum()
    balance = balance.reindex([s for s in (*SPLITS, PURGED) if s in balance.index])
    normal = balance.get("Normal", 0)
    balance["attack_share_pct"] = (100 * (balance["total"] - normal) / balance["total"]).round(4)
    balance.to_csv(res_dir / f"{ds}_split_balance.csv")
    print(f"\n=== {ds}: class balance per split ===\n{balance.to_string()}")
    if problems:
        log.error("LEAKAGE CHECK FAILED - no features written:\n  %s", "\n  ".join(problems))
        return 2
    log.info("Leakage check passed (%d files).", len(files))

    stats = acc.finalize(
        int(cfg["features"]["min_period_samples"]),
        meta={"dataset": ds, "split": TRAIN, "labels_used": ["Normal"]},
    )
    stats.to_json(art_dir / "per_id_stats.json")
    train_counts = {k: int(v) for k, v in balance.loc[TRAIN, label_cols].items()}
    weights = compute_class_weights(train_counts)
    (art_dir / "class_weights.json").write_text(
        json.dumps(
            {"method": "balanced", "train_counts": train_counts, "weights": weights}, indent=2
        ),
        "utf-8",
    )

    # ------------------------------------------------------------------ pass 2
    if out_dir.exists():
        shutil.rmtree(out_dir)
    scaler = StandardScaler()
    rows_written: Counter[str] = Counter()
    t_start = time.perf_counter()
    for n_file, path in enumerate(files, 1):
        t0 = time.perf_counter()
        df, split, _, _ = load_clean_split(path, ds, cfg)
        cap = capture_name(str(df["source_file"].iloc[0]))
        writers: dict[str, pq.ParquetWriter] = {}
        offset = 0
        for feats in extract_features(df, stats, win_n, win_ms, chunk_rows):
            sl = slice(offset, offset + len(feats))
            offset += len(feats)
            feats["timestamp"] = df["timestamp"].to_numpy()[sl]
            for col in ("label", "attack_type", "source_file"):
                feats[col] = df[col].astype(str).to_numpy()[sl]
            feats["split"] = split[sl]
            for name in SPLITS:
                part = feats[feats["split"] == name]
                if part.empty:
                    continue
                if name == TRAIN:
                    scaler.partial_fit(part[FEATURE_COLUMNS].to_numpy(dtype=np.float64))
                table = pa.Table.from_pandas(part.reset_index(drop=True), preserve_index=False)
                if name not in writers:
                    (out_dir / name).mkdir(parents=True, exist_ok=True)
                    writers[name] = pq.ParquetWriter(
                        out_dir / name / f"{cap}.parquet", table.schema
                    )
                writers[name].write_table(table)
                rows_written[name] += len(part)
        for w in writers.values():
            w.close()
        dt = time.perf_counter() - t0
        log.info(
            "[%d/%d] %s: %d frames in %.1fs (%.0f frames/s)",
            n_file,
            len(files),
            cap,
            len(df),
            dt,
            len(df) / max(dt, 1e-9),
        )
        del df

    joblib.dump(scaler, art_dir / "feature_scaler.joblib")
    meta = {
        "feature_version": FEATURE_VERSION,
        "dataset": ds,
        "feature_columns": FEATURE_COLUMNS,
        "feature_groups": FEATURE_GROUPS,
        "meta_columns_not_features": META_COLUMNS,
        "window_frames": win_n,
        "window_ms": win_ms,
        "split_config": cfg["splits"],
        "rows_written": dict(rows_written),
        "files": [p.name for p in files],
        "built_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "seconds": round(time.perf_counter() - t_start, 1),
    }
    (art_dir / "feature_metadata.json").write_text(json.dumps(meta, indent=2), "utf-8")
    print(f"\nRows written per split: {dict(rows_written)}")
    print(f"Feature extraction took {meta['seconds']}s; artifacts in {art_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
