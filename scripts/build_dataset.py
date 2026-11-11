"""Parse raw HCRL and ROAD files into unified-schema parquet (Phase 1).

Outputs
-------
data/interim/<dataset>/<file>.parquet          one parquet per source file
reports/results/dataset_summary.csv            rows per class per dataset
reports/results/dataset_summary_by_file.csv    rows per class/attack_type per file
reports/results/ingestion_report.csv           read / kept / dropped (+reasons) / skipped
reports/results/raw_checksums.csv              SHA256 of raw files (with --checksums)

Usage
-----
    python scripts/build_dataset.py                    # both datasets
    python scripts/build_dataset.py --dataset hcrl
    python scripts/build_dataset.py --max-rows 100000  # quick smoke run
    python scripts/build_dataset.py --checksums
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Iterable
from pathlib import Path

import pandas as pd

from canids.ingestion.hcrl_loader import load_hcrl_dir
from canids.ingestion.road_loader import load_road_dir
from canids.ingestion.schema import LoadReport, validate_frame_df
from canids.utils.config import load_config, resolve_path
from canids.utils.logger import get_logger

log = get_logger("scripts.build_dataset")


def sha256sum(path: Path, chunk: int = 1 << 20) -> str:
    """SHA256 of a file, read in 1 MB chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while block := fh.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def write_checksums(raw_dirs: dict[str, Path], out_csv: Path) -> pd.DataFrame:
    """Record SHA256 of every raw file so others can verify identical data."""
    rows = []
    for dataset, raw_dir in raw_dirs.items():
        for f in sorted(p for p in raw_dir.rglob("*") if p.is_file() and p.name != ".gitkeep"):
            rows.append(
                {
                    "dataset": dataset,
                    "file": f.relative_to(raw_dir).as_posix(),
                    "bytes": f.stat().st_size,
                    "sha256": sha256sum(f),
                }
            )
    df = pd.DataFrame(rows, columns=["dataset", "file", "bytes", "sha256"])
    df.to_csv(out_csv, index=False)
    log.info("Wrote %d checksums to %s", len(df), out_csv)
    return df


def process(
    results: Iterable[tuple[pd.DataFrame | None, LoadReport]], out_dir: Path
) -> tuple[list[LoadReport], list[pd.DataFrame]]:
    """Validate + write each loaded file; return reports and per-file count tables."""
    out_dir.mkdir(parents=True, exist_ok=True)
    reports: list[LoadReport] = []
    counts: list[pd.DataFrame] = []
    for df, report in results:
        reports.append(report)
        if df is None:
            continue
        validate_frame_df(df)
        out_name = Path(report.source_file).with_suffix(".parquet").as_posix().replace("/", "__")
        df.to_parquet(out_dir / out_name, index=False)
        c = df.groupby(["label", "attack_type"], observed=True).size().reset_index(name="rows")
        c.insert(0, "source_file", report.source_file)
        c.insert(0, "dataset", report.source_dataset)
        counts.append(c)
    return reports, counts


def main(argv: list[str] | None = None) -> int:
    """CLI entry point; returns a process exit code."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--dataset", choices=["hcrl", "road", "all"], default="all")
    ap.add_argument("--max-rows", type=int, default=None, help="limit lines per file (debug)")
    ap.add_argument("--checksums", action="store_true", help="write SHA256 of raw files")
    args = ap.parse_args(argv)

    cfg = load_config("data")
    paths = cfg["paths"]
    missing = int(cfg["schema"]["missing_byte_value"])
    interim = resolve_path(paths["interim"])
    results_dir = resolve_path(paths["results"])
    results_dir.mkdir(parents=True, exist_ok=True)
    raw = {"hcrl": resolve_path(paths["raw_hcrl"]), "road": resolve_path(paths["raw_road"])}
    selected = ["hcrl", "road"] if args.dataset == "all" else [args.dataset]

    if args.checksums:
        write_checksums({k: raw[k] for k in selected}, results_dir / "raw_checksums.csv")

    all_reports: list[LoadReport] = []
    all_counts: list[pd.DataFrame] = []
    for name in selected:
        if name == "hcrl":
            gen = load_hcrl_dir(raw["hcrl"], cfg["hcrl"]["files"], missing, args.max_rows)
        else:
            gen = load_road_dir(raw["road"], cfg["road"], missing, args.max_rows)
        reports, counts = process(gen, interim / name)
        all_reports += reports
        all_counts += counts

    report_df = pd.DataFrame(
        [
            {
                "dataset": r.source_dataset,
                "source_file": r.source_file,
                "skipped": r.skipped,
                "rows_read": r.rows_read,
                "rows_kept": r.rows_kept,
                "rows_dropped": r.rows_dropped,
                "drop_reasons": json.dumps(r.dropped, sort_keys=True) if r.dropped else "",
                "notes": " | ".join(r.notes),
            }
            for r in all_reports
        ]
    )
    report_df.to_csv(results_dir / "ingestion_report.csv", index=False)

    if not all_counts:
        log.error(
            "No raw data was parsed. Download the datasets first (scripts/download_data.md). "
            "No summary was written."
        )
        print(report_df.to_string(index=False))
        return 1

    by_file = pd.concat(all_counts, ignore_index=True)
    by_file.to_csv(results_dir / "dataset_summary_by_file.csv", index=False)
    summary = (
        by_file.groupby(["dataset", "label"], observed=True)["rows"]
        .sum()
        .unstack("label", fill_value=0)
    )
    summary["total"] = summary.sum(axis=1)
    summary.to_csv(results_dir / "dataset_summary.csv")
    if args.max_rows:
        log.warning("--max-rows=%d was used: counts are a PARTIAL sample.", args.max_rows)

    print("\n=== Rows per class per dataset ===")
    print(summary.to_string())
    print("\n=== Ingestion report ===")
    print(report_df.drop(columns=["notes"]).to_string(index=False))
    skipped = report_df[report_df["skipped"]]
    if not skipped.empty:
        print("\n=== Skipped files ===")
        print(skipped[["dataset", "source_file", "notes"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
