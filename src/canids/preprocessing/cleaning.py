"""Frame cleaning: ordering, de-duplication and validation of one source file.

Design decisions (viva):
* Cleaning is done PER SOURCE FILE. Different captures have unrelated clocks, so
  mixing them before sorting would interleave unrelated traffic and corrupt every
  timing feature.
* A *stable* sort by timestamp keeps the logged order of frames that share a
  timestamp (common at microsecond resolution).
* Exact duplicates (same timestamp, ID and payload) are logging artefacts, not
  real bus events, and would double-count frames; they are dropped and counted.
  Frames that repeat the same ID and payload at a different time are NOT
  duplicates - that is ordinary periodic CAN traffic.
* Missing payload bytes were already set to MISSING_BYTE by the loaders; the
  schema validation here re-checks that invariant.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from canids.ingestion.schema import BYTE_COLS, enforce_dtypes, validate_frame_df
from canids.utils.logger import get_logger

log = get_logger(__name__)

DUPLICATE_KEY: list[str] = ["timestamp", "can_id", "dlc", *BYTE_COLS]


@dataclass
class CleaningReport:
    """What cleaning changed in one file."""

    source_file: str
    rows_in: int = 0
    rows_out: int = 0
    duplicates_dropped: int = 0
    was_unsorted: bool = False
    out_of_order_frames: int = 0
    notes: list[str] = field(default_factory=list)


def clean_frames(df: pd.DataFrame) -> tuple[pd.DataFrame, CleaningReport]:
    """Clean the frames of ONE source file.

    Steps: validate schema -> stable sort by timestamp -> drop exact duplicates
    -> reset index -> validate again.

    Args:
        df: Unified-schema frames from a single ``source_file``.

    Returns:
        ``(clean_df, report)``.

    Raises:
        ValueError: if ``df`` mixes several source files.
        canids.ingestion.schema.SchemaError: if the input violates the schema.
    """
    files = df["source_file"].astype(str).unique()
    if len(files) > 1:
        raise ValueError(f"clean_frames expects one source file, got {len(files)}")
    name = files[0] if len(files) else ""
    report = CleaningReport(source_file=name, rows_in=len(df))
    validate_frame_df(df)

    ts = df["timestamp"].to_numpy()
    report.out_of_order_frames = int((ts[1:] < ts[:-1]).sum()) if len(ts) > 1 else 0
    report.was_unsorted = report.out_of_order_frames > 0
    out = df.sort_values("timestamp", kind="stable") if report.was_unsorted else df

    dup_mask = out.duplicated(subset=DUPLICATE_KEY, keep="first")
    report.duplicates_dropped = int(dup_mask.sum())
    if report.duplicates_dropped:
        out = out.loc[~dup_mask]

    out = enforce_dtypes(out.reset_index(drop=True))
    validate_frame_df(out)
    report.rows_out = len(out)
    if report.was_unsorted:
        report.notes.append(f"re-sorted {report.out_of_order_frames} out-of-order frames")
    log.info(
        "clean %s: in=%d out=%d duplicates=%d out_of_order=%d",
        name,
        report.rows_in,
        report.rows_out,
        report.duplicates_dropped,
        report.out_of_order_frames,
    )
    return out, report
