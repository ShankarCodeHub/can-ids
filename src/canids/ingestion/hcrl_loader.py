"""Loader for the HCRL Car-Hacking Dataset (Hacking and Countermeasure Research Lab).

Source formats (per the official dataset page; re-verified on the real files):

* Attack CSVs (``DoS_dataset.csv``, ``Fuzzy_dataset.csv``, ``gear_dataset.csv``,
  ``RPM_dataset.csv``), no header::

      1478198376.389427,0316,8,05,21,68,09,21,21,00,6f,R
      1478198376.389636,0080,2,00,17,T          <- DLC 2: flag moves left

  The flag (R = regular, T = injected) is the field right after the last data
  byte, so rows with DLC < 8 have fewer fields. A naive fixed-column read
  (``pd.read_csv`` with 12 columns) would put the flag inside DATA[2..7]. We
  therefore parse by DLC: ``data = fields[3 : 3+dlc]``, ``flag = fields[3+dlc]``.

* Attack-free log ``normal_run_data.txt``::

      Timestamp: 1479121434.850202        ID: 0350    000    DLC: 8    05 28 84 66 6d 00 00 a2

Label mapping: flag R -> Normal; flag T -> the file's attack label
(DoS, Fuzzy, or Spoofing for gear/RPM). Every row of the normal log is Normal.
Malformed rows are dropped and counted per reason (never silently).
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import pandas as pd

from canids.ingestion.schema import (
    ATTACK_NONE,
    LABEL_NORMAL,
    LABELS,
    MISSING_BYTE,
    N_BYTES,
    FrameAccumulator,
    LoadReport,
    ParseError,
    pad_payload,
    parse_can_id,
    parse_hex_byte,
)
from canids.utils.logger import get_logger

log = get_logger(__name__)

SOURCE_DATASET = "hcrl"
FLAG_REGULAR = "R"
FLAG_INJECTED = "T"

#: Default file -> (label, attack_type); overridden by ``data.yaml: hcrl.files``.
DEFAULT_FILES: dict[str, dict[str, str]] = {
    "DoS_dataset.csv": {"label": "DoS", "attack_type": "dos"},
    "Fuzzy_dataset.csv": {"label": "Fuzzy", "attack_type": "fuzzy"},
    "gear_dataset.csv": {"label": "Spoofing", "attack_type": "gear_spoofing"},
    "RPM_dataset.csv": {"label": "Spoofing", "attack_type": "rpm_spoofing"},
    "normal_run_data.txt": {"label": "Normal", "attack_type": ATTACK_NONE},
}

_TXT_RE = re.compile(
    r"^Timestamp:\s*(?P<ts>\S+)\s+ID:\s*(?P<id>\S+)\s+(?P<rtr>\S+)\s+"
    r"DLC:\s*(?P<dlc>\S+)\s*(?P<data>.*)$"
)

ParsedRow = tuple[float, int, int, tuple[int, ...]]


def _parse_timestamp(token: str) -> float:
    try:
        ts = float(token)
    except ValueError:
        raise ParseError("bad_timestamp", repr(token)) from None
    if not math.isfinite(ts) or ts < 0:
        raise ParseError("bad_timestamp", repr(token))
    return ts


def _parse_dlc(token: str) -> int:
    try:
        dlc = int(token)
    except ValueError:
        raise ParseError("bad_dlc", repr(token)) from None
    if not 0 <= dlc <= N_BYTES:
        raise ParseError("bad_dlc", repr(token))
    return dlc


def parse_hcrl_csv_line(line: str, missing_byte: int = MISSING_BYTE) -> tuple[ParsedRow, str]:
    """Parse one line of an HCRL attack CSV.

    Args:
        line: Raw text line.
        missing_byte: Fill value for byte positions beyond the DLC.

    Returns:
        ``((timestamp, can_id, dlc, payload8), flag)`` with flag ``"R"`` or ``"T"``.

    Raises:
        ParseError: with reasons ``empty_line``, ``too_few_fields``, ``bad_timestamp``,
            ``bad_id``, ``bad_dlc``, ``bad_hex``, ``bad_flag``, ``extra_fields``.
    """
    stripped = line.strip()
    if not stripped:
        raise ParseError("empty_line")
    fields = [f.strip() for f in stripped.split(",")]
    if len(fields) < 4:
        raise ParseError("too_few_fields", stripped)

    ts = _parse_timestamp(fields[0])
    can_id = parse_can_id(fields[1])
    dlc = _parse_dlc(fields[2])

    flag_idx = 3 + dlc
    if len(fields) <= flag_idx:
        raise ParseError("too_few_fields", stripped)
    if any(fields[flag_idx + 1 :]):  # allow trailing empty fields from extra commas only
        raise ParseError("extra_fields", stripped)

    data = [parse_hex_byte(tok) for tok in fields[3:flag_idx]]
    flag = fields[flag_idx].upper()
    if flag not in (FLAG_REGULAR, FLAG_INJECTED):
        raise ParseError("bad_flag", repr(fields[flag_idx]))
    return (ts, can_id, dlc, pad_payload(data, missing_byte)), flag


def parse_hcrl_txt_line(line: str, missing_byte: int = MISSING_BYTE) -> ParsedRow:
    """Parse one line of the HCRL attack-free text log (``normal_run_data.txt``).

    Raises:
        ParseError: ``empty_line``, ``bad_format``, ``bad_timestamp``, ``bad_id``,
            ``bad_dlc``, ``bad_hex`` or ``dlc_mismatch``.
    """
    stripped = line.strip()
    if not stripped:
        raise ParseError("empty_line")
    match = _TXT_RE.match(stripped)
    if match is None:
        raise ParseError("bad_format", stripped)
    ts = _parse_timestamp(match["ts"])
    can_id = parse_can_id(match["id"])
    dlc = _parse_dlc(match["dlc"])
    tokens = match["data"].split()
    if len(tokens) != dlc:
        raise ParseError("dlc_mismatch", f"DLC {dlc} but {len(tokens)} bytes")
    data = [parse_hex_byte(tok) for tok in tokens]
    return ts, can_id, dlc, pad_payload(data, missing_byte)


def load_hcrl_file(
    path: str | Path,
    label: str,
    attack_type: str,
    missing_byte: int = MISSING_BYTE,
    max_rows: int | None = None,
) -> tuple[pd.DataFrame, LoadReport]:
    """Parse one HCRL file into the unified schema.

    Args:
        path: ``.csv`` attack file or ``.txt`` normal log.
        label: Unified label applied to injected (flag T) rows.
        attack_type: Fine-grained attack name applied to injected rows.
        missing_byte: Fill value for bytes beyond the DLC.
        max_rows: Stop after this many lines (quick experiments / tests).

    Returns:
        ``(dataframe, report)``.
    """
    path = Path(path)
    if label not in LABELS:
        raise ValueError(f"label must be one of {LABELS}, got {label!r}")
    report = LoadReport(SOURCE_DATASET, path.name)
    acc = FrameAccumulator(SOURCE_DATASET, path.name)
    is_txt = path.suffix.lower() == ".txt"

    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for i, line in enumerate(fh):
            if max_rows is not None and i >= max_rows:
                break
            report.rows_read += 1
            try:
                if is_txt:
                    ts, can_id, dlc, payload = parse_hcrl_txt_line(line, missing_byte)
                    row_label, row_attack = LABEL_NORMAL, ATTACK_NONE
                else:
                    (ts, can_id, dlc, payload), flag = parse_hcrl_csv_line(line, missing_byte)
                    if flag == FLAG_INJECTED:
                        row_label, row_attack = label, attack_type
                    else:
                        row_label, row_attack = LABEL_NORMAL, ATTACK_NONE
            except ParseError as err:
                report.drop(err.reason)
                continue
            acc.append(ts, can_id, dlc, payload, row_label, row_attack)

    df = acc.to_dataframe()
    report.kept_per_label = {k: int(v) for k, v in df["label"].value_counts().items() if v}
    if is_txt and label != LABEL_NORMAL:
        report.notes.append("text log has no flag column; all rows labelled Normal")
    log.info(
        "HCRL %s: read=%d kept=%d dropped=%d %s per-class=%s",
        path.name,
        report.rows_read,
        report.rows_kept,
        report.rows_dropped,
        report.dropped or "",
        report.kept_per_label,
    )
    return df, report


def load_hcrl_dir(
    raw_dir: str | Path,
    files: Mapping[str, Mapping[str, Any]] | None = None,
    missing_byte: int = MISSING_BYTE,
    max_rows: int | None = None,
) -> Iterator[tuple[pd.DataFrame | None, LoadReport]]:
    """Load every configured HCRL file found in ``raw_dir`` (searched recursively).

    Yields ``(df, report)`` per configured file; ``df`` is ``None`` (and
    ``report.skipped`` True) when the file is not present.
    """
    raw_dir = Path(raw_dir)
    files = files or DEFAULT_FILES
    for name, spec in files.items():
        matches = sorted(raw_dir.rglob(name))
        if not matches:
            report = LoadReport(SOURCE_DATASET, name, skipped=True)
            report.notes.append(f"file not found under {raw_dir}")
            log.warning("HCRL file %s not found under %s - skipped", name, raw_dir)
            yield None, report
            continue
        if len(matches) > 1:
            log.warning("HCRL: several copies of %s found, using %s", name, matches[0])
        df, report = load_hcrl_file(
            matches[0],
            label=str(spec["label"]),
            attack_type=str(spec.get("attack_type", ATTACK_NONE)),
            missing_byte=missing_byte,
            max_rows=max_rows,
        )
        yield df, report
