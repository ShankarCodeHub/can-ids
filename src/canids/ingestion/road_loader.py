"""Loader for the ROAD dataset (Real ORNL Automotive Dynamometer, Oak Ridge National Lab).

Source format (per the Zenodo record, re-verified on the real files):

* Raw captures are can-utils ``candump -l`` logs, channel always ``can0``::

      (1592592355.150553) can0 0C1#0000000000000000

  Data fields are ALWAYS 8 bytes (zero-padded by the capture tool), so ROAD's
  true DLC is not recoverable; every ROAD frame gets ``dlc = 8``. This is a
  documented difference from HCRL that matters for cross-dataset features.
* Frames are NOT labelled. Each folder has a ``capture_metadata.json`` giving,
  per attack capture, the injection ID, injection data pattern and injection
  interval (elapsed seconds). ``X`` is a wildcard: in the data pattern it marks
  an unmodified nibble; as the ID it means "any ID" (fuzzing).

Frame-level labelling rule (attack captures)::

    attack  <=>  start <= t_elapsed <= end
                 AND (injection ID is a wildcard OR can_id == injection ID)
                 AND payload matches the data pattern (X = don't care)

Everything else in an attack capture is Normal; every ambient frame is Normal.
Fuzzing -> Fuzzy; targeted signal/ID injection (fabrication or masquerade) -> Spoofing.

Limitations (reported, not hidden):
* ROAD has no DoS flood attack, so DoS cannot be evaluated on ROAD.
* Accelerator attacks have no disclosed labels -> excluded.
* If a capture's metadata cannot identify injected frames precisely (e.g. both
  ID and data are full wildcards), the capture is skipped with a note rather
  than labelled by guesswork.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from canids.ingestion.schema import (
    ATTACK_NONE,
    BYTE_COLS,
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

SOURCE_DATASET = "road"

_CANDUMP_RE = re.compile(
    r"^\((?P<ts>[^)]+)\)\s+(?P<chan>\S+)\s+(?P<id>[0-9A-Fa-f]+)(?P<sep>##?)(?P<data>\S*)$"
)

DEFAULT_METADATA_KEYS = {
    "injection_id": "injection_id",
    "injection_data": "injection_data_str",
    "injection_interval": "injection_interval",
}
DEFAULT_FAMILY_MAP = {
    "fuzzing": "Fuzzy",
    "correlated_signal": "Spoofing",
    "max_speedometer": "Spoofing",
    "max_engine_coolant_temp": "Spoofing",
    "reverse_light": "Spoofing",
}

ParsedRow = tuple[float, int, int, tuple[int, ...]]


# =========================================================================== line parsing
def parse_candump_line(line: str, missing_byte: int = MISSING_BYTE) -> ParsedRow:
    """Parse one candump log line ``(ts) iface ID#DATA``.

    Args:
        line: Raw text line.
        missing_byte: Fill for byte positions beyond the payload length.

    Returns:
        ``(timestamp, can_id, dlc, payload8)``.

    Raises:
        ParseError: ``empty_line``, ``bad_format``, ``bad_timestamp``, ``bad_id``,
            ``unsupported_frame`` (CAN-FD ``##`` or remote ``R`` frames),
            ``bad_hex`` or ``bad_dlc``.
    """
    stripped = line.strip()
    if not stripped:
        raise ParseError("empty_line")
    match = _CANDUMP_RE.match(stripped)
    if match is None:
        raise ParseError("bad_format", stripped)
    try:
        ts = float(match["ts"])
    except ValueError:
        raise ParseError("bad_timestamp", match["ts"]) from None
    if not math.isfinite(ts) or ts < 0:
        raise ParseError("bad_timestamp", match["ts"])
    can_id = parse_can_id(match["id"])

    data = match["data"]
    if match["sep"] == "##" or data.upper().startswith("R"):
        raise ParseError("unsupported_frame", stripped)
    if len(data) % 2:
        raise ParseError("bad_hex", f"odd number of hex digits: {data!r}")
    if len(data) // 2 > N_BYTES:
        raise ParseError("bad_dlc", f"{len(data) // 2} bytes")
    payload = [parse_hex_byte(data[i : i + 2]) for i in range(0, len(data), 2)]
    return ts, can_id, len(payload), pad_payload(payload, missing_byte)


def parse_candump_file(
    path: str | Path,
    source_dataset: str = SOURCE_DATASET,
    source_file: str | None = None,
    missing_byte: int = MISSING_BYTE,
    max_rows: int | None = None,
) -> tuple[pd.DataFrame, LoadReport]:
    """Parse a whole candump log; every frame is provisionally labelled Normal.

    Also used by the replayer to replay arbitrary candump logs.
    """
    path = Path(path)
    name = source_file or path.name
    report = LoadReport(source_dataset, name)
    acc = FrameAccumulator(source_dataset, name)
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for i, line in enumerate(fh):
            if max_rows is not None and i >= max_rows:
                break
            report.rows_read += 1
            try:
                ts, can_id, dlc, payload = parse_candump_line(line, missing_byte)
            except ParseError as err:
                report.drop(err.reason)
                continue
            acc.append(ts, can_id, dlc, payload, LABEL_NORMAL, ATTACK_NONE)
    return acc.to_dataframe(), report


# =========================================================================== metadata
@dataclass(frozen=True)
class InjectionSpec:
    """Parsed attack metadata for one capture."""

    intervals: tuple[tuple[float, float], ...]
    can_id: int | None  # None = wildcard (any ID)
    data_pattern: str | None  # 16 chars of hex/X, None = no constraint

    @property
    def is_precise(self) -> bool:
        """True if the rule can tell injected frames apart from genuine ones."""
        return self.can_id is not None or (
            self.data_pattern is not None and set(self.data_pattern) != {"X"}
        )


def _is_wildcard(value: str) -> bool:
    return bool(value) and set(value.upper()) == {"X"}


def parse_injection_spec(entry: Mapping[str, Any], keys: Mapping[str, str]) -> InjectionSpec:
    """Turn one metadata entry into an :class:`InjectionSpec`.

    Raises:
        ValueError: if the entry lacks a usable interval or injection ID (e.g. the
            undisclosed accelerator attacks), with a human-readable reason.
    """
    raw_interval = entry.get(keys["injection_interval"])
    if raw_interval is None:
        raise ValueError("no injection interval in metadata")
    if (
        isinstance(raw_interval, Sequence)
        and len(raw_interval) == 2
        and all(isinstance(v, int | float) for v in raw_interval)
    ):
        intervals = ((float(raw_interval[0]), float(raw_interval[1])),)
    else:  # list of [start, end] pairs
        try:
            intervals = tuple((float(a), float(b)) for a, b in raw_interval)
        except (TypeError, ValueError):
            raise ValueError(f"unrecognised injection interval: {raw_interval!r}") from None
    if any(b < a for a, b in intervals):
        raise ValueError(f"interval end before start: {intervals}")

    raw_id = entry.get(keys["injection_id"])
    if raw_id is None:
        raise ValueError("injection ID is null (labels not disclosed)")
    if isinstance(raw_id, int):
        can_id: int | None = raw_id
    elif _is_wildcard(str(raw_id).strip()):
        can_id = None
    else:
        can_id = int(str(raw_id).strip(), 16)

    raw_data = entry.get(keys["injection_data"])
    pattern: str | None = None
    if raw_data is not None:
        pattern = str(raw_data).strip().upper().replace(" ", "")
        if len(pattern) > 2 * N_BYTES or not re.fullmatch(r"[0-9A-FX]*", pattern):
            raise ValueError(f"unrecognised injection data pattern: {raw_data!r}")
        pattern = pattern.ljust(2 * N_BYTES, "X")
    return InjectionSpec(intervals, can_id, pattern)


def load_metadata(path: str | Path) -> dict[str, Any]:
    """Read a ``capture_metadata.json`` file (a mapping capture-name -> entry)."""
    with Path(path).open("r", encoding="utf-8") as fh:
        meta = json.load(fh)
    if not isinstance(meta, dict):
        raise ValueError(f"{path}: expected a JSON object at top level")
    return meta


# =========================================================================== labelling
def payload_matches(df: pd.DataFrame, pattern: str) -> np.ndarray:
    """Vectorised nibble-wise match of payload bytes against a hex/X pattern."""
    payload = df[BYTE_COLS].to_numpy()
    mask = np.ones(len(df), dtype=bool)
    for nib, char in enumerate(pattern):
        if char == "X":
            continue
        byte = payload[:, nib // 2]
        value = (byte >> 4) if nib % 2 == 0 else (byte & 0x0F)
        mask &= (byte >= 0) & (value == int(char, 16))
    return mask


def label_attack_frames(
    df: pd.DataFrame,
    spec: InjectionSpec,
    label: str,
    attack_type: str,
    interval_reference: str = "first_frame",
) -> pd.DataFrame:
    """Return a copy of ``df`` with injected frames labelled ``label``/``attack_type``."""
    out = df.copy()
    if out.empty:
        return out
    ts = out["timestamp"].to_numpy()
    if interval_reference == "first_frame":
        elapsed = ts - ts.min()
    elif interval_reference == "absolute":
        elapsed = ts
    else:
        raise ValueError(f"interval_reference must be first_frame|absolute: {interval_reference}")

    in_window = np.zeros(len(out), dtype=bool)
    for start, end in spec.intervals:
        in_window |= (elapsed >= start) & (elapsed <= end)
    mask = in_window
    if spec.can_id is not None:
        mask &= out["can_id"].to_numpy() == spec.can_id
    if spec.data_pattern is not None:
        mask &= payload_matches(out, spec.data_pattern)

    labels = out["label"].astype(str).to_numpy(dtype=object)
    attacks = out["attack_type"].astype(str).to_numpy(dtype=object)
    labels[mask] = label
    attacks[mask] = attack_type
    out["label"] = pd.Categorical(labels)
    out["attack_type"] = pd.Categorical(attacks)
    return out


def classify_capture(stem: str, family_map: Mapping[str, str]) -> tuple[str, str] | None:
    """Map a capture name to ``(label, attack_type)`` via substring match, or None."""
    lower = stem.lower()
    for key, label in family_map.items():
        if key in lower:
            suffix = "_masquerade" if "masquerade" in lower else ""
            return label, f"{key}{suffix}"
    return None


# =========================================================================== directory loader
def _find_metadata(log_path: Path, raw_root: Path, file_name: str) -> Path | None:
    for directory in (log_path.parent, *log_path.parent.parents):
        candidate = directory / file_name
        if candidate.is_file():
            return candidate
        if directory == raw_root:
            break
    return None


def _metadata_entry(meta: Mapping[str, Any], log_path: Path) -> Mapping[str, Any] | None:
    for key in (log_path.stem, log_path.name):
        if key in meta and isinstance(meta[key], Mapping):
            return meta[key]
    return None


def load_road_file(
    log_path: str | Path,
    raw_root: str | Path,
    cfg: Mapping[str, Any] | None = None,
    missing_byte: int = MISSING_BYTE,
    max_rows: int | None = None,
) -> tuple[pd.DataFrame | None, LoadReport]:
    """Parse and label one ROAD capture.

    Returns ``(None, report)`` with ``report.skipped`` set when the capture is
    excluded or cannot be labelled reliably.
    """
    cfg = cfg or {}
    log_path, raw_root = Path(log_path), Path(raw_root)
    rel_name = log_path.relative_to(raw_root).as_posix()
    parts = {p.lower() for p in log_path.relative_to(raw_root).parts[:-1]}
    ambient_dir = str(cfg.get("ambient_dir", "ambient")).lower()
    attack_dir = str(cfg.get("attack_dir", "attacks")).lower()
    family_map = cfg.get("attack_family_map", DEFAULT_FAMILY_MAP)
    keys = {**DEFAULT_METADATA_KEYS, **cfg.get("metadata_keys", {})}

    def skip(reason: str) -> tuple[None, LoadReport]:
        rep = LoadReport(SOURCE_DATASET, rel_name, skipped=True)
        rep.notes.append(reason)
        log.warning("ROAD %s skipped: %s", rel_name, reason)
        return None, rep

    for pattern in cfg.get("exclude_patterns", ["accelerator"]):
        if pattern.lower() in log_path.stem.lower():
            return skip(f"excluded by pattern {pattern!r} (labels not disclosed by authors)")

    is_attack = attack_dir in parts
    if not is_attack and ambient_dir not in parts:
        return skip(f"not under '{ambient_dir}/' or '{attack_dir}/' - cannot tell ambient/attack")

    spec: InjectionSpec | None = None
    label = attack_type = None
    if is_attack:
        family = classify_capture(log_path.stem, family_map)
        if family is None:
            return skip("capture name matches no attack family in attack_family_map")
        label, attack_type = family
        if label not in LABELS:
            return skip(f"family label {label!r} not in {LABELS}")
        meta_path = _find_metadata(
            log_path, raw_root, cfg.get("metadata_file", "capture_metadata.json")
        )
        if meta_path is None:
            return skip("capture_metadata.json not found")
        entry = _metadata_entry(load_metadata(meta_path), log_path)
        if entry is None:
            return skip(f"no entry for this capture in {meta_path.name}")
        try:
            spec = parse_injection_spec(entry, keys)
        except ValueError as err:
            return skip(str(err))
        if not spec.is_precise:
            return skip("metadata cannot single out injected frames (ID and data both wildcard)")

    df, report = parse_candump_file(log_path, SOURCE_DATASET, rel_name, missing_byte, max_rows)
    if spec is not None and label is not None and attack_type is not None:
        df = label_attack_frames(
            df, spec, label, attack_type, str(cfg.get("interval_reference", "first_frame"))
        )
        n_attack = int((df["label"] != LABEL_NORMAL).sum())
        if n_attack == 0:
            report.notes.append("WARNING: 0 frames matched the injection rule - check metadata")
            log.warning("ROAD %s: no frames matched the injection rule", rel_name)
    report.kept_per_label = {k: int(v) for k, v in df["label"].value_counts().items() if v}
    log.info(
        "ROAD %s: read=%d kept=%d dropped=%d %s per-class=%s",
        rel_name,
        report.rows_read,
        report.rows_kept,
        report.rows_dropped,
        report.dropped or "",
        report.kept_per_label,
    )
    return df, report


def load_road_dir(
    raw_dir: str | Path,
    cfg: Mapping[str, Any] | None = None,
    missing_byte: int = MISSING_BYTE,
    max_rows: int | None = None,
) -> Iterator[tuple[pd.DataFrame | None, LoadReport]]:
    """Load every ``*.log`` capture under ``raw_dir`` (except ``ignore_dirs``)."""
    cfg = cfg or {}
    raw_dir = Path(raw_dir)
    ignore = {d.lower() for d in cfg.get("ignore_dirs", ["signal_extractions"])}
    logs = sorted(
        p
        for p in raw_dir.rglob("*.log")
        if not ignore & {part.lower() for part in p.relative_to(raw_dir).parts}
    )
    if not logs:
        log.warning("ROAD: no *.log captures found under %s", raw_dir)
    for log_path in logs:
        yield load_road_file(log_path, raw_dir, cfg, missing_byte, max_rows)
