"""Unified CAN frame schema shared by every loader, the replayer and the live reader.

Design decision (viva): HCRL (CSV/TXT) and ROAD (candump logs) look completely
different on disk. Converting both into ONE table layout means feature
engineering, models and the dashboard never need to know where a frame came
from. That is also what makes cross-dataset testing possible.

Columns
-------
==============  ========  ==========================================================
timestamp       float64   seconds (as recorded by the capture tool)
can_id          int32     arbitration ID (11-bit standard or 29-bit extended)
dlc             int8      number of data bytes actually present, 0..8
b0 .. b7        int16     payload bytes 0..255; bytes beyond the DLC = MISSING_BYTE
label           category  Normal | DoS | Fuzzy | Spoofing  (Unlabeled for live data)
attack_type     category  fine-grained scenario, e.g. "gear_spoofing", "none"
source_dataset  category  hcrl | road | live | replay
source_file     category  original file name (needed for per-file time splits)
==============  ========  ==========================================================

Missing payload bytes are padded with ``MISSING_BYTE = -1``, NOT 0, because 0x00
is a legitimate byte value; -1 keeps "byte absent" distinguishable (int16 is
used so that -1 fits).
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

MISSING_BYTE: int = -1
N_BYTES: int = 8
BYTE_COLS: list[str] = [f"b{i}" for i in range(N_BYTES)]

LABEL_NORMAL = "Normal"
LABEL_DOS = "DoS"
LABEL_FUZZY = "Fuzzy"
LABEL_SPOOFING = "Spoofing"
LABEL_UNLABELED = "Unlabeled"  # live / replayed traffic with unknown ground truth
LABELS: tuple[str, ...] = (LABEL_NORMAL, LABEL_DOS, LABEL_FUZZY, LABEL_SPOOFING)
ALLOWED_LABELS: frozenset[str] = frozenset((*LABELS, LABEL_UNLABELED))

ATTACK_NONE = "none"

COLUMNS: list[str] = [
    "timestamp",
    "can_id",
    "dlc",
    *BYTE_COLS,
    "label",
    "attack_type",
    "source_dataset",
    "source_file",
]

NUMERIC_DTYPES: dict[str, str] = {
    "timestamp": "float64",
    "can_id": "int32",
    "dlc": "int8",
    **{c: "int16" for c in BYTE_COLS},
}
CATEGORY_COLS: list[str] = ["label", "attack_type", "source_dataset", "source_file"]

MAX_EXTENDED_ID = 0x1FFFFFFF
MAX_STANDARD_ID = 0x7FF


class SchemaError(ValueError):
    """Raised when a DataFrame does not conform to the unified schema."""


class ParseError(ValueError):
    """Raised by line parsers; ``reason`` is a short machine-friendly tag used for drop counts."""

    def __init__(self, reason: str, detail: str = "") -> None:
        super().__init__(f"{reason}: {detail}" if detail else reason)
        self.reason = reason


# --------------------------------------------------------------------------- helpers
_HEX_BYTE = {f"{i:02x}": i for i in range(256)}
_HEX_BYTE.update({f"{i:02X}": i for i in range(256)})


def parse_hex_byte(token: str) -> int:
    """Parse a 1-2 character hex byte token such as ``"0f"`` or ``"F"``.

    Raises:
        ParseError: ``bad_hex`` if the token is not a valid byte.
    """
    tok = token.strip()
    if len(tok) == 1:
        tok = "0" + tok
    try:
        return _HEX_BYTE[tok]
    except KeyError:
        raise ParseError("bad_hex", repr(token)) from None


def parse_can_id(token: str) -> int:
    """Parse a hex CAN identifier (``"0316"``, ``"18FEF100"``, optional ``0x`` prefix).

    Raises:
        ParseError: ``bad_id`` if not hex or outside the 29-bit range.
    """
    tok = token.strip()
    try:
        value = int(tok, 16)
    except ValueError:
        raise ParseError("bad_id", repr(token)) from None
    if not 0 <= value <= MAX_EXTENDED_ID:
        raise ParseError("bad_id", f"out of range: {token!r}")
    return value


def pad_payload(data: Sequence[int], fill: int = MISSING_BYTE) -> tuple[int, ...]:
    """Pad a payload (length <= 8) to exactly 8 values with ``fill``."""
    if len(data) > N_BYTES:
        raise ParseError("bad_dlc", f"payload has {len(data)} bytes")
    return (*data, *([fill] * (N_BYTES - len(data))))


# --------------------------------------------------------------------------- record type
@dataclass(frozen=True, slots=True)
class CANFrame:
    """One CAN frame in the unified schema (used by the streaming / live path)."""

    timestamp: float
    can_id: int
    dlc: int
    payload: tuple[int, ...]  # always 8 values, MISSING_BYTE beyond dlc
    label: str = LABEL_UNLABELED
    attack_type: str = ATTACK_NONE
    source_dataset: str = "live"
    source_file: str = ""

    @property
    def data(self) -> bytes:
        """Actual payload bytes (first ``dlc`` bytes, no padding)."""
        return bytes(self.payload[: self.dlc])

    def to_row(self) -> dict[str, object]:
        """Convert to a flat dict with exactly the schema columns."""
        row: dict[str, object] = {
            "timestamp": self.timestamp,
            "can_id": self.can_id,
            "dlc": self.dlc,
        }
        row.update(zip(BYTE_COLS, self.payload, strict=True))
        row.update(
            label=self.label,
            attack_type=self.attack_type,
            source_dataset=self.source_dataset,
            source_file=self.source_file,
        )
        return row


def frames_to_dataframe(frames: Iterable[CANFrame]) -> pd.DataFrame:
    """Build a schema-conformant DataFrame from CANFrame objects."""
    rows = [f.to_row() for f in frames]
    if not rows:
        return empty_frame_df()
    return enforce_dtypes(pd.DataFrame(rows, columns=COLUMNS))


def empty_frame_df() -> pd.DataFrame:
    """Return an empty DataFrame with the unified columns and dtypes."""
    return enforce_dtypes(pd.DataFrame({c: [] for c in COLUMNS}))


def enforce_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Order columns and cast them to the schema dtypes (returns a new DataFrame)."""
    missing = [c for c in COLUMNS if c not in df.columns]
    if missing:
        raise SchemaError(f"missing columns: {missing}")
    out = df[COLUMNS].copy()
    out = out.astype(NUMERIC_DTYPES)
    for col in CATEGORY_COLS:
        out[col] = out[col].astype(str).astype("category")
    return out


def validate_frame_df(df: pd.DataFrame) -> None:
    """Check a DataFrame against the unified schema.

    Raises:
        SchemaError: describing the first violated rule.
    """
    if list(df.columns) != COLUMNS:
        raise SchemaError(f"columns must be {COLUMNS}, got {list(df.columns)}")
    for col, dtype in NUMERIC_DTYPES.items():
        if str(df[col].dtype) != dtype:
            raise SchemaError(f"column {col!r} must be {dtype}, got {df[col].dtype}")
    if df.empty:
        return
    if df["timestamp"].isna().any():
        raise SchemaError("timestamp contains NaN")
    if ((df["can_id"] < 0) | (df["can_id"] > MAX_EXTENDED_ID)).any():
        raise SchemaError("can_id out of 29-bit range")
    if ((df["dlc"] < 0) | (df["dlc"] > N_BYTES)).any():
        raise SchemaError("dlc outside 0..8")
    payload = df[BYTE_COLS].to_numpy()
    if ((payload < MISSING_BYTE) | (payload > 255)).any():
        raise SchemaError("payload byte outside -1..255")
    # A byte position < dlc must hold a real value; position >= dlc must be MISSING_BYTE
    # (ROAD always stores 8 bytes, so its dlc is 8 and nothing is MISSING).
    present = np.arange(N_BYTES)[None, :] < df["dlc"].to_numpy()[:, None]
    if (payload[present] == MISSING_BYTE).any():
        raise SchemaError("missing byte inside the DLC range")
    if (payload[~present] != MISSING_BYTE).any():
        raise SchemaError("byte present beyond the DLC")
    bad_labels = set(df["label"].astype(str).unique()) - ALLOWED_LABELS
    if bad_labels:
        raise SchemaError(f"unknown labels: {sorted(bad_labels)}")


# --------------------------------------------------------------------------- bulk builder
@dataclass
class FrameAccumulator:
    """Memory-friendly builder: collects parsed rows column-wise, emits DataFrames in chunks.

    Loaders parse millions of lines; storing them as Python tuples would need
    several GB of RAM. Column lists flushed to compact NumPy/pandas chunks keep
    memory proportional to the final DataFrame.
    """

    source_dataset: str
    source_file: str
    chunk_size: int = 500_000
    _ts: list[float] = field(default_factory=list)
    _ids: list[int] = field(default_factory=list)
    _dlc: list[int] = field(default_factory=list)
    _bytes: list[tuple[int, ...]] = field(default_factory=list)
    _label: list[str] = field(default_factory=list)
    _atype: list[str] = field(default_factory=list)
    _chunks: list[pd.DataFrame] = field(default_factory=list)

    def append(
        self,
        timestamp: float,
        can_id: int,
        dlc: int,
        payload: tuple[int, ...],
        label: str,
        attack_type: str,
    ) -> None:
        """Add one parsed frame (payload must already be padded to 8 values)."""
        self._ts.append(timestamp)
        self._ids.append(can_id)
        self._dlc.append(dlc)
        self._bytes.append(payload)
        self._label.append(label)
        self._atype.append(attack_type)
        if len(self._ts) >= self.chunk_size:
            self._flush()

    def __len__(self) -> int:
        return sum(len(c) for c in self._chunks) + len(self._ts)

    def _flush(self) -> None:
        if not self._ts:
            return
        payload = np.asarray(self._bytes, dtype=np.int16).reshape(-1, N_BYTES)
        chunk = pd.DataFrame(
            {
                "timestamp": np.asarray(self._ts, dtype=np.float64),
                "can_id": np.asarray(self._ids, dtype=np.int32),
                "dlc": np.asarray(self._dlc, dtype=np.int8),
                **{c: payload[:, i] for i, c in enumerate(BYTE_COLS)},
                "label": self._label,
                "attack_type": self._atype,
                "source_dataset": self.source_dataset,
                "source_file": self.source_file,
            }
        )
        self._chunks.append(enforce_dtypes(chunk))
        for buf in (self._ts, self._ids, self._dlc, self._bytes, self._label, self._atype):
            buf.clear()

    def to_dataframe(self) -> pd.DataFrame:
        """Return all collected frames as one schema-conformant DataFrame."""
        self._flush()
        if not self._chunks:
            return empty_frame_df()
        df = pd.concat(self._chunks, ignore_index=True)
        return enforce_dtypes(df)  # re-unify categories across chunks


# --------------------------------------------------------------------------- load report
@dataclass
class LoadReport:
    """What a loader did with one source file: kept rows per class and dropped rows per reason."""

    source_dataset: str
    source_file: str
    rows_read: int = 0
    dropped: dict[str, int] = field(default_factory=dict)
    kept_per_label: dict[str, int] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    skipped: bool = False

    def drop(self, reason: str) -> None:
        """Count one dropped row under ``reason``."""
        self.dropped[reason] = self.dropped.get(reason, 0) + 1

    @property
    def rows_kept(self) -> int:
        """Total rows written to the unified output."""
        return sum(self.kept_per_label.values())

    @property
    def rows_dropped(self) -> int:
        """Total rows discarded as malformed."""
        return sum(self.dropped.values())
