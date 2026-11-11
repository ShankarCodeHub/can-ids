"""Streaming-compatible feature engineering (one code path for offline AND real time).

Design decision (viva): a classic IDS bug is "train/serve skew", where features
are computed one way in a training notebook (vectorised pandas) and another way
in the live system, so the deployed model sees different numbers than it was
trained on. Here there is ONE implementation, :class:`StreamingFeatureExtractor`,
which consumes frames one at a time using only the past. Offline,
:func:`extract_features` simply feeds a recorded file through it; in Phase 6 the
real-time pipeline feeds live frames through the very same object.

Every feature is CAUSAL (uses the current and earlier frames only), so no future
information can leak into a prediction. All window statistics are maintained
incrementally in O(1) per frame (no re-scanning of the window), which keeps the
extractor fast enough for real-time use.

Per-ID statistics (known IDs, normal period per ID) are learned from the
TRAINING split's Normal frames only (:func:`fit_per_id_stats`) and saved as an
artifact; validation/test/live data only *use* them.

Sentinels: values that do not exist yet (e.g. inter-arrival time for the first
frame of an ID) are ``NA_VALUE = -1``. Every real value of those features is
>= 0, so -1 is unambiguous for tree models.
"""

from __future__ import annotations

import json
import math
from collections import deque
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from canids.ingestion.schema import BYTE_COLS, LABEL_NORMAL, MISSING_BYTE, N_BYTES, CANFrame

FEATURE_VERSION = "1.0"
NA_VALUE = -1.0
Z_CLIP = 100.0
RATE_CLIP = 1e6
MIN_SPAN_S = 1e-6

FRAME_FEATURES: list[str] = [
    "can_id",
    "dlc",
    *BYTE_COLS,
    "payload_entropy",
    "hamming_prev_id",
    "is_known_id",
]
TIMING_FEATURES: list[str] = ["iat_global", "iat_id", "iat_id_z"]
WINDOW_FEATURES: list[str] = [
    "win_rate_fps",
    "win_unique_ids",
    "win_top_id_share",
    "win_id0_share",
    "win_cur_id_share",
    "win_iat_mean",
    "win_iat_std",
    "win_id_entropy",
    "win_t_count",
]
FEATURE_COLUMNS: list[str] = [*FRAME_FEATURES, *TIMING_FEATURES, *WINDOW_FEATURES]

#: Feature groups used by the Phase 7 ablation study.
FEATURE_GROUPS: dict[str, list[str]] = {
    "identity": ["can_id", "dlc", "is_known_id"],
    "payload": [*BYTE_COLS, "payload_entropy", "hamming_prev_id"],
    "timing": TIMING_FEATURES,
    "window": WINDOW_FEATURES,
}

INT_FEATURES: dict[str, str] = {
    "can_id": "int32",
    "dlc": "int8",
    **{c: "int16" for c in BYTE_COLS},
    "is_known_id": "int8",
    "win_unique_ids": "int16",
    "win_t_count": "int32",
}


# =========================================================================== per-ID statistics
@dataclass
class PerIdStats:
    """Statistics of NORMAL training traffic, per CAN ID.

    Attributes:
        known_ids: IDs observed in Normal training frames.
        period_mean: Mean inter-arrival time (s) of each ID in Normal training traffic.
        period_std: Standard deviation of that inter-arrival time (s).
        meta: Provenance (dataset, split, frame counts, feature version).
    """

    known_ids: frozenset[int] = frozenset()
    period_mean: dict[int, float] = field(default_factory=dict)
    period_std: dict[int, float] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)

    def to_json(self, path: str | Path) -> None:
        """Save as human-readable JSON (IDs as hex strings)."""
        data = {
            "feature_version": FEATURE_VERSION,
            "meta": self.meta,
            "known_ids": sorted(f"0x{i:X}" for i in self.known_ids),
            "period_mean_s": {f"0x{k:X}": v for k, v in sorted(self.period_mean.items())},
            "period_std_s": {f"0x{k:X}": v for k, v in sorted(self.period_std.items())},
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")

    @classmethod
    def from_json(cls, path: str | Path) -> PerIdStats:
        """Load statistics saved by :meth:`to_json`."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            known_ids=frozenset(int(k, 16) for k in data["known_ids"]),
            period_mean={int(k, 16): float(v) for k, v in data["period_mean_s"].items()},
            period_std={int(k, 16): float(v) for k, v in data["period_std_s"].items()},
            meta=data.get("meta", {}),
        )


class PerIdStatsAccumulator:
    """Accumulates per-ID period statistics file by file (memory-bounded).

    Feed each source file's TRAIN frames with :meth:`add_file`; inter-arrival
    times are computed within a file between consecutive *Normal* frames of the
    same ID, so injected frames never distort the learned normal period.
    """

    def __init__(self) -> None:
        self._count: dict[int, int] = {}
        self._sum: dict[int, float] = {}
        self._sumsq: dict[int, float] = {}
        self._known: set[int] = set()
        self.frames_used = 0
        self.files_used = 0

    def add_file(self, frames: pd.DataFrame) -> None:
        """Add one file's training frames (any labels; only Normal ones are used)."""
        normal = frames[frames["label"].astype(str) == LABEL_NORMAL]
        if normal.empty:
            return
        self.files_used += 1
        self.frames_used += len(normal)
        self._known.update(int(i) for i in normal["can_id"].unique())
        iat = normal.groupby("can_id", observed=True)["timestamp"].diff()
        tmp = pd.DataFrame({"can_id": normal["can_id"].to_numpy(), "iat": iat.to_numpy()})
        agg = tmp.dropna().groupby("can_id")["iat"].agg(["count", "sum", lambda s: (s * s).sum()])
        for cid, (cnt, sm, sq) in zip(agg.index, agg.to_numpy(), strict=True):
            cid = int(cid)
            self._count[cid] = self._count.get(cid, 0) + int(cnt)
            self._sum[cid] = self._sum.get(cid, 0.0) + float(sm)
            self._sumsq[cid] = self._sumsq.get(cid, 0.0) + float(sq)

    def finalize(self, min_samples: int = 5, meta: dict[str, Any] | None = None) -> PerIdStats:
        """Produce :class:`PerIdStats`; IDs with fewer than ``min_samples`` gaps get no period."""
        mean, std = {}, {}
        for cid, n in self._count.items():
            if n < min_samples:
                continue
            m = self._sum[cid] / n
            var = max(self._sumsq[cid] / n - m * m, 0.0)
            mean[cid], std[cid] = m, math.sqrt(var)
        info = {"frames_used": self.frames_used, "files_used": self.files_used, **(meta or {})}
        return PerIdStats(frozenset(self._known), mean, std, info)


def fit_per_id_stats(
    train_files: Iterable[pd.DataFrame], min_samples: int = 5, meta: dict[str, Any] | None = None
) -> PerIdStats:
    """Fit :class:`PerIdStats` from an iterable of per-file TRAIN DataFrames."""
    acc = PerIdStatsAccumulator()
    for df in train_files:
        acc.add_file(df)
    return acc.finalize(min_samples, meta)


# =========================================================================== extractor
# -p*log2(p) contributions for byte entropy: _BYTE_ENT[n][c] with c occurrences out of n bytes
_BYTE_ENT = [[0.0] * (N_BYTES + 1)] + [
    [0.0 if c == 0 else -(c / n) * math.log2(c / n) for c in range(N_BYTES + 1)]
    for n in range(1, N_BYTES + 1)
]


def payload_entropy(payload: Iterable[int], dlc: int) -> float:
    """Shannon entropy (bits) of the byte values in the first ``dlc`` bytes (0..3 bits)."""
    if dlc <= 0:
        return 0.0
    vals = sorted(list(payload)[:dlc])
    table = _BYTE_ENT[dlc]
    ent, run = 0.0, 1
    for i in range(1, dlc):
        if vals[i] == vals[i - 1]:
            run += 1
        else:
            ent += table[run]
            run = 1
    return ent + table[run]


def pack_payload(payload: Iterable[int]) -> int:
    """Pack 8 payload values into one 64-bit int (missing bytes count as 0x00)."""
    packed = 0
    for b in payload:
        packed = (packed << 8) | (b if b > 0 else 0)
    return packed


class StreamingFeatureExtractor:
    """Computes :data:`FEATURE_COLUMNS` for one frame at a time, using only past frames.

    Args:
        stats: Per-ID statistics fitted on the training split.
        window_frames: Size N of the count-based rolling window (last N frames).
        window_ms: Length of the time-based window used for ``win_t_count``.

    Call :meth:`reset` between unrelated captures (state must not carry across files).
    """

    def __init__(self, stats: PerIdStats, window_frames: int = 100, window_ms: float = 50.0):
        if window_frames < 2:
            raise ValueError("window_frames must be >= 2")
        self.stats = stats
        self.window_frames = window_frames
        self.window_s = window_ms / 1000.0
        self._known = stats.known_ids
        # std floor avoids huge z-scores for IDs with near-perfect periodicity
        self._period = {
            cid: (m, max(stats.period_std.get(cid, 0.0), 0.05 * m, 1e-4))
            for cid, m in stats.period_mean.items()
        }
        self._clogc = [0.0] + [c * math.log2(c) for c in range(1, window_frames + 2)]
        self.reset()

    def reset(self) -> None:
        """Forget all history (start of a new capture / stream)."""
        self._prev_ts: float | None = None
        self._last_ts: dict[int, float] = {}
        self._last_packed: dict[int, int] = {}
        self._win: deque[tuple[int, float, float]] = deque()  # (can_id, ts, iat or NaN)
        self._counts: dict[int, int] = {}
        self._hist = [0] * (self.window_frames + 2)  # number of IDs having each count
        self._max_count = 0
        self._sum_clogc = 0.0
        self._n_zero = 0
        self._iat_n = 0
        self._iat_sum = 0.0
        self._iat_sq = 0.0
        self._evictions = 0
        self._tw: deque[float] = deque()
        self.frames_seen = 0

    # ------------------------------------------------------------------ window bookkeeping
    def _add(self, cid: int, ts: float, iat: float) -> None:
        self._win.append((cid, ts, iat))
        c = self._counts.get(cid, 0)
        if c:
            self._hist[c] -= 1
        self._hist[c + 1] += 1
        self._sum_clogc += self._clogc[c + 1] - self._clogc[c]
        self._counts[cid] = c + 1
        if c + 1 > self._max_count:
            self._max_count = c + 1
        if cid == 0:
            self._n_zero += 1
        if iat == iat:  # not NaN
            self._iat_n += 1
            self._iat_sum += iat
            self._iat_sq += iat * iat

    def _evict(self) -> None:
        cid, _, iat = self._win.popleft()
        c = self._counts[cid]
        self._hist[c] -= 1
        if c > 1:
            self._hist[c - 1] += 1
            self._counts[cid] = c - 1
        else:
            del self._counts[cid]
        self._sum_clogc += self._clogc[c - 1] - self._clogc[c]
        if c == self._max_count and self._hist[c] == 0:
            self._max_count = c - 1
        if cid == 0:
            self._n_zero -= 1
        if iat == iat:
            self._iat_n -= 1
            self._iat_sum -= iat
            self._iat_sq -= iat * iat
        self._evictions += 1
        if self._evictions % 100_000 == 0:  # cancel floating-point drift of running sums
            vals = [v for _, _, v in self._win if v == v]
            self._iat_sum = math.fsum(vals)
            self._iat_sq = math.fsum(v * v for v in vals)

    # ------------------------------------------------------------------ main entry
    def update(self, ts: float, can_id: int, dlc: int, payload: list[int]) -> tuple:
        """Ingest one frame and return its feature tuple (order = FEATURE_COLUMNS).

        Args:
            ts: Timestamp in seconds (non-decreasing within a stream).
            can_id: Arbitration ID.
            dlc: Number of data bytes.
            payload: 8 byte values, MISSING_BYTE (-1) beyond ``dlc``.
        """
        nan = math.nan
        # ---- timing
        iat_g = ts - self._prev_ts if self._prev_ts is not None else nan
        last = self._last_ts.get(can_id)
        if last is not None:
            iat_id = ts - last
            per = self._period.get(can_id)
            if per is not None:
                z = (iat_id - per[0]) / per[1]
                z = Z_CLIP if z > Z_CLIP else (-Z_CLIP if z < -Z_CLIP else z)
            else:
                z = 0.0
        else:
            iat_id, z = NA_VALUE, 0.0

        # ---- payload
        packed = pack_payload(payload)
        prev_packed = self._last_packed.get(can_id)
        hamming = float((packed ^ prev_packed).bit_count()) if prev_packed is not None else NA_VALUE
        ent = payload_entropy(payload, dlc)

        # ---- count window
        self._add(can_id, ts, iat_g)
        if len(self._win) > self.window_frames:
            self._evict()
        k = len(self._win)
        span = ts - self._win[0][1]
        if k > 1:
            rate = (k - 1) / (span if span > MIN_SPAN_S else MIN_SPAN_S)
            rate = RATE_CLIP if rate > RATE_CLIP else rate
        else:
            rate = 0.0
        if self._iat_n:
            m = self._iat_sum / self._iat_n
            var = self._iat_sq / self._iat_n - m * m
            iat_mean, iat_std = m, math.sqrt(var) if var > 0 else 0.0
        else:
            iat_mean = iat_std = NA_VALUE
        id_ent = math.log2(k) - self._sum_clogc / k
        id_ent = id_ent if id_ent > 0 else 0.0

        # ---- time window
        tw = self._tw
        tw.append(ts)
        while ts - tw[0] > self.window_s:
            tw.popleft()

        # ---- remember state for the next frame
        self._prev_ts = ts
        self._last_ts[can_id] = ts
        self._last_packed[can_id] = packed
        self.frames_seen += 1

        return (
            can_id,
            dlc,
            *payload,
            ent,
            hamming,
            1 if can_id in self._known else 0,
            iat_g if iat_g == iat_g else NA_VALUE,
            iat_id,
            z,
            rate,
            len(self._counts),
            self._max_count / k,
            self._n_zero / k,
            self._counts[can_id] / k,
            iat_mean,
            iat_std,
            id_ent,
            len(tw),
        )

    def update_frame(self, frame: CANFrame) -> dict[str, float]:
        """Convenience wrapper for the real-time path: CANFrame -> {feature: value}."""
        values = self.update(frame.timestamp, frame.can_id, frame.dlc, list(frame.payload))
        return dict(zip(FEATURE_COLUMNS, values, strict=True))


# =========================================================================== batch driver
def _to_frame(block: np.ndarray) -> pd.DataFrame:
    out = pd.DataFrame(block, columns=FEATURE_COLUMNS)
    for col in FEATURE_COLUMNS:
        out[col] = out[col].astype(INT_FEATURES.get(col, "float32"))
    return out


def extract_features(
    frames: pd.DataFrame,
    stats: PerIdStats,
    window_frames: int = 100,
    window_ms: float = 50.0,
    chunk_rows: int = 250_000,
) -> Iterator[pd.DataFrame]:
    """Feed ONE cleaned, time-sorted capture through a fresh extractor.

    Yields feature DataFrames of at most ``chunk_rows`` rows, aligned with
    ``frames`` row order (chunking only bounds memory; extractor state flows
    across chunks).
    """
    ex = StreamingFeatureExtractor(stats, window_frames, window_ms)
    ts = frames["timestamp"].to_numpy(dtype=np.float64)
    ids = frames["can_id"].to_numpy(dtype=np.int64)
    dlc = frames["dlc"].to_numpy(dtype=np.int64)
    payload_all = frames[BYTE_COLS].to_numpy(dtype=np.int64)
    n_feat = len(FEATURE_COLUMNS)
    update = ex.update
    for start in range(0, len(frames), chunk_rows):
        stop = min(start + chunk_rows, len(frames))
        ts_c = ts[start:stop].tolist()
        id_c = ids[start:stop].tolist()
        dlc_c = dlc[start:stop].tolist()
        pay_c = payload_all[start:stop].tolist()
        block = np.empty((stop - start, n_feat), dtype=np.float64)
        for i in range(stop - start):
            block[i] = update(ts_c[i], id_c[i], dlc_c[i], pay_c[i])
        yield _to_frame(block)


def features_dataframe(
    frames: pd.DataFrame, stats: PerIdStats, window_frames: int = 100, window_ms: float = 50.0
) -> pd.DataFrame:
    """Like :func:`extract_features` but returns one DataFrame (small inputs / tests)."""
    parts = list(extract_features(frames, stats, window_frames, window_ms))
    if not parts:
        return _to_frame(np.empty((0, len(FEATURE_COLUMNS))))
    return pd.concat(parts, ignore_index=True)


__all__ = [
    "FEATURE_COLUMNS",
    "FEATURE_GROUPS",
    "FEATURE_VERSION",
    "MISSING_BYTE",
    "NA_VALUE",
    "PerIdStats",
    "PerIdStatsAccumulator",
    "StreamingFeatureExtractor",
    "extract_features",
    "features_dataframe",
    "fit_per_id_stats",
    "pack_payload",
    "payload_entropy",
]
