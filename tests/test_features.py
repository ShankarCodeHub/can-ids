"""Feature tests on small hand-made sequences with hand-computed answers."""

from __future__ import annotations

import math
import random
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from canids.ingestion.schema import CANFrame, frames_to_dataframe, pad_payload
from canids.preprocessing.features import (
    FEATURE_COLUMNS,
    FEATURE_GROUPS,
    NA_VALUE,
    PerIdStats,
    StreamingFeatureExtractor,
    extract_features,
    features_dataframe,
    fit_per_id_stats,
    pack_payload,
    payload_entropy,
)

STATS = PerIdStats(known_ids=frozenset({0x100}), period_mean={0x100: 0.2}, period_std={0x100: 0.0})


def _df(rows: list[tuple[float, int, list[int]]], label: str = "Normal") -> pd.DataFrame:
    return frames_to_dataframe(
        CANFrame(t, cid, len(p), pad_payload(p), label=label, source_file="f.log")
        for t, cid, p in rows
    )


SEQ = [(0.0, 0x100, [1, 2]), (0.1, 0x200, [255]), (0.2, 0x100, [1, 3]), (0.3, 0x100, [1, 3])]


@pytest.fixture
def feats() -> pd.DataFrame:
    return features_dataframe(_df(SEQ), STATS, window_frames=3, window_ms=150)


# ------------------------------------------------------------------ helpers
def test_payload_entropy() -> None:
    assert payload_entropy([7] * 8, 8) == 0.0
    assert payload_entropy(list(range(8)), 8) == pytest.approx(3.0)
    assert payload_entropy([1, 2, -1, -1, -1, -1, -1, -1], 2) == pytest.approx(1.0)
    assert payload_entropy([-1] * 8, 0) == 0.0


def test_pack_payload() -> None:
    assert pack_payload([0, 0, 0, 0, 0, 0, 0, 1]) == 1
    assert pack_payload([1, -1, -1, -1, -1, -1, -1, -1]) == 1 << 56


def test_feature_groups_cover_all_columns() -> None:
    grouped = [c for g in FEATURE_GROUPS.values() for c in g]
    assert sorted(grouped) == sorted(FEATURE_COLUMNS)


def test_columns_and_dtypes(feats: pd.DataFrame) -> None:
    assert list(feats.columns) == FEATURE_COLUMNS
    assert feats["can_id"].dtype == np.int32 and feats["iat_global"].dtype == np.float32


# ------------------------------------------------------------------ hand-computed values
def test_frame_level_features(feats: pd.DataFrame) -> None:
    assert feats["payload_entropy"].tolist() == pytest.approx([1.0, 0.0, 1.0, 1.0])
    assert feats["hamming_prev_id"].tolist() == [NA_VALUE, NA_VALUE, 1.0, 0.0]  # 0x02^0x03
    assert feats["is_known_id"].tolist() == [1, 0, 1, 1]
    assert feats["b1"].tolist() == [2, -1, 3, 3]


def test_timing_features(feats: pd.DataFrame) -> None:
    assert feats["iat_global"].tolist() == pytest.approx([NA_VALUE, 0.1, 0.1, 0.1])
    assert feats["iat_id"].tolist() == pytest.approx([NA_VALUE, NA_VALUE, 0.2, 0.1])
    # z = (iat - 0.2) / max(std=0, 5% of mean=0.01, 1e-4) = (0.1-0.2)/0.01 = -10
    assert feats["iat_id_z"].tolist() == pytest.approx([0.0, 0.0, 0.0, -10.0], abs=1e-4)


def test_window_features(feats: pd.DataFrame) -> None:
    h23 = -(2 / 3) * math.log2(2 / 3) - (1 / 3) * math.log2(1 / 3)
    assert feats["win_rate_fps"].tolist() == pytest.approx([0.0, 10.0, 10.0, 10.0], rel=1e-4)
    assert feats["win_unique_ids"].tolist() == [1, 2, 2, 2]
    assert feats["win_top_id_share"].tolist() == pytest.approx([1.0, 0.5, 2 / 3, 2 / 3])
    assert feats["win_cur_id_share"].tolist() == pytest.approx([1.0, 0.5, 2 / 3, 2 / 3])
    assert feats["win_id0_share"].tolist() == [0.0, 0.0, 0.0, 0.0]
    assert feats["win_iat_mean"].tolist() == pytest.approx([NA_VALUE, 0.1, 0.1, 0.1])
    assert feats["win_iat_std"].tolist() == pytest.approx([NA_VALUE, 0.0, 0.0, 0.0], abs=1e-6)
    assert feats["win_id_entropy"].tolist() == pytest.approx([0.0, 1.0, h23, h23], rel=1e-5)
    assert feats["win_t_count"].tolist() == [1, 2, 2, 2]


def test_dos_like_burst_raises_id0_share_and_rate() -> None:
    rows = [(i * 0.01, 0x100, [1]) for i in range(10)] + [
        (0.1 + i * 0.0003, 0x000, [0] * 8) for i in range(20)
    ]
    f = features_dataframe(_df(rows), STATS, window_frames=10, window_ms=50)
    assert f["win_id0_share"].iloc[-1] == pytest.approx(1.0)
    assert f["win_rate_fps"].iloc[-1] > 30 * f["win_rate_fps"].iloc[9]
    assert f["is_known_id"].iloc[-1] == 0


# ------------------------------------------------------------------ correctness of O(1) bookkeeping
def _brute_window(rows: list[tuple[float, int]], i: int, n: int) -> dict[str, float]:
    win = rows[max(0, i - n + 1) : i + 1]
    ids = [c for _, c in win]
    k = len(win)
    counts = pd.Series(ids).value_counts()
    iats = [rows[j][0] - rows[j - 1][0] for j in range(max(1, i - n + 1), i + 1)]
    p = counts / k
    return {
        "win_unique_ids": len(counts),
        "win_top_id_share": counts.max() / k,
        "win_cur_id_share": counts[rows[i][1]] / k,
        "win_id0_share": ids.count(0) / k,
        "win_id_entropy": float(-(p * np.log2(p)).sum()),
        "win_iat_mean": float(np.mean(iats)) if iats else NA_VALUE,
        "win_iat_std": float(np.std(iats)) if iats else NA_VALUE,
    }


def test_incremental_window_matches_brute_force() -> None:
    rng = random.Random(42)
    t, rows = 0.0, []
    for _ in range(600):
        t += rng.choice([0.0, 0.0003, 0.001, 0.004])
        rows.append((t, rng.choice([0, 0x10, 0x20, 0x30, 0x40, 0x7FF])))
    ex = StreamingFeatureExtractor(STATS, window_frames=25, window_ms=5)
    for i, (ts, cid) in enumerate(rows):
        values = ex.update(ts, cid, 1, [5, -1, -1, -1, -1, -1, -1, -1])
        got = dict(zip(FEATURE_COLUMNS, values, strict=True))
        for key, expected in _brute_window(rows, i, 25).items():
            assert got[key] == pytest.approx(expected, abs=1e-9), (i, key)
        t_count = sum(1 for tt, _ in rows[: i + 1] if ts - tt <= 0.005)
        assert got["win_t_count"] == t_count


def test_chunking_does_not_change_results() -> None:
    rows = [(i * 0.001, 0x100 + (i % 7), [i % 256, 3]) for i in range(57)]
    df = _df(rows)
    whole = features_dataframe(df, STATS, window_frames=10)
    chunked = pd.concat(
        list(extract_features(df, STATS, 10, 50.0, chunk_rows=8)), ignore_index=True
    )
    pd.testing.assert_frame_equal(whole, chunked)


def test_reset_forgets_history() -> None:
    ex = StreamingFeatureExtractor(STATS, window_frames=5)
    first = ex.update(0.0, 0x100, 1, [1, -1, -1, -1, -1, -1, -1, -1])
    ex.update(0.1, 0x100, 1, [1, -1, -1, -1, -1, -1, -1, -1])
    ex.reset()
    assert ex.update(0.0, 0x100, 1, [1, -1, -1, -1, -1, -1, -1, -1]) == first


def test_update_frame_matches_update() -> None:
    frame = CANFrame(1.0, 0x100, 2, pad_payload([1, 2]))
    a = StreamingFeatureExtractor(STATS).update_frame(frame)
    b = StreamingFeatureExtractor(STATS).update(1.0, 0x100, 2, list(pad_payload([1, 2])))
    assert list(a.values()) == list(b)


# ------------------------------------------------------------------ per-ID statistics
def test_fit_per_id_stats_uses_normal_frames_only() -> None:
    normal = _df([(i * 0.1, 0x100, [1]) for i in range(11)])
    attack = _df([(i * 0.1 + 0.05, 0x100, [9]) for i in range(10)], label="Spoofing")
    mixed = pd.concat([normal, attack]).sort_values("timestamp", kind="stable")
    stats = fit_per_id_stats([mixed], min_samples=5)
    assert stats.period_mean[0x100] == pytest.approx(0.1)  # injected frames ignored
    assert stats.period_std[0x100] == pytest.approx(0.0, abs=1e-9)
    assert stats.known_ids == frozenset({0x100})
    assert stats.meta["frames_used"] == 11


def test_min_samples_and_json_roundtrip(tmp_path: Path) -> None:
    df = _df(
        [(0.0, 0x1, [1]), (0.1, 0x1, [1]), (0.0, 0x2, [1])]
        + [(i * 0.01, 0x3, [1]) for i in range(10)]
    )
    stats = fit_per_id_stats([df.sort_values("timestamp", kind="stable")], min_samples=5)
    assert 0x1 not in stats.period_mean and 0x3 in stats.period_mean
    assert stats.known_ids == frozenset({1, 2, 3})
    stats.to_json(tmp_path / "s.json")
    back = PerIdStats.from_json(tmp_path / "s.json")
    assert back.known_ids == stats.known_ids
    assert back.period_mean == pytest.approx(stats.period_mean)


def test_feature_dictionary_documents_every_feature(project_root: Path) -> None:
    text = (project_root / "reports" / "feature_dictionary.md").read_text(encoding="utf-8")
    missing = [
        c for c in FEATURE_COLUMNS if f"`{c}`" not in text and c not in {f"b{i}" for i in range(8)}
    ]
    assert not missing, f"undocumented features: {missing}"
    assert "`b0`" in text
