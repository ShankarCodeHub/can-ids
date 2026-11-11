"""Tests for time/capture splitting, leakage checks and imbalance helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from canids.ingestion.schema import CANFrame, frames_to_dataframe, pad_payload
from canids.preprocessing.splits import (
    PURGED,
    assign_capture_split,
    assign_splits,
    capture_name,
    chronological_split,
    class_balance_report,
    compute_class_weights,
    cross_split_duplicates,
    leakage_check,
    recording_group,
    split_index,
    undersample_normal,
)

RULES = [
    ["^max_engine_coolant_temp", "test"],
    ["_1(_masquerade)?$", "train"],
    ["_2(_masquerade)?$", "val"],
    ["_3(_masquerade)?$", "test"],
]


def _df(n: int, source: str = "f.log", labels: list[str] | None = None) -> pd.DataFrame:
    labels = labels or ["Normal"] * n
    return frames_to_dataframe(
        CANFrame(float(i), 0x100, 1, pad_payload([1]), label=labels[i], source_file=source)
        for i in range(n)
    )


def test_chronological_split_by_duration() -> None:
    s = chronological_split(np.arange(11, dtype=float))  # duration 10 -> boundaries at 6 and 8
    assert list(s) == ["train"] * 6 + ["val"] * 2 + ["test"] * 3


def test_chronological_uneven_rate_uses_time_not_counts() -> None:
    ts = np.array([0, 0.1, 0.2, 0.3, 0.4, 5, 10.0])  # most frames early
    s = chronological_split(ts)
    assert list(s) == ["train"] * 6 + ["test"]  # 6.0 and 8.0 boundaries: no val frame


def test_purge_marks_block_starts() -> None:
    s = chronological_split(np.arange(20, dtype=float), purge_frames=2)
    assert (s == PURGED).sum() == 4
    assert list(s[12:14]) == [PURGED, PURGED] and s[14] == "val"


def test_chronological_rejects_unsorted_and_bad_ratios() -> None:
    with pytest.raises(ValueError):
        chronological_split(np.array([0.0, 2.0, 1.0]))
    with pytest.raises(ValueError):
        chronological_split(np.arange(3.0), ratios=(0.5, 0.5, 0.5))


def test_capture_names_and_groups() -> None:
    assert capture_name("attacks/fuzzing_attack_1.log") == "fuzzing_attack_1"
    assert (
        recording_group("attacks__max_speedometer_attack_2_masquerade")
        == "max_speedometer_attack_2"
    )


@pytest.mark.parametrize(
    "name, split",
    [
        ("attacks/fuzzing_attack_1.log", "train"),
        ("attacks/reverse_light_on_attack_2_masquerade.log", "val"),
        ("attacks/correlated_signal_attack_3.log", "test"),
        ("attacks/max_engine_coolant_temp_attack_masquerade.log", "test"),
    ],
)
def test_capture_rules(name: str, split: str) -> None:
    assert assign_capture_split(name, RULES) == split


def test_unmatched_capture_raises() -> None:
    with pytest.raises(ValueError):
        assign_capture_split("attacks/new_attack.log", RULES)


def test_assign_splits_capture_strategy() -> None:
    df = _df(5, "attacks/fuzzing_attack_2.log")
    assert set(assign_splits(df, "capture", capture_rules=RULES)) == {"val"}


def test_leakage_check_passes_and_detects() -> None:
    a = _df(20, "ambient/a.log")
    sa = assign_splits(a, "chronological", purge_frames=1)
    b = _df(5, "attacks/x_attack_1.log")
    c = _df(5, "attacks/x_attack_1_masquerade.log")
    good = pd.concat(
        [
            split_index(a, sa, "chronological"),
            split_index(b, np.array(["train"] * 5), "capture"),
            split_index(c, np.array(["train"] * 5), "capture"),
        ]
    )
    assert leakage_check(good) == []

    bad_time = sa.copy()
    bad_time[0] = "test"  # a test frame before the training block
    bad_masq = split_index(c, np.array(["val"] * 5), "capture")
    split_in_two = split_index(b, np.array(["train"] * 3 + ["val"] * 2), "capture")
    problems = leakage_check(
        pd.concat([split_index(a, bad_time, "chronological"), split_in_two, bad_masq])
    )
    assert any("overlaps" in p for p in problems)
    assert any("capture split across" in p for p in problems)
    assert any("original/masquerade" in p for p in problems)


def test_cross_split_duplicates() -> None:
    a = _df(4, "attacks/x_attack_1.log")
    m = _df(4, "attacks/x_attack_1_masquerade.log")
    both = pd.concat([a, m], ignore_index=True)
    assert cross_split_duplicates(both, np.array(["train"] * 8)) == 0
    assert cross_split_duplicates(both, np.array(["train"] * 4 + ["test"] * 4)) == 4


def test_class_balance_report() -> None:
    labels = pd.Series(["Normal", "Normal", "DoS", "Normal", "DoS", "Normal"])
    rep = class_balance_report(labels, ["train"] * 4 + ["test"] * 2)
    assert rep.loc["train", "DoS"] == 1 and rep.loc["train", "total"] == 4
    assert rep.loc["test", "attack_share_pct"] == pytest.approx(50.0)


def test_compute_class_weights() -> None:
    w = compute_class_weights({"Normal": 90, "DoS": 10, "Fuzzy": 0})
    assert w == pytest.approx({"Normal": 100 / (2 * 90), "DoS": 100 / (2 * 10)})


def test_undersample_normal_keeps_attacks_and_order() -> None:
    labels = ["Normal"] * 50 + ["DoS"] * 5 + ["Normal"] * 45
    df = _df(100, labels=labels)
    out = undersample_normal(df, ratio=2.0, seed=0)
    assert (out["label"] == "DoS").sum() == 5
    assert (out["label"] == "Normal").sum() == 10
    assert out["timestamp"].is_monotonic_increasing
    again = undersample_normal(df, ratio=2.0, seed=0)
    assert out.index.equals(again.index)  # seeded -> reproducible
