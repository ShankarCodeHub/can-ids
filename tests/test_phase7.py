"""Phase 7 tests: cross-dataset helpers, ablation summary, runtime helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from canids.evaluation.ablation import ABLATIONS, summarise
from canids.evaluation.cross_dataset import common_features, map_labels, train_and_test
from canids.evaluation.runtime import can_bus_capacity, model_size_bytes
from canids.preprocessing.features import FEATURE_COLUMNS, FEATURE_GROUPS


def test_common_features_drop_vehicle_specific() -> None:
    kept, dropped = common_features(FEATURE_COLUMNS)
    assert "can_id" in dropped and "b0" in dropped and "dlc" in dropped
    assert "is_known_id" in dropped and "iat_id_z" in dropped
    assert "win_rate_fps" in kept and "payload_entropy" in kept and "iat_id" in kept
    assert sorted(kept + list(dropped)) == sorted(FEATURE_COLUMNS)


def test_map_labels() -> None:
    y = np.array(["Normal", "DoS", "Fuzzy", "Spoofing"])
    m, keep = map_labels(y, "multiclass")
    assert keep.tolist() == [True, False, True, True]
    b, keep_b = map_labels(y, "binary")
    assert b.tolist() == ["Normal", "Attack", "Attack", "Attack"] and keep_b.all()
    with pytest.raises(ValueError):
        map_labels(y, "other")


def _synthetic(seed: int, shift: float) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = 600
    df = pd.DataFrame({"a": rng.normal(0, 1, n), "b": rng.normal(0, 1, n)})
    lab = np.array(["Normal"] * 500 + ["Spoofing"] * 50 + ["DoS"] * 50)
    df.loc[500:549, "a"] += 4 + shift
    df.loc[550:, "b"] += 4
    df["label"] = lab
    return df


def test_train_and_test_shows_transfer_gap() -> None:
    src, tgt = _synthetic(0, 0.0), _synthetic(1, -4.0)  # target attacks look different
    params = {"n_estimators": 20, "max_depth": 3}
    within = train_and_test(src, _synthetic(2, 0.0), ["a", "b"], "multiclass", "xgboost", params, 0)
    cross = train_and_test(src, tgt, ["a", "b"], "multiclass", "xgboost", params, 0)
    assert within["train_classes"] == ["Normal", "Spoofing"]  # DoS removed in multiclass
    assert within["macro_f1"] > cross["macro_f1"]
    binary = train_and_test(src, tgt, ["a", "b"], "binary", "xgboost", params, 0)
    assert binary["train_classes"] == ["Normal", "Attack"]
    assert "roc_auc" in binary


def test_ablation_definitions_and_summary() -> None:
    assert set(ABLATIONS) == {"full", "no_iforest", "no_timing", "no_window", "no_payload"}
    assert ABLATIONS["no_payload"]["drop"] == FEATURE_GROUPS["payload"]
    y = np.array(["Normal"] * 6 + ["Spoofing"] * 4)
    lab = np.array(["Normal"] * 5 + ["Unknown"] + ["Spoofing"] * 3 + ["Normal"])
    at = np.array(["none"] * 6 + ["x"] * 2 + ["y"] * 2)
    row, det = summarise("v", y, lab, at, ["Normal", "Spoofing"])
    assert row["detection_recall"] == pytest.approx(0.75)
    assert row["alarm_fpr"] == pytest.approx(1 / 6)
    assert row["unknown_frames"] == 1
    assert det.set_index("attack_type").loc["y", "detection_rate"] == 0.5


def test_can_bus_capacity() -> None:
    cap = can_bus_capacity(500_000)
    assert cap["dlc8_worst_stuffing_fps"] < cap["dlc8_no_stuffing_fps"] < cap["dlc0_fps"]
    assert cap["dlc8_no_stuffing_fps"] == pytest.approx(500_000 / 111)


def test_model_size(tmp_path) -> None:
    (tmp_path / "m").mkdir()
    (tmp_path / "m" / "a.bin").write_bytes(b"x" * 10)
    (tmp_path / "m" / "b.json").write_bytes(b"x" * 5)
    assert model_size_bytes(tmp_path / "m") == 15
