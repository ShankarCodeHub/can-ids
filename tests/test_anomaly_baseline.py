"""Tests for the Isolation Forest detector and the rule baseline (synthetic data)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from canids.models.anomaly import AnomalyDetector, param_grid, tune_isolation_forest
from canids.models.baseline import RuleBaseline, tune_baseline

FEATS = ["f1", "f2", "f3"]


def _data(seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    train = pd.DataFrame(rng.normal(0, 1, (3000, 3)), columns=FEATS)
    val_n = pd.DataFrame(rng.normal(0, 1, (2000, 3)), columns=FEATS)
    val_a = pd.DataFrame(rng.normal(6, 1, (100, 3)), columns=FEATS)  # clear outliers
    val = pd.concat([val_n, val_a], ignore_index=True)
    y = np.r_[np.zeros(2000, bool), np.ones(100, bool)]
    return train, val, y


SEARCH = {"n_estimators": [50], "max_samples": [128, 256], "max_features": [1.0]}


def test_param_grid() -> None:
    assert len(param_grid({"a": [1, 2], "b": [3, 4, 5]})) == 6


def test_tuning_respects_fpr_and_finds_outliers(tmp_path: Path) -> None:
    train, val, y = _data()
    det, table = tune_isolation_forest(train, val, y, FEATS, SEARCH, 0.01, seed=1, n_jobs=1)
    assert len(table) == 2
    assert det.meta["val_metrics"]["fpr"] <= 0.01
    assert det.meta["val_metrics"]["recall"] > 0.95
    scores, flag = det.predict(val)
    assert scores[y].mean() > scores[~y].mean()  # higher = more anomalous
    det.save(tmp_path / "anom")
    back = AnomalyDetector.load(tmp_path / "anom")
    np.testing.assert_allclose(back.score(val), scores)
    assert back.threshold == det.threshold and back.features == FEATS


def test_tuning_is_reproducible() -> None:
    train, val, y = _data()
    a, _ = tune_isolation_forest(train, val, y, FEATS, SEARCH, 0.01, seed=7, n_jobs=1)
    b, _ = tune_isolation_forest(train, val, y, FEATS, SEARCH, 0.01, seed=7, n_jobs=1)
    np.testing.assert_allclose(a.score(val), b.score(val))


# ------------------------------------------------------------------ baseline
def _frames(rows: list[tuple[int, int, float, int]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["can_id", "is_known_id", "iat_id", "win_t_count"])


PERIOD = {0x100: 0.010}


def test_rules_fire() -> None:
    bl = RuleBaseline(rate_threshold=120, iat_ratio=0.5, period_mean=PERIOD)
    X = _frames(
        [
            (0x100, 1, 0.010, 100),  # normal
            (0x100, 1, 0.004, 100),  # too fast: 0.5*0.01/0.004 = 1.25 > 1
            (0x7FF, 0, -1.0, 100),  # unknown ID
            (0x100, 1, 0.010, 130),  # bus load 130/120 > 1
            (0x100, 1, -1.0, 100),  # first frame of the ID: interval rule silent
        ]
    )
    s, alert = bl.predict(X)
    assert alert.tolist() == [False, True, True, True, False]
    assert s[1] == pytest.approx(1.25) and s[2] == 2.0


def test_baseline_tuning_and_roundtrip(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    n = 5000
    train = _frames([(0x100, 1, 0.01, int(c)) for c in rng.integers(90, 111, n)])
    val_n = [
        (0x100, 1, float(i), int(c))
        for i, c in zip(rng.normal(0.01, 0.001, n), rng.integers(90, 111, n), strict=True)
    ]
    val_a = [(0x100, 1, 0.003, 100)] * 50  # injected: arrives 3x too fast
    val = _frames(val_n + val_a)
    y = np.r_[np.zeros(n, bool), np.ones(50, bool)]
    bl, table = tune_baseline(train, val, y, PERIOD, [0.99, 1.0], [0.3, 0.5, 0.7], 0.01)
    assert len(table) == 6
    assert bl.meta["val_metrics"]["fpr"] <= 0.01 and bl.meta["val_metrics"]["recall"] == 1.0
    bl.save(tmp_path / "b.json")
    back = RuleBaseline.load(tmp_path / "b.json")
    np.testing.assert_allclose(back.score(val), bl.score(val))
