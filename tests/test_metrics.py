"""Tests for evaluation metrics."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from canids.evaluation.metrics import (
    binary_metrics,
    curve_points,
    detection_rate_by_group,
    false_positive_rate_by_group,
    plot_roc_pr,
    recall_at_fpr,
    save_json,
    threshold_at_fpr,
)


def test_threshold_at_fpr_bounds_false_positives() -> None:
    normal = np.arange(1000, dtype=float)
    thr = threshold_at_fpr(normal, 0.01)
    assert (normal > thr).mean() <= 0.01
    assert (normal > thr).mean() >= 0.009


def test_threshold_validation() -> None:
    with pytest.raises(ValueError):
        threshold_at_fpr(np.arange(3.0), 0.0)
    with pytest.raises(ValueError):
        threshold_at_fpr(np.array([]), 0.1)


def test_binary_metrics_hand_example() -> None:
    y = np.array([1, 1, 1, 0, 0, 0, 0, 0])
    p = np.array([1, 1, 0, 1, 0, 0, 0, 0])
    m = binary_metrics(y, p, scores=np.array([0.9, 0.8, 0.3, 0.7, 0.2, 0.1, 0.1, 0.0]))
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (2, 1, 1, 4)
    assert m["precision"] == pytest.approx(2 / 3)
    assert m["recall"] == pytest.approx(2 / 3)
    assert m["fpr"] == pytest.approx(1 / 5)
    assert m["f1"] == pytest.approx(2 / 3)
    # ranking: 0.9, 0.8 > 0.7(neg) > 0.3(pos) -> 14 of 15 positive/negative pairs ordered
    assert m["roc_auc"] == pytest.approx(14 / 15)
    assert 0 < m["pr_auc"] <= 1


def test_binary_metrics_degenerate_no_auc() -> None:
    m = binary_metrics(np.zeros(4), np.zeros(4), scores=np.zeros(4))
    assert m["precision"] == 0.0 and "roc_auc" not in m


def test_recall_at_fpr() -> None:
    y = np.array([0] * 100 + [1] * 10)
    s = np.concatenate([np.linspace(0, 1, 100), np.full(10, 5.0)])
    recall, thr = recall_at_fpr(y, s, 0.05)
    assert recall == 1.0 and thr < 5.0


def test_group_tables() -> None:
    y = np.array([1, 1, 1, 0, 0, 0])
    p = np.array([1, 0, 1, 1, 0, 0])
    g = np.array(["dos", "dos", "fuzzy", "amb", "amb", "atk"])
    d = detection_rate_by_group(y, p, g).set_index("group")
    assert d.loc["dos", "detection_rate"] == 0.5 and d.loc["fuzzy", "detection_rate"] == 1.0
    f = false_positive_rate_by_group(y, p, g).set_index("group")
    assert f.loc["amb", "fpr"] == 0.5 and f.loc["atk", "fpr"] == 0.0


def test_curves_plot_and_json(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    y = np.r_[np.zeros(500), np.ones(50)]
    s = np.r_[rng.normal(0, 1, 500), rng.normal(2, 1, 50)]
    c = curve_points(y, s, max_points=100)
    assert len(c["roc"]["fpr"]) <= 100 and c["roc"]["tpr"][-1] == 1.0
    plot_roc_pr({"m": c}, tmp_path / "f.png", "t", fpr_marker=0.01)
    assert (tmp_path / "f.png").stat().st_size > 0
    save_json({"a": np.float32(1.5), "b": np.arange(2)}, tmp_path / "m.json")
    assert '"a": 1.5' in (tmp_path / "m.json").read_text()
