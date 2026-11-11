"""Tests for the stage-2 classifier, the hybrid decision rules and multiclass metrics."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from canids.evaluation.metrics import (
    alarm_metrics,
    expected_calibration_error,
    multiclass_metrics,
    plot_confusion_matrix,
)
from canids.models.classifier import AttackClassifier, balanced_sample_weight, search_classifiers
from canids.models.hybrid import UNKNOWN, HybridThresholds, hybrid_decide, tune_hybrid

CLASSES = ["Normal", "Fuzzy", "Spoofing"]


# ------------------------------------------------------------------ hybrid rules
def test_each_rule_fires_as_documented() -> None:
    th = HybridThresholds(t_normal=0.9, t_attack=0.7, t_suppress=0.95)
    is_anom = np.array([False, True, False, True, True, False])
    proba = np.array(
        [
            [0.95, 0.03, 0.02],  # 1: quiet + confident Normal       -> Normal (rule 1)
            [0.10, 0.80, 0.10],  # 2: confident Fuzzy (anomalous)     -> Fuzzy (rule 2)
            [0.20, 0.05, 0.75],  # 2: confident Spoofing, IF quiet    -> Spoofing (rule 2)
            [0.97, 0.02, 0.01],  # 3: anomaly overruled by P(N)>=0.95 -> Normal (rule 3)
            [0.50, 0.30, 0.20],  # 4: anomaly, nobody confident       -> Unknown (rule 4)
            [0.60, 0.25, 0.15],  # 5: quiet, unsure                   -> Normal low-conf (rule 5)
        ]
    )
    labels, rule = hybrid_decide(is_anom, proba, CLASSES, th)
    assert labels.tolist() == ["Normal", "Fuzzy", "Spoofing", "Normal", UNKNOWN, "Normal"]
    assert rule.tolist() == [1, 2, 2, 3, 4, 5]


def test_disabled_overrule_gives_unknown() -> None:
    th = HybridThresholds(t_suppress=1.01)
    labels, rule = hybrid_decide(np.array([True]), np.array([[0.99, 0.005, 0.005]]), CLASSES, th)
    assert labels[0] == UNKNOWN and rule[0] == 4


def test_unseen_attack_becomes_unknown_not_normal() -> None:
    """Classifier thinks 'Normal' (never saw this attack) but stage 1 fires -> Unknown."""
    labels, _ = hybrid_decide(
        np.array([True]), np.array([[0.80, 0.10, 0.10]]), CLASSES, HybridThresholds()
    )
    assert labels[0] == UNKNOWN


def test_tune_hybrid_respects_fpr_budget() -> None:
    rng = np.random.default_rng(0)
    n_norm, n_att = 2000, 100
    y = np.array(["Normal"] * n_norm + ["Spoofing"] * n_att)
    is_anom = np.r_[rng.random(n_norm) < 0.05, np.ones(n_att, bool)]
    p_sp = np.r_[rng.uniform(0, 0.3, n_norm), rng.uniform(0.6, 1.0, n_att)]
    proba = np.column_stack([1 - p_sp, np.zeros_like(p_sp), p_sp])
    th, table = tune_hybrid(
        is_anom, proba, CLASSES, y, np.ones(len(y)), 0.9, [0.5, 0.7, 0.9], [0.5, 0.9, 1.01], 0.01
    )
    best = table[(table.t_attack == th.t_attack) & (table.t_suppress == th.t_suppress)].iloc[0]
    assert best["alarm_fpr"] <= 0.01
    assert len(table) == 9


# ------------------------------------------------------------------ metrics
def test_multiclass_metrics_with_unknown_and_weights() -> None:
    y = np.array(["Normal", "Normal", "Fuzzy", "Spoofing"])
    p = np.array(["Normal", UNKNOWN, "Fuzzy", "Normal"])
    m = multiclass_metrics(y, p, CLASSES, extra_pred_labels=(UNKNOWN,))
    assert m["per_class"]["Fuzzy"]["f1"] == 1.0
    assert m["per_class"]["Spoofing"]["recall"] == 0.0
    assert m["per_class"]["Normal"]["precision"] == pytest.approx(0.5)
    assert m["confusion_matrix"]["labels_pred"] == [*CLASSES, UNKNOWN]
    assert m["confusion_matrix"]["values"][0] == [1, 0, 0, 1]
    w = multiclass_metrics(y, p, CLASSES, sample_weight=np.array([10, 1, 1, 1]))
    assert w["per_class"]["Normal"]["recall"] == pytest.approx(10 / 11)


def test_alarm_metrics() -> None:
    y = np.array(["Normal"] * 4 + ["Fuzzy"] * 2)
    p = np.array(["Normal", "Normal", "Normal", UNKNOWN, "Fuzzy", "Normal"])
    m = alarm_metrics(y, p, "Normal")
    assert m == pytest.approx({"detection_recall": 0.5, "alarm_fpr": 0.25, "alarm_precision": 0.5})


def test_ece_perfect_and_overconfident() -> None:
    y = np.array([0, 1, 0, 1])
    assert expected_calibration_error(y, np.eye(2)[y]) == pytest.approx(0.0)
    wrong = np.eye(2)[1 - y] * 0.98 + 0.01
    assert expected_calibration_error(y, wrong) > 0.9


def test_confusion_plot(tmp_path: Path) -> None:
    m = multiclass_metrics(
        np.array(["Normal", "Fuzzy"]),
        np.array(["Normal", UNKNOWN]),
        CLASSES,
        extra_pred_labels=(UNKNOWN,),
    )
    plot_confusion_matrix(m["confusion_matrix"], tmp_path / "cm.png", "t")
    assert (tmp_path / "cm.png").stat().st_size > 0


# ------------------------------------------------------------------ classifier
def test_balanced_sample_weight() -> None:
    w = balanced_sample_weight(np.array([0, 0, 0, 1]))
    assert w[:3].sum() == pytest.approx(w[3])


def _synthetic(seed: int) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    n = {"Normal": 1500, "Fuzzy": 60, "Spoofing": 60}
    centers = {"Normal": 0.0, "Fuzzy": 4.0, "Spoofing": -4.0}
    X = pd.concat(
        [
            pd.DataFrame(rng.normal(centers[c], 1, (k, 3)), columns=["a", "b", "c"])
            for c, k in n.items()
        ],
        ignore_index=True,
    )
    y = np.concatenate([[c] * k for c, k in n.items()])
    return X, y


@pytest.mark.parametrize("kind", ["random_forest", "xgboost"])
def test_search_and_roundtrip(kind: str, tmp_path: Path) -> None:
    Xtr, ytr = _synthetic(0)
    Xv, yv = _synthetic(1)
    space = {
        "random_forest": {"n_estimators": [20], "max_depth": [None, 5]},
        "xgboost": {"n_estimators": [30], "max_depth": [3], "learning_rate": [0.3]},
    }
    clf, table = search_classifiers(
        Xtr, ytr, Xv, yv, np.ones(len(yv)), ["a", "b", "c"], space, [kind], 2, seed=0, n_jobs=1
    )
    assert clf.classes == ["Normal", "Fuzzy", "Spoofing"]
    assert clf.meta["val_metrics"]["macro_f1"] > 0.9
    proba = clf.predict_proba(Xv)
    assert proba.shape == (len(Xv), 3) and np.allclose(proba.sum(axis=1), 1)
    clf.save(tmp_path / "c")
    back = AttackClassifier.load(tmp_path / "c")
    np.testing.assert_allclose(back.predict_proba(Xv), proba)
    assert set(table["model"]) == {kind}
