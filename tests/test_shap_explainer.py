"""Tests for the SHAP explainer (native XGBoost TreeSHAP backend; shap package optional)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from xgboost import XGBClassifier

from canids.explain.shap_explainer import (
    CAUSALITY_NOTE,
    AlertExplainer,
    describe_feature,
    plot_beeswarm,
    plot_explanation,
    plot_global_bar,
    reason_text,
)
from canids.models.classifier import AttackClassifier

FEATS = ["iat_id_z", "payload_entropy", "b0", "can_id"]


@pytest.fixture(scope="module")
def clf() -> tuple[AttackClassifier, pd.DataFrame]:
    rng = np.random.default_rng(0)
    n = 600
    X = pd.DataFrame(
        {
            "iat_id_z": rng.normal(0, 1, n),
            "payload_entropy": rng.uniform(0, 3, n),
            "b0": rng.integers(0, 256, n).astype(float),
            "can_id": rng.choice([0x100, 0x200], n).astype(float),
        }
    )
    y = np.where(X["payload_entropy"] > 2.5, 1, np.where(X["iat_id_z"] < -1, 2, 0))
    model = XGBClassifier(n_estimators=30, max_depth=3, random_state=0, n_jobs=1).fit(
        X.to_numpy(np.float32), y
    )
    return AttackClassifier(model, FEATS, ["Normal", "Fuzzy", "Spoofing"]), X


def test_additivity(clf) -> None:
    c, X = clf
    ex = AlertExplainer(c, prefer_shap_package=False)
    values, base = ex.shap_values(X)
    margin = c.model.get_booster().predict(
        __import__("xgboost").DMatrix(X.to_numpy(np.float32)), output_margin=True
    )
    np.testing.assert_allclose(values.sum(axis=2) + base, margin, atol=1e-4)


def test_explain_structure_and_reason(clf) -> None:
    c, X = clf
    ex = AlertExplainer(c, top_k=3, prefer_shap_package=False)
    row = X.loc[X["payload_entropy"].idxmax()]
    e = ex.explain(row)
    assert e.predicted_class == "Fuzzy"
    assert len(e.top_features) == 3
    assert e.top_features[0]["feature"] == "payload_entropy"
    assert "Fuzzy-like" in e.reason and "entropy" in e.reason
    assert e.note == CAUSALITY_NOTE and e.latency_ms > 0
    assert abs(e.top_features[0]["shap"]) >= abs(e.top_features[1]["shap"])
    assert e.to_dict()["backend"].startswith(("xgboost", "shap"))


def test_explain_specific_class(clf) -> None:
    c, X = clf
    e = AlertExplainer(c, prefer_shap_package=False).explain(X.iloc[0], target_class="Spoofing")
    assert e.explained_class == "Spoofing"


def test_latency_profile(clf) -> None:
    c, X = clf
    lat = AlertExplainer(c, prefer_shap_package=False).latency_profile(X, n=20)
    assert lat["n"] == 20 and 0 < lat["p50_ms"] <= lat["max_ms"]


def test_global_importance_and_plots(clf, tmp_path: Path) -> None:
    c, X = clf
    ex = AlertExplainer(c, prefer_shap_package=False)
    table, values = ex.global_importance(X)
    assert table.index[0] in ("payload_entropy", "iat_id_z")
    assert values.shape == (len(X), 3, len(FEATS))
    plot_global_bar(table, tmp_path / "bar.png", "t")
    plot_beeswarm(values[:, 1, :], X, FEATS, tmp_path / "bee.png", "t")
    plot_explanation(ex.explain(X.iloc[0]), tmp_path / "alert.png", "t")
    for f in ("bar.png", "bee.png", "alert.png"):
        assert (tmp_path / f).stat().st_size > 0


def test_binary_model_gets_mirrored_attributions(clf) -> None:
    """Regression: binary XGBoost returns one margin; both classes must be explainable."""
    _, X = clf
    y = (X["payload_entropy"] > 2.5).astype(int)
    model = XGBClassifier(n_estimators=20, max_depth=3, random_state=0, n_jobs=1).fit(
        X.to_numpy(np.float32), y
    )
    ex = AlertExplainer(
        AttackClassifier(model, FEATS, ["Normal", "Attack"]), prefer_shap_package=False
    )
    values, base = ex.shap_values(X)
    assert values.shape == (len(X), 2, len(FEATS))
    np.testing.assert_allclose(values[:, 0], -values[:, 1])
    assert ex.explain(X.iloc[0], target_class="Attack").explained_class == "Attack"
    assert ex.explain(X.iloc[0], target_class="Normal").explained_class == "Normal"


def test_phrases() -> None:
    assert describe_feature("can_id", 0x6E0, 1.0) == "CAN ID 0x6E0 is typical of this class"
    assert describe_feature("b3", 255, 0.5) == "payload byte 3 = 0xFF"
    assert describe_feature("b3", -1, 0.5) == "payload byte 3 = absent"
    assert "never seen" in describe_feature("is_known_id", 0, 1.0)
    assert "deviates" in describe_feature("iat_id_z", -8.2, 0.4)
    top = [
        {"feature": "b5", "value": 255.0, "shap": 2.0},
        {"feature": "dlc", "value": 8.0, "shap": -1.0},
    ]
    assert reason_text("Spoofing", top) == "payload byte 5 = 0xFF -> Spoofing-like"
