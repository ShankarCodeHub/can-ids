"""Ablation study: which parts of the system matter?

Variants (each compared on the SAME held-out test set):

=============  ==============================================================
full           all 25 features, Isolation Forest + XGBoost (reproduces Phase 4)
no_iforest     hybrid WITHOUT stage 1 (stage-1 flag forced to False)
no_timing      drop iat_global, iat_id, iat_id_z
no_window      drop the 9 rolling-window features
no_payload     drop b0..b7, payload_entropy, hamming_prev_id
=============  ==============================================================

Protocol (viva): every variant uses the hyperparameters already selected in
Phases 3-4 (Isolation Forest grid winner, XGBoost random-search winner). It is
REFIT on the same seeded training samples with its feature subset. Its
thresholds (IF threshold at 1 % validation FPR, hybrid ``t_attack`` /
``t_suppress``) are RE-TUNED on validation exactly as in Phase 4. Only then is
it scored once on test. Re-running the full search per variant was skipped to
fit the laptop budget; this is stated in the results.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from canids.evaluation.metrics import alarm_metrics, multiclass_metrics, threshold_at_fpr
from canids.ingestion.schema import LABEL_NORMAL
from canids.models.anomaly import AnomalyDetector, fit_isolation_forest
from canids.models.classifier import AttackClassifier, balanced_sample_weight, build_model
from canids.models.hybrid import UNKNOWN, HybridThresholds, hybrid_decide, tune_hybrid
from canids.preprocessing.features import FEATURE_COLUMNS, FEATURE_GROUPS
from canids.utils.logger import get_logger

log = get_logger(__name__)

ABLATIONS: dict[str, dict[str, Any]] = {
    "full": {"drop": [], "use_iforest": True},
    "no_iforest": {"drop": [], "use_iforest": False},
    "no_timing": {"drop": FEATURE_GROUPS["timing"], "use_iforest": True},
    "no_window": {"drop": FEATURE_GROUPS["window"], "use_iforest": True},
    "no_payload": {"drop": FEATURE_GROUPS["payload"], "use_iforest": True},
}


@dataclass
class Variant:
    """A fitted ablation variant."""

    name: str
    features: list[str]
    anomaly: AnomalyDetector | None
    classifier: AttackClassifier
    thresholds: HybridThresholds
    val: dict[str, Any]

    def decide(self, X: pd.DataFrame) -> np.ndarray:
        """Hybrid labels for ``X``."""
        if self.anomaly is not None:
            _, is_anom = self.anomaly.predict(X)
        else:
            is_anom = np.zeros(len(X), dtype=bool)
        proba = self.classifier.predict_proba(X)
        labels, _ = hybrid_decide(is_anom, proba, self.classifier.classes, self.thresholds)
        return labels


def fit_variant(
    name: str,
    drop: Sequence[str],
    use_iforest: bool,
    train_normal: pd.DataFrame,
    train: pd.DataFrame,
    y_train: np.ndarray,
    val: pd.DataFrame,
    y_val: np.ndarray,
    w_val: np.ndarray,
    if_params: Mapping[str, Any],
    clf_kind: str,
    clf_params: Mapping[str, Any],
    t_normal: float,
    t_attack_grid: Sequence[float],
    t_suppress_grid: Sequence[float],
    target_fpr: float,
    seed: int,
) -> Variant:
    """Refit both stages on a feature subset and re-tune thresholds on validation."""
    features = [f for f in FEATURE_COLUMNS if f not in set(drop)]
    detector = None
    if use_iforest:
        model = fit_isolation_forest(train_normal[features].to_numpy(np.float32), if_params, seed)
        scores = -model.score_samples(val[features].to_numpy(np.float32))
        thr = threshold_at_fpr(scores[y_val == LABEL_NORMAL], target_fpr)
        detector = AnomalyDetector(model, features, thr, {"params": dict(if_params)})
        is_anom = scores > thr
    else:
        is_anom = np.zeros(len(val), dtype=bool)

    classes = sorted(np.unique(y_train).tolist(), key=lambda c: (c != LABEL_NORMAL, c))
    idx = {c: i for i, c in enumerate(classes)}
    ytr = np.array([idx[c] for c in y_train])
    clf_model = build_model(clf_kind, clf_params, len(classes), seed)
    clf_model.fit(
        train[features].to_numpy(np.float32), ytr, sample_weight=balanced_sample_weight(ytr)
    )
    clf = AttackClassifier(
        clf_model, features, classes, {"algorithm": clf_kind, "params": dict(clf_params)}
    )
    proba = clf.predict_proba(val)

    grid_suppress = t_suppress_grid if use_iforest else [1.01]
    th, _ = tune_hybrid(
        is_anom, proba, classes, y_val, w_val, t_normal, t_attack_grid, grid_suppress, target_fpr
    )
    labels, _ = hybrid_decide(is_anom, proba, classes, th)
    val_summary = {
        "thresholds": th.__dict__,
        "if_threshold": detector.threshold if detector else None,
        **{
            k: v
            for k, v in multiclass_metrics(y_val, labels, classes, w_val, (UNKNOWN,)).items()
            if k in ("macro_f1", "weighted_f1")
        },
        **alarm_metrics(y_val, labels, LABEL_NORMAL, w_val),
    }
    log.info(
        "ablation %s: %d features, val macro-F1=%.4f detection=%.4f fpr=%.4f",
        name,
        len(features),
        val_summary["macro_f1"],
        val_summary["detection_recall"],
        val_summary["alarm_fpr"],
    )
    return Variant(name, features, detector, clf, th, val_summary)


def summarise(
    name: str, y_true: np.ndarray, labels: np.ndarray, attack_type: np.ndarray, classes: list[str]
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Test metrics for one variant + per-attack-type detection table."""
    mc = multiclass_metrics(y_true, labels, classes, extra_pred_labels=(UNKNOWN,))
    am = alarm_metrics(y_true, labels, LABEL_NORMAL)
    attack = y_true != LABEL_NORMAL
    alarm = labels != LABEL_NORMAL
    tp = int((attack & alarm).sum())
    row = {
        "variant": name,
        "macro_f1": mc["macro_f1"],
        "weighted_f1": mc["weighted_f1"],
        **{f"f1_{c}": mc["per_class"][c]["f1"] for c in classes},
        "detection_recall": am["detection_recall"],
        "alarm_fpr": am["alarm_fpr"],
        "alarm_precision": am["alarm_precision"],
        "binary_f1": (
            (2 * tp / (2 * tp + int((~attack & alarm).sum()) + int((attack & ~alarm).sum())))
            if attack.any()
            else 0.0
        ),
        "unknown_frames": int((labels == UNKNOWN).sum()),
    }
    det = (
        pd.DataFrame({"attack_type": attack_type[attack], "hit": alarm[attack]})
        .groupby("attack_type")["hit"]
        .agg(frames="size", detected="sum")
        .reset_index()
    )
    det["detection_rate"] = det["detected"] / det["frames"]
    det.insert(0, "variant", name)
    return row, det
