"""Hybrid decision logic: stage 1 (anomaly detector) + stage 2 (attack classifier).

Rules, applied in order to every frame::

    1. not anomaly and P(Normal) >= t_normal      -> Normal
    2. max P(attack class) >= t_attack             -> that attack class
    3. anomaly and P(Normal) >= t_suppress         -> Normal   (stage-1 false alarm overruled)
    4. anomaly                                     -> Unknown / Suspicious
    5. otherwise                                   -> Normal, flagged low-confidence

Why this design helps with UNSEEN attacks (viva)
------------------------------------------------
A supervised classifier can only output classes it was trained on. A new attack
is usually mapped to "Normal" or to the wrong attack class. The anomaly
detector never saw any attack, so it fires on anything unusual. When the
detector fires but the classifier cannot confidently name an attack (rule 4),
the frame becomes **Unknown / Suspicious** instead of being forced into a
label. The operator learns that something is wrong without a misleading
category. Rule 2 still lets the classifier catch known attacks that look
statistically normal (for example masquerade frames) even when stage 1 is quiet.

Rule 3 is a tunable addition. Isolation Forest fires on about 1-2 % of normal
traffic, and when the classifier is almost certain such a frame is Normal, the
alarm can be overruled. ``t_suppress > 1`` disables it, which gives exactly
the 4-rule design. Both variants are evaluated, because overruling trades
unseen-attack sensitivity for fewer false alarms.

``t_normal`` changes only the low-confidence FLAG (rule 1 vs rule 5 both output
Normal), so it is fixed in the config rather than tuned.
"""

from __future__ import annotations

import itertools
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from canids.evaluation.metrics import alarm_metrics, multiclass_metrics
from canids.ingestion.schema import LABEL_NORMAL
from canids.models.anomaly import AnomalyDetector
from canids.models.classifier import AttackClassifier

UNKNOWN = "Unknown"
RULE_NAMES = {
    1: "normal_confident",
    2: "attack_classified",
    3: "anomaly_overruled",
    4: "unknown_suspicious",
    5: "normal_low_confidence",
}
HYBRID_FILE = "hybrid_metadata.json"


@dataclass(frozen=True)
class HybridThresholds:
    """Decision thresholds (see module docstring)."""

    t_normal: float = 0.90
    t_attack: float = 0.70
    t_suppress: float = 1.01  # > 1 disables rule 3


def hybrid_decide(
    is_anomaly: np.ndarray, proba: np.ndarray, classes: Sequence[str], th: HybridThresholds
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorised hybrid decision.

    Args:
        is_anomaly: Stage-1 flags, shape (n,).
        proba: Stage-2 probabilities, shape (n, k), columns in ``classes`` order.
        classes: Class names; must contain ``"Normal"``.
        th: Thresholds.

    Returns:
        ``(labels, rule)``: final label per frame (object array) and the rule
        number (1-5) that decided it.
    """
    classes = list(classes)
    n_idx = classes.index(LABEL_NORMAL)
    is_anomaly = np.asarray(is_anomaly, dtype=bool)
    p_normal = proba[:, n_idx]
    attack_cols = [i for i in range(len(classes)) if i != n_idx]
    labels = np.full(len(proba), LABEL_NORMAL, dtype=object)
    rule = np.full(len(proba), 5, dtype=np.int8)

    if attack_cols:
        att = proba[:, attack_cols]
        best = att.argmax(axis=1)
        p_att = att[np.arange(len(att)), best]
        att_names = np.asarray([classes[i] for i in attack_cols], dtype=object)[best]
    else:
        p_att = np.zeros(len(proba))
        att_names = np.full(len(proba), LABEL_NORMAL, dtype=object)

    r1 = ~is_anomaly & (p_normal >= th.t_normal)
    r2 = ~r1 & (p_att >= th.t_attack)
    r3 = ~r1 & ~r2 & is_anomaly & (p_normal >= th.t_suppress)
    r4 = ~r1 & ~r2 & ~r3 & is_anomaly
    rule[r1], rule[r2], rule[r3], rule[r4] = 1, 2, 3, 4
    labels[r2] = att_names[r2]
    labels[r4] = UNKNOWN
    return labels, rule


def tune_hybrid(
    is_anomaly: np.ndarray,
    proba: np.ndarray,
    classes: Sequence[str],
    y_true: np.ndarray,
    weights: np.ndarray,
    t_normal: float,
    t_attack_grid: Sequence[float],
    t_suppress_grid: Sequence[float],
    target_fpr: float,
) -> tuple[HybridThresholds, pd.DataFrame]:
    """Grid-search ``t_attack`` and ``t_suppress`` on VALIDATION.

    Objective: maximise weighted macro-F1 (over the true classes; Unknown counts
    as an error) among settings whose alarm FPR is at most ``target_fpr``, where
    Unknown counts as an alarm. Ties go to the higher detection recall. If no
    setting meets the FPR budget, the lowest-FPR setting is returned and the
    table shows it.
    """
    rows, feasible, all_rows = [], [], []
    for ta, ts in itertools.product(t_attack_grid, t_suppress_grid):
        th = HybridThresholds(t_normal, float(ta), float(ts))
        labels, rule = hybrid_decide(is_anomaly, proba, classes, th)
        mc = multiclass_metrics(y_true, labels, list(classes), weights, (UNKNOWN,))
        am = alarm_metrics(y_true, labels, LABEL_NORMAL, weights)
        unknown_share = float(np.average(labels == UNKNOWN, weights=weights))
        row = {
            "t_attack": ta,
            "t_suppress": ts,
            "macro_f1": mc["macro_f1"],
            **am,
            "unknown_share": unknown_share,
        }
        rows.append(row)
        all_rows.append((th, row))
        if am["alarm_fpr"] <= target_fpr:
            feasible.append((th, row))
    pool = feasible or [min(all_rows, key=lambda t: t[1]["alarm_fpr"])]
    th, _ = max(pool, key=lambda t: (t[1]["macro_f1"], t[1]["detection_recall"]))
    return th, pd.DataFrame(rows)


@dataclass
class HybridDetector:
    """Stage 1 + stage 2 + thresholds, applied together."""

    anomaly: AnomalyDetector
    classifier: AttackClassifier
    thresholds: HybridThresholds
    meta: dict[str, Any] = field(default_factory=dict)

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """Per-frame decision table (one row per input row)."""
        score, is_anom = self.anomaly.predict(X)
        proba = self.classifier.predict_proba(X)
        labels, rule = hybrid_decide(is_anom, proba, self.classifier.classes, self.thresholds)
        top = proba.argmax(axis=1)
        return pd.DataFrame(
            {
                "predicted_class": labels,
                "anomaly_score": score,
                "is_anomaly": is_anom,
                "classifier_class": np.asarray(self.classifier.classes, dtype=object)[top],
                "class_probability": proba[np.arange(len(proba)), top],
                "rule": rule,
                "low_confidence": rule == 5,
            }
        )

    def save(self, directory: str | Path) -> None:
        """Save thresholds + provenance (the two models are saved separately)."""
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        data = {"thresholds": asdict(self.thresholds), "rules": RULE_NAMES, **self.meta}
        (d / HYBRID_FILE).write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")

    @classmethod
    def load(cls, model_root: str | Path) -> HybridDetector:
        """Load from ``models/<dataset>/`` (expects anomaly/, classifier/, hybrid/)."""
        root = Path(model_root)
        data = json.loads((root / "hybrid" / HYBRID_FILE).read_text(encoding="utf-8"))
        return cls(
            AnomalyDetector.load(root / "anomaly"),
            AttackClassifier.load(root / "classifier"),
            HybridThresholds(**data["thresholds"]),
            {k: v for k, v in data.items() if k not in ("thresholds", "rules")},
        )
