"""Cross-dataset evaluation: train on one vehicle's data, test on another's.

Why the feature and label sets must be reduced (viva)
-----------------------------------------------------
HCRL and ROAD come from different vehicles, so CAN IDs, payload layouts and
signal meanings are unrelated. Features whose values only make sense for one
vehicle would let a model "transfer" nothing but noise, so only the common,
vehicle-agnostic subset is used:

* DROPPED ``can_id``: arbitrary per vehicle (0x0D0 means different things).
* DROPPED ``b0``..``b7``: payload layout and signal semantics are vehicle-specific.
* DROPPED ``dlc``: ROAD logs store every frame as 8 bytes, so DLC is constant there.
* DROPPED ``is_known_id``, ``iat_id_z``: computed against the SOURCE training
  vehicle's ID table, so they are meaningless (all-unknown / zero) on the target.
* KEPT: timing (``iat_global``, ``iat_id``), all rolling-window statistics,
  ``payload_entropy``, ``hamming_prev_id``. These describe traffic behaviour,
  not one vehicle's encoding.

Labels: only classes present in BOTH datasets are compared, i.e. {Normal, Fuzzy,
Spoofing}. HCRL's DoS has no ROAD counterpart, so DoS frames are removed in the
multi-class setting and counted as "Attack" in the binary Normal-vs-Attack
setting.

Expected outcome (stated before running): a large drop compared with
within-dataset results, because of different vehicles, ID sets, bus loads,
capture conditions (road vs dynamometer) and attack definitions (HCRL fuzzing
uses random payloads; ROAD fuzzing uses all-0xFF payloads).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from canids.evaluation.metrics import alarm_metrics, binary_metrics, multiclass_metrics
from canids.ingestion.schema import LABEL_NORMAL
from canids.models.classifier import balanced_sample_weight, build_model
from canids.preprocessing.features import FEATURE_COLUMNS

COMMON_LABELS: tuple[str, ...] = ("Normal", "Fuzzy", "Spoofing")
ATTACK = "Attack"

DROP_REASONS: dict[str, str] = {
    "can_id": "CAN ID assignment is vehicle-specific",
    **{f"b{i}": "payload byte layout/semantics are vehicle-specific" for i in range(8)},
    "dlc": "ROAD stores every frame as 8 bytes (constant)",
    "is_known_id": "computed against the source vehicle's ID table",
    "iat_id_z": "normalised by the source vehicle's per-ID periods",
}


def common_features(features: Sequence[str] = FEATURE_COLUMNS) -> tuple[list[str], dict[str, str]]:
    """Split features into (kept, {dropped: reason}) for cross-dataset use."""
    kept = [f for f in features if f not in DROP_REASONS]
    dropped = {f: DROP_REASONS[f] for f in features if f in DROP_REASONS}
    return kept, dropped


def map_labels(labels: np.ndarray, mode: str) -> tuple[np.ndarray, np.ndarray]:
    """Map labels for ``mode`` = 'multiclass' or 'binary'.

    Returns ``(mapped_labels, keep_mask)``. Multiclass keeps only COMMON_LABELS
    (DoS rows are dropped); binary maps every attack to "Attack" and keeps all rows.
    """
    labels = np.asarray(labels).astype(str)
    if mode == "multiclass":
        keep = np.isin(labels, COMMON_LABELS)
        return labels, keep
    if mode == "binary":
        return np.where(labels == LABEL_NORMAL, LABEL_NORMAL, ATTACK), np.ones(len(labels), bool)
    raise ValueError(f"mode must be 'multiclass' or 'binary', got {mode!r}")


def train_and_test(
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: Sequence[str],
    mode: str,
    clf_kind: str,
    clf_params: Mapping[str, Any],
    seed: int,
    test_weights: np.ndarray | None = None,
) -> dict[str, Any]:
    """Fit a classifier on ``train`` and score ``test`` (both: features + 'label' column).

    The decision is the argmax class (no thresholds), so source-tuned thresholds
    cannot hide or exaggerate the transfer gap.
    """
    ytr, ktr = map_labels(train["label"].to_numpy(), mode)
    yte, kte = map_labels(test["label"].to_numpy(), mode)
    ytr, Xtr = ytr[ktr], train.loc[ktr, list(features)].to_numpy(np.float32)
    yte, Xte = yte[kte], test.loc[kte, list(features)].to_numpy(np.float32)
    w_te = None if test_weights is None else np.asarray(test_weights)[kte]
    classes = sorted(np.unique(ytr).tolist(), key=lambda c: (c != LABEL_NORMAL, c))
    idx = {c: i for i, c in enumerate(classes)}
    yi = np.array([idx[c] for c in ytr])
    model = build_model(clf_kind, clf_params, len(classes), seed)
    model.fit(Xtr, yi, sample_weight=balanced_sample_weight(yi))
    proba = model.predict_proba(Xte)
    if proba.ndim == 1 or proba.shape[1] == 1:
        proba = np.column_stack([1 - proba.ravel(), proba.ravel()])
    pred = np.asarray(classes, dtype=object)[proba.argmax(axis=1)]
    eval_classes = [c for c in classes if c in set(yte)] or classes
    out: dict[str, Any] = {
        "mode": mode,
        "train_rows": int(len(ytr)),
        "test_rows": int(len(yte)),
        "train_classes": classes,
        "test_class_counts": {c: int((yte == c).sum()) for c in np.unique(yte)},
        **{
            k: v
            for k, v in multiclass_metrics(yte, pred, eval_classes, w_te).items()
            if k in ("macro_f1", "weighted_f1", "per_class", "confusion_matrix")
        },
        **alarm_metrics(yte, pred, LABEL_NORMAL, w_te),
    }
    attack = yte != LABEL_NORMAL
    if 0 < attack.sum() < len(attack):
        bm = binary_metrics(attack, pred != LABEL_NORMAL, 1 - proba[:, idx[LABEL_NORMAL]])
        out["roc_auc"], out["pr_auc"] = bm.get("roc_auc"), bm.get("pr_auc")
    return out
