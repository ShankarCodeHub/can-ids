"""Stage 2: supervised attack classifier (RandomForest and XGBoost).

Design decisions (viva)
-----------------------
* **Tree ensembles.** CAN features mix discrete IDs, byte values, rates and
  sentinels. Trees split on raw thresholds, so no scaling or encoding is needed.
  They are fast on a CPU, and SHAP's TreeExplainer gives exact explanations for
  them (Phase 5).
* **RandomForest vs XGBoost.** Bagging (variance reduction, robust defaults) is
  compared with boosting (bias reduction, often more accurate). Both are tuned
  by a seeded random search on the FIXED validation split. A time-ordered
  dataset cannot be shuffled into CV folds without leaking neighbouring frames,
  so the predefined split is used. The winner is the model with the highest
  validation macro-F1.
* **Macro-F1 for selection.** Each class counts equally, so a model cannot win
  by being good only at the 99.6 % Normal class.
* **Imbalance.** TRAIN Normal frames are subsampled (compute), then
  ``balanced`` sample weights equalise the total weight of each class.
  Validation keeps its natural distribution via inverse-sampling weights.
* **Only the classes present in the training data are learned.** On ROAD these
  are Normal, Fuzzy and Spoofing: ROAD has no DoS.
"""

from __future__ import annotations

import json
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import ParameterSampler
from xgboost import XGBClassifier

from canids.evaluation.metrics import expected_calibration_error, multiclass_metrics
from canids.utils.logger import get_logger

log = get_logger(__name__)

CANDIDATES = ("random_forest", "xgboost")
MODEL_FILE = "classifier.joblib"
META_FILE = "classifier_metadata.json"


def balanced_sample_weight(y: np.ndarray) -> np.ndarray:
    """Per-row weights so every class has the same total weight (sklearn 'balanced' formula)."""
    classes, counts = np.unique(y, return_counts=True)
    per_class = {c: len(y) / (len(classes) * n) for c, n in zip(classes, counts, strict=True)}
    return np.array([per_class[v] for v in y], dtype=np.float64)


def build_model(kind: str, params: Mapping[str, Any], n_classes: int, seed: int, n_jobs: int = -1):
    """Instantiate an untrained classifier of the given kind."""
    if kind == "random_forest":
        return RandomForestClassifier(**params, random_state=seed, n_jobs=n_jobs)
    if kind == "xgboost":
        return XGBClassifier(
            **params,
            objective="multi:softprob" if n_classes > 2 else "binary:logistic",
            tree_method="hist",
            eval_metric="mlogloss" if n_classes > 2 else "logloss",
            random_state=seed,
            n_jobs=n_jobs,
        )
    raise ValueError(f"unknown classifier kind {kind!r}")


@dataclass
class AttackClassifier:
    """Fitted classifier + feature list + class names (index i of predict_proba = classes[i])."""

    model: Any
    features: list[str]
    classes: list[str]
    meta: dict[str, Any] = field(default_factory=dict)

    def predict_proba(self, X: pd.DataFrame, batch_rows: int = 500_000) -> np.ndarray:
        """Class probabilities, shape (n, len(classes))."""
        values = X[self.features].to_numpy(dtype=np.float32)
        out = np.empty((len(values), len(self.classes)), dtype=np.float64)
        for start in range(0, len(values), batch_rows):
            p = self.model.predict_proba(values[start : start + batch_rows])
            if p.ndim == 1 or p.shape[1] == 1:  # binary XGBoost edge case
                p = np.column_stack([1 - p.ravel(), p.ravel()])
            out[start : start + batch_rows] = p
        return out

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Most probable class name per row."""
        return np.asarray(self.classes, dtype=object)[self.predict_proba(X).argmax(axis=1)]

    def save(self, directory: str | Path) -> None:
        """Save model (.joblib) + metadata JSON."""
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, d / MODEL_FILE, compress=3)
        meta = {**self.meta, "features": self.features, "classes": self.classes}
        (d / META_FILE).write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")

    @classmethod
    def load(cls, directory: str | Path) -> AttackClassifier:
        """Load a classifier saved with :meth:`save`."""
        d = Path(directory)
        meta = json.loads((d / META_FILE).read_text(encoding="utf-8"))
        return cls(joblib.load(d / MODEL_FILE), meta["features"], meta["classes"], meta)


def search_classifiers(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    w_val: np.ndarray,
    features: list[str],
    spaces: Mapping[str, Mapping[str, Sequence[Any]]],
    candidates: Sequence[str],
    n_iter: int,
    seed: int,
    n_jobs: int = -1,
) -> tuple[AttackClassifier, pd.DataFrame]:
    """Random search per candidate on the fixed validation split; best by weighted macro-F1.

    Args:
        X_train, y_train: Training rows and string labels (already subsampled).
        X_val, y_val, w_val: Validation rows, labels and inverse-sampling weights.
        features: Feature columns to use.
        spaces: ``{kind: {param: [values]}}`` from model.yaml.
        candidates: Model kinds to try.
        n_iter: Settings sampled per kind.
        seed: Seed for the sampler and the models.
    """
    classes = sorted(np.unique(y_train).tolist(), key=lambda c: (c != "Normal", c))
    idx = {c: i for i, c in enumerate(classes)}
    ytr = np.array([idx[c] for c in y_train])
    wtr = balanced_sample_weight(ytr)
    Xtr = X_train[features].to_numpy(dtype=np.float32)
    Xv = X_val[features].to_numpy(dtype=np.float32)
    yv_idx = np.array([idx.get(c, -1) for c in y_val])

    rows, best, best_f1 = [], None, -1.0
    for kind in candidates:
        sampler = ParameterSampler(dict(spaces[kind]), n_iter=n_iter, random_state=seed)
        for params in sampler:
            t0 = time.perf_counter()
            model = build_model(kind, params, len(classes), seed, n_jobs)
            model.fit(Xtr, ytr, sample_weight=wtr)
            proba = model.predict_proba(Xv)
            pred = np.asarray(classes, dtype=object)[proba.argmax(axis=1)]
            m = multiclass_metrics(y_val, pred, classes, w_val)
            seconds = time.perf_counter() - t0
            row = {
                "model": kind,
                **{f"param_{k}": v for k, v in params.items()},
                "val_macro_f1": m["macro_f1"],
                "val_weighted_f1": m["weighted_f1"],
                **{f"val_f1_{c}": m["per_class"][c]["f1"] for c in classes},
                "seconds": round(seconds, 1),
            }
            rows.append(row)
            log.info("%s %s -> val macro-F1=%.4f (%.0fs)", kind, params, m["macro_f1"], seconds)
            if m["macro_f1"] > best_f1:
                best_f1 = m["macro_f1"]
                best = (kind, params, model, m, proba)
    assert best is not None
    kind, params, model, m, proba = best
    ece = (
        expected_calibration_error(yv_idx, proba, sample_weight=w_val)
        if (yv_idx >= 0).all()
        else None
    )
    clf = AttackClassifier(
        model,
        list(features),
        classes,
        meta={
            "algorithm": kind,
            "params": params,
            "selection": "max weighted val macro-F1 over random search",
            "n_iter_per_candidate": n_iter,
            "val_metrics": m,
            "val_ece": ece,
            "train_rows": len(Xtr),
            "train_class_counts": {c: int((ytr == i).sum()) for c, i in idx.items()},
            "imbalance": "Normal subsampled + balanced sample weights",
            "seed": seed,
        },
    )
    return clf, pd.DataFrame(rows)
