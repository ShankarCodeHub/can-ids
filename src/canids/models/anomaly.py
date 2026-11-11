"""Stage 1: Isolation Forest anomaly detector.

Design decisions (viva)
-----------------------
* **Why Isolation Forest?** It isolates points with random axis-aligned splits.
  Rare, unusual points are isolated in few splits (a short average path
  length), so path length becomes an anomaly score. It needs no attack labels,
  so it can flag attack types never seen in training, which is the point of
  stage 1. It is fast (each tree sees only ``max_samples`` rows), handles mixed
  feature scales without normalisation, and runs on a laptop or a Raspberry Pi.
* **Trained on NORMAL frames only.** The detector learns what normal traffic
  looks like. If attack frames were included they would partly become "normal"
  to the model. Attack labels are used only to tune hyperparameters and the
  threshold on the validation split.
* **Threshold from validation Normal scores.** Instead of trusting
  ``contamination``, the threshold is the score that flags at most
  ``target_fpr`` (1 %) of validation Normal frames. Among all hyperparameter
  settings, the one with the highest validation attack recall at that FPR wins.
* **Score convention:** ``anomaly_score = -IsolationForest.score_samples(X)``,
  so a higher score means more anomalous; ``is_anomaly = anomaly_score > threshold``.
"""

from __future__ import annotations

import itertools
import json
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from canids.evaluation.metrics import binary_metrics, threshold_at_fpr
from canids.utils.logger import get_logger

log = get_logger(__name__)

MODEL_FILE = "isolation_forest.joblib"
META_FILE = "isolation_forest_metadata.json"


@dataclass
class AnomalyDetector:
    """A fitted Isolation Forest + the feature list and threshold it must be used with."""

    model: IsolationForest
    features: list[str]
    threshold: float
    meta: dict[str, Any] = field(default_factory=dict)

    def score(self, X: pd.DataFrame, batch_rows: int = 500_000) -> np.ndarray:
        """Continuous anomaly score (higher = more anomalous)."""
        values = X[self.features].to_numpy(dtype=np.float32)
        out = np.empty(len(values), dtype=np.float64)
        for start in range(0, len(values), batch_rows):
            stop = start + batch_rows
            out[start:stop] = -self.model.score_samples(values[start:stop])
        return out

    def predict(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(anomaly_score, is_anomaly)``."""
        s = self.score(X)
        return s, s > self.threshold

    def save(self, directory: str | Path) -> None:
        """Save the model (.joblib) and a human-readable metadata JSON."""
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, d / MODEL_FILE, compress=3)
        meta = {**self.meta, "features": self.features, "threshold": self.threshold}
        (d / META_FILE).write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")

    @classmethod
    def load(cls, directory: str | Path) -> AnomalyDetector:
        """Load a detector saved with :meth:`save`."""
        d = Path(directory)
        meta = json.loads((d / META_FILE).read_text(encoding="utf-8"))
        return cls(joblib.load(d / MODEL_FILE), meta["features"], float(meta["threshold"]), meta)


def fit_isolation_forest(
    X_normal: np.ndarray, params: Mapping[str, Any], seed: int, n_jobs: int = -1
) -> IsolationForest:
    """Fit an Isolation Forest on Normal rows only."""
    model = IsolationForest(
        n_estimators=int(params["n_estimators"]),
        max_samples=min(int(params["max_samples"]), len(X_normal)),
        max_features=float(params["max_features"]),
        contamination="auto",  # unused: our threshold comes from validation data
        random_state=seed,
        n_jobs=n_jobs,
    )
    return model.fit(X_normal)


def param_grid(search: Mapping[str, Sequence[Any]]) -> list[dict[str, Any]]:
    """Cartesian product of the search space in ``model.yaml: anomaly.search``."""
    keys = list(search)
    return [dict(zip(keys, combo, strict=True)) for combo in itertools.product(*search.values())]


def tune_isolation_forest(
    X_train_normal: pd.DataFrame,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    features: list[str],
    search: Mapping[str, Sequence[Any]],
    target_fpr: float,
    seed: int,
    n_jobs: int = -1,
) -> tuple[AnomalyDetector, pd.DataFrame]:
    """Grid-search on VALIDATION: maximise attack recall subject to FPR <= ``target_fpr``.

    For each setting, the threshold is the (1 - target_fpr) quantile of the
    validation Normal scores. Ties on recall are broken by validation PR-AUC.

    Returns:
        ``(best_detector, results_table)``.
    """
    y_val = np.asarray(y_val).astype(bool)
    Xtr = X_train_normal[features].to_numpy(dtype=np.float32)
    Xv = X_val[features].to_numpy(dtype=np.float32)
    rows, best, best_key = [], None, (-1.0, -1.0)
    for params in param_grid(search):
        t0 = time.perf_counter()
        model = fit_isolation_forest(Xtr, params, seed, n_jobs)
        scores = -model.score_samples(Xv)
        thr = threshold_at_fpr(scores[~y_val], target_fpr)
        m = binary_metrics(y_val, scores > thr, scores)
        row = {**params, "threshold": thr, **m, "seconds": round(time.perf_counter() - t0, 1)}
        rows.append(row)
        log.info(
            "IF %s -> val recall=%.4f fpr=%.4f pr_auc=%.4f (%.1fs)",
            params,
            m["recall"],
            m["fpr"],
            m.get("pr_auc", float("nan")),
            row["seconds"],
        )
        key = (m["recall"], m.get("pr_auc", 0.0))
        if key > best_key:
            best_key, best = key, (model, params, thr, m)
    assert best is not None
    model, params, thr, m = best
    detector = AnomalyDetector(
        model,
        list(features),
        thr,
        meta={
            "algorithm": "IsolationForest",
            "params": params,
            "target_fpr": target_fpr,
            "selection": "max val recall at FPR<=target, tie-break val PR-AUC",
            "val_metrics": m,
            "train_normal_rows": len(Xtr),
            "val_rows": len(Xv),
            "seed": seed,
        },
    )
    return detector, pd.DataFrame(rows)
