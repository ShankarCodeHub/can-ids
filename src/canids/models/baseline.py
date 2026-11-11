"""Rule-based baseline detector (what a simple, non-ML IDS would do).

Three classic rules from the CAN IDS literature, each using a feature computed
by the same streaming extractor:

1. **Unknown ID.** ``is_known_id == 0``: an ID never seen in normal training
   traffic. This catches fuzzing and HCRL's ``0x000`` DoS.
2. **Too-fast ID (interval rule).** The time since the previous frame of the
   same ID is below ``iat_ratio`` x that ID's normal period. Injecting extra
   frames of an ID shortens its inter-arrival time. This is the timing-interval
   idea of Song, Kim and Kim (2016).
3. **Bus-load rule.** ``win_t_count`` (frames in the last 50 ms) exceeds a
   high quantile of normal training traffic. This catches floods.

The baseline is tuned exactly like the ML model: thresholds come from TRAIN
Normal frames, and the rule parameters are chosen on VALIDATION to maximise
recall at FPR <= 1 %. The comparison is therefore fair.

For ROC/PR curves the rules are combined into one continuous score whose
alarm condition is exactly ``score > 1``::

    score = max(win_t_count / rate_threshold,
                iat_ratio * period_mean[id] / iat_id   (if both are known),
                2.0 if the ID is unknown)
"""

from __future__ import annotations

import itertools
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from canids.evaluation.metrics import binary_metrics
from canids.utils.logger import get_logger

log = get_logger(__name__)

BASELINE_FEATURES = ["can_id", "is_known_id", "iat_id", "win_t_count"]
UNKNOWN_SCORE = 2.0
SCORE_CLIP = 1e3


@dataclass
class RuleBaseline:
    """Unknown-ID + interval + bus-load rules."""

    rate_threshold: float
    iat_ratio: float
    period_mean: dict[int, float]
    flag_unknown: bool = True
    meta: dict[str, Any] = field(default_factory=dict)

    def score(self, X: pd.DataFrame) -> np.ndarray:
        """Continuous rule score; an alert is raised iff score > 1."""
        rate = X["win_t_count"].to_numpy(dtype=np.float64) / max(self.rate_threshold, 1e-9)
        mean = X["can_id"].map(self.period_mean).to_numpy(dtype=np.float64)  # NaN if unknown
        iat = X["iat_id"].to_numpy(dtype=np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            interval = np.where(
                (iat >= 0) & ~np.isnan(mean),
                self.iat_ratio * mean / np.maximum(iat, 1e-9),
                0.0,
            )
        s = np.maximum(rate, np.minimum(interval, SCORE_CLIP))
        if self.flag_unknown:
            s = np.where(X["is_known_id"].to_numpy() == 0, np.maximum(s, UNKNOWN_SCORE), s)
        return s

    def predict(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(score, is_alert)``."""
        s = self.score(X)
        return s, s > 1.0

    def save(self, path: str | Path) -> None:
        """Save parameters as JSON (the baseline has no binary model)."""
        data = {
            "rate_threshold": self.rate_threshold,
            "iat_ratio": self.iat_ratio,
            "flag_unknown": self.flag_unknown,
            "period_mean": {f"0x{k:X}": v for k, v in sorted(self.period_mean.items())},
            "meta": self.meta,
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> RuleBaseline:
        """Load parameters saved with :meth:`save`."""
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            float(d["rate_threshold"]),
            float(d["iat_ratio"]),
            {int(k, 16): float(v) for k, v in d["period_mean"].items()},
            bool(d["flag_unknown"]),
            d.get("meta", {}),
        )


def tune_baseline(
    train_normal: pd.DataFrame,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    period_mean: Mapping[int, float],
    rate_quantiles: Sequence[float],
    iat_ratios: Sequence[float],
    target_fpr: float,
    flag_unknown: bool = True,
) -> tuple[RuleBaseline, pd.DataFrame]:
    """Choose (rate quantile, interval ratio) on VALIDATION: max recall with FPR <= target.

    If no setting meets the FPR target, the setting with the lowest FPR is kept
    and the shortfall is recorded in ``meta``.
    """
    y_val = np.asarray(y_val).astype(bool)
    counts = train_normal["win_t_count"].to_numpy(dtype=np.float64)
    rows, candidates = [], []
    for q, r in itertools.product(rate_quantiles, iat_ratios):
        rate_thr = float(np.quantile(counts, q))
        model = RuleBaseline(rate_thr, float(r), dict(period_mean), flag_unknown)
        s, pred = model.predict(X_val)
        m = binary_metrics(y_val, pred, s)
        rows.append({"rate_quantile": q, "rate_threshold": rate_thr, "iat_ratio": r, **m})
        candidates.append((model, q, m))
    feasible = [c for c in candidates if c[2]["fpr"] <= target_fpr]
    if feasible:
        model, q, m = max(feasible, key=lambda c: (c[2]["recall"], -c[2]["fpr"]))
        note = "met FPR target"
    else:
        model, q, m = min(candidates, key=lambda c: c[2]["fpr"])
        note = f"no setting reached FPR <= {target_fpr}; lowest-FPR setting kept"
        log.warning("Baseline: %s", note)
    model.meta = {
        "rate_quantile": q,
        "target_fpr": target_fpr,
        "selection": "max val recall at FPR<=target",
        "note": note,
        "val_metrics": m,
    }
    return model, pd.DataFrame(rows)
