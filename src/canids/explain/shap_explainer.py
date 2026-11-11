"""SHAP explanations for the stage-2 tree classifier: global importance and per-alert reasons.

What SHAP gives (viva)
----------------------
For one prediction, SHAP splits the model's raw output (the log-odds margin of a
class, for XGBoost) into ``base value + sum of per-feature contributions``.
The contributions are Shapley values from cooperative game theory: each
feature's average marginal contribution over all orders in which features
could be added. They are the only attribution satisfying local accuracy (they
add up exactly), missingness and consistency. TreeSHAP computes them EXACTLY
for tree ensembles in polynomial time, so no sampling approximation is needed.

**SHAP shows MODEL ATTRIBUTION, NOT CAUSAL PROOF.** A large contribution from
``iat_id_z`` means the *model* relied on that feature for this decision. It
does not prove the timing anomaly *caused* the attack, and correlated features
can share or swap credit. Explanations are aids for an analyst, not evidence.

Backends
--------
1. ``shap.TreeExplainer`` (the ``shap`` package) - preferred when importable.
2. XGBoost's built-in TreeSHAP (``Booster.predict(pred_contribs=True)``), which
   implements the same exact algorithm in C++. It is used automatically when
   ``shap`` cannot be imported. On the development laptop, Windows Application
   Control blocks the numba DLL that ``shap`` depends on, so the native
   backend is used there. Additivity is verified in the tests.

The explainer is built once and cached. A single explanation takes a few
milliseconds, measured by :meth:`AlertExplainer.latency_profile`.
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from canids.models.classifier import AttackClassifier
from canids.utils.logger import get_logger

log = get_logger(__name__)

CAUSALITY_NOTE = (
    "SHAP values show which features the model relied on (attribution), "
    "not proof of what caused the attack."
)


# =========================================================================== backends
def _load_shap_backend():
    try:
        import shap  # noqa: PLC0415  (optional, may be blocked by OS policy)

        return shap
    except Exception as err:  # ImportError or OS-level DLL block (OSError/ImportError)
        log.info("shap package unavailable (%s); using XGBoost native TreeSHAP", err)
        return None


class _ShapPackageBackend:
    name = "shap.TreeExplainer"

    def __init__(self, shap_mod: Any, model: Any) -> None:
        self._ex = shap_mod.TreeExplainer(model)

    def contributions(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        vals = self._ex.shap_values(X)
        arr = np.asarray(vals)
        if isinstance(vals, list):  # older shap: list of (n, f) per class
            arr = np.stack(vals, axis=-1)
        if arr.ndim == 2:  # binary model: (n, f)
            arr = arr[:, :, None]
        base = np.atleast_1d(np.asarray(self._ex.expected_value, dtype=np.float64))
        return arr.transpose(0, 2, 1), base  # -> (n, k, f), (k,)


class _XGBoostNativeBackend:
    name = "xgboost native TreeSHAP (pred_contribs)"

    def __init__(self, model: Any) -> None:
        self._booster = model.get_booster()

    def contributions(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        import xgboost as xgb  # noqa: PLC0415

        c = self._booster.predict(xgb.DMatrix(X), pred_contribs=True)
        if c.ndim == 2:  # binary: (n, f+1)
            c = c[:, None, :]
        return c[:, :, :-1].astype(np.float64), c[0, :, -1].astype(np.float64)


# =========================================================================== reasons
_PHRASES: dict[str, tuple[str, str]] = {
    # feature: (phrase when it pushes TOWARDS the class, phrase when AWAY)
    "can_id": ("CAN ID {v} is typical of this class", "CAN ID {v} is atypical"),
    "dlc": ("data length {v}", "data length {v}"),
    "payload_entropy": ("payload entropy {v} bits", "payload entropy {v} bits"),
    "hamming_prev_id": (
        "{v} bits changed vs previous frame of this ID",
        "{v} bits changed vs previous frame of this ID",
    ),
    "is_known_id": (
        "ID {known} in normal training traffic",
        "ID {known} in normal training traffic",
    ),
    "iat_global": ("gap since previous frame {v} s", "gap since previous frame {v} s"),
    "iat_id": (
        "gap since previous frame of this ID {v} s",
        "gap since previous frame of this ID {v} s",
    ),
    "iat_id_z": (
        "ID timing deviates from its normal period (z={v})",
        "ID timing close to its normal period (z={v})",
    ),
    "win_rate_fps": ("bus rate {v} frames/s", "bus rate {v} frames/s"),
    "win_unique_ids": (
        "{v} distinct IDs in last 100 frames",
        "{v} distinct IDs in last 100 frames",
    ),
    "win_top_id_share": (
        "top ID holds {v} of last 100 frames",
        "top ID holds {v} of last 100 frames",
    ),
    "win_id0_share": ("ID 0x000 share {v}", "ID 0x000 share {v}"),
    "win_cur_id_share": (
        "this ID is {v} of the last 100 frames",
        "this ID is {v} of the last 100 frames",
    ),
    "win_iat_mean": ("mean frame gap {v} s", "mean frame gap {v} s"),
    "win_iat_std": ("frame-gap jitter {v} s", "frame-gap jitter {v} s"),
    "win_id_entropy": ("ID entropy {v} bits", "ID entropy {v} bits"),
    "win_t_count": ("{v} frames in last 50 ms", "{v} frames in last 50 ms"),
}


def _fmt(feature: str, value: float) -> str:
    if feature == "can_id":
        return f"0x{int(value):03X}"
    if feature.startswith("b") and feature[1:].isdigit():
        return "absent" if value < 0 else f"0x{int(value):02X}"
    if value == -1 and feature in (
        "iat_global",
        "iat_id",
        "hamming_prev_id",
        "win_iat_mean",
        "win_iat_std",
    ):
        return "n/a"
    if abs(value) >= 100 or float(value).is_integer():
        return f"{value:.0f}"
    return f"{value:.3g}"


def describe_feature(feature: str, value: float, contribution: float) -> str:
    """Human-readable phrase for one feature's contribution."""
    v = _fmt(feature, value)
    if feature.startswith("b") and feature[1:].isdigit():
        return f"payload byte {feature[1:]} = {v}"
    towards, away = _PHRASES.get(feature, (f"{feature} = {{v}}", f"{feature} = {{v}}"))
    template = towards if contribution > 0 else away
    return template.format(v=v, known="seen" if value >= 1 else "never seen")


def reason_text(predicted: str, top: Sequence[Mapping[str, Any]], n: int = 2) -> str:
    """One-line reason from the strongest contributions that SUPPORT the predicted class."""
    support = [c for c in top if c["shap"] > 0][:n] or list(top[:1])
    parts = " and ".join(describe_feature(c["feature"], c["value"], c["shap"]) for c in support)
    return f"{parts} -> {predicted}-like"


# =========================================================================== explainer
@dataclass
class Explanation:
    """Explanation of one frame's classifier decision."""

    predicted_class: str
    explained_class: str
    probability: float
    base_value: float
    margin: float
    top_features: list[dict[str, float | str]]
    reason: str
    latency_ms: float
    backend: str
    note: str = CAUSALITY_NOTE
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable dict."""
        return asdict(self)


class AlertExplainer:
    """Cached TreeSHAP explainer around a fitted :class:`AttackClassifier`.

    Args:
        classifier: The stage-2 model.
        top_k: Number of features returned per explanation.
        prefer_shap_package: Try the ``shap`` package first.
    """

    def __init__(
        self, classifier: AttackClassifier, top_k: int = 5, prefer_shap_package: bool = True
    ) -> None:
        self.classifier = classifier
        self.features = list(classifier.features)
        self.top_k = top_k
        shap_mod = _load_shap_backend() if prefer_shap_package else None
        if shap_mod is not None:
            self._backend: Any = _ShapPackageBackend(shap_mod, classifier.model)
        elif hasattr(classifier.model, "get_booster"):
            self._backend = _XGBoostNativeBackend(classifier.model)
        else:
            raise RuntimeError(
                "shap is unavailable and the model is not XGBoost; no TreeSHAP backend available"
            )
        self.backend_name = self._backend.name

    # ------------------------------------------------------------------ raw values
    def shap_values(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(values (n, k, f), base (k,))`` in the model's margin space.

        Binary models produce ONE margin (log-odds of the positive class). For a
        2-class model it is expanded to [-margin, +margin], because
        logit(P(class 0)) = -logit(P(class 1)), so every class has its own attribution.
        """
        values, base = self._backend.contributions(X[self.features].to_numpy(dtype=np.float32))
        if values.shape[1] == 1 and len(self.classifier.classes) == 2:
            values = np.concatenate([-values, values], axis=1)
            base = np.array([-base[0], base[0]])
        return values, base

    # ------------------------------------------------------------------ single alert
    def explain(
        self, row: pd.Series | Mapping[str, float], target_class: str | None = None
    ) -> Explanation:
        """Explain one frame.

        Args:
            row: Feature values (at least ``self.features``).
            target_class: Class to explain; default = the predicted class.
        """
        t0 = time.perf_counter()
        X = pd.DataFrame([{f: row[f] for f in self.features}])
        values, base = self.shap_values(X)
        proba = self.classifier.predict_proba(X)[0]
        pred = self.classifier.classes[int(proba.argmax())]
        target = target_class or pred
        k = self.classifier.classes.index(target)
        contrib = values[0, k]
        order = np.argsort(-np.abs(contrib))[: self.top_k]
        top = [
            {"feature": self.features[i], "value": float(X.iloc[0, i]), "shap": float(contrib[i])}
            for i in order
        ]
        latency = (time.perf_counter() - t0) * 1000
        return Explanation(
            predicted_class=pred,
            explained_class=target,
            probability=float(proba[k]),
            base_value=float(base[k] if base.size > 1 else base[0]),
            margin=float(base[k if base.size > 1 else 0] + contrib.sum()),
            top_features=top,
            reason=reason_text(target, top),
            latency_ms=latency,
            backend=self.backend_name,
        )

    def latency_profile(self, X: pd.DataFrame, n: int = 200) -> dict[str, float]:
        """Per-explanation latency (ms) over ``n`` single-row calls (after one warm-up call)."""
        rows = X.iloc[: max(n, 1)]
        self.explain(rows.iloc[0])
        times = [self.explain(rows.iloc[i % len(rows)]).latency_ms for i in range(n)]
        return {
            "n": n,
            "p50_ms": float(np.percentile(times, 50)),
            "p95_ms": float(np.percentile(times, 95)),
            "p99_ms": float(np.percentile(times, 99)),
            "max_ms": float(np.max(times)),
            "mean_ms": float(np.mean(times)),
            "backend": self.backend_name,
        }

    # ------------------------------------------------------------------ global
    def global_importance(self, X: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
        """Mean |SHAP| per feature and class over ``X``. Returns ``(table, values)``."""
        values, _ = self.shap_values(X)
        imp = np.abs(values).mean(axis=0)  # (k, f)
        table = pd.DataFrame(imp.T, index=self.features, columns=self.classifier.classes)
        table["mean_all_classes"] = table.mean(axis=1)
        return table.sort_values("mean_all_classes", ascending=False), values


# =========================================================================== plots
def _plt():
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    return plt


def plot_global_bar(table: pd.DataFrame, path: str | Path, title: str, top_n: int = 15) -> None:
    """Grouped horizontal bar chart of mean |SHAP| per class (top features)."""
    plt = _plt()
    t = table.head(top_n).drop(columns="mean_all_classes").iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 0.42 * len(t) + 1.5))
    h = 0.8 / t.shape[1]
    for j, cls in enumerate(t.columns):
        ax.barh(np.arange(len(t)) + j * h, t[cls], height=h, label=cls)
    ax.set_yticks(np.arange(len(t)) + 0.4 - h / 2, t.index)
    ax.set(xlabel="mean |SHAP value| (log-odds)", title=title)
    ax.legend(fontsize=8)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_beeswarm(
    values: np.ndarray,
    X: pd.DataFrame,
    features: list[str],
    path: str | Path,
    title: str,
    top_n: int = 12,
    seed: int = 0,
) -> None:
    """Beeswarm-style summary for ONE class: SHAP per frame, coloured by feature-value rank."""
    plt = _plt()
    order = np.argsort(-np.abs(values).mean(axis=0))[:top_n][::-1]
    rng = np.random.default_rng(seed)
    fig, ax = plt.subplots(figsize=(8, 0.45 * len(order) + 1.5))
    for row, i in enumerate(order):
        x = values[:, i]
        col = X[features[i]].rank(pct=True).to_numpy()
        ax.scatter(
            x,
            row + rng.uniform(-0.3, 0.3, len(x)),
            c=col,
            cmap="coolwarm",
            s=6,
            alpha=0.6,
            linewidths=0,
        )
    ax.axvline(0, c="grey", lw=1)
    ax.set_yticks(range(len(order)), [features[i] for i in order])
    ax.set(xlabel="SHAP value (impact on log-odds)", title=title)
    sm = plt.cm.ScalarMappable(cmap="coolwarm")
    fig.colorbar(sm, ax=ax, label="feature value (rank: low -> high)", fraction=0.04)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_explanation(exp: Explanation, path: str | Path, title: str) -> None:
    """Waterfall-style bar chart of the top contributions for one alert."""
    plt = _plt()
    top = exp.top_features[::-1]
    labels = [f"{t['feature']} = {_fmt(str(t['feature']), float(t['value']))}" for t in top]
    vals = [float(t["shap"]) for t in top]
    fig, ax = plt.subplots(figsize=(8.5, 0.5 * len(top) + 2.2))
    ax.barh(labels, vals, color=["#d62728" if v > 0 else "#1f77b4" for v in vals])
    ax.axvline(0, c="grey", lw=1)
    ax.set(
        xlabel=f"SHAP contribution to '{exp.explained_class}' log-odds "
        f"(base {exp.base_value:.2f} -> {exp.margin:.2f})"
    )
    ax.set_title(
        f"{title}\nP({exp.explained_class}) = {exp.probability:.3f}   |   {exp.reason}", fontsize=9
    )
    fig.text(0.01, 0.01, CAUSALITY_NOTE, fontsize=7, color="grey")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
