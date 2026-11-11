"""Detection metrics shared by every experiment (Phases 3-7).

Design decisions (viva):
* **FPR is reported next to precision and recall.** On a CAN bus carrying about
  2,000 frames/s, an FPR of only 1 % means about 20 false alarms PER SECOND.
  Accuracy is useless here: on ROAD, predicting "Normal" for everything already
  scores more than 99.7 %.
* **PR-AUC (average precision) is reported alongside ROC-AUC.** With attacks
  below 1 % of frames, ROC-AUC can look excellent while precision is poor;
  PR-AUC exposes that.
* **Thresholds are chosen on validation data only.** :func:`threshold_at_fpr`
  is applied to validation Normal scores; test data never influences a threshold.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


def threshold_at_fpr(normal_scores: np.ndarray, target_fpr: float) -> float:
    """Smallest threshold such that ``score > threshold`` flags at most ``target_fpr`` of Normal.

    Args:
        normal_scores: Anomaly scores (higher = more anomalous) of Normal frames.
        target_fpr: Allowed false-positive rate, e.g. 0.01.
    """
    if not 0 < target_fpr < 1:
        raise ValueError("target_fpr must be in (0, 1)")
    s = np.asarray(normal_scores, dtype=np.float64)
    if s.size == 0:
        raise ValueError("no normal scores")
    return float(np.quantile(s, 1.0 - target_fpr, method="higher"))


def binary_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, scores: np.ndarray | None = None
) -> dict[str, float]:
    """Precision, recall, F1, FPR, confusion counts, and (if scores given) ROC-AUC / PR-AUC.

    ``y_true`` / ``y_pred``: 1 = attack (positive), 0 = normal.
    """
    y_true = np.asarray(y_true).astype(bool)
    y_pred = np.asarray(y_pred).astype(bool)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[False, True]).ravel()
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    out: dict[str, float] = {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "fpr": float(fp / (fp + tn)) if fp + tn else 0.0,
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "n": int(len(y_true)),
        "attack_share": float(y_true.mean()) if len(y_true) else 0.0,
    }
    if scores is not None and 0 < y_true.sum() < len(y_true):
        out["roc_auc"] = float(roc_auc_score(y_true, scores))
        out["pr_auc"] = float(average_precision_score(y_true, scores))
    return out


def recall_at_fpr(y_true: np.ndarray, scores: np.ndarray, target_fpr: float) -> tuple[float, float]:
    """(recall, threshold) when the threshold is set on the Normal scores of the same data."""
    y_true = np.asarray(y_true).astype(bool)
    thr = threshold_at_fpr(scores[~y_true], target_fpr)
    return float((scores[y_true] > thr).mean()) if y_true.any() else 0.0, thr


def detection_rate_by_group(
    y_true: np.ndarray, y_pred: np.ndarray, groups: np.ndarray
) -> pd.DataFrame:
    """Per-group frames, detected count and detection rate (recall) for ATTACK frames only."""
    y_true = np.asarray(y_true).astype(bool)
    df = pd.DataFrame({"group": np.asarray(groups)[y_true], "hit": np.asarray(y_pred)[y_true]})
    out = df.groupby("group")["hit"].agg(frames="size", detected="sum")
    out["detection_rate"] = out["detected"] / out["frames"]
    return out.reset_index()


def false_positive_rate_by_group(
    y_true: np.ndarray, y_pred: np.ndarray, groups: np.ndarray
) -> pd.DataFrame:
    """Per-group FPR over NORMAL frames (e.g. ambient vs normal frames in attack captures)."""
    y_true = np.asarray(y_true).astype(bool)
    df = pd.DataFrame({"group": np.asarray(groups)[~y_true], "fp": np.asarray(y_pred)[~y_true]})
    out = df.groupby("group")["fp"].agg(frames="size", false_positives="sum")
    out["fpr"] = out["false_positives"] / out["frames"]
    return out.reset_index()


def curve_points(
    y_true: np.ndarray, scores: np.ndarray, max_points: int = 2000
) -> dict[str, dict[str, list[float]]]:
    """Thinned ROC and PR curve points (full curves can have millions of points)."""
    fpr, tpr, _ = roc_curve(y_true, scores, drop_intermediate=True)
    prec, rec, _ = precision_recall_curve(y_true, scores)

    def thin(*arrs: np.ndarray) -> list[list[float]]:
        idx = np.unique(np.linspace(0, len(arrs[0]) - 1, min(max_points, len(arrs[0]))).astype(int))
        return [a[idx].round(6).tolist() for a in arrs]

    f, t = thin(fpr, tpr)
    p, r = thin(prec, rec)
    return {"roc": {"fpr": f, "tpr": t}, "pr": {"precision": p, "recall": r}}


def multiclass_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    classes: list[str],
    sample_weight: np.ndarray | None = None,
    extra_pred_labels: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Per-class precision/recall/F1, macro and weighted F1, and the confusion matrix.

    Args:
        y_true: True labels (strings).
        y_pred: Predicted labels; may contain ``extra_pred_labels`` such as ``"Unknown"``,
            which never equal a true label (so they count as errors for F1) but get
            their own confusion-matrix column.
        classes: True classes to score (F1 is averaged over these only).
        sample_weight: Optional per-row weights (e.g. inverse Normal sampling rate).
        extra_pred_labels: Prediction-only labels.
    """
    y_true = np.asarray(y_true).astype(str)
    y_pred = np.asarray(y_pred).astype(str)
    w = np.ones(len(y_true)) if sample_weight is None else np.asarray(sample_weight, float)
    per_class, f1s, supports = {}, [], []
    for c in classes:
        tp = w[(y_true == c) & (y_pred == c)].sum()
        fp = w[(y_true != c) & (y_pred == c)].sum()
        fn = w[(y_true == c) & (y_pred != c)].sum()
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f = 2 * p * r / (p + r) if p + r else 0.0
        support = float(w[y_true == c].sum())
        per_class[c] = {"precision": p, "recall": r, "f1": f, "support": support}
        f1s.append(f)
        supports.append(support)
    sup = np.array(supports)
    cols = [*classes, *[e for e in extra_pred_labels if e not in classes]]
    cm = pd.DataFrame(0.0, index=classes, columns=cols)
    for t in classes:
        for p in cols:
            cm.loc[t, p] = w[(y_true == t) & (y_pred == p)].sum()
    return {
        "per_class": per_class,
        "macro_f1": float(np.mean(f1s)) if f1s else 0.0,
        "weighted_f1": float(np.dot(f1s, sup) / sup.sum()) if sup.sum() else 0.0,
        "confusion_matrix": {
            "labels_true": classes,
            "labels_pred": cols,
            "values": cm.values.tolist(),
        },
    }


def alarm_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    normal_label: str,
    sample_weight: np.ndarray | None = None,
) -> dict[str, float]:
    """Binary view of a multiclass decision: any non-Normal prediction is an alarm.

    Returns weighted detection recall (attack frames that raised an alarm),
    alarm FPR (Normal frames that raised one) and alarm precision.
    """
    y_true = np.asarray(y_true).astype(str)
    alarm = np.asarray(y_pred).astype(str) != normal_label
    attack = y_true != normal_label
    w = np.ones(len(y_true)) if sample_weight is None else np.asarray(sample_weight, float)
    tp, fp = w[attack & alarm].sum(), w[~attack & alarm].sum()
    return {
        "detection_recall": float(tp / w[attack].sum()) if attack.any() else 0.0,
        "alarm_fpr": float(fp / w[~attack].sum()) if (~attack).any() else 0.0,
        "alarm_precision": float(tp / (tp + fp)) if tp + fp else 0.0,
    }


def expected_calibration_error(
    y_true_idx: np.ndarray,
    proba: np.ndarray,
    n_bins: int = 15,
    sample_weight: np.ndarray | None = None,
) -> float:
    """Expected calibration error of the top-class probability (0 = perfectly calibrated)."""
    conf = proba.max(axis=1)
    correct = (proba.argmax(axis=1) == np.asarray(y_true_idx)).astype(float)
    w = np.ones(len(conf)) if sample_weight is None else np.asarray(sample_weight, float)
    edges = np.linspace(0, 1, n_bins + 1)
    ece, total = 0.0, w.sum()
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            wb = w[m].sum()
            ece += (
                wb
                / total
                * abs(np.average(correct[m], weights=w[m]) - np.average(conf[m], weights=w[m]))
            )
    return float(ece)


def plot_confusion_matrix(cm: Mapping[str, Any], path: str | Path, title: str) -> None:
    """Row-normalised confusion-matrix heatmap with raw counts annotated."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    vals = np.asarray(cm["values"], dtype=float)
    norm = vals / np.maximum(vals.sum(axis=1, keepdims=True), 1e-12)
    fig, ax = plt.subplots(figsize=(1.6 + 1.3 * vals.shape[1], 1.2 + 1.0 * vals.shape[0]))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(vals.shape[1]), cm["labels_pred"], rotation=30, ha="right")
    ax.set_yticks(range(vals.shape[0]), cm["labels_true"])
    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            ax.text(
                j,
                i,
                f"{norm[i, j]:.3f}\n({int(round(vals[i, j])):,})",
                ha="center",
                va="center",
                fontsize=7,
                color="white" if norm[i, j] > 0.5 else "black",
            )
    ax.set(xlabel="Predicted", ylabel="True", title=title)
    fig.colorbar(im, ax=ax, fraction=0.046, label="row share")
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_json(obj: Mapping[str, Any], path: str | Path) -> None:
    """Write metrics as pretty JSON (numpy types converted)."""

    def default(o: Any) -> Any:
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(type(o))

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2, default=default), encoding="utf-8")


def plot_roc_pr(
    curves: Mapping[str, Mapping[str, Mapping[str, list[float]]]],
    path: str | Path,
    title: str,
    fpr_marker: float | None = None,
) -> None:
    """Save a side-by-side ROC / PR figure for several models (``curves[name] = curve_points``)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    for name, c in curves.items():
        ax1.plot(c["roc"]["fpr"], c["roc"]["tpr"], label=name)
        ax2.plot(c["pr"]["recall"], c["pr"]["precision"], label=name)
    if fpr_marker:
        ax1.axvline(fpr_marker, ls="--", c="grey", lw=1, label=f"FPR = {fpr_marker:g}")
    ax1.set(
        xscale="log",
        xlim=(1e-5, 1),
        ylim=(0, 1.02),
        xlabel="False-positive rate (log)",
        ylabel="True-positive rate",
        title="ROC",
    )
    ax2.set(
        xlim=(0, 1), ylim=(0, 1.02), xlabel="Recall", ylabel="Precision", title="Precision-Recall"
    )
    for ax in (ax1, ax2):
        ax.grid(alpha=0.3)
        ax.legend(loc="lower right" if ax is ax1 else "upper right", fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
