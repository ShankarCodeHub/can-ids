"""Train / validation / test splitting, leakage checks and class-balance reporting.

Design decisions (viva)
-----------------------
Random row-level splitting is WRONG for CAN logs: neighbouring frames are almost
identical, so a random split puts near-copies of test frames into training and
inflates every metric. Two leakage-safe strategies are provided, chosen per
dataset in ``data.yaml`` BEFORE any model is trained:

``chronological`` (HCRL files, ROAD ambient captures)
    Each source file is cut by time into contiguous blocks:
    first 60 % of its duration -> train, next 20 % -> val, last 20 % -> test.
    Training never sees a frame recorded after a validation/test frame of the
    same file. The first ``purge_frames`` frames of the val and test blocks are
    marked ``purged`` and excluded, so no evaluation frame's rolling window
    (which looks back ``window_frames`` frames) overlaps the training block.

``capture`` (ROAD attack captures)
    ROAD's injection windows sit at very different points of each capture, so
    a per-file 60/20/20 cut left validation with ZERO fuzzing frames and 13 of
    29 attack captures with no attack frames in test (measured, 2026-11-11).
    ROAD provides 3 independent recordings of most attacks, so whole captures
    are assigned instead (``_1`` -> train, ``_2`` -> val, ``_3`` -> test). This
    is stricter than a time split: test captures are completely unseen
    recordings. A ``_masquerade`` file always goes to the same split as its
    original, because both are derived from the same physical recording.

Class imbalance is handled on the TRAIN split only (class weights or controlled
undersampling of Normal); validation and test keep their natural distribution
so the reported metrics reflect real traffic.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from canids.ingestion.schema import LABEL_NORMAL

TRAIN, VAL, TEST, PURGED = "train", "val", "test", "purged"
SPLITS: tuple[str, ...] = (TRAIN, VAL, TEST)
MASQUERADE_SUFFIX = "_masquerade"


# =========================================================================== assignment
def _check_ratios(ratios: Sequence[float]) -> tuple[float, float, float]:
    if len(ratios) != 3 or any(r <= 0 for r in ratios) or not np.isclose(sum(ratios), 1.0):
        raise ValueError(f"ratios must be 3 positive numbers summing to 1, got {ratios}")
    return float(ratios[0]), float(ratios[1]), float(ratios[2])


def chronological_split(
    timestamps: np.ndarray, ratios: Sequence[float] = (0.6, 0.2, 0.2), purge_frames: int = 0
) -> np.ndarray:
    """Assign train/val/test by time within ONE file (timestamps must be sorted).

    Boundaries are placed at 60 % and 80 % of the file's DURATION. The first
    ``purge_frames`` frames after each boundary are labelled ``purged``.

    Returns:
        Array of split names, same length as ``timestamps``.
    """
    r_train, r_val, _ = _check_ratios(ratios)
    ts = np.asarray(timestamps, dtype=np.float64)
    out = np.full(len(ts), TRAIN, dtype=object)
    if len(ts) == 0:
        return out
    if np.any(np.diff(ts) < 0):
        raise ValueError("timestamps must be sorted (run clean_frames first)")
    t0, t1 = ts[0], ts[-1]
    b1 = t0 + r_train * (t1 - t0)
    b2 = t0 + (r_train + r_val) * (t1 - t0)
    i1 = int(np.searchsorted(ts, b1, side="left"))
    i2 = int(np.searchsorted(ts, b2, side="left"))
    out[i1:i2] = VAL
    out[i2:] = TEST
    if purge_frames > 0:
        out[i1 : min(i1 + purge_frames, i2)] = PURGED
        out[i2 : min(i2 + purge_frames, len(ts))] = PURGED
    return out


def capture_name(source_file: str) -> str:
    """Bare capture name from a source path or interim file stem.

    ``'attacks/fuzzing_attack_1.log'`` and ``'attacks__fuzzing_attack_1'`` (the
    interim parquet naming) both give ``'fuzzing_attack_1'``.
    """
    base = str(source_file).replace("\\", "/").split("/")[-1]
    return re.sub(r"\.[^.]+$", "", base).split("__")[-1]


def recording_group(source_file: str) -> str:
    """Physical recording a file comes from (masquerade files share their original's group)."""
    name = capture_name(source_file)
    return name[: -len(MASQUERADE_SUFFIX)] if name.endswith(MASQUERADE_SUFFIX) else name


def assign_capture_split(source_file: str, rules: Sequence[Sequence[str]]) -> str:
    """Pick a split for a whole capture using ordered ``[regex, split]`` rules (first match wins).

    Raises:
        ValueError: if no rule matches (every capture must be assigned explicitly).
    """
    name = capture_name(source_file)
    for pattern, split in rules:
        if split not in SPLITS:
            raise ValueError(f"rule {pattern!r} maps to unknown split {split!r}")
        if re.search(pattern, name):
            return split
    raise ValueError(f"no capture_assignment rule matches {name!r}")


def assign_splits(
    df: pd.DataFrame,
    strategy: str,
    ratios: Sequence[float] = (0.6, 0.2, 0.2),
    purge_frames: int = 0,
    capture_rules: Sequence[Sequence[str]] | None = None,
) -> np.ndarray:
    """Return the split of every frame of ONE cleaned source file."""
    if strategy == "chronological":
        return chronological_split(df["timestamp"].to_numpy(), ratios, purge_frames)
    if strategy == "capture":
        if not capture_rules:
            raise ValueError("strategy 'capture' needs capture_rules")
        name = str(df["source_file"].iloc[0]) if len(df) else ""
        return np.full(len(df), assign_capture_split(name, capture_rules), dtype=object)
    raise ValueError(f"unknown split strategy {strategy!r}")


# =========================================================================== leakage checks
def split_index(df: pd.DataFrame, split: np.ndarray, strategy: str) -> pd.DataFrame:
    """Compact per-(file, split) summary used by :func:`leakage_check`."""
    tmp = pd.DataFrame(
        {
            "source_file": df["source_file"].astype(str).to_numpy(),
            "timestamp": df["timestamp"].to_numpy(),
            "split": split,
        }
    )
    idx = (
        tmp.groupby(["source_file", "split"])["timestamp"]
        .agg(ts_min="min", ts_max="max", frames="size")
        .reset_index()
    )
    idx["strategy"] = strategy
    return idx


def leakage_check(index: pd.DataFrame) -> list[str]:
    """Return a list of problems (empty list = no leakage detected).

    Checks:
    1. chronological files: train block ends before val starts, val before test.
    2. capture files: each file lies in exactly one split.
    3. a recording's original and ``_masquerade`` files share one split.
    """
    problems: list[str] = []
    order = {TRAIN: 0, VAL: 1, TEST: 2}
    for (source_file, strategy), g in index[index["split"] != PURGED].groupby(
        ["source_file", "strategy"]
    ):
        if strategy == "chronological":
            blocks = g.assign(o=g["split"].map(order)).sort_values("o")
            for (_, a), (_, b) in zip(
                blocks.iloc[:-1].iterrows(), blocks.iloc[1:].iterrows(), strict=True
            ):
                if a["ts_max"] >= b["ts_min"]:
                    problems.append(
                        f"{source_file}: {a['split']} block (ends {a['ts_max']}) overlaps "
                        f"{b['split']} block (starts {b['ts_min']})"
                    )
        elif strategy == "capture" and g["split"].nunique() != 1:
            problems.append(f"{source_file}: capture split across {sorted(g['split'])}")

    caps = index[(index["strategy"] == "capture") & (index["split"] != PURGED)]
    groups = caps.assign(group=caps["source_file"].map(recording_group))
    for group, g in groups.groupby("group"):
        if g["split"].nunique() > 1:
            problems.append(f"recording {group}: original/masquerade in different splits")
    return problems


def cross_split_duplicates(df: pd.DataFrame, split: np.ndarray) -> int:
    """Count frames of the same recording that appear (identically) in more than one split."""
    key = pd.DataFrame(
        {
            "group": df["source_file"].astype(str).map(recording_group).to_numpy(),
            "timestamp": df["timestamp"].to_numpy(),
            "can_id": df["can_id"].to_numpy(),
            "split": split,
        }
    )
    key = key[key["split"] != PURGED]
    n_splits = key.groupby(["group", "timestamp", "can_id"])["split"].nunique()
    return int((n_splits > 1).sum())


# =========================================================================== balance
def class_balance_report(labels: pd.Series, split: Iterable[str]) -> pd.DataFrame:
    """Rows = split, columns = label counts + total + attack share (%)."""
    tab = pd.crosstab(pd.Series(list(split), name="split"), labels.astype(str).to_numpy())
    tab = tab.reindex([s for s in (*SPLITS, PURGED) if s in tab.index])
    tab["total"] = tab.sum(axis=1)
    normal = tab.get(LABEL_NORMAL, 0)
    tab["attack_share_pct"] = (100 * (tab["total"] - normal) / tab["total"]).round(4)
    return tab


def compute_class_weights(counts: Mapping[str, int]) -> dict[str, float]:
    """'Balanced' weights ``n_total / (n_classes * n_class)`` from TRAIN counts.

    Same formula as scikit-learn's ``class_weight="balanced"``; saved as an
    artifact so the weighting is documented and reused consistently.
    """
    present = {k: int(v) for k, v in counts.items() if v > 0}
    total = sum(present.values())
    return {k: total / (len(present) * v) for k, v in present.items()}


def undersample_normal(
    train: pd.DataFrame, ratio: float, seed: int, label_col: str = "label"
) -> pd.DataFrame:
    """Keep all attack frames and at most ``ratio`` Normal frames per attack frame (TRAIN only).

    Sampling is random but seeded. Row order (time order) is preserved so the
    result can still be inspected chronologically.
    """
    if ratio <= 0:
        raise ValueError("ratio must be > 0")
    is_normal = (train[label_col].astype(str) == LABEL_NORMAL).to_numpy()
    n_attack = int((~is_normal).sum())
    normal_idx = np.flatnonzero(is_normal)
    keep_n = min(len(normal_idx), int(np.ceil(ratio * n_attack)))
    rng = np.random.default_rng(seed)
    keep = rng.choice(normal_idx, size=keep_n, replace=False)
    mask = ~is_normal
    mask[keep] = True
    return train.loc[mask]
