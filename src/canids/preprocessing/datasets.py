"""Memory-bounded loading of processed feature splits.

ROAD's processed training split has about 16 million rows, more than this
16 GB laptop can comfortably hold together with models. :func:`load_split`
streams each parquet file in record batches and can keep a seeded random
subsample of Normal rows while keeping EVERY attack row. Subsampling Normal
frames only reduces compute (Isolation Forest itself only looks at
``max_samples`` rows per tree); it never removes attack evidence.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from canids.ingestion.schema import LABEL_NORMAL
from canids.utils.config import load_config, resolve_path


def split_dir(dataset: str, split: str) -> Path:
    """``data/processed/<dataset>/<split>``."""
    return resolve_path(load_config("data")["paths"]["processed"]) / dataset / split


def count_labels(dataset: str, split: str) -> pd.Series:
    """Rows per label in a processed split (reads only the label column)."""
    total: pd.Series | None = None
    for f in sorted(split_dir(dataset, split).glob("*.parquet")):
        c = pq.read_table(f, columns=["label"]).column("label").to_pandas().value_counts()
        total = c if total is None else total.add(c, fill_value=0)
    if total is None:
        raise FileNotFoundError(f"no processed files for {dataset}/{split} - run build_features.py")
    return total.astype(int)


def load_split(
    dataset: str,
    split: str,
    columns: Sequence[str] | None = None,
    max_normal: int | None = None,
    seed: int = 42,
    normal_only: bool = False,
) -> pd.DataFrame:
    """Load a processed split, optionally subsampling Normal rows.

    Args:
        dataset: ``"road"`` or ``"hcrl"``.
        split: ``"train"``, ``"val"`` or ``"test"``.
        columns: Columns to read (``label`` is always added).
        max_normal: Keep about this many Normal rows (Bernoulli sampling with a
            fixed seed); ``None`` keeps all of them.
        seed: RNG seed for the subsample.
        normal_only: Drop attack rows (used to train the anomaly detector).
    """
    cols = None if columns is None else list(dict.fromkeys([*columns, "label"]))
    files = sorted(split_dir(dataset, split).glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no processed files for {dataset}/{split}")
    p_keep = 1.0
    if max_normal is not None:
        n_normal = int(count_labels(dataset, split).get(LABEL_NORMAL, 0))
        p_keep = min(1.0, max_normal / max(n_normal, 1))
    rng = np.random.default_rng(seed)
    parts = []
    for f in files:
        for batch in pq.ParquetFile(f).iter_batches(batch_size=500_000, columns=cols):
            df = batch.to_pandas()
            is_normal = (df["label"] == LABEL_NORMAL).to_numpy()
            keep = ~is_normal if not normal_only else np.zeros(len(df), dtype=bool)
            if p_keep >= 1.0:
                keep |= is_normal
            else:
                keep |= is_normal & (rng.random(len(df)) < p_keep)
            if keep.any():
                parts.append(df.loc[keep])
    out = pd.concat(parts, ignore_index=True)
    for col in ("label", "attack_type", "source_file", "split"):
        if col in out.columns:
            out[col] = out[col].astype("category")
    return out
