"""Global random seeding for reproducible experiments.

Design decision (viva): randomness enters the pipeline in several places -
Isolation Forest sub-sampling, RandomForest bootstrapping, XGBoost column
sampling, undersampling of the Normal class, hyperparameter search and LSTM
weight initialisation. Fixing every seed means rerunning an experiment gives
the same numbers, so reported results can be verified.

Note: scikit-learn and XGBoost estimators should ALSO receive the seed
explicitly through ``random_state=`` (read from ``model.yaml``); the global
NumPy seed alone is not a reliable guarantee for them.
"""

from __future__ import annotations

import os
import random

import numpy as np

_MAX_SEED = 2**32 - 1  # NumPy's legacy seeding accepts 0 .. 2**32 - 1


def set_seed(seed: int = 42, deterministic_torch: bool = True) -> int:
    """Seed Python, NumPy and (if installed) PyTorch.

    Args:
        seed: Non-negative integer seed.
        deterministic_torch: Ask PyTorch for deterministic kernels (slightly
            slower, but repeatable). Ignored when torch is not installed.

    Returns:
        The seed that was applied.

    Raises:
        ValueError: If ``seed`` is not an int in ``[0, 2**32 - 1]``.
    """
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= _MAX_SEED:
        raise ValueError(f"seed must be an int in [0, {_MAX_SEED}], got {seed!r}")

    # Only affects subprocesses started after this point (hashing in the
    # current interpreter is already initialised), e.g. joblib workers.
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch  # optional dependency (requirements-dl.txt)
    except ImportError:
        return seed

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic_torch:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
    return seed


def get_rng(seed: int = 42) -> np.random.Generator:
    """Return an independent NumPy ``Generator`` (preferred over the global state).

    Args:
        seed: Seed for the generator.

    Returns:
        A seeded ``numpy.random.Generator``.
    """
    return np.random.default_rng(seed)
