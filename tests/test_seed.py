"""Tests for canids.utils.seed."""

from __future__ import annotations

import random

import numpy as np
import pytest

from canids.utils.seed import get_rng, set_seed


def test_set_seed_makes_python_and_numpy_repeatable() -> None:
    set_seed(123)
    first = (random.random(), np.random.rand(3).tolist())
    set_seed(123)
    second = (random.random(), np.random.rand(3).tolist())
    assert first == second


def test_different_seeds_differ() -> None:
    set_seed(1)
    a = np.random.rand(5)
    set_seed(2)
    b = np.random.rand(5)
    assert not np.allclose(a, b)


@pytest.mark.parametrize("bad", [-1, 2**32, 1.5, "42", True])
def test_invalid_seed_rejected(bad: object) -> None:
    with pytest.raises(ValueError):
        set_seed(bad)  # type: ignore[arg-type]


def test_get_rng_is_independent_and_repeatable() -> None:
    assert get_rng(7).integers(0, 1000, 5).tolist() == get_rng(7).integers(0, 1000, 5).tolist()
