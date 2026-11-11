"""Smoke tests: the package and every (placeholder) module import cleanly."""

from __future__ import annotations

import importlib
import pkgutil

import pytest

import canids


def test_version_is_set() -> None:
    assert canids.__version__ == "0.1.0"


def _all_module_names() -> list[str]:
    return [m.name for m in pkgutil.walk_packages(canids.__path__, prefix="canids.")]


def test_expected_subpackages_exist() -> None:
    names = set(_all_module_names())
    for sub in (
        "ingestion",
        "preprocessing",
        "models",
        "explain",
        "alerting",
        "evaluation",
        "pipeline",
        "utils",
    ):
        assert f"canids.{sub}" in names


@pytest.mark.parametrize("module_name", _all_module_names())
def test_module_imports_and_has_docstring(module_name: str) -> None:
    module = importlib.import_module(module_name)
    assert module.__doc__ and module.__doc__.strip(), f"{module_name} needs a docstring"
