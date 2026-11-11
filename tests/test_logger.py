"""Tests for canids.utils.logger."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path

import pytest

from canids.utils.logger import ROOT_LOGGER_NAME, get_logger, setup_logging


@pytest.fixture(autouse=True)
def _reset_canids_logger() -> Iterator[None]:
    """Leave the canids logger without handlers after each test."""
    yield
    root = logging.getLogger(ROOT_LOGGER_NAME)
    for handler in list(root.handlers):
        root.removeHandler(handler)
        handler.close()


def test_writes_to_rotating_file(tmp_path: Path) -> None:
    setup_logging(log_dir=tmp_path, console=False, force=True)
    get_logger("tests.demo").info("hello file")
    for handler in logging.getLogger(ROOT_LOGGER_NAME).handlers:
        handler.flush()
    text = (tmp_path / "canids.log").read_text(encoding="utf-8")
    assert "hello file" in text
    assert "canids.tests.demo" in text


def test_setup_is_idempotent(tmp_path: Path) -> None:
    setup_logging(log_dir=tmp_path, force=True)
    n = len(logging.getLogger(ROOT_LOGGER_NAME).handlers)
    setup_logging(log_dir=tmp_path)  # no force -> must not add duplicate handlers
    assert len(logging.getLogger(ROOT_LOGGER_NAME).handlers) == n == 2


def test_get_logger_naming(tmp_path: Path) -> None:
    setup_logging(log_dir=tmp_path, console=False, force=True)
    assert get_logger("canids.models.hybrid").name == "canids.models.hybrid"
    assert get_logger("scripts.train").name == "canids.scripts.train"
