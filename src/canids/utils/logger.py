"""Project-wide logging: console + size-rotating log file.

Design decision (viva): all modules log through children of one ``"canids"``
logger (e.g. ``canids.ingestion.hcrl_loader``). Configuring that single parent
once gives every module the same format and destinations. A rotating file
handler caps disk usage (important for long replays and the Raspberry Pi),
while the file keeps a permanent record of data-cleaning decisions such as
"dropped 12 malformed rows" for the report.
"""

from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

ROOT_LOGGER_NAME = "canids"
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging(
    level: int | str = "INFO",
    log_dir: str | Path | None = None,
    log_file: str = "canids.log",
    max_bytes: int = 5 * 1024 * 1024,
    backup_count: int = 3,
    console: bool = True,
    force: bool = False,
) -> logging.Logger:
    """Configure the ``canids`` root logger.

    Calling it again is a no-op unless ``force=True`` (avoids duplicated lines
    when modules or Streamlit reruns import it repeatedly).

    Args:
        level: Logging level name or number.
        log_dir: Directory for the rotating log file; ``None`` = console only.
        log_file: Log file name inside ``log_dir``.
        max_bytes: Rotate when the file exceeds this size.
        backup_count: Number of rotated files to keep.
        console: Also log to stderr.
        force: Replace any existing handlers.

    Returns:
        The configured ``canids`` logger.
    """
    root = logging.getLogger(ROOT_LOGGER_NAME)
    if root.handlers and not force:
        return root

    for handler in list(root.handlers):
        root.removeHandler(handler)
        handler.close()

    root.setLevel(level.upper() if isinstance(level, str) else level)
    # Don't also pass records to Python's global root logger (prevents double printing
    # when third-party libraries call logging.basicConfig()).
    root.propagate = False
    formatter = logging.Formatter(LOG_FORMAT, DATE_FORMAT)

    if console:
        stream_handler = logging.StreamHandler(sys.stderr)
        stream_handler.setFormatter(formatter)
        root.addHandler(stream_handler)

    if log_dir is not None:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            log_path / log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
            delay=True,  # create the file only when the first record is written
        )
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)

    return root


def get_logger(name: str) -> logging.Logger:
    """Return a logger under the ``canids`` hierarchy.

    If logging has not been configured yet, it is set up with defaults
    (INFO, console + ``<project root>/logs/canids.log``).

    Args:
        name: Usually ``__name__``; names outside ``canids`` are prefixed with it.

    Returns:
        A ``logging.Logger``.
    """
    root = logging.getLogger(ROOT_LOGGER_NAME)
    if not root.handlers:
        # Imported lazily to avoid a circular import (config -> logger -> config).
        from canids.utils.config import find_project_root

        setup_logging(log_dir=find_project_root() / "logs")

    if name == ROOT_LOGGER_NAME or name.startswith(ROOT_LOGGER_NAME + "."):
        return logging.getLogger(name)
    return logging.getLogger(f"{ROOT_LOGGER_NAME}.{name}")
