"""Provenance helpers: record which code version produced an artifact or result."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from canids.utils.config import find_project_root

_GIT_FALLBACKS = (r"C:\Program Files\Git\cmd\git.exe",)


def _git_executable() -> str | None:
    found = shutil.which("git")
    if found:
        return found
    return next((p for p in _GIT_FALLBACKS if Path(p).exists()), None)


def git_commit() -> str:
    """Current git commit hash; ``'uncommitted'`` if git or a commit is unavailable."""
    git = _git_executable()
    if git is None:
        return "uncommitted"
    try:
        r = subprocess.run(
            [git, "rev-parse", "HEAD"],
            cwd=find_project_root(),
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
        return r.stdout.strip() or "uncommitted"
    except (OSError, subprocess.SubprocessError):
        return "uncommitted"
