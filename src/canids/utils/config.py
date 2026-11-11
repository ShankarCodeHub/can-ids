"""YAML configuration loading.

Design decision (viva): every tunable value - paths, split ratios, window sizes,
hyperparameters, thresholds - lives in ``configs/*.yaml`` instead of being
hard-coded. An experiment is then fully described by (code version + config
files + seed), which is what makes results reproducible and auditable.

``yaml.safe_load`` is used (never ``yaml.load``) because the full loader can
construct arbitrary Python objects from a malicious file.
"""

from __future__ import annotations

import copy
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

#: Environment variable that overrides automatic project-root detection.
ROOT_ENV_VAR = "CANIDS_ROOT"
#: Environment variable that overrides the configs directory.
CONFIG_DIR_ENV_VAR = "CANIDS_CONFIG_DIR"

_MISSING: Any = object()


def find_project_root(start: str | Path | None = None) -> Path:
    """Locate the repository root (the folder holding ``pyproject.toml`` and ``configs/``).

    Search order: ``$CANIDS_ROOT`` -> upward from ``start`` (default: current
    working directory) -> upward from this file's location -> current directory.

    Args:
        start: Directory to start searching from.

    Returns:
        Absolute path of the project root.
    """
    env_root = os.environ.get(ROOT_ENV_VAR)
    if env_root:
        return Path(env_root).expanduser().resolve()

    candidates = [Path(start) if start is not None else Path.cwd(), Path(__file__).parent]
    for candidate in candidates:
        candidate = candidate.resolve()
        for directory in (candidate, *candidate.parents):
            if (directory / "pyproject.toml").is_file() and (directory / "configs").is_dir():
                return directory
    return Path.cwd().resolve()


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Read a YAML file whose top level is a mapping.

    Args:
        path: Path to the YAML file.

    Returns:
        Parsed mapping (an empty dict for an empty file).

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the top-level YAML value is not a mapping.
    """
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"Config file not found: {file_path}")
    with file_path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"Top level of {file_path} must be a mapping, got {type(data).__name__}")
    return data


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` into ``base`` without mutating either.

    Nested mappings are merged key by key; any other value in ``override``
    replaces the value in ``base``.

    Args:
        base: Default values.
        override: Values that take precedence.

    Returns:
        A new merged dictionary.
    """
    merged: dict[str, Any] = copy.deepcopy(dict(base))
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def load_config(
    name: str,
    config_dir: str | Path | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Load a named project config, e.g. ``load_config("model")``.

    Args:
        name: Config name (``"data"``, ``"model"``, ``"app"``) or a file name
            ending in ``.yaml``/``.yml``.
        config_dir: Directory holding the configs. Defaults to
            ``$CANIDS_CONFIG_DIR`` or ``<project root>/configs``.
        overrides: Optional nested values merged on top (handy for experiments
            and tests, e.g. ``{"splits": {"train": 0.7}}``).

    Returns:
        The configuration dictionary.
    """
    if config_dir is None:
        config_dir = os.environ.get(CONFIG_DIR_ENV_VAR) or find_project_root() / "configs"
    file_name = name if name.endswith((".yaml", ".yml")) else f"{name}.yaml"
    cfg = load_yaml(Path(config_dir) / file_name)
    if overrides:
        cfg = deep_merge(cfg, overrides)
    return cfg


def get_value(cfg: Mapping[str, Any], dotted_key: str, default: Any = _MISSING) -> Any:
    """Fetch a nested value with a dotted path, e.g. ``"hybrid.t_attack"``.

    Args:
        cfg: Configuration mapping.
        dotted_key: Keys separated by dots.
        default: Returned when the key is missing. If omitted, a KeyError is raised.

    Returns:
        The value found at the path.

    Raises:
        KeyError: If the path is missing and no default was given.
    """
    node: Any = cfg
    for part in dotted_key.split("."):
        if isinstance(node, Mapping) and part in node:
            node = node[part]
        elif default is _MISSING:
            raise KeyError(f"Config key not found: {dotted_key!r} (failed at {part!r})")
        else:
            return default
    return node


def resolve_path(path: str | Path, root: str | Path | None = None) -> Path:
    """Turn a (possibly relative) config path into an absolute path.

    Relative paths in configs are interpreted relative to the project root, so
    scripts behave the same no matter which directory they are launched from.

    Args:
        path: Path from a config file.
        root: Base directory; defaults to the project root.

    Returns:
        Absolute path.
    """
    p = Path(path).expanduser()
    if p.is_absolute():
        return p
    base = Path(root) if root is not None else find_project_root()
    return (base / p).resolve()
