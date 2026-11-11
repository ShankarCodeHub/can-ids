"""Tests for canids.utils.config and the shipped YAML configs."""

from __future__ import annotations

from pathlib import Path

import pytest

from canids.utils.config import (
    deep_merge,
    find_project_root,
    get_value,
    load_config,
    load_yaml,
    resolve_path,
)


# ------------------------------------------------------------ load_yaml
def test_load_yaml_reads_mapping(tmp_path: Path) -> None:
    f = tmp_path / "c.yaml"
    f.write_text("a: 1\nb:\n  c: two\n", encoding="utf-8")
    assert load_yaml(f) == {"a": 1, "b": {"c": "two"}}


def test_load_yaml_empty_file_returns_empty_dict(tmp_path: Path) -> None:
    f = tmp_path / "empty.yaml"
    f.write_text("", encoding="utf-8")
    assert load_yaml(f) == {}


def test_load_yaml_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_yaml(tmp_path / "nope.yaml")


def test_load_yaml_non_mapping_raises(tmp_path: Path) -> None:
    f = tmp_path / "list.yaml"
    f.write_text("- 1\n- 2\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_yaml(f)


# ------------------------------------------------------------ helpers
def test_deep_merge_is_recursive_and_non_mutating() -> None:
    base = {"a": 1, "nested": {"x": 1, "y": 2}}
    merged = deep_merge(base, {"nested": {"y": 20, "z": 30}, "b": 2})
    assert merged == {"a": 1, "b": 2, "nested": {"x": 1, "y": 20, "z": 30}}
    assert base == {"a": 1, "nested": {"x": 1, "y": 2}}


def test_get_value_dotted_and_default() -> None:
    cfg = {"hybrid": {"t_attack": 0.7}}
    assert get_value(cfg, "hybrid.t_attack") == 0.7
    assert get_value(cfg, "hybrid.missing", default=None) is None
    with pytest.raises(KeyError):
        get_value(cfg, "hybrid.missing")


def test_resolve_path(tmp_path: Path) -> None:
    assert resolve_path("data/raw", root=tmp_path) == (tmp_path / "data/raw").resolve()
    assert resolve_path(tmp_path) == tmp_path


def test_find_project_root(project_root: Path) -> None:
    assert find_project_root(project_root / "src" / "canids") == project_root


# ------------------------------------------------------------ shipped configs
def test_data_config_is_valid(project_root: Path) -> None:
    cfg = load_config("data", config_dir=project_root / "configs")
    for key in ("paths", "splits", "windows", "schema", "seed"):
        assert key in cfg
    s = cfg["splits"]
    assert s["train"] + s["val"] + s["test"] == pytest.approx(1.0)
    assert set(cfg["schema"]["labels"]) == {"Normal", "DoS", "Fuzzy", "Spoofing"}


def test_model_config_thresholds_are_probabilities(project_root: Path) -> None:
    cfg = load_config("model", config_dir=project_root / "configs")
    assert 0.0 < cfg["anomaly"]["target_fpr"] < 1.0
    assert 0.0 <= cfg["hybrid"]["t_attack"] <= 1.0
    assert 0.0 <= cfg["hybrid"]["t_normal"] <= 1.0


def test_app_config_live_reader_is_listen_only(project_root: Path) -> None:
    cfg = load_config("app", config_dir=project_root / "configs")
    assert cfg["live"]["listen_only"] is True


def test_load_config_overrides(project_root: Path) -> None:
    cfg = load_config(
        "data", config_dir=project_root / "configs", overrides={"windows": {"time_ms": 100}}
    )
    assert cfg["windows"]["time_ms"] == 100
    assert cfg["windows"]["frame_count"] == 100  # untouched sibling key kept
