"""Integration test: build_dataset.py on a tiny synthetic project tree."""

from __future__ import annotations

import importlib.util
import json
import shutil
from pathlib import Path
from types import ModuleType

import pandas as pd
import pytest

from canids.ingestion.schema import validate_frame_df


def _load_script(project_root: Path) -> ModuleType:
    path = project_root / "scripts" / "build_dataset.py"
    spec = importlib.util.spec_from_file_location("build_dataset", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.integration
def test_build_dataset_end_to_end(
    tmp_path: Path, project_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    shutil.copytree(project_root / "configs", tmp_path / "configs")
    (tmp_path / "pyproject.toml").write_text("", encoding="utf-8")
    hcrl = tmp_path / "data/raw/hcrl"
    road = tmp_path / "data/raw/road"
    hcrl.mkdir(parents=True)
    (hcrl / "DoS_dataset.csv").write_text(
        "1.0,0316,8,05,21,68,09,21,21,00,6f,R\n1.1,0000,8,00,00,00,00,00,00,00,00,T\n"
        "1.2,0080,2,00,17,R\nbad,line\n",
        encoding="utf-8",
    )
    (road / "attacks").mkdir(parents=True)
    (road / "ambient").mkdir()
    (road / "ambient/ambient_1.log").write_text("(5.0) can0 0D0#0000000000000000\n", "utf-8")
    (road / "attacks/max_speedometer_attack_1.log").write_text(
        "(10.0) can0 0D0#0000000000000000\n(11.0) can0 0D0#0000FFFF00000000\n", "utf-8"
    )
    (road / "attacks/capture_metadata.json").write_text(
        json.dumps(
            {
                "max_speedometer_attack_1": {
                    "injection_id": "0D0",
                    "injection_data_str": "XXXXFFFF",
                    "injection_interval": [0.5, 2.0],
                }
            }
        ),
        "utf-8",
    )
    monkeypatch.setenv("CANIDS_ROOT", str(tmp_path))

    module = _load_script(project_root)
    assert module.main(["--checksums"]) == 0

    results = tmp_path / "reports/results"
    summary = pd.read_csv(results / "dataset_summary.csv", index_col="dataset")
    assert summary.loc["hcrl", "Normal"] == 2 and summary.loc["hcrl", "DoS"] == 1
    assert summary.loc["road", "Normal"] == 2 and summary.loc["road", "Spoofing"] == 1

    report = pd.read_csv(results / "ingestion_report.csv")
    hcrl_dos = report[report["source_file"] == "DoS_dataset.csv"].iloc[0]
    assert hcrl_dos["rows_dropped"] == 1
    assert report["skipped"].sum() == 4  # four HCRL files absent from the fixture

    parquet = tmp_path / "data/interim/road/attacks__max_speedometer_attack_1.parquet"
    validate_frame_df(pd.read_parquet(parquet).pipe(lambda d: d.astype({"label": "category"})))
    assert (results / "raw_checksums.csv").exists()


@pytest.mark.integration
def test_build_dataset_without_data_writes_no_summary(
    tmp_path: Path, project_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    shutil.copytree(project_root / "configs", tmp_path / "configs")
    (tmp_path / "pyproject.toml").write_text("", encoding="utf-8")
    monkeypatch.setenv("CANIDS_ROOT", str(tmp_path))
    module = _load_script(project_root)
    assert module.main([]) == 1
    assert not (tmp_path / "reports/results/dataset_summary.csv").exists()
