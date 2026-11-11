"""Phase 9: integration test on the shipped models/demo data + coverage of thin modules."""

from __future__ import annotations

import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from canids.alerting.store import AlertStore
from canids.evaluation.ablation import fit_variant
from canids.evaluation.runtime import ResourceMonitor, percentiles
from canids.ingestion.schema import CANFrame, frames_to_dataframe, pad_payload
from canids.preprocessing.datasets import count_labels, load_split
from canids.preprocessing.features import FEATURE_COLUMNS
from canids.utils.provenance import git_commit
from canids.utils.seed import set_seed

ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "data" / "demo" / "road_fuzzing_attack_3.parquet"
MODELS = ROOT / "models" / "road"


# ------------------------------------------------------------------ integration (shipped artefacts)
@pytest.mark.integration
@pytest.mark.skipif(
    not (DEMO.exists() and (MODELS / "hybrid").exists()),
    reason="needs committed demo data + trained ROAD models",
)
def test_replay_10k_frames_produces_alerts(tmp_path: Path) -> None:
    """Replay >= 10k frames of a fuzzing capture through the real models -> fuzzing alerts."""
    from canids.pipeline.realtime import MonitorSession

    store = AlertStore(tmp_path / "it.sqlite")
    session = MonitorSession(MODELS, store, 100, 50.0, 500.0, {}, explain=False)
    session.start(DEMO, speed=0, interface="virtual", virtual_channel=f"it{time.time_ns()}")
    session.wait(timeout=120)
    s = session.snapshot()
    assert s["error"] == ""
    assert s["frames"] >= 10_000
    hits = s["truth_confusion"].get(("Fuzzy", "Fuzzy"), 0)
    total_fuzzy = sum(v for (t, _), v in s["truth_confusion"].items() if t == "Fuzzy")
    assert total_fuzzy > 0 and hits / total_fuzzy > 0.9
    ev = store.query(session=s["session"])
    assert (ev["predicted_class"] == "Fuzzy").any()


# ------------------------------------------------------------------ datasets loader
def _processed_tree(root: Path) -> None:
    shutil.copytree(ROOT / "configs", root / "configs")
    (root / "pyproject.toml").write_text("", "utf-8")
    rng = np.random.default_rng(0)
    for split in ("train", "val"):
        d = root / "data" / "processed" / "toy" / split
        d.mkdir(parents=True)
        n = 1000
        df = pd.DataFrame(rng.normal(size=(n, len(FEATURE_COLUMNS))), columns=FEATURE_COLUMNS)
        df["label"] = ["Normal"] * 950 + ["Spoofing"] * 50
        df["attack_type"] = np.where(df["label"] == "Normal", "none", "x")
        df["source_file"], df["split"], df["timestamp"] = "a.log", split, np.arange(n) * 0.01
        df.to_parquet(d / "a.parquet")


def test_load_split_subsamples_normal_keeps_attacks(tmp_path: Path, monkeypatch) -> None:
    _processed_tree(tmp_path)
    monkeypatch.setenv("CANIDS_ROOT", str(tmp_path))
    assert count_labels("toy", "train").to_dict() == {"Normal": 950, "Spoofing": 50}
    df = load_split("toy", "train", ["win_rate_fps"], max_normal=100, seed=1)
    assert (df["label"] == "Spoofing").sum() == 50
    assert 50 <= (df["label"] == "Normal").sum() <= 160
    only = load_split("toy", "train", ["win_rate_fps"], normal_only=True)
    assert set(only["label"]) == {"Normal"} and len(only) == 950
    with pytest.raises(FileNotFoundError):
        load_split("toy", "test")


# ------------------------------------------------------------------ ablation variant fit
def test_fit_variant_on_synthetic() -> None:
    rng = np.random.default_rng(0)
    n = 1200
    X = pd.DataFrame(rng.normal(size=(n, len(FEATURE_COLUMNS))), columns=FEATURE_COLUMNS)
    y = np.array(["Normal"] * 1100 + ["Spoofing"] * 100)
    X.loc[1100:, "win_cur_id_share"] += 5
    common = dict(
        train_normal=X[y == "Normal"],
        train=X,
        y_train=y,
        val=X,
        y_val=y,
        w_val=np.ones(n),
        if_params={"n_estimators": 20, "max_samples": 128, "max_features": 1.0},
        clf_kind="xgboost",
        clf_params={"n_estimators": 20, "max_depth": 3},
        t_normal=0.9,
        t_attack_grid=[0.5, 0.9],
        t_suppress_grid=[0.5, 1.01],
        target_fpr=0.05,
        seed=0,
    )
    v = fit_variant("no_window", ["win_rate_fps"], True, **common)
    assert "win_rate_fps" not in v.features and v.anomaly is not None
    assert (v.decide(X) == "Spoofing").sum() > 50
    v2 = fit_variant("no_iforest", [], False, **common)
    assert v2.anomaly is None and v2.thresholds.t_suppress == 1.01


# ------------------------------------------------------------------ runtime / provenance / seed
def test_resource_monitor_and_percentiles() -> None:
    with ResourceMonitor(interval_s=0.01) as mon:
        sum(i * i for i in range(200_000))
    r = mon.result()
    assert r["wall_s"] > 0 and r["peak_rss_mb"] > 0 and r["cpu_percent"] >= 0
    p = percentiles([1.0, 2.0, 3.0, 4.0])
    assert p["p50_ms"] == pytest.approx(2.5) and p["max_ms"] == 4.0
    assert np.isnan(percentiles([])["p50_ms"])


def test_git_commit_returns_string() -> None:
    c = git_commit()
    assert isinstance(c, str) and c


def test_set_seed_with_torch_absent_or_present() -> None:
    assert set_seed(5, deterministic_torch=False) == 5


def test_frames_to_dataframe_empty() -> None:
    assert frames_to_dataframe([]).empty
    df = frames_to_dataframe([CANFrame(0.0, 1, 1, pad_payload([1]))])
    assert len(df) == 1
