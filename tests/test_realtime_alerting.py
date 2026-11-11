"""Phase 6 tests: alerts, aggregation, SQLite store, streaming pipeline, monitor session."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import IsolationForest
from xgboost import XGBClassifier

from canids.alerting.alert import Alert, AlertAggregator, severity_for
from canids.alerting.store import AlertStore
from canids.ingestion.schema import CANFrame, frames_to_dataframe, pad_payload
from canids.models.anomaly import AnomalyDetector
from canids.models.classifier import AttackClassifier
from canids.models.hybrid import HybridDetector, HybridThresholds
from canids.pipeline.realtime import MonitorSession, RealtimePipeline
from canids.preprocessing.features import FEATURE_COLUMNS, PerIdStats, features_dataframe


# ------------------------------------------------------------------ severity + aggregation
def test_severity_rules() -> None:
    assert severity_for("Spoofing", 0.95) == "High"
    assert severity_for("Spoofing", 0.6) == "Medium"  # uncertain -> one level down
    assert severity_for("Spoofing", 0.95, count=150) == "Critical"  # burst -> one level up
    assert severity_for("DoS", 0.99, count=500) == "Critical"  # capped
    assert severity_for("Unknown", 0.3) == "Medium"  # Unknown is never down-graded


def test_aggregator_merges_bursts_and_splits_on_gap() -> None:
    agg = AlertAggregator(window_ms=100)
    e1, new1 = agg.add(0.00, 0x0D0, "Spoofing", class_probability=0.9, anomaly_score=0.5)
    e2, new2 = agg.add(0.05, 0x0D0, "Spoofing", class_probability=0.95, anomaly_score=0.7)
    e3, new3 = agg.add(0.06, 0x6E0, "Spoofing", class_probability=0.9)  # other ID -> new event
    assert new1 and not new2 and new3 and e1 is e2
    assert e1.count == 2 and e1.last_timestamp == 0.05
    assert e1.class_probability == 0.95 and e1.anomaly_score == 0.7
    e4, new4 = agg.add(0.30, 0x0D0, "Spoofing", class_probability=0.9)  # gap > 100 ms
    assert new4 and e4 is not e1
    closed = agg.expire(0.5)
    assert {e.id for e in closed} >= {e3.id}


def test_fuzzy_aggregated_across_ids() -> None:
    agg = AlertAggregator(window_ms=100)
    for i, cid in enumerate([0x10, 0x20, 0x30, 0x20]):
        ev, _ = agg.add(i * 0.01, cid, "Fuzzy", class_probability=0.99)
    assert ev.count == 4 and ev.distinct_ids == 3
    assert len(agg.flush()) == 1


# ------------------------------------------------------------------ store
def test_store_roundtrip_filters_and_export(tmp_path: Path) -> None:
    store = AlertStore(tmp_path / "a.sqlite")
    a = Alert.new(
        timestamp=1.0,
        can_id=0x0D0,
        predicted_class="Spoofing",
        anomaly_score=0.5,
        class_probability=0.9,
        severity="High",
        session="s1",
        shap_top_features=[{"feature": "b1", "value": 1.0, "shap": 0.5}],
    )
    b = Alert.new(
        timestamp=2.0,
        can_id=0x123,
        predicted_class="Fuzzy",
        anomaly_score=0.6,
        class_probability=0.99,
        severity="High",
        session="s2",
    )
    assert store.upsert([a, b]) == 2
    a.count = 10
    store.upsert([a])  # update, not duplicate
    assert store.count() == 2
    df = store.query(classes=["Spoofing"])
    assert len(df) == 1 and df.iloc[0]["count"] == 10
    assert df.iloc[0]["shap_top_features"][0]["feature"] == "b1"
    assert len(store.query(can_ids=[0x123])) == 1
    assert len(store.query(start_ts=1.5)) == 1
    assert len(store.query(session="s1")) == 1
    assert store.export_csv(tmp_path / "e.csv") == 2
    assert len(pd.read_csv(tmp_path / "e.csv")) == 2
    store.clear(session="s1")
    assert store.count() == 1


def test_store_migrates_old_schema(tmp_path: Path) -> None:
    import sqlite3

    p = tmp_path / "old.sqlite"
    con = sqlite3.connect(p)
    con.execute(
        "CREATE TABLE alerts (id TEXT PRIMARY KEY, timestamp REAL NOT NULL, "
        "last_timestamp REAL NOT NULL, wall_time TEXT NOT NULL, can_id INTEGER NOT NULL, "
        "predicted_class TEXT NOT NULL, severity TEXT NOT NULL, count INTEGER NOT NULL, "
        "anomaly_score REAL, class_probability REAL, shap_top_features TEXT, "
        "explanation_text TEXT, source TEXT, session TEXT, true_label TEXT, rule TEXT, "
        "frame TEXT)"
    )
    con.commit()
    con.close()
    store = AlertStore(p)
    store.upsert(
        [
            Alert.new(
                timestamp=1.0,
                can_id=1,
                predicted_class="Fuzzy",
                anomaly_score=0.0,
                class_probability=0.9,
                severity="High",
                distinct_ids=4,
            )
        ]
    )
    assert store.query().iloc[0]["distinct_ids"] == 4


# ------------------------------------------------------------------ synthetic models
STATS = PerIdStats(
    known_ids=frozenset({0x100, 0x200}),
    period_mean={0x100: 0.01, 0x200: 0.02},
    period_std={0x100: 0.001, 0x200: 0.002},
)


def _traffic(n: int = 1500, seed: int = 0) -> pd.DataFrame:
    """Normal periodic traffic with an injected burst of 0x100 frames carrying 0xFF."""
    rng = np.random.default_rng(seed)
    frames, t = [], 0.0
    for i in range(n):
        t += 0.005
        cid = 0x100 if i % 2 else 0x200
        attack = 600 <= i < 700 and i % 2 == 1
        payload = [255] * 8 if attack else list(rng.integers(0, 50, 8))
        frames.append(
            CANFrame(
                t,
                cid,
                8,
                pad_payload(payload),
                label="Spoofing" if attack else "Normal",
                source_file="syn.log",
            )
        )
    return frames_to_dataframe(frames)


@pytest.fixture(scope="module")
def hybrid() -> HybridDetector:
    df = _traffic()
    X = features_dataframe(df, STATS)
    y = df["label"].astype(str).to_numpy()
    feats = list(FEATURE_COLUMNS)
    iforest = IsolationForest(n_estimators=50, random_state=0).fit(
        X[y == "Normal"][feats].to_numpy(np.float32)
    )
    scores = -iforest.score_samples(X[y == "Normal"][feats].to_numpy(np.float32))
    det = AnomalyDetector(iforest, feats, float(np.quantile(scores, 0.99)))
    ynum = (y == "Spoofing").astype(int)
    xgb = XGBClassifier(n_estimators=30, max_depth=3, random_state=0, n_jobs=1).fit(
        X[feats].to_numpy(np.float32), ynum
    )
    clf = AttackClassifier(xgb, feats, ["Normal", "Spoofing"])
    return HybridDetector(det, clf, HybridThresholds(0.9, 0.6, 0.5))


def test_streaming_matches_offline(hybrid: HybridDetector) -> None:
    """Train/serve parity: frame-by-frame pipeline == offline batch features + model."""
    df = _traffic(seed=1)
    offline = hybrid.predict(features_dataframe(df, STATS))["predicted_class"].tolist()
    frames = [
        CANFrame(
            r.timestamp,
            int(r.can_id),
            int(r.dlc),
            tuple(int(getattr(r, f"b{i}")) for i in range(8)),
        )
        for r in df.itertuples()
    ]
    pipe = RealtimePipeline(hybrid, STATS, batch_size=64)
    streamed = [d.predicted_class for d in pipe.process(frames)]
    assert streamed == offline
    assert pipe.stats.frames == len(df)
    assert "Spoofing" in streamed
    assert all(d >= 0 for d in pipe.stats.latencies_ms)


def test_pipeline_writes_aggregated_events(hybrid: HybridDetector, tmp_path: Path) -> None:
    store = AlertStore(tmp_path / "p.sqlite")
    df = _traffic(seed=2)
    frames = [
        CANFrame(
            r.timestamp,
            int(r.can_id),
            int(r.dlc),
            tuple(int(getattr(r, f"b{i}")) for i in range(8)),
        )
        for r in df.itertuples()
    ]
    pipe = RealtimePipeline(
        hybrid, STATS, aggregator=AlertAggregator(500), store=store, session="t"
    )
    list(pipe.process(frames))
    ev = store.query(session="t")
    assert not ev.empty
    assert ev["count"].sum() == pipe.stats.alert_frames
    assert len(ev) < pipe.stats.alert_frames  # bursts were collapsed


@pytest.mark.integration
def test_monitor_session_end_to_end(hybrid: HybridDetector, tmp_path: Path) -> None:
    root = tmp_path / "models"
    hybrid.anomaly.save(root / "anomaly")
    hybrid.classifier.save(root / "classifier")
    hybrid.save(root / "hybrid")
    STATS.to_json(root / "feature_artifacts" / "per_id_stats.json")
    log = tmp_path / "replay.parquet"
    _traffic(seed=3).to_parquet(log)
    store = AlertStore(tmp_path / "s.sqlite")
    session = MonitorSession(root, store, 100, 50.0, 500.0, {}, explain=True)
    session.start(log, speed=0, interface="virtual", virtual_channel=f"t{time.time_ns()}")
    session.wait(timeout=60)
    s = session.snapshot()
    assert s["error"] == ""
    assert s["frames"] == 1500 and s["replay_frames_sent"] == 1500
    assert s["truth_confusion"].get(("Spoofing", "Spoofing"), 0) > 0
    ev = store.query(session=s["session"])
    assert not ev.empty and ev["explanation_text"].str.len().max() > 0
    assert not session.running
