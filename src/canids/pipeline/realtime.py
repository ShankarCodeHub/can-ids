"""Real-time inference: frames -> streaming features -> hybrid model -> aggregated alerts.

Architecture (viva)
-------------------
::

    Replayer thread --(virtual bus / vcan0)--> PassiveReader
                                                  |
                         RealtimePipeline thread  v
       StreamingFeatureExtractor (per frame, same code as training)
                      -> micro-batch (<= batch_size frames or <= max_wait_ms)
                      -> Isolation Forest + XGBoost + hybrid rules (vectorised)
                      -> AlertAggregator -> AlertStore (SQLite)  ->  dashboard reads
                                         \-> explain queue -> SHAP thread -> store

* **Micro-batching.** Calling the models once per frame wastes most of the time
  on Python and library call overhead. Collecting frames for at most
  ``max_wait_ms`` (default 20 ms) and scoring them together raises throughput
  by about 2 orders of magnitude, while keeping decision latency in
  milliseconds. The batching wait is INCLUDED in the reported latency.
* **Threads plus queues.** The UI never blocks: replay, inference and SHAP
  explanation each run in their own thread, and the dashboard only reads
  snapshots and the database.
* **Passive.** The pipeline only consumes frames; only the replayer
  transmits, and only on an isolated virtual bus (see ``replayer.py``).
"""

from __future__ import annotations

import queue
import threading
import time
import uuid
from collections import Counter, deque
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from canids.alerting.alert import Alert, AlertAggregator
from canids.alerting.store import AlertStore
from canids.ingestion.live_reader import PassiveReader
from canids.ingestion.replayer import (
    Replayer,
    load_replay_frames,
    open_replay_bus,
    resolve_interface,
)
from canids.ingestion.schema import LABEL_NORMAL, CANFrame
from canids.models.hybrid import RULE_NAMES, HybridDetector, hybrid_decide
from canids.preprocessing.features import FEATURE_COLUMNS, PerIdStats, StreamingFeatureExtractor
from canids.utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class FrameDecision:
    """Decision for one frame (kept in a ring buffer for the live table)."""

    timestamp: float
    can_id: int
    predicted_class: str
    anomaly_score: float
    class_probability: float
    rule: str
    latency_ms: float
    true_label: str = ""
    event_id: str = ""


@dataclass
class PipelineStats:
    """Live counters; :meth:`snapshot` is safe to call from the UI thread."""

    frames: int = 0
    alert_frames: int = 0
    events: int = 0
    class_counts: Counter = field(default_factory=Counter)
    latencies_ms: deque = field(default_factory=lambda: deque(maxlen=50_000))
    per_second: deque = field(default_factory=lambda: deque(maxlen=600))  # (wall_s, frames, alerts)
    recent: deque = field(default_factory=lambda: deque(maxlen=500))  # FrameDecision of alerts
    started_wall: float = field(default_factory=time.time)
    last_alert_wall: float = 0.0
    last_alert_class: str = ""
    truth_confusion: Counter = field(default_factory=Counter)  # (true, predicted) when known

    def record_second(self, frames: int, alerts: int) -> None:
        """Accumulate counts into the current wall-clock second."""
        sec = int(time.time())
        if self.per_second and self.per_second[-1][0] == sec:
            s, f, a = self.per_second[-1]
            self.per_second[-1] = (s, f + frames, a + alerts)
        else:
            self.per_second.append((sec, frames, alerts))

    def latency_percentiles(self) -> dict[str, float]:
        """p50/p95/p99/max decision latency (ms) over the recent window."""
        if not self.latencies_ms:
            return {
                "p50": float("nan"),
                "p95": float("nan"),
                "p99": float("nan"),
                "max": float("nan"),
            }
        a = np.fromiter(self.latencies_ms, dtype=np.float64)
        return {
            "p50": float(np.percentile(a, 50)),
            "p95": float(np.percentile(a, 95)),
            "p99": float(np.percentile(a, 99)),
            "max": float(a.max()),
        }


class RealtimePipeline:
    """Streaming detector. Feed frames with :meth:`process` (sync) or :meth:`run` (loop).

    Args:
        hybrid: Loaded hybrid detector (stage 1 + stage 2 + thresholds).
        per_id_stats: Feature artifact fitted on the training split.
        window_frames / window_ms: Must match the training feature config.
        batch_size / max_wait_ms: Micro-batching limits.
        aggregator: Alert burst aggregator (None = no alert events).
        store: SQLite store for events (None = keep in memory only).
        explain_queue: Queue receiving new events for SHAP explanation (optional).
        session / source: Labels written into every alert.
        inline_explainer: Explain new events synchronously instead (benchmarks only).
    """

    def __init__(
        self,
        hybrid: HybridDetector,
        per_id_stats: PerIdStats,
        window_frames: int = 100,
        window_ms: float = 50.0,
        batch_size: int = 256,
        max_wait_ms: float = 20.0,
        aggregator: AlertAggregator | None = None,
        store: AlertStore | None = None,
        explain_queue: queue.Queue | None = None,
        session: str = "",
        source: str = "replay",
        inline_explainer: Any | None = None,
    ) -> None:
        self.hybrid = hybrid
        # Optional AlertExplainer called synchronously for each NEW event (benchmarking the
        # cost of SHAP in the critical path). Default: explanations run in a separate thread.
        self.inline_explainer = inline_explainer
        # Small batches: avoid per-call thread-pool start-up inside the models.
        if hasattr(hybrid.anomaly.model, "n_jobs"):
            hybrid.anomaly.model.n_jobs = 1
        self.extractor = StreamingFeatureExtractor(per_id_stats, window_frames, window_ms)
        self.batch_size = batch_size
        self.max_wait_s = max_wait_ms / 1000.0
        self.aggregator = aggregator
        self.store = store
        self.explain_queue = explain_queue
        self.session = session
        self.source = source
        self.stats = PipelineStats()
        self._rows: list[tuple] = []
        self._meta: list[tuple[float, int, float, str]] = []  # ts, can_id, arrival, truth
        self._first_arrival = 0.0
        self._classes = hybrid.classifier.classes

    # ------------------------------------------------------------------ core
    def add(
        self, frame: CANFrame, arrival: float | None = None, truth: str = ""
    ) -> list[FrameDecision]:
        """Featurise one frame; returns decisions if this frame completed a batch."""
        arrival = time.perf_counter() if arrival is None else arrival
        if not self._rows:
            self._first_arrival = arrival
        self._rows.append(
            self.extractor.update(frame.timestamp, frame.can_id, frame.dlc, list(frame.payload))
        )
        self._meta.append((frame.timestamp, frame.can_id, arrival, truth))
        if len(self._rows) >= self.batch_size:
            return self.flush()
        return []

    def due(self) -> bool:
        """True if the oldest buffered frame has waited at least ``max_wait_ms``."""
        return bool(self._rows) and time.perf_counter() - self._first_arrival >= self.max_wait_s

    def flush(self) -> list[FrameDecision]:
        """Score all buffered frames and emit decisions / alert events."""
        if not self._rows:
            return []
        X = pd.DataFrame(np.asarray(self._rows, dtype=np.float64), columns=FEATURE_COLUMNS)
        meta = self._meta
        self._rows, self._meta = [], []
        score, is_anom = self.hybrid.anomaly.predict(X)
        proba = self.hybrid.classifier.predict_proba(X)
        labels, rule = hybrid_decide(is_anom, proba, self._classes, self.hybrid.thresholds)

        decisions, touched = [], {}
        n_alert = 0
        for i, (ts, cid, _arrival, truth) in enumerate(meta):
            lab = str(labels[i])
            k = self._classes.index(lab) if lab in self._classes else int(proba[i].argmax())
            dec = FrameDecision(
                ts,
                cid,
                lab,
                float(score[i]),
                float(proba[i, k]),
                RULE_NAMES[int(rule[i])],
                0.0,
                truth,
            )
            self.stats.class_counts[lab] += 1
            if truth:
                self.stats.truth_confusion[(truth, lab)] += 1
            if lab != LABEL_NORMAL:
                n_alert += 1
                if self.aggregator is not None:
                    ev, is_new = self.aggregator.add(
                        ts,
                        cid,
                        lab,
                        anomaly_score=dec.anomaly_score,
                        class_probability=dec.class_probability,
                        source=self.source,
                        session=self.session,
                        true_label=truth,
                        rule=dec.rule,
                        frame=dict(zip(FEATURE_COLUMNS, X.iloc[i].tolist(), strict=True)),
                    )
                    dec.event_id = ev.id
                    touched[ev.id] = ev
                    if is_new:
                        self.stats.events += 1
                        if self.inline_explainer is not None:  # SHAP in the critical path
                            exp = self.inline_explainer.explain(
                                ev.frame, target_class=lab if lab in self._classes else None
                            )
                            ev.shap_top_features, ev.explanation_text = exp.top_features, exp.reason
                        elif self.explain_queue is not None:
                            self.explain_queue.put(ev)
                self.stats.recent.append(dec)
                self.stats.last_alert_wall = time.time()
                self.stats.last_alert_class = lab
            decisions.append(dec)

        # Latency = arrival -> decision available, INCLUDING batching wait, models,
        # aggregation and (if enabled) inline SHAP.
        done = time.perf_counter()
        for dec, (_, _, arrival, _) in zip(decisions, meta, strict=True):
            dec.latency_ms = (done - arrival) * 1000
            self.stats.latencies_ms.append(dec.latency_ms)
        self.stats.frames += len(meta)
        self.stats.alert_frames += n_alert
        self.stats.record_second(len(meta), n_alert)
        if self.aggregator is not None:
            closed = self.aggregator.expire(meta[-1][0])
            for ev in closed:
                touched[ev.id] = ev
            if self.store is not None and touched:
                self.store.upsert(touched.values())
        return decisions

    def process(self, frames: Iterable[CANFrame]) -> Iterator[FrameDecision]:
        """Synchronous helper: run frames through the pipeline (used by tests/benchmarks)."""
        for f in frames:
            yield from self.add(f)
        yield from self.flush()
        self.close()

    def close(self) -> None:
        """Flush open alert events to the store."""
        if self.aggregator is not None:
            events = self.aggregator.flush()
            if self.store is not None and events:
                self.store.upsert(events)

    def run(
        self,
        frames: Iterable[CANFrame | None],
        stop_event: threading.Event,
        truth: deque | None = None,
    ) -> None:
        """Consume a (possibly endless) frame source until ``stop_event`` is set.

        ``None`` items mean "bus idle" and trigger a time-based flush.
        """
        try:
            for f in frames:
                if stop_event.is_set():
                    break
                if f is not None:
                    t = truth.popleft()[0] if truth else ""
                    self.add(f, truth=t)
                if self.due():
                    self.flush()
        finally:
            self.flush()
            self.close()


# =========================================================================== orchestration
class MonitorSession:
    """Owns replay + reader + pipeline + explainer threads for one dashboard/CLI session."""

    def __init__(
        self,
        model_root: str | Path,
        store: AlertStore,
        window_frames: int,
        window_ms: float,
        aggregation_ms: float,
        severity_base: dict[str, str],
        batch_size: int = 256,
        max_wait_ms: float = 20.0,
        explain: bool = True,
        top_k: int = 5,
    ) -> None:
        root = Path(model_root)
        self.hybrid = HybridDetector.load(root)
        self.stats_artifact = PerIdStats.from_json(root / "feature_artifacts" / "per_id_stats.json")
        self.store = store
        self.cfg = dict(
            window_frames=window_frames,
            window_ms=window_ms,
            batch_size=batch_size,
            max_wait_ms=max_wait_ms,
        )
        self.aggregation_ms = aggregation_ms
        self.severity_base = severity_base
        self.explain_enabled = explain
        self.top_k = top_k
        self._explainer = None
        self.pipeline: RealtimePipeline | None = None
        self.replay_stats = None
        self._threads: list[threading.Thread] = []
        self._stop = threading.Event()
        self.session_id = ""
        self.info: dict[str, Any] = {}
        self.error = ""

    # ------------------------------------------------------------------ control
    @property
    def running(self) -> bool:
        """True while the pipeline thread is alive."""
        return any(t.is_alive() for t in self._threads if t.name == "pipeline")

    def start(
        self,
        file_path: str | Path,
        speed: float = 1.0,
        interface: str = "auto",
        channel: str = "vcan0",
        virtual_channel: str = "canids_demo",
        max_frames: int | None = None,
    ) -> str:
        """Start replaying ``file_path`` and detecting on the resulting traffic."""
        self.stop()
        self._stop = threading.Event()
        self.error = ""
        iface, chan = resolve_interface(interface, channel, virtual_channel)
        frames = load_replay_frames(file_path)
        if max_frames:
            frames = frames.iloc[:max_frames]
        self.session_id = uuid.uuid4().hex[:8]
        truth: deque | None = deque() if iface == "virtual" else None
        explain_q: queue.Queue = queue.Queue()
        self.pipeline = RealtimePipeline(
            self.hybrid,
            self.stats_artifact,
            **self.cfg,
            aggregator=AlertAggregator(self.aggregation_ms, self.severity_base),
            store=self.store,
            explain_queue=explain_q if self.explain_enabled else None,
            session=self.session_id,
            source=f"replay:{Path(file_path).name}",
        )
        reader = PassiveReader(iface, chan, source_dataset="replay").open()
        bus = open_replay_bus(iface, chan)
        replayer = Replayer(frames, bus, speed, truth_queue=truth)
        self.info = {
            "file": str(file_path),
            "frames": len(frames),
            "speed": speed,
            "interface": iface,
            "channel": chan,
            "session": self.session_id,
            "started": time.time(),
        }

        def replay_job() -> None:
            try:
                self.replay_stats = replayer.run(stop_event=self._stop)
            except Exception as err:  # surfaced in the UI
                self.error = f"replay: {err}"
                log.exception("replay failed")
            finally:
                bus.shutdown()

        def pipeline_job() -> None:
            try:
                assert self.pipeline is not None
                src = reader.frames(
                    timeout=0.01, stop_event=self._stop, yield_idle=True, idle_timeout=None
                )
                self.pipeline.run(self._until_replay_done(src, replay_thread), self._stop, truth)
            except Exception as err:
                self.error = f"pipeline: {err}"
                log.exception("pipeline failed")
            finally:
                reader.close()

        def explain_job() -> None:
            while not (self._stop.is_set() and explain_q.empty()):
                try:
                    ev: Alert = explain_q.get(timeout=0.2)
                except queue.Empty:
                    if not any(t.is_alive() for t in self._threads if t.name == "pipeline"):
                        break
                    continue
                try:
                    exp = self.explainer.explain(
                        ev.frame,
                        target_class=(
                            ev.predicted_class
                            if ev.predicted_class in self.hybrid.classifier.classes
                            else None
                        ),
                    )
                    ev.shap_top_features = exp.top_features
                    ev.explanation_text = exp.reason
                    self.store.upsert([ev])
                except Exception:
                    log.exception("explanation failed")

        replay_thread = threading.Thread(target=replay_job, name="replay", daemon=True)
        self._threads = [
            replay_thread,
            threading.Thread(target=pipeline_job, name="pipeline", daemon=True),
        ]
        if self.explain_enabled:
            self._threads.append(threading.Thread(target=explain_job, name="explain", daemon=True))
        for t in self._threads[1:]:
            t.start()
        replay_thread.start()
        log.info("session %s started: %s", self.session_id, self.info)
        return self.session_id

    @staticmethod
    def _until_replay_done(
        src: Iterator[CANFrame | None], replay_thread: threading.Thread, grace_s: float = 0.5
    ) -> Iterator[CANFrame | None]:
        """Pass frames through; stop once the replay finished and the bus stayed idle."""
        idle_since = None
        for f in src:
            if f is None:
                if not replay_thread.is_alive():
                    idle_since = idle_since or time.perf_counter()
                    if time.perf_counter() - idle_since > grace_s:
                        return
            else:
                idle_since = None
            yield f

    def stop(self, timeout: float = 5.0) -> None:
        """Stop all threads (idempotent)."""
        self._stop.set()
        for t in self._threads:
            t.join(timeout)
        self._threads = []

    def wait(self, timeout: float | None = None) -> None:
        """Block until the session finishes on its own (end of the replayed file)."""
        for t in self._threads:
            t.join(timeout)

    @property
    def explainer(self):
        """Lazily created SHAP explainer (created on first use)."""
        if self._explainer is None:
            from canids.explain.shap_explainer import AlertExplainer  # noqa: PLC0415

            self._explainer = AlertExplainer(self.hybrid.classifier, top_k=self.top_k)
        return self._explainer

    # ------------------------------------------------------------------ read side
    def snapshot(self) -> dict[str, Any]:
        """Thread-safe summary for the UI."""
        p = self.pipeline
        if p is None:
            return {"running": False, "frames": 0}
        s = p.stats
        elapsed = max(time.time() - s.started_wall, 1e-9)
        recent_s = [x for x in s.per_second if x[0] >= int(time.time()) - 3]
        fps_now = sum(f for _, f, _ in recent_s) / max(len(recent_s), 1)
        since_alert = time.time() - s.last_alert_wall if s.last_alert_wall else None
        if since_alert is not None and since_alert < 5:
            status = "red" if s.last_alert_class in ("DoS", "Fuzzy", "Spoofing") else "amber"
        else:
            status = "green"
        return {
            "running": self.running,
            "session": self.session_id,
            "info": self.info,
            "frames": s.frames,
            "alert_frames": s.alert_frames,
            "events": s.events,
            "fps_now": fps_now,
            "fps_avg": s.frames / elapsed,
            "status": status,
            "class_counts": dict(s.class_counts),
            "latency": s.latency_percentiles(),
            "per_second": list(s.per_second),
            "recent": list(s.recent)[-200:],
            "truth_confusion": dict(s.truth_confusion),
            "error": self.error,
            "replay_frames_sent": self.replay_stats.frames_sent if self.replay_stats else None,
        }
