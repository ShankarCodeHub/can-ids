"""Headless edge IDS runner (Raspberry Pi 4 / any Linux box / laptop for testing).

Reads frames PASSIVELY, runs the same streaming pipeline as the dashboard, stores alert
events in local SQLite and optionally forwards them to the laptop collector.

Sources:
    --source can0                     physical SocketCAN (must be LISTEN-ONLY; verified)
    --source vcan0                    virtual SocketCAN (e.g. fed by canplayer on the bench)
    --source file:<log.parquet|.log>  replay a recorded log on python-can's virtual bus

Examples:
    python scripts/edge_runner.py --source can0 --forward http://192.168.1.10:8765/alerts
    python scripts/edge_runner.py --source file:data/demo/road_fuzzing_attack_3.parquet --speed 0

Stop with Ctrl+C. Prints a status line every --report-s seconds.
"""

from __future__ import annotations

import argparse
import signal
import threading
import time
from pathlib import Path

from canids.alerting.alert import DEFAULT_SEVERITY, AlertAggregator
from canids.alerting.store import AlertStore
from canids.ingestion.live_reader import PassiveReader
from canids.ingestion.replayer import Replayer, load_replay_frames, open_replay_bus
from canids.models.hybrid import HybridDetector
from canids.pipeline.edge import AlertForwarder, check_listen_only
from canids.pipeline.realtime import RealtimePipeline
from canids.preprocessing.features import PerIdStats
from canids.utils.config import load_config, resolve_path


class ForwardingStore(AlertStore):
    """SQLite store that also forwards every upserted event batch."""

    def __init__(self, path, forwarder: AlertForwarder) -> None:
        super().__init__(path)
        self.forwarder = forwarder

    def upsert(self, alerts) -> int:  # type: ignore[override]
        alerts = list(alerts)
        n = super().upsert(alerts)
        self.forwarder.send(alerts)
        return n


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    app, data = load_config("app"), load_config("data")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--source", required=True, help="can0 | vcan0 | file:<path>")
    ap.add_argument("--models", default=None, help="model dir (default models/<dataset> or bundle)")
    ap.add_argument("--dataset", default=app["dashboard"].get("model_dataset", "road"))
    ap.add_argument("--db", default="reports/edge_alerts.sqlite")
    ap.add_argument("--forward", default=None, help="collector URL, e.g. http://laptop:8765/alerts")
    ap.add_argument("--node", default="pi4")
    ap.add_argument("--speed", type=float, default=1.0, help="replay speed for file: sources")
    ap.add_argument("--report-s", type=float, default=5.0)
    args = ap.parse_args(argv)

    root = (
        Path(args.models)
        if args.models
        else resolve_path(data["paths"]["artifacts"]) / args.dataset
    )
    hybrid = HybridDetector.load(root)
    stats = PerIdStats.from_json(root / "feature_artifacts" / "per_id_stats.json")
    forwarder = AlertForwarder(args.forward, node=args.node)
    store = ForwardingStore(resolve_path(args.db), forwarder)
    pipeline = RealtimePipeline(
        hybrid,
        stats,
        int(data["windows"]["frame_count"]),
        float(data["windows"]["time_ms"]),
        int(app["pipeline"]["batch_size"]),
        float(app["pipeline"]["max_wait_ms"]),
        aggregator=AlertAggregator(
            float(app["alerting"]["aggregation_window_ms"]),
            app["alerting"].get("severity", DEFAULT_SEVERITY),
        ),
        store=store,
        session=f"edge-{int(time.time())}",
        source=f"edge:{args.node}",
    )

    stop = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    replay_thread = None
    if args.source.startswith("file:"):
        channel = f"edge{time.time_ns()}"
        frames = load_replay_frames(args.source[5:])
        reader = PassiveReader("virtual", channel, source_dataset="replay").open()
        bus = open_replay_bus("virtual", channel)

        def replay() -> None:
            Replayer(frames, bus, args.speed).run(stop_event=stop)
            bus.shutdown()

        replay_thread = threading.Thread(target=replay, daemon=True)
        replay_thread.start()
    else:
        check_listen_only("socketcan", args.source)  # refuses physical buses not in listen-only
        reader = PassiveReader("socketcan", args.source, source_dataset="live").open()

    print(f"edge IDS on {args.source} | models {root} | db {args.db} | forward {args.forward}")
    last = time.perf_counter()
    idle_since = None
    try:
        for f in reader.frames(timeout=0.01, stop_event=stop, yield_idle=True):
            if f is not None:
                pipeline.add(f)
                idle_since = None
            elif replay_thread is not None and not replay_thread.is_alive():
                idle_since = idle_since or time.perf_counter()
                if time.perf_counter() - idle_since > 0.5:
                    break
            if pipeline.due():
                pipeline.flush()
            if time.perf_counter() - last >= args.report_s:
                last = time.perf_counter()
                s, lat = pipeline.stats, pipeline.stats.latency_percentiles()
                print(
                    f"  frames={s.frames:,} alert-frames={s.alert_frames:,} events={s.events:,} "
                    f"p95={lat['p95']:.1f} ms forwarded={forwarder.sent} failed={forwarder.failed}"
                )
    finally:
        pipeline.flush()
        pipeline.close()
        reader.close()
    s, lat = pipeline.stats, pipeline.stats.latency_percentiles()
    print(
        f"done: frames={s.frames:,} alert-frames={s.alert_frames:,} events={s.events:,} "
        f"latency p50={lat['p50']:.2f} p95={lat['p95']:.2f} ms forwarded={forwarder.sent} "
        f"failed={forwarder.failed}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
