"""Headless real-time demo: replay a log onto the virtual bus and detect live (no UI).

Same engine as the dashboard (``canids.pipeline.realtime.MonitorSession``).

Usage:
    python scripts/run_realtime.py --file data/interim/road/attacks__fuzzing_attack_3.parquet
    python scripts/run_realtime.py --file <log> --speed 10 --max-frames 200000
    python scripts/run_realtime.py --file <log> --interface socketcan --channel vcan0   # Linux
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

from canids.alerting.alert import DEFAULT_SEVERITY
from canids.alerting.store import AlertStore
from canids.pipeline.realtime import MonitorSession
from canids.utils.config import load_config, resolve_path


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    app, data = load_config("app"), load_config("data")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--file", required=True, help="unified .parquet or candump .log to replay")
    ap.add_argument("--dataset", default=app["dashboard"].get("model_dataset", "road"))
    ap.add_argument("--speed", type=float, default=float(app["replay"]["speed"]))
    ap.add_argument("--interface", default=app["replay"]["interface"])
    ap.add_argument("--channel", default=app["replay"]["channel"])
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--no-explain", action="store_true")
    ap.add_argument("--db", default=app["alerting"]["db_path"])
    args = ap.parse_args(argv)

    store = AlertStore(resolve_path(args.db))
    session = MonitorSession(
        resolve_path(data["paths"]["artifacts"]) / args.dataset,
        store,
        window_frames=int(data["windows"]["frame_count"]),
        window_ms=float(data["windows"]["time_ms"]),
        aggregation_ms=float(app["alerting"]["aggregation_window_ms"]),
        severity_base=app["alerting"].get("severity", DEFAULT_SEVERITY),
        batch_size=int(app["pipeline"]["batch_size"]),
        max_wait_ms=float(app["pipeline"]["max_wait_ms"]),
        explain=not args.no_explain,
    )
    sid = session.start(
        Path(args.file),
        args.speed,
        args.interface,
        args.channel,
        app["replay"]["virtual_channel"],
        args.max_frames,
    )
    print(f"session {sid}: replaying {args.file} at {args.speed}x on {session.info['interface']}")
    try:
        while session.running:
            time.sleep(1.0)
            s = session.snapshot()
            lat = s["latency"]
            print(
                f"  frames={s['frames']:>9,}  fps={s['fps_now']:>8,.0f}  "
                f"alert-frames={s['alert_frames']:>7,}  events={s['events']:>5,}  "
                f"p95={lat['p95']:.1f} ms  status={s['status']}"
            )
    except KeyboardInterrupt:
        print("stopping...")
    session.stop()
    session.wait(5)
    s = session.snapshot()
    lat = s["latency"]
    print(
        f"\nDone. frames={s['frames']:,} (sent {s['replay_frames_sent']}), events={s['events']:,}, "
        f"avg throughput={s['fps_avg']:,.0f} frames/s"
    )
    print(
        f"Decision latency (incl. batching): p50={lat['p50']:.2f} p95={lat['p95']:.2f} "
        f"p99={lat['p99']:.2f} max={lat['max']:.2f} ms"
    )
    print(f"Predicted frame counts: {s['class_counts']}")
    if s["truth_confusion"]:
        cm = pd.Series(s["truth_confusion"]).unstack(fill_value=0)
        cm.index.name, cm.columns.name = "true", "predicted"
        print("Ground truth vs prediction (frames, replayed labels):")
        print(cm.to_string())
    ev = store.query(session=sid, limit=10)
    if not ev.empty:
        cols = [
            "predicted_class",
            "severity",
            "can_id",
            "count",
            "distinct_ids",
            "class_probability",
            "true_label",
            "explanation_text",
        ]
        ev["can_id"] = ev["can_id"].map(lambda v: f"0x{v:03X}")
        print("\nLatest alert events:")
        print(ev[cols].to_string(index=False, max_colwidth=70))
    if s["error"]:
        print(f"ERROR: {s['error']}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
