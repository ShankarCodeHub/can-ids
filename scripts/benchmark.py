"""Phase 7 runtime benchmark on about 1M frames (state the hardware with every number).

Scenarios
---------
1. features_only        StreamingFeatureExtractor alone (us/frame)
2. pipeline             features + IF + XGBoost + hybrid + aggregation, max speed, no SHAP
3. pipeline_inline_shap same, but SHAP computed synchronously for every new alert event
4. pipeline_batch1      per-frame model calls (batch_size=1) on a small slice, to justify
                        micro-batching
5. end_to_end_bus       replayer -> python-can virtual bus -> reader -> pipeline (one process)
6. shap_single          latency of one explanation
Also: model size on disk, peak RAM, CPU %, and CAN bus capacity for comparison.

The input is the cleaned frames of an attack capture followed by an ambient capture,
truncated to --frames (default 1,000,000). The extractor is reset between captures.

Usage:
    python scripts/benchmark.py --dataset road --frames 1000000
"""

from __future__ import annotations

import argparse
import time
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from canids.alerting.alert import AlertAggregator
from canids.alerting.store import AlertStore
from canids.evaluation.metrics import save_json
from canids.evaluation.runtime import (
    ResourceMonitor,
    can_bus_capacity,
    hardware_info,
    model_size_bytes,
    percentiles,
)
from canids.explain.shap_explainer import AlertExplainer
from canids.ingestion.schema import BYTE_COLS, CANFrame, enforce_dtypes
from canids.models.hybrid import HybridDetector
from canids.pipeline.realtime import MonitorSession, RealtimePipeline
from canids.preprocessing.cleaning import clean_frames
from canids.preprocessing.features import (
    PerIdStats,
    StreamingFeatureExtractor,
    features_dataframe,
)
from canids.utils.config import load_config, resolve_path
from canids.utils.logger import get_logger
from canids.utils.provenance import git_commit

log = get_logger("scripts.benchmark")
DEFAULT_FILES = [
    "attacks__max_speedometer_attack_3.parquet",
    "ambient__ambient_highway_street_driving_long.parquet",
]


def load_frames(interim: Path, names: list[str], n_total: int) -> list[pd.DataFrame]:
    """Cleaned captures, truncated so that the total is ``n_total`` frames."""
    out, left = [], n_total
    for name in names:
        if left <= 0:
            break
        df, _ = clean_frames(enforce_dtypes(pd.read_parquet(interim / name)))
        out.append(df.iloc[:left].reset_index(drop=True))
        left -= len(out[-1])
    return out


def to_frames(df: pd.DataFrame) -> list[CANFrame]:
    """DataFrame -> CANFrame objects (built BEFORE timing starts)."""
    pay = df[BYTE_COLS].to_numpy().tolist()
    return [
        CANFrame(t, int(c), int(d), tuple(p))
        for t, c, d, p in zip(
            df["timestamp"].tolist(), df["can_id"].tolist(), df["dlc"].tolist(), pay, strict=True
        )
    ]


def run_pipeline(
    captures: list[list[CANFrame]],
    hybrid: HybridDetector,
    stats: PerIdStats,
    cfg: dict,
    batch_size: int,
    inline_explainer=None,
) -> dict:
    """Time the pipeline over all captures (fresh pipeline per capture)."""
    lat, frames, alerts, events = [], 0, 0, 0
    with ResourceMonitor() as mon:
        for frames_list in captures:
            p = RealtimePipeline(
                hybrid,
                stats,
                cfg["window_frames"],
                cfg["window_ms"],
                batch_size,
                cfg["max_wait_ms"],
                aggregator=AlertAggregator(cfg["agg_ms"]),
                inline_explainer=inline_explainer,
            )
            for _ in p.process(frames_list):
                pass
            lat.extend(p.stats.latencies_ms)
            frames += p.stats.frames
            alerts += p.stats.alert_frames
            events += p.stats.events
    r = mon.result()
    return {
        "frames": frames,
        "alert_frames": alerts,
        "events": events,
        "throughput_fps": frames / r["wall_s"],
        **r,
        **percentiles(lat),
    }


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--dataset", default="road")
    ap.add_argument("--frames", type=int, default=1_000_000)
    ap.add_argument("--files", nargs="*", default=DEFAULT_FILES)
    ap.add_argument(
        "--data-dir",
        default=None,
        help="directory with the input parquet files (default data/interim/<dataset>)",
    )
    ap.add_argument("--models", default=None, help="model dir or edge bundle (default models/<ds>)")
    ap.add_argument("--tag", default="", help="suffix for result files, e.g. 'pi4'")
    args = ap.parse_args(argv)
    app, data = load_config("app"), load_config("data")
    root = (
        resolve_path(args.models)
        if args.models
        else resolve_path(data["paths"]["artifacts"]) / args.dataset
    )
    interim = (
        resolve_path(args.data_dir)
        if args.data_dir
        else resolve_path(data["paths"]["interim"]) / args.dataset
    )
    results = resolve_path(data["paths"]["results"])
    suffix = f"_{args.tag}" if args.tag else ""
    cfg = {
        "window_frames": int(data["windows"]["frame_count"]),
        "window_ms": float(data["windows"]["time_ms"]),
        "max_wait_ms": float(app["pipeline"]["max_wait_ms"]),
        "agg_ms": float(app["alerting"]["aggregation_window_ms"]),
    }
    batch = int(app["pipeline"]["batch_size"])

    hybrid = HybridDetector.load(root)
    stats = PerIdStats.from_json(root / "feature_artifacts" / "per_id_stats.json")
    dfs = load_frames(interim, args.files, args.frames)
    captures = [to_frames(d) for d in dfs]
    n = sum(len(c) for c in captures)
    log.info("benchmark input: %d frames from %s", n, [len(c) for c in captures])
    out: dict = {
        "hardware": hardware_info(),
        "git_commit": git_commit(),
        "run_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "input": {
            "files": args.files[: len(dfs)],
            "frames": n,
            "per_file": [len(c) for c in captures],
        },
        "config": {**cfg, "batch_size": batch},
    }

    # 1. features only
    t0 = time.perf_counter()
    for c in captures:
        ex = StreamingFeatureExtractor(stats, cfg["window_frames"], cfg["window_ms"])
        for f in c:
            ex.update(f.timestamp, f.can_id, f.dlc, list(f.payload))
    dt = time.perf_counter() - t0
    out["features_only"] = {
        "frames": n,
        "seconds": dt,
        "us_per_frame": 1e6 * dt / n,
        "throughput_fps": n / dt,
    }
    log.info("features only: %.1f us/frame", out["features_only"]["us_per_frame"])

    # 2. pipeline, no SHAP   3. pipeline with inline SHAP
    out["pipeline"] = run_pipeline(captures, hybrid, stats, cfg, batch)
    log.info("pipeline: %.0f fps", out["pipeline"]["throughput_fps"])
    explainer = AlertExplainer(hybrid.classifier, top_k=5)
    out["pipeline_inline_shap"] = run_pipeline(captures, hybrid, stats, cfg, batch, explainer)
    log.info("pipeline + inline SHAP: %.0f fps", out["pipeline_inline_shap"]["throughput_fps"])

    # 4. batch size 1 on a 5k slice (per-frame model calls are slow; 5k is enough to compare)
    slice_ = [captures[0][:5_000]]
    out["pipeline_batch1_5k"] = run_pipeline(slice_, hybrid, stats, cfg, 1)
    out["pipeline_batch256_5k"] = run_pipeline(slice_, hybrid, stats, cfg, batch)

    # 5. end-to-end through the virtual bus (no SHAP)
    tmp = results / "_bench_e2e.parquet"
    pd.concat(dfs[-1:]).to_parquet(tmp)  # ambient capture (one continuous clock)
    store = AlertStore(":memory:")
    sess = MonitorSession(
        root,
        store,
        cfg["window_frames"],
        cfg["window_ms"],
        cfg["agg_ms"],
        {},
        batch,
        cfg["max_wait_ms"],
        explain=False,
    )
    with ResourceMonitor() as mon:
        sess.start(tmp, speed=0, interface="virtual", virtual_channel=f"bench{time.time_ns()}")
        sess.wait()
    s = sess.snapshot()
    tmp.unlink(missing_ok=True)
    r = mon.result()
    out["end_to_end_bus"] = {
        "frames": s["frames"],
        "frames_sent": s["replay_frames_sent"],
        "throughput_fps": s["frames"] / r["wall_s"],
        **r,
        **{f"{k}_ms": v for k, v in s["latency"].items()},
    }
    log.info("end-to-end: %.0f fps", out["end_to_end_bus"]["throughput_fps"])

    # 6. single SHAP explanation latency on attack frames
    feats = features_dataframe(dfs[0].iloc[:50_000], stats)
    attack_rows = feats[(dfs[0]["label"].iloc[:50_000] != "Normal").to_numpy()]
    out["shap_single"] = explainer.latency_profile(
        attack_rows if len(attack_rows) else feats, n=500
    )

    out["model_size_bytes"] = {
        "isolation_forest": model_size_bytes(root / "anomaly"),
        "xgboost_classifier": model_size_bytes(root / "classifier"),
        "feature_artifacts": model_size_bytes(root / "feature_artifacts"),
    }
    out["can_bus_capacity_500kbps"] = can_bus_capacity(500_000)
    save_json(out, results / f"{args.dataset}_runtime_benchmark{suffix}.json")

    rows = []
    for k in (
        "pipeline",
        "pipeline_inline_shap",
        "pipeline_batch1_5k",
        "pipeline_batch256_5k",
        "end_to_end_bus",
    ):
        v = out[k]
        rows.append(
            {
                "scenario": k,
                "frames": v["frames"],
                "throughput_fps": v["throughput_fps"],
                "p50_ms": v.get("p50_ms"),
                "p95_ms": v.get("p95_ms"),
                "p99_ms": v.get("p99_ms"),
                "max_ms": v.get("max_ms"),
                "cpu_percent": v["cpu_percent"],
                "peak_rss_mb": v["peak_rss_mb"],
                "peak_temp_c": v.get("peak_temp_c"),
            }
        )
    table = pd.DataFrame(rows)
    table.to_csv(results / f"{args.dataset}_runtime_benchmark{suffix}.csv", index=False)
    pd.set_option("display.width", 200)
    hw = out["hardware"]
    print(
        f"\nHardware: {hw['cpu']}, {hw['logical_cpus']} logical CPUs, {hw['ram_gb']} GB RAM, "
        f"{hw['os']}, Python {hw['python']}"
    )
    print(
        f"Input: {n:,} frames  |  features only: "
        f"{out['features_only']['us_per_frame']:.1f} us/frame"
    )
    print(table.round(2).to_string(index=False))
    sh = out["shap_single"]
    print(
        f"SHAP single explanation: p50={sh['p50_ms']:.2f} p95={sh['p95_ms']:.2f} "
        f"p99={sh['p99_ms']:.2f} max={sh['max_ms']:.2f} ms"
    )
    print(f"Model size: {out['model_size_bytes']}")
    cap = out["can_bus_capacity_500kbps"]
    print(
        f"CAN 500 kbps capacity: 8-byte worst {cap['dlc8_worst_stuffing_fps']:,.0f} / "
        f"8-byte {cap['dlc8_no_stuffing_fps']:,.0f} / 0-byte {cap['dlc0_fps']:,.0f} frames/s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
