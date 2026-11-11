# Phase 6: Real-time pipeline, alerting and dashboard

## What was built

| Component | File | Notes |
|---|---|---|
| Streaming pipeline | `src/canids/pipeline/realtime.py` | Frame → `StreamingFeatureExtractor` (the same code used in training) → micro-batch (≤ 256 frames or ≤ 20 ms) → Isolation Forest + XGBoost + hybrid rules → alerts. The batching wait is included in the reported latency. |
| Orchestration | `MonitorSession` (same file) | Runs replay, inference and SHAP explanation in three background threads. The UI only reads snapshots and SQLite. |
| Alert model | `src/canids/alerting/alert.py` | `Alert` dataclass with severity rules: base severity per class, one level down if probability < 0.8, one level up for bursts of ≥ 100 frames. **Aggregation**: same class + CAN ID within 500 ms becomes one event. Fuzzy is aggregated across IDs, recording `distinct_ids`. |
| Event store | `src/canids/alerting/store.py` | SQLite in WAL mode (concurrent reader and writer), upsert per event, filters (time, class, ID, session), CSV export, forward-compatible schema migration. |
| Dashboard | `app/dashboard.py` | Seven tabs: Live Monitor, Attack Analytics, Alert Detail (SHAP), Event History (filters + CSV export), Model Performance (saved results only), System Health (latency, throughput, CPU/RAM), About (team + safety). |
| Headless runner | `scripts/run_realtime.py` | The same engine, printing to the terminal. |
| Linux demo | `scripts/replay_demo.sh` | Brings up `vcan0` and launches the dashboard, or a headless run on `vcan0`. |

**Replay design**
- **Windows** uses python-can's in-process `virtual` bus. **Linux** can use `vcan0` (set `CANIDS_REPLAY_INTERFACE=socketcan`).
- The virtual bus is opened with `preserve_timestamps=True`, so frames keep their original capture timestamps and **features are identical at any replay speed**. With SocketCAN, frames carry kernel timestamps, so only 1× speed is faithful there; the code logs this.
- **Ground truth for demos:** the replayer passes each frame's true label through a FIFO queue. The virtual bus is lossless and in order, so the dashboard can show the true label next to each decision.

## Verification (all numbers measured on this laptop: AMD Ryzen 5 7520U, 8 logical CPUs, 15.3 GB RAM, Windows 11)

| Check | Result |
|---|---|
| Train/serve parity (`test_streaming_matches_offline`) | Frame-by-frame streaming decisions are **identical** to offline features + model on 1,500 synthetic frames |
| End-to-end virtual bus (`test_monitor_session_end_to_end`) | 1,500 of 1,500 frames delivered; attacks detected; events stored with SHAP text |
| Live replay `fuzzing_attack_3` at 4× (headless) | 114/114 fuzzing frames detected; 2 false positives and 4 Unknown on normal frames. This matches the offline Phase 4 test result for that capture. |
| Live replay `max_speedometer_attack_3` at 5× (dashboard, browser) | Alerts appeared live. The main burst on `0x0D0` collapsed into **one event of 2,793 frames**, escalated to Critical, with explanation "this ID is 0.08 of the last 100 frames and payload byte 1 = 0x72 → Spoofing-like" |
| Max-speed run: 300,000 ambient frames (`--speed 0`, no SHAP) | **11,198 frames/s end-to-end** (replayer + virtual bus + reader + pipeline sharing one Python process). Latency **p50 13.3 ms, p95 17.7 ms, p99 20.4 ms, max 36.2 ms**. 9 alert events out of 300,000 frames (0.003%). |
| Fuzzing at 4× | Decision latency p50 20.2 ms, p95 35.7 ms, max 49.8 ms (dominated by the 20 ms batching wait when traffic is light) |
| All dashboard tabs | Render with no exceptions (checked in the in-app browser) |

A 500 kbps CAN bus carries roughly 2,000–8,000 frames/s, and ROAD's real rate is about 2,000/s. The end-to-end rate of about 11,200 frames/s therefore gives headroom. Phase 7 benchmarks this formally on 1M frames.

## Findings

1. **Burst aggregation matters.** The speedometer attack produced 2,793 alert frames but one event. Without class-level aggregation, fuzzing produced 120 events for 116 alert frames, because each frame uses a new random ID. Aggregating Fuzzy by class fixed this, and each event now records how many distinct IDs it touched.
2. **Cold-start alerts.** In the 300k-frame ambient run, 7 of the 9 alerts occurred in the first 0.3 s (plus 1 at 18.6 s), all on the **first frame of an ID** (`iat_id = −1`, no history yet). A warm-up rule (no alerts until each ID has been seen once, or until the first N frames) would remove most of them. It is not applied, because it would change the evaluated decision logic; it is recorded as future work.
3. **ROAD label noise at injection boundaries.** All-`FF` payloads occur **0 times in 24.1M ambient frames**. Yet 6 all-`FF` frames in the fuzzing captures are labelled Normal: fuzzing_1 has 1, fuzzing_2 has 3 and fuzzing_3 has 2. Each lies on or just outside the metadata interval edge, for example 13.3524 s against an interval end of 13.3468 s. They are almost certainly injected frames that the interval boundaries excluded. The 2 "false positives" on `fuzzing_attack_3` in the Phase 4 test are therefore very likely correct detections. **Labels were not changed after the test evaluation**; this is reported as a threat to validity.
4. **Explanations run off the critical path.** SHAP (p95 about 11 ms, Phase 5) runs in its own thread for each new event, so it never delays detection.

## How to run

```bash
streamlit run app/dashboard.py
```
```bash
python scripts/run_realtime.py --file data/interim/road/attacks__max_speedometer_attack_3.parquet --speed 5
```
On Linux with `vcan0`:
```bash
bash scripts/replay_demo.sh
```
