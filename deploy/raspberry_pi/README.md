# Raspberry Pi 4 edge deployment (optional, Phase 8)

> **Status:** the software, model bundle and procedure are ready and were tested on the laptop
> (virtual bus). **No Pi benchmark has been run yet**, because no Raspberry Pi 4 + MCP2515 hardware was
> available during development. `reports/RESULTS.md` says "Not run" until
> `reports/results/road_runtime_benchmark_pi4.json` exists.

## 0. Safety checklist (do this every time)

- [ ] **Isolated bench only.** The Pi is connected to a USB-CAN adapter or a second test node on
      a short twisted pair, **never to a vehicle** or OBD-II port.
- [ ] The bus is terminated with **120 Ω at both ends** (most MCP2515 boards have a jumper).
- [ ] `can0` is configured **listen-only** (step 3). `edge_runner.py` refuses to start otherwise.
- [ ] Only recorded public datasets are transmitted, and only by the **laptop-side bench tool**
      (step 6). The Pi never transmits.
- [ ] Power the Pi from its own 5 V / 3 A supply. Do not back-power it from the CAN board.

## 1. Hardware

| Item | Notes |
|---|---|
| Raspberry Pi 4 Model B (2 GB or more) | 64-bit Raspberry Pi OS (Bookworm, Python 3.11) |
| MCP2515 + TJA1050/SN65HVD230 CAN board | SPI0, interrupt on **GPIO25** |
| Crystal on the CAN board | Read the metal can: **8.000** → 8 MHz, **16.000** → 16 MHz |
| USB-CAN adapter on the laptop (bench feed) | Any SocketCAN-compatible adapter (e.g. CANable/gs_usb) |

Wiring (MCP2515 → Pi header): VCC→3V3 (or 5 V if the board needs it, with 3.3 V logic), GND→GND,
SI→GPIO10 (MOSI), SO→GPIO9 (MISO), SCK→GPIO11 (SCLK), CS→GPIO8 (CE0), INT→**GPIO25**.

## 2. Enable SPI and the MCP2515 overlay

Edit `/boot/firmware/config.txt` (older images use `/boot/config.txt`) and add **one** of these
overlay lines, matching your crystal:

```ini
dtparam=spi=on
# 8 MHz crystal:
dtoverlay=mcp2515-can0,oscillator=8000000,interrupt=25
# 16 MHz crystal (use this line INSTEAD of the one above):
# dtoverlay=mcp2515-can0,oscillator=16000000,interrupt=25
```

Reboot, then check the controller was found:

```bash
dmesg | grep -i mcp251
```

You should see `mcp251x spi0.0 can0: MCP2515 successfully initialized`.

## 3. Bring up can0 at 500 kbit/s in LISTEN-ONLY mode

```bash
sudo ip link set can0 down
```
```bash
sudo ip link set can0 type can bitrate 500000 listen-only on
```
```bash
sudo ip link set can0 up
```
```bash
ip -details link show can0
```

The output **must** contain `<LISTEN-ONLY>`. In this mode the CAN controller never drives the
bus: no data, no ACK, no error frames. `edge_runner.py` reads the same information
(`ip -details -json link show can0`) and refuses to run if listen-only is not set.

## 4. Install the IDS on the Pi

```bash
sudo apt update && sudo apt install -y python3-venv git can-utils
```
```bash
git clone <your-repo-url> can-ids && cd can-ids
```
```bash
python3 -m venv .venv && source .venv/bin/activate
```
```bash
pip install -r requirements-pi.txt && pip install --no-deps -e .
```

Copy the model bundle from the laptop. On the **laptop**:

```bash
python scripts/export_edge_bundle.py --dataset road
```
```bash
scp -r deploy/raspberry_pi/bundle/road pi@<pi-ip>:~/can-ids/deploy/raspberry_pi/bundle/
```

On the Pi, check the bundle against its manifest. `manifest.json` lists SHA256 sums and the
library versions it was built with, which must match `requirements-pi.txt`.

## 5. Run headless

Optional: on the **laptop**, start the collector so Pi alerts appear in the dashboard:

```bash
python scripts/alert_collector.py --port 8765
```

On the **Pi**:

```bash
python scripts/edge_runner.py --source can0 --models deploy/raspberry_pi/bundle/road --forward http://<laptop-ip>:8765/alerts --node pi4
```

The Pi stores alerts locally in `reports/edge_alerts.sqlite` and forwards each event as JSON.
If the laptop is unreachable, detection continues and the failures are counted.

Dry run without any CAN hardware (replays a demo log on the virtual bus):

```bash
python scripts/edge_runner.py --source file:data/demo/road_max_speedometer_attack_3.parquet --models deploy/raspberry_pi/bundle/road --speed 0
```

## 6. Bench test setup (laptop transmits recorded data, Pi listens)

```
Laptop --USB--> USB-CAN adapter ==twisted pair (CAN_H/CAN_L, 120 Ω each end)== MCP2515 --SPI--> Pi 4
           (replays ROAD log)                                                 (listen-only can0)
```

On the laptop (Linux), with the adapter as `can1`, replay a recorded log at real time:

```bash
sudo ip link set can1 type can bitrate 500000 && sudo ip link set can1 up
```
```bash
canplayer -I data/raw/road/attacks/max_speedometer_attack_3.log can1=can0
```

`canplayer` is part of `can-utils`. The `can1=can0` argument sends frames logged on `can0`
(as in ROAD's logs) out of the adapter `can1`. This is the only transmitting step, and it runs on an isolated
two-node bench bus. The IDS's own replayer refuses physical interfaces by design.

> A listen-only node does not acknowledge frames. With only two nodes (adapter + Pi), the
> adapter sees no ACK and retransmits. Either add a third ACK-ing node, or put the adapter in a
> mode that tolerates missing ACKs, if your adapter supports one.

## 7. Benchmark on the Pi

```bash
python scripts/benchmark.py --data-dir data/demo --files road_max_speedometer_attack_3.parquet road_ambient_basic_short_60k.parquet --frames 273551 --models deploy/raspberry_pi/bundle/road --tag pi4
```

This writes `reports/results/road_runtime_benchmark_pi4.json|csv`, including the board model,
CPU %, peak RAM and SoC temperature (start and peak). Copy both files back to the laptop and run
`python scripts/make_results.py`; the Pi-vs-laptop table then appears in `RESULTS.md`.
**For a fair comparison, run the laptop benchmark with the same input and `--tag laptop_demo`.**

## 8. Model export notes

- The bundle has the joblib models (Isolation Forest about 2.3 MiB, XGBoost about 0.26 MiB) plus
  `classifier.ubj`, XGBoost's native, version-portable format.
- **ONNX export was not used.** It would need extra converter packages that are not in the
  agreed stack (skl2onnx/onnxmltools), and the joblib/UBJ models already fit easily in the Pi's RAM.
