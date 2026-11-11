#!/usr/bin/env bash
# One-command demo (Linux): bring up vcan0 and launch the dashboard, which replays
# a recorded attack log onto vcan0 and detects it live.
#
#   bash scripts/replay_demo.sh                 # dashboard (choose file + press Start)
#   bash scripts/replay_demo.sh --headless FILE # no UI, prints alerts in the terminal
#
# vcan0 is a purely VIRTUAL kernel interface: no hardware, no vehicle. On Windows,
# skip this script; the dashboard uses python-can's in-process 'virtual' bus.
set -euo pipefail
cd "$(dirname "$0")/.."

PY=".venv/bin/python"
[ -x "$PY" ] || PY="python3"

# --- 1. virtual CAN interface (needs sudo once per boot)
if ! ip link show vcan0 &>/dev/null; then
    sudo modprobe vcan
    sudo ip link add dev vcan0 type vcan
fi
sudo ip link set up vcan0
echo "vcan0 is up."

# --- 2. run
if [[ "${1:-}" == "--headless" ]]; then
    FILE="${2:?usage: replay_demo.sh --headless <file.parquet|file.log>}"
    exec "$PY" scripts/run_realtime.py --file "$FILE" --interface socketcan --channel vcan0 --speed 1
else
    export CANIDS_REPLAY_INTERFACE=socketcan
    exec "$PY" -m streamlit run app/dashboard.py
fi
