"""Runtime measurement helpers: throughput, latency, CPU, RAM, model size, bus capacity.

CAN bus capacity (for comparison with our throughput)
-----------------------------------------------------
A classic CAN 2.0A data frame with n data bytes has 44 + 8n bits plus a 3-bit
inter-frame space (47 + 8n). Bit stuffing can add up to floor((34 + 8n - 1)/4)
bits. At 500 kbit/s and 100 % bus load:

* 8-byte frames, worst-case stuffing: 500000 / 135 = about 3,700 frames/s
* 8-byte frames, no stuffing: 500000 / 111 = about 4,500 frames/s
* 0-byte frames (DoS-style floods): 500000 / 47 = about 10,600 frames/s

Real buses run well below 100 % load (ROAD: about 2,000 frames/s).
"""

from __future__ import annotations

import os
import platform
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import psutil


def can_bus_capacity(bitrate: int = 500_000) -> dict[str, float]:
    """Theoretical max frames/s of classic CAN at 100 % load (standard 11-bit IDs)."""

    def bits(n: int, stuffing: bool) -> int:
        base = 47 + 8 * n
        return base + ((34 + 8 * n - 1) // 4 if stuffing else 0)

    return {
        "bitrate": bitrate,
        "dlc0_fps": bitrate / bits(0, False),
        "dlc8_no_stuffing_fps": bitrate / bits(8, False),
        "dlc8_worst_stuffing_fps": bitrate / bits(8, True),
    }


def model_size_bytes(path: str | Path) -> int:
    """Total size of all files under ``path``."""
    p = Path(path)
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) if p.exists() else 0


def cpu_name() -> str:
    """Human-readable CPU model name (best effort, cross-platform)."""
    if platform.system() == "Windows":
        try:
            import winreg  # noqa: PLC0415

            key = winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
            )
            return str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
        except OSError:
            pass
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text().splitlines():
            if line.lower().startswith(("model name", "hardware")):
                return line.split(":", 1)[1].strip()
    return platform.processor() or platform.machine()


def os_name() -> str:
    """OS name; Windows 11 reports release '10', so it is detected by build number (>= 22000)."""
    system, release, version = platform.system(), platform.release(), platform.version()
    if system == "Windows":
        try:
            build = int(version.split(".")[-1])
            release = "11" if build >= 22000 else release
        except ValueError:
            pass
    return f"{system} {release} (build {version})"


def board_model() -> str | None:
    """Single-board computer model (e.g. 'Raspberry Pi 4 Model B Rev 1.4'), else None."""
    p = Path("/proc/device-tree/model")
    try:
        return p.read_bytes().rstrip(b"\x00").decode() if p.exists() else None
    except OSError:
        return None


def cpu_temperature_c() -> float | None:
    """SoC temperature in degC from the Linux thermal zone (Raspberry Pi), else None."""
    p = Path("/sys/class/thermal/thermal_zone0/temp")
    try:
        return int(p.read_text().strip()) / 1000.0 if p.exists() else None
    except (OSError, ValueError):
        return None


def hardware_info() -> dict[str, Any]:
    """Machine description to report next to every runtime number."""
    return {
        "board": board_model(),
        "cpu": cpu_name(),
        "logical_cpus": psutil.cpu_count(logical=True),
        "physical_cpus": psutil.cpu_count(logical=False),
        "ram_gb": round(psutil.virtual_memory().total / 2**30, 1),
        "os": os_name(),
        "python": platform.python_version(),
    }


class ResourceMonitor:
    """Samples this process's RSS in a thread; reports peak RAM and average CPU.

    CPU % is process CPU time / wall time x 100, so values above 100 % mean that
    more than one core was busy.
    """

    def __init__(self, interval_s: float = 0.05) -> None:
        self.interval_s = interval_s
        self._proc = psutil.Process(os.getpid())
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self.peak_rss = 0
        self.start_temp_c: float | None = None
        self.peak_temp_c: float | None = None

    def _run(self) -> None:
        while not self._stop.is_set():
            self.peak_rss = max(self.peak_rss, self._proc.memory_info().rss)
            t = cpu_temperature_c()
            if t is not None:
                self.peak_temp_c = max(self.peak_temp_c or t, t)
            time.sleep(self.interval_s)

    def __enter__(self) -> ResourceMonitor:
        self._t0 = time.perf_counter()
        self._cpu0 = sum(self._proc.cpu_times()[:2])
        self.peak_rss = self._proc.memory_info().rss
        self.start_temp_c = cpu_temperature_c()
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        self._thread.join()
        self.wall_s = time.perf_counter() - self._t0
        self.cpu_percent = (
            100 * (sum(self._proc.cpu_times()[:2]) - self._cpu0) / max(self.wall_s, 1e-9)
        )
        self.peak_rss = max(self.peak_rss, self._proc.memory_info().rss)

    def result(self) -> dict[str, float | None]:
        """Wall time, average CPU %, peak RSS (MB) and SoC temperature (if available)."""
        return {
            "wall_s": self.wall_s,
            "cpu_percent": self.cpu_percent,
            "peak_rss_mb": self.peak_rss / 2**20,
            "start_temp_c": self.start_temp_c,
            "peak_temp_c": self.peak_temp_c,
        }


def percentiles(values_ms: Any) -> dict[str, float]:
    """p50/p95/p99/max/mean of a latency sample (ms)."""
    a = np.asarray(list(values_ms), dtype=np.float64)
    if a.size == 0:
        return {k: float("nan") for k in ("p50_ms", "p95_ms", "p99_ms", "max_ms", "mean_ms")}
    return {
        "p50_ms": float(np.percentile(a, 50)),
        "p95_ms": float(np.percentile(a, 95)),
        "p99_ms": float(np.percentile(a, 99)),
        "max_ms": float(a.max()),
        "mean_ms": float(a.mean()),
    }
