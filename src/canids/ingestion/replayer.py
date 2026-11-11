"""Replay recorded CAN traffic onto an ISOLATED virtual bus (demo / testing only).

Safety design (viva): the replayer is the only component that transmits frames,
so it is locked down in code, not just by convention. ``open_replay_bus``
accepts exactly two targets:

* python-can's in-process ``virtual`` interface (works on Windows and Linux;
  frames never leave the Python process), or
* Linux SocketCAN on a ``vcan*`` interface (a kernel loopback with no hardware).

Any other interface or channel (e.g. ``can0``, a USB-CAN adapter) raises
:class:`UnsafeBusError`. Note that python-can's virtual bus only connects buses
inside the SAME process, which is why the dashboard runs the replayer and the
detector in one process (Phase 6).
"""

from __future__ import annotations

import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import can
import numpy as np
import pandas as pd

from canids.ingestion.road_loader import parse_candump_file
from canids.ingestion.schema import BYTE_COLS, MAX_STANDARD_ID, enforce_dtypes
from canids.utils.logger import get_logger

log = get_logger(__name__)

VIRTUAL_INTERFACE = "virtual"
SOCKETCAN_INTERFACE = "socketcan"


class UnsafeBusError(RuntimeError):
    """Raised when asked to transmit on anything other than an isolated virtual bus."""


def is_safe_target(interface: str, channel: str) -> bool:
    """True only for python-can 'virtual' or SocketCAN 'vcan*' interfaces."""
    if interface == VIRTUAL_INTERFACE:
        return True
    return interface == SOCKETCAN_INTERFACE and str(channel).startswith("vcan")


def resolve_interface(
    interface: str = "auto", channel: str = "vcan0", virtual_channel: str = "canids_demo"
) -> tuple[str, str]:
    """Resolve ``"auto"``: SocketCAN vcan0 on Linux if it exists, else python-can virtual."""
    if interface != "auto":
        return interface, channel if interface == SOCKETCAN_INTERFACE else virtual_channel
    if sys.platform.startswith("linux") and Path(f"/sys/class/net/{channel}").exists():
        return SOCKETCAN_INTERFACE, channel
    return VIRTUAL_INTERFACE, virtual_channel


def open_replay_bus(interface: str, channel: str) -> can.BusABC:
    """Open a bus for replay, refusing anything that is not an isolated virtual bus.

    Raises:
        UnsafeBusError: for any physical or unknown interface/channel.
    """
    if not is_safe_target(interface, channel):
        raise UnsafeBusError(
            f"Refusing to transmit on interface={interface!r} channel={channel!r}. "
            "Replay is allowed only on python-can 'virtual' or SocketCAN 'vcan*'."
        )
    # Virtual bus: keep the ORIGINAL capture timestamps on delivered frames, so timing
    # features stay identical to training even when replaying at N x speed.
    # (SocketCAN vcan0 stamps frames with kernel receive time -> use speed 1.0 there.)
    kwargs = (
        {"receive_own_messages": False, "preserve_timestamps": True}
        if interface == VIRTUAL_INTERFACE
        else {}
    )
    if interface == SOCKETCAN_INTERFACE:
        log.info("SocketCAN replay: timing features are only faithful at speed 1.0")
    return can.Bus(interface=interface, channel=channel, **kwargs)


def load_replay_frames(path: str | Path) -> pd.DataFrame:
    """Load frames to replay from a unified ``.parquet`` file or a candump ``.log``."""
    path = Path(path)
    if path.suffix.lower() == ".parquet":
        return enforce_dtypes(pd.read_parquet(path))
    if path.suffix.lower() == ".log":
        df, _ = parse_candump_file(path, source_dataset="replay")
        return df
    raise ValueError(f"unsupported replay file type: {path.suffix}")


@dataclass
class ReplayStats:
    """Summary of one replay run."""

    frames_sent: int = 0
    wall_seconds: float = 0.0
    log_seconds: float = 0.0
    stopped_early: bool = False

    @property
    def frames_per_second(self) -> float:
        """Achieved send rate."""
        return self.frames_sent / self.wall_seconds if self.wall_seconds > 0 else 0.0


class Replayer:
    """Send frames from a unified DataFrame onto a (virtual) bus, preserving timing.

    Args:
        frames: Unified-schema DataFrame in the order to replay.
        bus: Bus returned by :func:`open_replay_bus`.
        speed: 1.0 = real time, N = N x faster, 0 = as fast as possible.
    """

    def __init__(
        self,
        frames: pd.DataFrame,
        bus: can.BusABC,
        speed: float = 1.0,
        truth_queue: deque | None = None,
    ) -> None:
        if speed < 0:
            raise ValueError("speed must be >= 0")
        self.bus = bus
        self.speed = speed
        # Optional ground truth for demos: (label, attack_type) appended per frame SENT.
        # The in-process virtual bus is lossless and FIFO, so a consumer can pop one
        # entry per frame received to show the true label next to each decision.
        self.truth_queue = truth_queue
        self._labels = frames["label"].astype(str).to_numpy() if "label" in frames else None
        self._atypes = (
            frames["attack_type"].astype(str).to_numpy() if "attack_type" in frames else None
        )
        self._ts = frames["timestamp"].to_numpy(dtype=np.float64)
        self._ids = frames["can_id"].to_numpy(dtype=np.int64)
        self._dlc = frames["dlc"].to_numpy(dtype=np.int64)
        self._payload = frames[BYTE_COLS].to_numpy(dtype=np.int64)

    def __len__(self) -> int:
        return len(self._ts)

    def message(self, i: int) -> can.Message:
        """Build the python-can message for frame ``i``."""
        dlc = int(self._dlc[i])
        can_id = int(self._ids[i])
        return can.Message(
            arbitration_id=can_id,
            is_extended_id=can_id > MAX_STANDARD_ID,
            data=bytes(int(b) for b in self._payload[i, :dlc]),
            timestamp=float(self._ts[i]),
        )

    def run(
        self, stop_event: threading.Event | None = None, max_frames: int | None = None
    ) -> ReplayStats:
        """Replay frames until done, ``max_frames`` reached, or ``stop_event`` set."""
        n = len(self) if max_frames is None else min(len(self), max_frames)
        stats = ReplayStats()
        if n == 0:
            return stats
        t0_wall = time.perf_counter()
        t0_log = self._ts[0]
        for i in range(n):
            if stop_event is not None and stop_event.is_set():
                stats.stopped_early = True
                break
            if self.speed > 0:
                due = (self._ts[i] - t0_log) / self.speed
                ahead = due - (time.perf_counter() - t0_wall)
                if ahead > 0.002:  # sleep only when meaningfully early (OS timer granularity)
                    time.sleep(ahead)
            if self.truth_queue is not None:
                self.truth_queue.append(
                    (
                        self._labels[i] if self._labels is not None else "",
                        self._atypes[i] if self._atypes is not None else "",
                    )
                )
            self.bus.send(self.message(i))
            stats.frames_sent += 1
        stats.wall_seconds = time.perf_counter() - t0_wall
        stats.log_seconds = (
            float(self._ts[stats.frames_sent - 1] - t0_log) if stats.frames_sent else 0.0
        )
        log.info(
            "Replay sent %d frames in %.2fs (%.0f fps, speed=%s)",
            stats.frames_sent,
            stats.wall_seconds,
            stats.frames_per_second,
            self.speed,
        )
        return stats
