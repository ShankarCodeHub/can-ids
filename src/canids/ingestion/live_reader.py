"""Passive CAN listener that yields frames in the unified schema.

Safety design (viva): the reader is passive by construction.
* It wraps the python-can bus privately and exposes NO send method.
* ``listen_only=False`` is rejected outright.
* For physical interfaces it additionally requests ``BusState.PASSIVE`` where
  the driver supports it. For SocketCAN hardware, listen-only must be set at the
  OS level (``ip link set can0 type can bitrate 500000 listen-only on``), which
  the code cannot verify, so it logs a reminder. The controller then never
  even sends ACK bits.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from types import TracebackType
from typing import Any

import can

from canids.ingestion.schema import (
    ATTACK_NONE,
    LABEL_UNLABELED,
    N_BYTES,
    CANFrame,
    pad_payload,
)
from canids.utils.logger import get_logger

log = get_logger(__name__)

_VIRTUAL_INTERFACES = {"virtual"}


def message_to_frame(
    msg: can.Message, source_dataset: str = "live", source_file: str = ""
) -> CANFrame | None:
    """Convert a python-can message to a :class:`CANFrame`.

    Returns ``None`` for error frames, remote frames and CAN-FD frames longer
    than 8 bytes (outside the scope of this classic-CAN IDS).
    """
    if msg.is_error_frame or msg.is_remote_frame or len(msg.data) > N_BYTES:
        return None
    data = list(msg.data)
    return CANFrame(
        timestamp=float(msg.timestamp),
        can_id=int(msg.arbitration_id),
        dlc=len(data),
        payload=pad_payload(data),
        label=LABEL_UNLABELED,
        attack_type=ATTACK_NONE,
        source_dataset=source_dataset,
        source_file=source_file,
    )


class PassiveReader:
    """Read-only CAN listener.

    Args:
        interface: python-can interface name (``virtual``, ``socketcan``, ...).
        channel: Channel name (``canids_demo``, ``vcan0``, ``can0``...).
        bitrate: Bitrate for hardware interfaces that take one (ignored otherwise).
        listen_only: Must be True; kept as an explicit, auditable parameter.
        source_dataset: Value written to the ``source_dataset`` column.
    """

    def __init__(
        self,
        interface: str,
        channel: str,
        bitrate: int | None = None,
        listen_only: bool = True,
        source_dataset: str = "live",
    ) -> None:
        if not listen_only:
            raise ValueError("PassiveReader is listen-only by design; listen_only=False refused.")
        self.interface = interface
        self.channel = channel
        self.bitrate = bitrate
        self.source_dataset = source_dataset
        self._bus: can.BusABC | None = None
        self.skipped_frames = 0

    # ------------------------------------------------------------------ lifecycle
    def open(self) -> PassiveReader:
        """Open the underlying bus (idempotent)."""
        if self._bus is not None:
            return self
        kwargs: dict[str, Any] = {}
        is_virtual = self.interface in _VIRTUAL_INTERFACES or str(self.channel).startswith("vcan")
        if self.interface in _VIRTUAL_INTERFACES:
            kwargs["receive_own_messages"] = False
        if self.bitrate and self.interface not in ("socketcan", *_VIRTUAL_INTERFACES):
            kwargs["bitrate"] = self.bitrate
        self._bus = can.Bus(interface=self.interface, channel=self.channel, **kwargs)
        if not is_virtual:
            try:
                self._bus.state = can.BusState.PASSIVE
            except (NotImplementedError, AttributeError, can.CanError):
                log.warning(
                    "Driver %s cannot be switched to PASSIVE in software; make sure the "
                    "interface itself is configured listen-only (e.g. "
                    "'ip link set %s type can bitrate %s listen-only on').",
                    self.interface,
                    self.channel,
                    self.bitrate or 500000,
                )
        log.info("PassiveReader listening on %s/%s", self.interface, self.channel)
        return self

    def close(self) -> None:
        """Shut down the bus."""
        if self._bus is not None:
            self._bus.shutdown()
            self._bus = None

    def __enter__(self) -> PassiveReader:
        return self.open()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    # ------------------------------------------------------------------ reading
    def frames(
        self,
        timeout: float = 1.0,
        max_frames: int | None = None,
        stop_event: threading.Event | None = None,
        idle_timeout: float | None = None,
        yield_idle: bool = False,
    ) -> Iterator[CANFrame | None]:
        """Yield received frames.

        Args:
            timeout: Seconds per receive attempt (keeps ``stop_event`` responsive).
            max_frames: Stop after yielding this many frames.
            stop_event: Stop when this event is set.
            idle_timeout: Stop after this many seconds without any frame
                (``None`` = wait forever).
            yield_idle: Yield ``None`` after each empty receive attempt, so a consumer
                can flush partial batches while the bus is quiet.
        """
        self.open()
        assert self._bus is not None
        yielded = 0
        idle = 0.0
        while stop_event is None or not stop_event.is_set():
            if max_frames is not None and yielded >= max_frames:
                return
            msg = self._bus.recv(timeout=timeout)
            if msg is None:
                idle += timeout
                if idle_timeout is not None and idle >= idle_timeout:
                    return
                if yield_idle:
                    yield None
                continue
            idle = 0.0
            frame = message_to_frame(msg, self.source_dataset, self.channel)
            if frame is None:
                self.skipped_frames += 1
                continue
            yielded += 1
            yield frame
