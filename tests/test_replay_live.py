"""Replayer safety + virtual-bus round trip through the passive reader."""

from __future__ import annotations

import time
import uuid

import can
import pytest

from canids.ingestion.live_reader import PassiveReader, message_to_frame
from canids.ingestion.replayer import (
    Replayer,
    UnsafeBusError,
    is_safe_target,
    open_replay_bus,
    resolve_interface,
)
from canids.ingestion.schema import CANFrame, frames_to_dataframe, pad_payload


def _frames(n: int = 5, step: float = 0.01) -> list[CANFrame]:
    return [
        CANFrame(
            timestamp=100.0 + i * step,
            can_id=0x100 + i,
            dlc=2,
            payload=pad_payload([i, 255 - i]),
            label="Normal",
            source_dataset="hcrl",
            source_file="t.csv",
        )
        for i in range(n)
    ]


def test_safe_target_rules() -> None:
    assert is_safe_target("virtual", "anything")
    assert is_safe_target("socketcan", "vcan0")
    assert not is_safe_target("socketcan", "can0")
    assert not is_safe_target("pcan", "PCAN_USBBUS1")


@pytest.mark.parametrize("iface, chan", [("socketcan", "can0"), ("pcan", "PCAN_USBBUS1")])
def test_open_replay_bus_refuses_physical(iface: str, chan: str) -> None:
    with pytest.raises(UnsafeBusError):
        open_replay_bus(iface, chan)


def test_resolve_interface_explicit_virtual() -> None:
    assert resolve_interface("virtual", "vcan0", "demo") == ("virtual", "demo")


def test_reader_refuses_non_listen_only() -> None:
    with pytest.raises(ValueError):
        PassiveReader("virtual", "x", listen_only=False)


def test_reader_has_no_send_method() -> None:
    assert not hasattr(PassiveReader, "send")


def test_message_to_frame_skips_remote_and_error() -> None:
    assert message_to_frame(can.Message(arbitration_id=1, is_remote_frame=True)) is None
    assert message_to_frame(can.Message(arbitration_id=1, is_error_frame=True)) is None
    f = message_to_frame(can.Message(arbitration_id=0x316, data=b"\x01\x02", timestamp=3.0))
    assert f is not None and f.dlc == 2 and f.payload[:3] == (1, 2, -1)


@pytest.mark.integration
def test_virtual_bus_roundtrip() -> None:
    channel = f"canids_test_{uuid.uuid4().hex[:8]}"
    df = frames_to_dataframe(_frames(5))
    with PassiveReader("virtual", channel, source_dataset="replay") as reader:
        bus = open_replay_bus("virtual", channel)
        try:
            stats = Replayer(df, bus, speed=0).run()
        finally:
            bus.shutdown()
        got = list(reader.frames(timeout=0.2, max_frames=5, idle_timeout=1.0))
    assert stats.frames_sent == 5
    assert [f.can_id for f in got] == df["can_id"].tolist()
    assert [f.payload for f in got] == [f.payload for f in _frames(5)]
    assert all(f.source_dataset == "replay" for f in got)


@pytest.mark.integration
def test_real_time_pacing_and_stop() -> None:
    channel = f"canids_test_{uuid.uuid4().hex[:8]}"
    df = frames_to_dataframe(_frames(11, step=0.02))  # 0.2 s of log time
    bus = open_replay_bus("virtual", channel)
    try:
        t0 = time.perf_counter()
        stats = Replayer(df, bus, speed=1.0).run()
        elapsed = time.perf_counter() - t0
        fast = Replayer(df, bus, speed=4.0).run(max_frames=3)
    finally:
        bus.shutdown()
    assert stats.frames_sent == 11
    assert elapsed >= 0.18  # respects recorded timing
    assert fast.frames_sent == 3
