"""Phase 8 tests: listen-only enforcement, alert forwarding/collector, edge bundle, demo runner."""

from __future__ import annotations

import json
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

from canids.alerting.alert import Alert
from canids.alerting.store import AlertStore
from canids.evaluation.runtime import board_model, cpu_temperature_c, hardware_info
from canids.pipeline.edge import (
    AlertForwarder,
    ListenOnlyError,
    alerts_from_json,
    check_listen_only,
    is_virtual_interface,
    parse_listen_only,
)

IP_LISTEN_ONLY = json.dumps(
    [
        {
            "ifname": "can0",
            "linkinfo": {
                "info_kind": "can",
                "info_data": {"ctrlmode": ["LISTEN-ONLY"], "state": "ERROR-ACTIVE"},
            },
        }
    ]
)
IP_NORMAL = json.dumps(
    [{"ifname": "can0", "linkinfo": {"info_kind": "can", "info_data": {"state": "ERROR-ACTIVE"}}}]
)


def test_parse_listen_only() -> None:
    assert parse_listen_only(IP_LISTEN_ONLY, "can0")
    assert not parse_listen_only(IP_NORMAL, "can0")
    assert not parse_listen_only(IP_LISTEN_ONLY, "can1")  # other interface
    assert not parse_listen_only("not json", "can0")


def test_virtual_interfaces_exempt() -> None:
    assert is_virtual_interface("virtual", "x") and is_virtual_interface("socketcan", "vcan0")
    assert not is_virtual_interface("socketcan", "can0")
    check_listen_only("virtual", "demo")  # no exception
    check_listen_only("socketcan", "vcan0")


def test_non_socketcan_physical_refused() -> None:
    with pytest.raises(ListenOnlyError):
        check_listen_only("pcan", "PCAN_USBBUS1")


def test_physical_without_listen_only_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    import subprocess

    import canids.pipeline.edge as edge

    monkeypatch.setattr(edge.shutil, "which", lambda _: "/sbin/ip")
    monkeypatch.setattr(
        edge.subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a, 0, IP_NORMAL, "")
    )
    with pytest.raises(ListenOnlyError, match="listen-only on"):
        check_listen_only("socketcan", "can0")
    monkeypatch.setattr(
        edge.subprocess,
        "run",
        lambda *a, **k: subprocess.CompletedProcess(a, 0, IP_LISTEN_ONLY, ""),
    )
    check_listen_only("socketcan", "can0")  # confirmed -> allowed


def _alert(i: int) -> Alert:
    return Alert.new(
        timestamp=float(i),
        can_id=0x0D0,
        predicted_class="Spoofing",
        anomaly_score=0.5,
        class_probability=0.9,
        severity="High",
        session="edge",
    )


def test_forwarder_to_collector_roundtrip(tmp_path: Path) -> None:
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "alert_collector", Path(__file__).resolve().parents[1] / "scripts" / "alert_collector.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    store = AlertStore(tmp_path / "c.sqlite")
    server = ThreadingHTTPServer(("127.0.0.1", 0), mod.make_handler(store))
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    try:
        fwd = AlertForwarder(f"http://127.0.0.1:{server.server_port}/alerts", node="pi4")
        fwd.send([_alert(1), _alert(2)])
        assert fwd.sent == 2 and fwd.failed == 0
        df = store.query()
        assert len(df) == 2 and set(df["source"]) == {"edge:pi4"}
    finally:
        server.shutdown()


def test_forwarder_failure_is_counted_not_raised() -> None:
    fwd = AlertForwarder("http://127.0.0.1:9/alerts", timeout_s=0.5)
    fwd.send([_alert(1)])
    assert fwd.failed == 1 and fwd.sent == 0


def test_alerts_from_json_ignores_unknown_keys() -> None:
    a = alerts_from_json([{**_alert(3).to_dict(), "unexpected": 1}])[0]
    assert a.predicted_class == "Spoofing" and a.timestamp == 3.0


def test_hardware_helpers_do_not_crash() -> None:
    info = hardware_info()
    assert info["logical_cpus"] >= 1 and "os" in info
    assert board_model() is None or isinstance(board_model(), str)
    t = cpu_temperature_c()
    assert t is None or 0 < t < 120
