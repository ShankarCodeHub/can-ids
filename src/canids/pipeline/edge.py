"""Headless edge deployment helpers (Raspberry Pi 4 + MCP2515, isolated bench only).

Safety design (viva)
--------------------
The edge node must never influence a CAN bus. In addition to the passive
reader (no send path), the runner checks that a PHYSICAL SocketCAN interface
is configured in the controller's **listen-only** mode BEFORE it opens it.
In listen-only mode the CAN controller does not transmit at all: no ACK bits
and no error frames. The check reads the kernel's view (``ip -details -json
link show <if>``); if listen-only is not confirmed, the runner refuses to
start. Virtual interfaces (python-can ``virtual``, ``vcan*``) carry no
electrical bus and are exempt.

Alerts are written to a local SQLite store and can optionally be pushed to the
laptop's collector (``scripts/alert_collector.py``) with a plain HTTP POST of
JSON, so the existing dashboard can display edge alerts.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import urllib.error
import urllib.request
from collections.abc import Iterable
from typing import Any

from canids.alerting.alert import Alert
from canids.utils.logger import get_logger

log = get_logger(__name__)


class ListenOnlyError(RuntimeError):
    """Raised when a physical CAN interface is not confirmed to be in listen-only mode."""


def is_virtual_interface(interface: str, channel: str) -> bool:
    """True for interfaces without an electrical bus (python-can virtual, vcan*)."""
    return interface == "virtual" or str(channel).startswith("vcan")


def parse_listen_only(ip_json: str, channel: str) -> bool:
    """Parse ``ip -details -json link show <channel>`` output; True if LISTEN-ONLY is on.

    iproute2 reports CAN control modes under
    ``linkinfo.info_data.ctrlmode`` (a list such as ``["LISTEN-ONLY"]``).
    """
    try:
        data = json.loads(ip_json)
    except json.JSONDecodeError:
        return False
    for link in data if isinstance(data, list) else [data]:
        if link.get("ifname") != channel:
            continue
        info = (link.get("linkinfo") or {}).get("info_data") or {}
        modes = info.get("ctrlmode") or []
        if isinstance(modes, str):
            modes = [modes]
        return any(str(m).upper() == "LISTEN-ONLY" for m in modes)
    return False


def check_listen_only(interface: str, channel: str) -> None:
    """Refuse to continue unless a physical interface is confirmed listen-only.

    Raises:
        ListenOnlyError: if the interface is physical and listen-only is not confirmed.
    """
    if is_virtual_interface(interface, channel):
        return
    if interface != "socketcan":
        raise ListenOnlyError(
            f"Only SocketCAN physical interfaces are supported on the edge node "
            f"(got {interface!r}); listen-only mode cannot be verified for other drivers."
        )
    ip = shutil.which("ip")
    if ip is None:
        raise ListenOnlyError("'ip' command not found; cannot verify listen-only mode")
    out = subprocess.run(
        [ip, "-details", "-json", "link", "show", channel],
        capture_output=True,
        text=True,
        timeout=10,
    )
    if out.returncode != 0 or not parse_listen_only(out.stdout, channel):
        raise ListenOnlyError(
            f"{channel} is not in LISTEN-ONLY mode. Configure it first, e.g.:\n"
            f"  sudo ip link set {channel} down\n"
            f"  sudo ip link set {channel} type can bitrate 500000 listen-only on\n"
            f"  sudo ip link set {channel} up"
        )
    log.info("%s confirmed LISTEN-ONLY by the kernel", channel)


class AlertForwarder:
    """POST alert events as JSON to a collector URL (best effort, never blocks detection long)."""

    def __init__(self, url: str | None, timeout_s: float = 2.0, node: str = "edge") -> None:
        self.url = url
        self.timeout_s = timeout_s
        self.node = node
        self.sent = 0
        self.failed = 0

    def send(self, alerts: Iterable[Alert]) -> None:
        """Send a batch of events; failures are counted and logged, not raised."""
        if not self.url:
            return
        payload = [{**a.to_dict(), "source": a.source or f"edge:{self.node}"} for a in alerts]
        if not payload:
            return
        req = urllib.request.Request(
            self.url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:  # noqa: S310
                resp.read()
            self.sent += len(payload)
        except (urllib.error.URLError, TimeoutError, OSError) as err:
            self.failed += len(payload)
            log.warning("alert forward to %s failed: %s", self.url, err)


def alerts_from_json(items: list[dict[str, Any]]) -> list[Alert]:
    """Rebuild Alert objects from forwarded JSON (unknown keys are ignored)."""
    fields = set(Alert.__dataclass_fields__)
    return [Alert(**{k: v for k, v in it.items() if k in fields}) for it in items]
