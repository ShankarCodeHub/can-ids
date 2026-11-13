"""Tests for the animated Live Car View (app/car_view.py)."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

from car_view import (  # noqa: E402
    ECUS,
    IDS,
    OBD,
    CarState,
    ecu_for_id,
    keyframes,
    render_car_html,
    render_car_svg,
    route,
    state_from_snapshot,
)


def test_ecu_placement_is_stable_and_valid() -> None:
    assert ecu_for_id(0x0D0) == ecu_for_id(0x0D0)
    assert {ecu_for_id(i).key for i in range(200)} <= {e.key for e in ECUS}


def test_route_runs_along_the_bus() -> None:
    pts = route(OBD, IDS)
    assert pts[1][1] == pts[2][1]  # middle leg is horizontal (the bus)
    assert pts[0][0] == OBD.cx and pts[-1][0] == IDS.cx


def test_keyframes_constant_speed_percentages() -> None:
    kf = keyframes("r_x", [(0, 0), (0, 100), (300, 100)])
    assert kf.startswith("@keyframes r_x{0.0%")
    assert "25.0%" in kf and "100.0%" in kf  # 100 of 400 units = 25%


def test_idle_state_has_no_packets() -> None:
    out = render_car_svg(CarState(running=False, status="green", fps=0, alert_fps=0))
    assert "IDLE" in out and "<circle" not in out


def test_normal_traffic_draws_teal_packets_no_attack() -> None:
    out = render_car_svg(CarState(running=True, status="green", fps=2000, alert_fps=0), seed=1)
    assert "ALL CLEAR" in out and 'class="pkt"' in out and "atk" not in out


def test_attack_state_shows_class_id_and_attack_path() -> None:
    st = CarState(
        running=True,
        status="red",
        fps=2000,
        alert_fps=120,
        alert_class="Spoofing",
        alert_can_id=0x0D0,
        alert_count=2793,
    )
    out = render_car_svg(st, seed=2)
    assert "SPOOFING DETECTED" in out and "0x0D0" in out and "2,793" in out
    assert "r_attack" in out and 'class="pkt atk"' in out and 'class="glow"' in out


def test_label_text_is_escaped() -> None:
    st = CarState(
        running=True, status="amber", fps=10, alert_fps=1, alert_class="<script>", alert_can_id=1
    )
    out = render_car_svg(st)
    assert "<script>" not in out and "&lt;SCRIPT&gt;" in out


def test_state_from_snapshot() -> None:
    dec = SimpleNamespace(predicted_class="Fuzzy", can_id=0x49F)
    s = {
        "running": True,
        "status": "red",
        "fps_now": 1500.0,
        "alert_frames": 116,
        "per_second": [(1, 1500, 30), (2, 1500, 60)],
        "recent": [dec],
    }
    st = state_from_snapshot(s)
    assert st.alert_class == "Fuzzy" and st.alert_can_id == 0x49F and st.alert_fps == 45
    calm = state_from_snapshot({"running": True, "status": "green", "recent": [dec]})
    assert calm.alert_class == "" and calm.alert_can_id is None


def test_page_html_is_a_single_svg_image() -> None:
    """Streamlit strips inline <svg>, so the page HTML must carry it as an <img>."""
    out = render_car_html(CarState(running=True, status="green", fps=100, alert_fps=0))
    assert "<img" in out and "data:image/svg+xml;base64," in out and "<svg" not in out
    assert render_car_svg(CarState(running=False, status="green", fps=0, alert_fps=0)).startswith(
        "<svg xmlns="
    )
