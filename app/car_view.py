"""Animated 'Live Car View' for the dashboard: a top-down car whose ECUs talk on a CAN bus.

Pure SVG + CSS keyframes (no JavaScript), so it renders through ``st.html`` (not
an iframe, no flicker on refresh) and passes Streamlit's HTML sanitiser.

What is REAL and what is ILLUSTRATIVE (stated on the picture too):
* real   - the status (normal / suspicious / attack), the predicted class, the CAN ID
           and frame count of the latest alert, frames/s and the alert rate: all read
           from the running pipeline's snapshot;
* illustrative - which ECU box a CAN ID is drawn at. ROAD's IDs are anonymised, so
           IDs are spread over the ECU boxes by a stable hash; the box names are
           generic vehicle functions, not decoded semantics.

The packet animation restarts each refresh (about 1 s); every packet starts mid-route
(negative delay), so the traffic looks continuous.
"""

from __future__ import annotations

import base64
import html
import math
import random
from collections.abc import Sequence
from dataclasses import dataclass

# ---------------------------------------------------------------- palette (dark panel)
BG = "#0F1B2D"
BODY = "#1B2A41"
BODY_EDGE = "#3A4E6E"
BUS = "#F2A541"
ECU_FILL = "#22344F"
ECU_EDGE = "#4A6285"
TEXT = "#E8EEF6"
MUTED = "#9FB0C8"
GOOD = "#2EC4B6"
COLOURS = {
    "Normal": "#2EC4B6",
    "Spoofing": "#FF5A6A",
    "Fuzzy": "#FFA94D",
    "DoS": "#FF5A6A",
    "Unknown": "#B983FF",
}

W, H = 1000, 470
BUS_Y = 240
PACKET_SECONDS = 1.6


@dataclass(frozen=True)
class Ecu:
    """One drawn ECU box (centre x, centre y, label)."""

    key: str
    cx: int
    cy: int
    label: str
    icon: str


ECUS: tuple[Ecu, ...] = (
    Ecu("engine", 770, 135, "Engine", "⚙"),
    Ecu("cluster", 590, 135, "Instrument cluster", "⏱"),
    Ecu("body", 250, 135, "Body / lights", "💡"),
    Ecu("brakes", 770, 345, "Brakes (ABS)", "⛔"),
    Ecu("steering", 590, 345, "Steering", "🛞"),
    Ecu("infotainment", 420, 345, "Infotainment", "🎵"),
)
IDS = Ecu("ids", 420, 135, "IDS (this project)", "🛡")
OBD = Ecu("obd", 250, 345, "OBD-II port", "🔌")
BOX_W, BOX_H = 132, 48


def ecu_for_id(can_id: int) -> Ecu:
    """Stable, ILLUSTRATIVE placement of a CAN ID on one of the ECU boxes."""
    return ECUS[(can_id * 2654435761) % len(ECUS)]


def _edge_y(e: Ecu) -> float:
    """Where a box's stub meets it (bottom edge for top-row boxes, top edge otherwise)."""
    return e.cy + BOX_H / 2 if e.cy < BUS_Y else e.cy - BOX_H / 2


def route(src: Ecu, dst: Ecu) -> list[tuple[float, float]]:
    """Polyline along the bus: src stub down/up to the bus, along it, then up/down to dst."""
    return [(src.cx, _edge_y(src)), (src.cx, BUS_Y), (dst.cx, BUS_Y), (dst.cx, _edge_y(dst))]


def keyframes(name: str, pts: Sequence[tuple[float, float]]) -> str:
    """CSS keyframes moving an element along ``pts`` at constant speed."""
    seg = [math.dist(a, b) for a, b in zip(pts[:-1], pts[1:], strict=True)]
    total = sum(seg) or 1.0
    acc, frames = 0.0, []
    for i, (x, y) in enumerate(pts):
        pct = 100.0 * acc / total
        frames.append(f"{pct:.1f}%{{transform:translate({x:.0f}px,{y:.0f}px)}}")
        if i < len(seg):
            acc += seg[i]
    return f"@keyframes {name}{{{''.join(frames)}}}"


@dataclass(frozen=True)
class CarState:
    """Everything the picture needs, taken from the pipeline snapshot."""

    running: bool
    status: str  # green | amber | red
    fps: float
    alert_fps: float
    alert_class: str = ""
    alert_can_id: int | None = None
    alert_count: int = 0


def state_from_snapshot(s: dict) -> CarState:
    """Build a :class:`CarState` from ``MonitorSession.snapshot()``."""
    per_second = s.get("per_second") or []
    last = per_second[-3:] if per_second else []
    alert_fps = sum(a for _, _, a in last) / max(len(last), 1)
    recent = s.get("recent") or []
    latest = recent[-1] if recent else None
    status = s.get("status", "green")
    alerting = status in ("red", "amber") and latest is not None
    return CarState(
        running=bool(s.get("running")),
        status=status,
        # the per-second window is empty for the first second: fall back to the session average
        fps=float(s.get("fps_now") or s.get("fps_avg") or 0.0),
        alert_fps=float(alert_fps),
        alert_class=str(latest.predicted_class) if alerting else "",
        alert_can_id=int(latest.can_id) if alerting else None,
        alert_count=int(s.get("alert_frames") or 0),
    )


def _box(e: Ecu, stroke: str, extra: str = "", glow: bool = False) -> str:
    x, y = e.cx - BOX_W / 2, e.cy - BOX_H / 2
    cls = ' class="glow"' if glow else ""
    return (
        f'<g{cls}><rect x="{x:.0f}" y="{y:.0f}" width="{BOX_W}" height="{BOX_H}" rx="10" '
        f'fill="{ECU_FILL}" stroke="{stroke}" stroke-width="{2.5 if glow else 1.5}"/>'
        f'<text x="{e.cx}" y="{e.cy + 6}" text-anchor="middle" font-size="15" fill="{TEXT}">'
        f'<tspan font-size="17">{e.icon}</tspan> {html.escape(e.label)}</text>{extra}</g>'
    )


def _stub(e: Ecu, colour: str = ECU_EDGE, width: float = 2.0) -> str:
    return (
        f'<line x1="{e.cx}" y1="{_edge_y(e):.0f}" x2="{e.cx}" y2="{BUS_Y}" '
        f'stroke="{colour}" stroke-width="{width}"/>'
    )


def render_car_html(state: CarState, seed: int = 0) -> str:
    """Return the self-contained HTML (SVG + CSS) for the current state."""
    rng = random.Random(seed)
    attacking = bool(state.alert_class) and state.status in ("red", "amber")
    attack_colour = COLOURS.get(state.alert_class, COLOURS["Unknown"])
    target = (
        ecu_for_id(state.alert_can_id) if attacking and state.alert_can_id is not None else None
    )

    css, packets = [], []
    # normal traffic: ECU -> IDS (the IDS hears every frame on the bus)
    if state.running:
        n_normal = max(4, min(14, int(state.fps / 250) + 4))
        routes_used = set()
        for i in range(n_normal):
            src = ECUS[rng.randrange(len(ECUS))]
            name = f"r_{src.key}"
            if name not in routes_used:
                css.append(keyframes(name, route(src, IDS)))
                routes_used.add(name)
            delay = -(i / n_normal) * PACKET_SECONDS - rng.random() * 0.2
            packets.append(
                f'<circle r="6" fill="{COLOURS["Normal"]}" class="pkt" '
                f'style="animation:{name} {PACKET_SECONDS}s linear {delay:.2f}s infinite"/>'
            )
    # injected traffic: OBD-II -> target ECU
    if attacking and target is not None:
        css.append(keyframes("r_attack", route(OBD, target)))
        n_attack = max(4, min(10, int(state.alert_fps / 20) + 4))
        for i in range(n_attack):
            delay = -(i / n_attack) * PACKET_SECONDS
            packets.append(
                f'<circle r="7.5" fill="{attack_colour}" class="pkt atk" '
                f'style="animation:r_attack {PACKET_SECONDS}s linear {delay:.2f}s infinite"/>'
            )

    # ---------------------------------------------------------------- banner text
    if attacking:
        cid = f"0x{state.alert_can_id:03X}" if state.alert_can_id is not None else "?"
        banner = f"⚠ {html.escape(state.alert_class.upper())} DETECTED · CAN ID {cid}"
        banner_colour, sub = attack_colour, f"{state.alert_count:,} alert frames so far"
    elif state.running:
        banner, banner_colour = "✓ ALL CLEAR · traffic looks normal", GOOD
        sub = f"{state.fps:,.0f} frames/s monitored"
    else:
        banner, banner_colour = "IDLE · pick a log and press Start to replay it", MUTED
        sub = "ECUs are silent until a replay runs"

    ids_stroke = attack_colour if attacking else (GOOD if state.running else ECU_EDGE)
    boxes = "".join(
        _box(
            e,
            attack_colour if (target is not None and e.key == target.key) else ECU_EDGE,
            glow=target is not None and e.key == target.key,
        )
        for e in ECUS
    )
    obd_box = _box(OBD, attack_colour if attacking else ECU_EDGE, glow=attacking)
    ids_box = _box(IDS, ids_stroke, glow=attacking)
    stubs = (
        "".join(_stub(e) for e in ECUS)
        + _stub(IDS, ids_stroke, 2.5)
        + _stub(OBD, attack_colour if attacking else ECU_EDGE, 2.5 if attacking else 2.0)
    )

    style = (
        "<style>"
        ".canviz .pkt{filter:drop-shadow(0 0 4px currentColor)}"
        ".canviz .glow{animation:canviz-pulse 0.9s ease-in-out infinite}"
        "@keyframes canviz-pulse{0%,100%{opacity:1}50%{opacity:.45}}"
        ".canviz .bus{stroke-dasharray:14 10;animation:canviz-flow 1.2s linear infinite}"
        "@keyframes canviz-flow{to{stroke-dashoffset:-48}}" + "".join(css) + "</style>"
    )
    bus_anim = ' class="bus"' if state.running else ""
    note = (
        "ECU positions are illustrative: ROAD CAN IDs are anonymised, so IDs are spread over "
        "the boxes by a stable hash. Status, class, CAN ID and rates are live."
    )
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" class="canviz" viewBox="0 0 {W} {H}" font-family="Segoe UI, Arial, sans-serif">
  {style}
  <rect x="0" y="0" width="{W}" height="{H}" rx="18" fill="{BG}"/>
  <text x="28" y="38" font-size="20" font-weight="700" fill="{banner_colour}">{banner}</text>
  <text x="28" y="62" font-size="14" fill="{MUTED}">{html.escape(sub)}</text>
  <!-- car body (top-down, front to the right) -->
  <rect x="150" y="82" width="740" height="316" rx="120" fill="{BODY}" stroke="{BODY_EDGE}" stroke-width="3"/>
  <rect x="215" y="62" width="104" height="26" rx="8" fill="#0B1220"/>
  <rect x="215" y="392" width="104" height="26" rx="8" fill="#0B1220"/>
  <rect x="700" y="62" width="104" height="26" rx="8" fill="#0B1220"/>
  <rect x="700" y="392" width="104" height="26" rx="8" fill="#0B1220"/>
  <rect x="868" y="120" width="14" height="44" rx="6" fill="#FFE08A"/>
  <rect x="868" y="316" width="14" height="44" rx="6" fill="#FFE08A"/>
  <rect x="158" y="120" width="12" height="44" rx="6" fill="#FF5A6A" opacity="0.8"/>
  <rect x="158" y="316" width="12" height="44" rx="6" fill="#FF5A6A" opacity="0.8"/>
  <text x="884" y="250" font-size="12" fill="{MUTED}" transform="rotate(90 884 250)" text-anchor="middle">FRONT</text>
  <!-- CAN bus backbone -->
  <line x1="200" y1="{BUS_Y}" x2="840" y2="{BUS_Y}" stroke="{BUS}" stroke-width="5" opacity="0.35"/>
  <line x1="200" y1="{BUS_Y}" x2="840" y2="{BUS_Y}" stroke="{BUS}" stroke-width="5"{bus_anim}/>
  <text x="840" y="{BUS_Y - 10}" font-size="13" fill="{BUS}" text-anchor="end">CAN bus · 500 kbit/s</text>
  {stubs}{boxes}{ids_box}{obd_box}
  {''.join(packets)}
  <text x="28" y="{H - 16}" font-size="12" fill="{MUTED}">{html.escape(note)}</text>
</svg>"""
    legend = "".join(
        f'<span style="display:inline-flex;align-items:center;gap:6px;margin-right:16px">'
        f'<span style="width:11px;height:11px;border-radius:50%;background:{c};display:inline-block"></span>'
        f"{html.escape(k)}</span>"
        for k, c in (
            ("Normal frame", COLOURS["Normal"]),
            ("Spoofing", COLOURS["Spoofing"]),
            ("Fuzzy", COLOURS["Fuzzy"]),
            ("Unknown / suspicious", COLOURS["Unknown"]),
        )
    )
    # Streamlit's HTML sanitiser drops inline <svg>, but keeps an <img>; CSS animations
    # inside an SVG image still run (scripts would not, and none are used).
    data = base64.b64encode(svg.encode("utf-8")).decode("ascii")
    alt = html.escape(f"Live car view. {banner}. {sub}.")
    return (
        f'<div style="width:100%"><img src="data:image/svg+xml;base64,{data}" alt="{alt}" '
        f'style="width:100%;height:auto;display:block;border-radius:18px"/>'
        f'<div style="font-size:13px;color:{MUTED};margin-top:6px">{legend}</div></div>'
    )


def render_car_svg(state: CarState, seed: int = 0) -> str:
    """The raw SVG markup (decoded from the image), for tests and offline use."""
    out = render_car_html(state, seed)
    data = out.split("base64,", 1)[1].split('"', 1)[0]
    return base64.b64decode(data).decode("utf-8")
