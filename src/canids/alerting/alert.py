"""Alert objects, severity rules and burst aggregation.

Why aggregate (viva): an injection attack produces hundreds of malicious frames
per second. One alert per frame would flood the operator (alert fatigue) and
the database. Frames with the same predicted class and CAN ID that arrive
within ``window_ms`` of each other are collapsed into ONE event that carries a
``count`` and first/last timestamps. An event is closed after ``window_ms``
of silence.

Classes listed in ``class_level`` (default: Fuzzy) are aggregated across all
IDs, because a fuzzer uses a fresh random ID on almost every frame; the event
records how many ``distinct_ids`` it touched.

Severity rules (simple and explainable):
* base severity per class from ``app.yaml`` (e.g. Unknown -> Medium, Spoofing -> High,
  DoS -> Critical);
* one level DOWN if the classifier probability is below 0.8 (uncertain);
* one level UP if the event has accumulated >= ``burst_escalation`` frames.
"""

from __future__ import annotations

import uuid
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any

SEVERITIES: tuple[str, ...] = ("Low", "Medium", "High", "Critical")
DEFAULT_SEVERITY = {"Unknown": "Medium", "Fuzzy": "High", "Spoofing": "High", "DoS": "Critical"}


def _shift(severity: str, steps: int) -> str:
    i = SEVERITIES.index(severity) + steps
    return SEVERITIES[min(max(i, 0), len(SEVERITIES) - 1)]


def severity_for(
    predicted_class: str,
    class_probability: float,
    count: int = 1,
    base: Mapping[str, str] | None = None,
    burst_escalation: int = 100,
) -> str:
    """Severity of an event (see module docstring)."""
    sev = (base or DEFAULT_SEVERITY).get(predicted_class, "Medium")
    if predicted_class != "Unknown" and class_probability < 0.8:
        sev = _shift(sev, -1)
    if count >= burst_escalation:
        sev = _shift(sev, +1)
    return sev


@dataclass
class Alert:
    """One alert EVENT (possibly aggregating many frames)."""

    id: str
    timestamp: float  # capture time of the first frame (s)
    can_id: int
    predicted_class: str
    anomaly_score: float
    class_probability: float
    severity: str
    shap_top_features: list[dict[str, Any]] = field(default_factory=list)
    explanation_text: str = ""
    source: str = ""
    count: int = 1
    last_timestamp: float = 0.0
    wall_time: str = ""
    session: str = ""
    true_label: str = ""  # ground truth when replaying a labelled log (demo only)
    rule: str = ""
    frame: dict[str, Any] = field(default_factory=dict)  # raw features of the first frame
    distinct_ids: int = 1  # >1 for class-level events (e.g. fuzzing hits many IDs)

    @classmethod
    def new(cls, **kwargs: Any) -> Alert:
        """Create an alert with a fresh id and wall-clock time."""
        kwargs.setdefault("id", uuid.uuid4().hex[:12])
        kwargs.setdefault("wall_time", datetime.now(UTC).isoformat(timespec="milliseconds"))
        kwargs.setdefault("last_timestamp", kwargs.get("timestamp", 0.0))
        return cls(**kwargs)

    def to_dict(self) -> dict[str, Any]:
        """Plain dict (JSON-friendly)."""
        return asdict(self)


class AlertAggregator:
    """Collapse bursts of identical alerts (same class + CAN ID within ``window_ms``).

    Args:
        window_ms: Max gap (capture time) between frames of one event.
        severity_base: Class -> base severity map.
        burst_escalation: Frame count that raises severity one level.
    """

    def __init__(
        self,
        window_ms: float = 500.0,
        severity_base: Mapping[str, str] | None = None,
        burst_escalation: int = 100,
        class_level: tuple[str, ...] = ("Fuzzy",),
    ) -> None:
        self.window_s = window_ms / 1000.0
        self.severity_base = dict(severity_base or DEFAULT_SEVERITY)
        self.burst_escalation = burst_escalation
        # Classes aggregated across ALL IDs: a fuzzer uses a new random ID per frame,
        # so (class, ID) keys would never merge its burst.
        self.class_level = set(class_level)
        self.open: dict[tuple[str, int], Alert] = {}
        self._ids: dict[str, set[int]] = {}

    def add(
        self, ts: float, can_id: int, predicted_class: str, **fields: Any
    ) -> tuple[Alert, bool]:
        """Add one alerting frame. Returns ``(event, is_new_event)``."""
        key = (predicted_class, -1 if predicted_class in self.class_level else int(can_id))
        ev = self.open.get(key)
        if ev is not None and ts - ev.last_timestamp <= self.window_s:
            ev.count += 1
            ev.last_timestamp = ts
            ids = self._ids.setdefault(ev.id, {ev.can_id})
            ids.add(int(can_id))
            ev.distinct_ids = len(ids)
            ev.anomaly_score = max(ev.anomaly_score, float(fields.get("anomaly_score", 0.0)))
            ev.class_probability = max(
                ev.class_probability, float(fields.get("class_probability", 0.0))
            )
            ev.severity = severity_for(
                predicted_class,
                ev.class_probability,
                ev.count,
                self.severity_base,
                self.burst_escalation,
            )
            return ev, False
        prob = float(fields.pop("class_probability", 0.0))
        ev = Alert.new(
            timestamp=ts,
            can_id=int(can_id),
            predicted_class=predicted_class,
            class_probability=prob,
            anomaly_score=float(fields.pop("anomaly_score", 0.0)),
            severity=severity_for(
                predicted_class, prob, 1, self.severity_base, self.burst_escalation
            ),
            **fields,
        )
        self.open[key] = ev
        return ev, True

    def expire(self, now_ts: float) -> list[Alert]:
        """Close and return events that have been silent for longer than the window."""
        done = [k for k, ev in self.open.items() if now_ts - ev.last_timestamp > self.window_s]
        closed = [self.open.pop(k) for k in done]
        for ev in closed:
            self._ids.pop(ev.id, None)
        return closed

    def flush(self) -> list[Alert]:
        """Close all open events (end of stream)."""
        out = list(self.open.values())
        self.open.clear()
        self._ids.clear()
        return out
