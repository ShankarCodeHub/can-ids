"""SQLite event store for alerts.

Design decisions (viva):
* **SQLite.** It is a single file, needs no server, is ACID, and is built into
  Python. That is enough for one IDS node, and the Raspberry Pi can use it too.
* **WAL mode.** The pipeline thread writes while the dashboard reads; write-ahead
  logging lets readers and the writer work concurrently without blocking.
* **One row per aggregated EVENT** (``INSERT OR REPLACE`` updates the count and
  last timestamp as a burst grows), so the table stays small under attack.
* **Indexes** on time, class and CAN ID match the dashboard's filters.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from collections.abc import Iterable, Sequence
from pathlib import Path

import pandas as pd

from canids.alerting.alert import Alert

COLUMNS = [
    "id",
    "timestamp",
    "last_timestamp",
    "wall_time",
    "can_id",
    "predicted_class",
    "severity",
    "count",
    "anomaly_score",
    "class_probability",
    "shap_top_features",
    "explanation_text",
    "source",
    "session",
    "true_label",
    "rule",
    "frame",
    "distinct_ids",
]

_SCHEMA = """
CREATE TABLE IF NOT EXISTS alerts (
    id TEXT PRIMARY KEY,
    timestamp REAL NOT NULL,
    last_timestamp REAL NOT NULL,
    wall_time TEXT NOT NULL,
    can_id INTEGER NOT NULL,
    predicted_class TEXT NOT NULL,
    severity TEXT NOT NULL,
    count INTEGER NOT NULL,
    anomaly_score REAL,
    class_probability REAL,
    shap_top_features TEXT,
    explanation_text TEXT,
    source TEXT,
    session TEXT,
    true_label TEXT,
    rule TEXT,
    frame TEXT,
    distinct_ids INTEGER DEFAULT 1
);
CREATE INDEX IF NOT EXISTS idx_alerts_ts ON alerts(timestamp);
CREATE INDEX IF NOT EXISTS idx_alerts_class ON alerts(predicted_class);
CREATE INDEX IF NOT EXISTS idx_alerts_canid ON alerts(can_id);
CREATE INDEX IF NOT EXISTS idx_alerts_session ON alerts(session);
"""


class AlertStore:
    """Thread-safe SQLite store (one connection per thread)."""

    def __init__(self, path: str | Path) -> None:
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._shared: sqlite3.Connection | None = None  # used for :memory: databases
        with self._conn() as con:
            con.executescript(_SCHEMA)
            # forward-compatible migration for databases created by older versions
            have = {r[1] for r in con.execute("PRAGMA table_info(alerts)")}
            if "distinct_ids" not in have:
                con.execute("ALTER TABLE alerts ADD COLUMN distinct_ids INTEGER DEFAULT 1")

    def _conn(self) -> sqlite3.Connection:
        if self.path == ":memory:":
            if self._shared is None:
                self._shared = sqlite3.connect(":memory:", check_same_thread=False)
            return self._shared
        con = getattr(self._local, "con", None)
        if con is None:
            con = sqlite3.connect(self.path, timeout=10, check_same_thread=False)
            con.execute("PRAGMA journal_mode=WAL")
            con.execute("PRAGMA synchronous=NORMAL")
            self._local.con = con
        return con

    # ------------------------------------------------------------------ writes
    def upsert(self, alerts: Iterable[Alert]) -> int:
        """Insert new events or update existing ones (same id). Returns rows written."""
        rows = []
        for a in alerts:
            d = a.to_dict()
            d["shap_top_features"] = json.dumps(d["shap_top_features"])
            d["frame"] = json.dumps(d["frame"])
            rows.append(tuple(d[c] for c in COLUMNS))
        if not rows:
            return 0
        con = self._conn()
        with con:
            con.executemany(
                f"INSERT OR REPLACE INTO alerts ({','.join(COLUMNS)}) "
                f"VALUES ({','.join('?' * len(COLUMNS))})",
                rows,
            )
        return len(rows)

    def clear(self, session: str | None = None) -> None:
        """Delete all events (or one session's)."""
        con = self._conn()
        with con:
            if session is None:
                con.execute("DELETE FROM alerts")
            else:
                con.execute("DELETE FROM alerts WHERE session = ?", (session,))

    # ------------------------------------------------------------------ reads
    def query(
        self,
        start_ts: float | None = None,
        end_ts: float | None = None,
        classes: Sequence[str] | None = None,
        can_ids: Sequence[int] | None = None,
        session: str | None = None,
        limit: int | None = None,
    ) -> pd.DataFrame:
        """Events matching all given filters, newest (wall-clock) first.

        ``start_ts``/``end_ts`` filter on capture timestamps.
        """
        where, args = [], []
        if start_ts is not None:
            where.append("timestamp >= ?")
            args.append(start_ts)
        if end_ts is not None:
            where.append("timestamp <= ?")
            args.append(end_ts)
        if classes:
            where.append(f"predicted_class IN ({','.join('?' * len(classes))})")
            args.extend(classes)
        if can_ids:
            where.append(f"can_id IN ({','.join('?' * len(can_ids))})")
            args.extend(int(c) for c in can_ids)
        if session:
            where.append("session = ?")
            args.append(session)
        sql = "SELECT * FROM alerts"
        if where:
            sql += " WHERE " + " AND ".join(where)
        # Newest first by WALL time: capture clocks of different logs are unrelated
        # (every ROAD capture starts near the same synthetic timestamp).
        sql += " ORDER BY wall_time DESC, timestamp DESC"
        if limit:
            sql += f" LIMIT {int(limit)}"
        df = pd.read_sql_query(sql, self._conn(), params=args)
        if not df.empty:
            df["shap_top_features"] = df["shap_top_features"].map(json.loads)
            df["frame"] = df["frame"].map(json.loads)
        return df

    def count(self, session: str | None = None) -> int:
        """Number of stored events."""
        sql, args = "SELECT COUNT(*) FROM alerts", ()
        if session:
            sql, args = sql + " WHERE session = ?", (session,)
        return int(self._conn().execute(sql, args).fetchone()[0])

    def export_csv(self, path: str | Path, **filters) -> int:
        """Write matching events to CSV; returns rows written."""
        df = self.query(**filters)
        if not df.empty:
            df["shap_top_features"] = df["shap_top_features"].map(json.dumps)
            df["frame"] = df["frame"].map(json.dumps)
        df.to_csv(path, index=False)
        return len(df)
