"""Laptop-side collector: receives alert events from edge nodes and stores them for the dashboard.

Edge runners POST a JSON list of alert events to ``http://<laptop>:<port>/alerts``.
Events are upserted into the dashboard's SQLite store (app.yaml: alerting.db_path),
so they appear in Attack Analytics / Event History with source ``edge:<node>``.
Standard library only (http.server). Intended for an isolated bench network.

Usage:
    python scripts/alert_collector.py --port 8765
"""

from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from canids.alerting.store import AlertStore
from canids.pipeline.edge import alerts_from_json
from canids.utils.config import load_config, resolve_path

MAX_BODY = 5 * 2**20  # 5 MB per request


def make_handler(store: AlertStore) -> type[BaseHTTPRequestHandler]:
    """Request handler bound to ``store``."""

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802 (http.server API)
            if self.path != "/alerts":
                self.send_error(404)
                return
            length = int(self.headers.get("Content-Length", 0))
            if length <= 0 or length > MAX_BODY:
                self.send_error(413 if length > MAX_BODY else 400)
                return
            try:
                items = json.loads(self.rfile.read(length))
                alerts = alerts_from_json(items if isinstance(items, list) else [items])
                n = store.upsert(alerts)
            except (ValueError, TypeError) as err:
                self.send_error(400, str(err))
                return
            body = json.dumps({"stored": n}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            body = json.dumps({"status": "ok", "events": store.count()}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, fmt: str, *args: object) -> None:  # quieter console
            pass

    return Handler


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    app = load_config("app")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--db", default=app["alerting"]["db_path"])
    args = ap.parse_args(argv)
    store = AlertStore(resolve_path(args.db))
    server = ThreadingHTTPServer((args.host, args.port), make_handler(store))
    print(f"collector listening on http://{args.host}:{args.port}/alerts -> {args.db}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
