"""Local viewer. Bind to localhost and serve the experimenter console.

python viz/server.py
"""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from viz.board import fault_index, scenario_index, scenario_view
from viz.witness import witness_call

WITNESS = Path(__file__).with_name("witness.html")
BENCH = Path(__file__).with_name("index.html")
HOST = "127.0.0.1"
PORT = 8765


class Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self._send(200, "text/html; charset=utf-8", WITNESS.read_bytes())
            return
        if parsed.path == "/bench":
            self._send(200, "text/html; charset=utf-8", BENCH.read_bytes())
            return
        if parsed.path == "/api/witness":
            query = parse_qs(parsed.query)
            try:
                seed = int(query.get("seed", ["1"])[0])
            except ValueError:
                self._send(404, "application/json", b'{"error":"bad seed"}')
                return
            payload = witness_call(
                query.get("id", ["s007"])[0],
                model=query.get("model", ["claude-opus-5-5"])[0],
                condition=query.get("condition", ["C1a"])[0],
                seed=seed,
            )
            self._send(200, "application/json", json.dumps(payload).encode())
            return
        if parsed.path == "/api/index":
            payload = {"scenarios": scenario_index(), "faults": fault_index()}
            self._send(200, "application/json", json.dumps(payload).encode())
            return
        if parsed.path == "/api/scenario":
            query = parse_qs(parsed.query)
            scenario_id = query.get("id", ["s014"])[0]
            fault_id = query.get("fault", [""])[0] or None
            try:
                payload = scenario_view(scenario_id, fault_id)
            except KeyError:
                self._send(404, "application/json", b'{"error":"unknown scenario or fault"}')
                return
            self._send(200, "application/json", json.dumps(payload).encode())
            return
        self._send(404, "text/plain", b"not found")

    def _send(self, status: int, content_type: str, body: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt: str, *args: object) -> None:
        return


def main() -> None:
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"http://{HOST}:{PORT}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
