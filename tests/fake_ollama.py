"""A tiny stand-in for Ollama's HTTP API, used only by the tests.

It implements the two endpoints the project uses (/api/tags and
/api/generate) and answers deterministically, so the whole two-machine
flow can be tested on one computer without downloading a model.
"""

from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import List


class FakeOllama:
    def __init__(self, name: str, models: List[str], delay_s: float = 0.0):
        self.name = name
        self.models = models
        self.delay_s = delay_s
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):  # keep test output quiet
                pass

            def _reply(self, status: int, body: dict) -> None:
                data = json.dumps(body, ensure_ascii=False).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                if self.path == "/api/tags":
                    self._reply(200, {"models": [{"name": m} for m in fake.models]})
                else:
                    self._reply(404, {"error": "not found"})

            def do_POST(self):
                length = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(length))
                if self.path != "/api/generate":
                    self._reply(404, {"error": "not found"})
                elif body.get("model") not in fake.models:
                    self._reply(404, {"error": f"model '{body.get('model')}' not found"})
                else:
                    time.sleep(fake.delay_s)
                    self._reply(200, {
                        "model": body["model"],
                        "response": f"[{fake.name}/{body['model']}] {body['prompt']}",
                        "done": True,
                    })

        self._http = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self._http.server_address[1]}"
        threading.Thread(target=self._http.serve_forever, daemon=True).start()

    def close(self) -> None:
        self._http.shutdown()
        self._http.server_close()
