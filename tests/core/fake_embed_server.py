#!/usr/bin/env python3
"""
A stand-in for llama-server --embedding that answers /health, /v1/embeddings and /tokenize

It takes the command line the embedding pool builds and reads only --port from it
The pool's supervision then runs against a real child process with no GPU or weights
"""

import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

DIMS = 4


class Handler(BaseHTTPRequestHandler):
    """Serves the three routes the pool and its tests call."""

    def _send(self, code: int, body: object):
        payload = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):
        if self.path == "/health":
            self._send(200, {"status": "ok"})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        if self.path == "/v1/embeddings":
            texts = body.get("input", [])
            texts = texts if isinstance(texts, list) else [texts]
            self._send(200, {"data": [{"index": i, "embedding": [0.5] * DIMS} for i in range(len(texts))]})
        elif self.path == "/tokenize":
            self._send(200, {"tokens": list(range(len(str(body.get("content", "")).split())))})
        else:
            self._send(404, {"error": "not found"})

    def log_message(self, format, *args):
        pass


def main():
    """Serves on the port after --port until killed."""
    port = int(sys.argv[sys.argv.index("--port") + 1])
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()


if __name__ == "__main__":
    main()
