#!/usr/bin/env python3
"""
Frontend demo harness — feeds fake data to the real router SPAs.

Serves the actual `/dash`, `/chat`, `/translate` pages and their shared
`/webui` assets straight from `../src`, then answers every API endpoint those
pages poll with a live, evolving *simulation* — no llama-server, no GPUs, no
router. A background thread advances a fake fleet (GPU util/VRAM, per-GPU status,
request history, model load/unload with eviction) so the dashboard is busy the
moment it opens and stays lively while you poke at it.

Run it:

    python frontend-demo/launch.py           # -> http://127.0.0.1:11500/dash

Env knobs:
    DEMO_PORT   listen port                 (default 11500)
    DEMO_HOST   listen host                 (default 127.0.0.1)
    DEMO_AUTO   auto load/unload models     (default 1; set 0 to drive by hand)

The mock fleet (GPUs + models) lives in mock_data.py next to this file.
"""

import json
import mimetypes
import os
import random
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs

import mock_data as MD

SRC = Path(__file__).resolve().parent.parent / "src"

HOST = os.environ.get("DEMO_HOST", "127.0.0.1")
PORT = int(os.environ.get("DEMO_PORT", "11500"))
AUTO = os.environ.get("DEMO_AUTO", "1") not in ("0", "false", "no")

GPU_WINDOW = 7200      # seconds of GPU history to keep (matches the real 2h)
GPU_FLUSH = 5.0        # append a GPU history point this often
HIST_WINDOW = 30 * 86400  # keep ~30d of request history so every range button has data

STATUSES = ("idle", "serving", "swapping")

mimetypes.add_type("font/woff2", ".woff2")
mimetypes.add_type("application/javascript", ".js")
mimetypes.add_type("text/css", ".css")


class Sim:
    """
    The evolving fake fleet.

    Owns all mutable demo state behind a single lock and mutates it once per
    second from a background thread: progresses in-flight load/unload swaps,
    fabricates request traffic (which lights GPUs `serving` and appends history
    rows), samples GPU util/VRAM, and records per-GPU status-change timelines.
    HTTP handlers read consistent snapshots under the same lock.
    """

    def __init__(self):
        self.lock = threading.RLock()
        self.running = True                                  # router lifecycle
        self.loaded: set[str] = set()                        # resident model ids
        self.port_of: dict[str, int] = {}                    # model id -> port
        self._next_port = MD.LLM_BASE_PORT
        # In-flight swaps: model id -> {"op": "load"|"unload", "done": ts}.
        self.swaps: dict[str, dict] = {}
        self.serving_until: dict[int, float] = {g["index"]: 0.0 for g in MD.GPUS}
        self.util_history: dict[int, list] = {g["index"]: [] for g in MD.GPUS}
        self.vram_history: dict[int, list] = {g["index"]: [] for g in MD.GPUS}
        self.timeline: dict[int, list] = {}                  # gpu -> [(ts, status)]
        self._last_status: dict[int, str] = {}
        self.history: list[dict] = []                        # request rows, newest last
        self._next_hist_id = 1
        self._last_flush = 0.0
        self._last_auto = time.time()
        self.by_id = {m["id"]: m for m in MD.MODELS}
        self._seed()

    # --- seeding -----------------------------------------------------------

    def _seed(self):
        """Backfills history + GPU windows and loads the initial resident set."""
        now = time.time()
        for mid in MD.INITIAL_LOADED:
            if mid in self.by_id:
                self.loaded.add(mid)
                self.port_of[mid] = self._alloc_port()

        # Backfill request history across the widest range button (~30d), denser
        # toward now so the zoomable chart has both breadth and detail.
        span = HIST_WINDOW
        n = 4000
        for _ in range(n):
            # Bias timestamps toward the present (squared uniform).
            age = span * (random.random() ** 2)
            self._add_history_row(now - age)
        self.history.sort(key=lambda r: r["request_time"])

        # Backfill GPU util/VRAM history across the 2h window at flush cadence.
        t = now - GPU_WINDOW
        while t < now:
            for g in MD.GPUS:
                gi = g["index"]
                self.util_history[gi].append((round(t, 1), round(random.uniform(2, 18), 1)))
                self.vram_history[gi].append((round(t, 1), round(self._vram_for(gi) + random.uniform(-120, 120), 1)))
            t += GPU_FLUSH
        self._last_flush = now
        # Prime the timeline so every lane starts labeled.
        self._record_timeline(now)

    # --- helpers -----------------------------------------------------------

    def _alloc_port(self) -> int:
        """Returns the next free replica port."""
        p = self._next_port
        self._next_port += 1
        return p

    def _gpu_status(self, gpu: int, now: float, swapping: set[int]) -> str:
        """Computes one GPU's running status (swapping > serving > idle)."""
        if gpu in swapping:
            return "swapping"
        if self.serving_until.get(gpu, 0.0) > now:
            return "serving"
        return "idle"

    def _swapping_gpus(self) -> set[int]:
        """GPUs touched by an in-flight load/unload swap."""
        out: set[int] = set()
        for mid in self.swaps:
            out.update(self.by_id[mid]["gpus"])
        return out

    def _vram_for(self, gpu: int) -> float:
        """Baseline resident VRAM (MB) on a GPU from the models loaded on it."""
        used = 380.0  # driver / cuda context floor
        for mid in self.loaded:
            m = self.by_id[mid]
            if gpu in m["gpus"]:
                used += m["vram_mb"] / len(m["gpus"])
        # Loading models also occupy VRAM as they come up.
        for mid, s in self.swaps.items():
            if s["op"] == "load":
                m = self.by_id[mid]
                if gpu in m["gpus"]:
                    used += (m["vram_mb"] / len(m["gpus"])) * 0.6
        total = next(g["total_vram_mb"] for g in MD.GPUS if g["index"] == gpu)
        return min(used, total - 60)

    def _add_history_row(self, completed_at: float):
        """Appends one fabricated request row that finished at `completed_at`."""
        pool = list(self.loaded) or [m["id"] for m in MD.MODELS]
        mid = random.choice(pool)
        tps = self.by_id[mid]["tps"] * random.uniform(0.8, 1.15)
        prompt_n = random.randint(200, 60000)
        predicted_n = random.randint(20, 4000)
        # duration = prefill (~2500 tok/s) + decode at the model's tok/s.
        dur = prompt_n / 2500.0 + predicted_n / max(tps, 1.0)
        self.history.append({
            "id": self._next_hist_id,
            "model": mid,
            "request_time": round(completed_at - dur, 3),
            "response_time": round(completed_at, 3),
            "prompt_n": prompt_n,
            "predicted_n": predicted_n,
        })
        self._next_hist_id += 1

    def _record_timeline(self, now: float):
        """Appends a timeline entry per GPU whose status changed."""
        swapping = self._swapping_gpus()
        for g in MD.GPUS:
            gi = g["index"]
            st = self._gpu_status(gi, now, swapping) if self.running else "inactive"
            if st != self._last_status.get(gi):
                self.timeline.setdefault(gi, []).append((round(now, 1), st))
                self._last_status[gi] = st
                cutoff = now - GPU_WINDOW
                self.timeline[gi] = [(t, s) for t, s in self.timeline[gi] if t > cutoff]

    # --- load / unload -----------------------------------------------------

    def request_load(self, mid: str):
        """Schedules a load, evicting any resident sharing its GPUs first."""
        with self.lock:
            if mid not in self.by_id or mid in self.loaded or mid in self.swaps:
                return
            now = time.time()
            gpus = set(self.by_id[mid]["gpus"])
            for other in list(self.loaded):
                if set(self.by_id[other]["gpus"]) & gpus and other not in self.swaps:
                    self.swaps[other] = {"op": "unload", "done": now + random.uniform(1.5, 2.5)}
            self.swaps[mid] = {"op": "load", "done": now + random.uniform(2.5, 4.0)}

    def request_unload(self, mid: str):
        """Schedules an unload of a resident model."""
        with self.lock:
            if mid in self.loaded and mid not in self.swaps:
                self.swaps[mid] = {"op": "unload", "done": time.time() + random.uniform(1.5, 2.5)}

    # --- the tick ----------------------------------------------------------

    def tick(self):
        """Advances the simulation one step; called ~once per second."""
        with self.lock:
            now = time.time()

            # 1) finish any swaps whose timer elapsed.
            for mid, s in list(self.swaps.items()):
                if now >= s["done"]:
                    if s["op"] == "load":
                        self.loaded.add(mid)
                        self.port_of.setdefault(mid, self._alloc_port())
                    else:
                        self.loaded.discard(mid)
                        self.port_of.pop(mid, None)
                    del self.swaps[mid]

            # 2) auto load/unload to keep the board moving.
            if AUTO and self.running and not self.swaps and now - self._last_auto > random.uniform(12, 26):
                self._last_auto = now
                candidates = [m["id"] for m in MD.MODELS if m["id"] not in self.loaded]
                if candidates and random.random() < 0.75:
                    self.request_load(random.choice(candidates))
                elif self.loaded:
                    self.request_unload(random.choice(list(self.loaded)))

            # 3) fabricate traffic: each resident model may fire a request this
            #    tick, lighting its GPUs `serving` and dropping a history row.
            if self.running:
                for mid in list(self.loaded):
                    if mid in self.swaps:
                        continue
                    if random.random() < 0.35:
                        m = self.by_id[mid]
                        busy = random.uniform(1.5, 6.0)
                        for gi in m["gpus"]:
                            self.serving_until[gi] = max(self.serving_until.get(gi, 0), now + busy)
                        self._add_history_row(now)

            # 4) trim history to the retained window.
            cutoff = now - HIST_WINDOW
            if len(self.history) > 6000:
                self.history = [r for r in self.history if r["request_time"] > cutoff]

            # 5) flush a GPU util/VRAM sample on cadence.
            if now - self._last_flush >= GPU_FLUSH:
                self._last_flush = now
                swapping = self._swapping_gpus()
                gcut = now - GPU_WINDOW
                for g in MD.GPUS:
                    gi = g["index"]
                    st = self._gpu_status(gi, now, swapping) if self.running else "inactive"
                    if st == "serving":
                        util = random.uniform(55, 99)
                    elif st == "swapping":
                        util = random.uniform(25, 70)
                    else:
                        util = random.uniform(1, 9)
                    self.util_history[gi].append((round(now, 1), round(util, 1)))
                    self.vram_history[gi].append((round(now, 1), round(self._vram_for(gi) + random.uniform(-100, 100), 1)))
                    self.util_history[gi] = [(t, v) for t, v in self.util_history[gi] if t > gcut]
                    self.vram_history[gi] = [(t, v) for t, v in self.vram_history[gi] if t > gcut]

            # 6) record status transitions for the timeline lanes.
            self._record_timeline(now)

    # --- API views ---------------------------------------------------------

    def overall_status(self) -> str:
        """Router-wide status the navbar shows (serving > swapping > idle)."""
        with self.lock:
            if not self.running:
                return "inactive"
            now = time.time()
            swapping = self._swapping_gpus()
            sts = [self._gpu_status(g["index"], now, swapping) for g in MD.GPUS]
            if "serving" in sts:
                return "serving"
            if "swapping" in sts:
                return "swapping"
            return "idle"

    def router_view(self) -> dict:
        """Payload for GET /router."""
        with self.lock:
            now = time.time()
            swapping = self._swapping_gpus()
            gpu_status = {
                str(g["index"]): (self._gpu_status(g["index"], now, swapping) if self.running else "inactive")
                for g in MD.GPUS
            }
            instances = {str(self.port_of[m]): m for m in sorted(self.loaded) if m in self.port_of}
            return {
                "status": self.overall_status(),
                "gpu_status": gpu_status,
                "ports": sorted(self.port_of.values()),
                "instances": instances,
                "num_gpus": len(MD.GPUS),
                "max_models_per_gpu": MD.MAX_MODELS_PER_GPU,
                "eviction_policy": MD.EVICTION_POLICY,
                "model_gpus": {m["id"]: m["gpus"] for m in MD.MODELS},
            }

    def models_view(self) -> dict:
        """Payload for GET /router/models."""
        with self.lock:
            return {
                "object": "list",
                "data": [
                    {
                        "id": m["id"],
                        "status": {"value": "loaded" if m["id"] in self.loaded else "unloaded"},
                        "gpus": m["gpus"],
                        "ports": [self.port_of[m["id"]]] if m["id"] in self.port_of else [],
                    }
                    for m in MD.MODELS
                ],
            }

    def gpu_view(self) -> dict:
        """Payload for GET /router/gpu."""
        with self.lock:
            return {
                "gpus": [
                    {
                        "index": g["index"],
                        "name": g["name"],
                        "total_vram_mb": g["total_vram_mb"],
                        "util_history": list(self.util_history[g["index"]]),
                        "vram_history": list(self.vram_history[g["index"]]),
                    }
                    for g in MD.GPUS
                ]
            }

    def timeline_view(self) -> dict:
        """Payload for GET /router/status_timeline."""
        with self.lock:
            return {"entries": {str(g): list(lane) for g, lane in self.timeline.items()}}

    def history_view(self, model=None, since=None, until=None, limit=10000) -> list:
        """Payload for GET /router/history (newest first, windowed + filtered)."""
        with self.lock:
            rows = self.history
            out = []
            for r in reversed(rows):  # newest first
                if model and r["model"] != model:
                    continue
                if since is not None and r["request_time"] < since:
                    continue
                if until is not None and r["request_time"] > until:
                    continue
                out.append(r)
                if len(out) >= limit:
                    break
            return out

    def reset_history(self):
        """Clears all request history."""
        with self.lock:
            self.history.clear()


SIM = Sim()
LANGUAGES = None


def load_languages():
    """Reads the real translate language table so /translate isn't empty."""
    global LANGUAGES
    if LANGUAGES is not None:
        return LANGUAGES
    out = []
    csv_path = SRC / "translate" / "languages.csv"
    try:
        import csv
        with open(csv_path, newline="") as f:
            for row in csv.reader(f):
                if len(row) >= 2:
                    out.append({"lang_id": row[0].strip(), "lang_name": row[1].strip()})
    except Exception:
        out = [{"lang_id": "en-US", "lang_name": "English"}, {"lang_id": "ja-JP", "lang_name": "Japanese"}]
    LANGUAGES = out
    return out


def sim_loop():
    """Background driver: ticks the simulation once a second forever."""
    while True:
        try:
            SIM.tick()
        except Exception as e:  # never let the demo die on a sim hiccup
            print(f"[demo] tick error: {e}", flush=True)
        time.sleep(1.0)


# Minimal placeholder served for the /chat iframe (there is no real llama.cpp UI
# behind these ports in the demo). Keeps the tab structure/embedding exercised.
INSTANCE_PAGE = """<!doctype html><html><head><meta charset=utf-8>
<style>html,body{height:100%;margin:0;font-family:ui-monospace,monospace;
background:#12121a;color:#a6accd;display:flex;align-items:center;justify-content:center}
.b{text-align:center;opacity:.75;padding:2rem}.p{font-size:2rem;color:#8aadf4}</style></head>
<body><div class=b><div class=p>demo instance :{port}</div>
<div>{model}</div><div style="margin-top:.6rem;opacity:.5">no real llama.cpp UI in demo mode</div>
</div></body></html>"""


class Handler(BaseHTTPRequestHandler):
    """Routes static SPA/asset requests and the faked router API."""

    server_version = "llama-router-demo"

    def log_message(self, fmt, *args):
        """Silences per-request logging (keep the console for [demo] lines)."""
        pass

    # --- response helpers --------------------------------------------------

    def _json(self, obj, status=200):
        """Writes a JSON response."""
        body = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _html(self, text, status=200):
        """Writes an HTML response."""
        body = text.encode() if isinstance(text, str) else text
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _file(self, path: Path):
        """Serves a file from disk, guarding against path escapes."""
        try:
            path = path.resolve()
            path.relative_to(SRC.resolve())  # containment check
            data = path.read_bytes()
        except (ValueError, FileNotFoundError, IsADirectoryError):
            return self._json({"error": "not found"}, 404)
        ctype = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _spa(self, page: str):
        """Serves one of the real SPA index.html files from ../src."""
        html = SRC / page / "index.html"
        if not html.exists():
            return self._json({"error": f"{page} not found"}, 404)
        return self._html(html.read_text())

    # --- routing -----------------------------------------------------------

    def do_GET(self):
        u = urlparse(self.path)
        p = u.path
        q = parse_qs(u.query)

        if p == "/":
            self.send_response(302)
            self.send_header("Location", "/dash")
            self.end_headers()
            return
        if p in ("/dash", "/chat", "/translate"):
            return self._spa(p.lstrip("/"))
        if p.startswith("/webui/"):
            return self._file(SRC / "webui" / p[len("/webui/"):])

        # --- router API ---
        if p == "/router":
            return self._json(SIM.router_view())
        if p == "/router/models":
            return self._json(SIM.models_view())
        if p == "/router/gpu":
            return self._json(SIM.gpu_view())
        if p == "/router/status_timeline":
            return self._json(SIM.timeline_view())
        if p == "/v1/models":
            return self._json({"object": "list", "data": [{"id": m["id"], "object": "model"} for m in MD.MODELS]})
        if p == "/router/history":
            return self._json(SIM.history_view(
                model=(q.get("model", [None])[0] or None),
                since=_f(q.get("since", [None])[0]),
                until=_f(q.get("until", [None])[0]),
                limit=int(q.get("limit", ["10000"])[0]),
            ))
        if p == "/router/reset_history":
            SIM.reset_history()
            return self._json({"success": True})
        if p == "/router/languages":
            langs = load_languages()
            name = q.get("lang", [None])[0]
            return self._json([l for l in langs if l["lang_name"] == name] if name else langs)
        if p in ("/router/start", "/router/stop", "/router/restart"):
            SIM.running = (p != "/router/stop")
            return self._json({"success": True, "status": SIM.overall_status()})

        # --- chat iframe placeholder ---
        if p.startswith("/instance/"):
            parts = p.split("/")
            port = parts[2] if len(parts) > 2 else "?"
            if p.rstrip("/").endswith("/props"):
                return self._json({"default_generation_settings": {"n_ctx": 262144}})
            model = next((m for m, pt in SIM.port_of.items() if str(pt) == port), "—")
            return self._html(INSTANCE_PAGE.replace("{port}", str(port)).replace("{model}", model))

        return self._json({"error": "not found", "path": p}, 404)

    def do_POST(self):
        u = urlparse(self.path)
        p = u.path
        length = int(self.headers.get("Content-Length", "0") or "0")
        raw = self.rfile.read(length) if length else b""
        try:
            body = json.loads(raw) if raw else {}
        except Exception:
            body = {}

        if p == "/models/load":
            mid = body.get("model")
            if mid not in SIM.by_id:
                return self._json({"success": False, "error": {"message": "unknown model"}}, 404)
            SIM.request_load(mid)
            return self._json({"success": True})
        if p == "/models/unload":
            mid = body.get("model")
            if mid not in SIM.by_id:
                return self._json({"success": False, "error": {"message": "unknown model"}}, 404)
            SIM.request_unload(mid)
            return self._json({"success": True})
        if p == "/router/translate":
            return self._fake_translate(body)

        return self._json({"error": "not found", "path": p}, 404)

    def _fake_translate(self, body: dict):
        """Streams a fabricated OpenAI-style translation so /translate works."""
        text = (body.get("text") or "").strip()
        # A dummy "translation": echo with a marker, chunked into tokens.
        out = f"[demo translation] {text}" if text else "[demo translation]"
        pieces = out.split(" ")
        stream = bool(body.get("stream"))
        if not stream:
            return self._json({"choices": [{"message": {"content": out}}]})

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        for i, w in enumerate(pieces):
            chunk = {"choices": [{"delta": {"content": (w if i == 0 else " " + w)}}]}
            try:
                self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                return
            time.sleep(0.04)
        try:
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass


def _f(v):
    """Parses a float query param, tolerating None/blank."""
    try:
        return float(v) if v not in (None, "") else None
    except (TypeError, ValueError):
        return None


def main():
    """Starts the sim thread and serves the demo until interrupted."""
    threading.Thread(target=sim_loop, daemon=True).start()
    httpd = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"[demo] frontend demo on http://{HOST}:{PORT}/dash", flush=True)
    print(f"[demo] {len(MD.GPUS)} GPUs · {len(MD.MODELS)} models · auto={'on' if AUTO else 'off'}", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[demo] bye", flush=True)
        httpd.shutdown()


if __name__ == "__main__":
    main()
