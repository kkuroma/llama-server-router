# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A router in front of `llama.cpp`. It spawns/supervises `llama-server` child processes on demand, routes OpenAI-compatible API requests to the right one (hot-swapping models to fit VRAM), and serves a monitoring dashboard (`/dash`) plus a translation UI (`/translate`). Pure-Python, no build step; ships three ways: Nix package/flake, NixOS module, and Docker.

## Running & developing

There is no test suite, linter, or build step. The app is `src/main.py` run under `asyncio`.

```bash
# Run locally (needs the deps from docker/requirements.txt + a llama-server binary)
python src/main.py

# Nix
nix run .                       # or: nix build .
nix run git+https://git.kuroma.dev/kkuroma/llama-router

# Docker (from repo root; Dockerfile layers the router onto llama.cpp:server-cuda)
docker build -f docker/Dockerfile .
docker compose up -d --build    # standalone; needs docker-compose.override.yml (see README)
```

Config comes entirely from env vars + two config files (no CLI flags):
- `ROUTER_CONFIG_PATH` (default `/configs/config.json`) — models, scheduler tunables, ports, llama-server path.
- `LLAMA_PRESETS_PATH` (default `/configs/presets.ini`) — llama.cpp per-model settings, passed to `llama-server --models-preset`.
- `ROUTER_HOST` (default `0.0.0.0`), `HISTORY_DB_PATH` (default `/webui/monitor/history.db`).

`examples/config.json` and `examples/presets.ini` are the canonical templates. When developing without GPUs, `pynvml` is optional — GPU monitoring and NVML GPU-count detection degrade gracefully.

## Architecture

Four modules under `src/`, wired together in `main.py`:

- **`router.py` — `LLMRouter`**: the whole scheduler and process supervisor. This is where nearly all logic lives (~800 lines). Everything else is thin.
- **`api.py`**: FastAPI app. Custom endpoints (`/router/*`, `/models/load`, `/models/unload`, `/dash`, `/translate`, `/chat`, `/instance/{port}/*`, `/v1/models`), a **`/webui` StaticFiles mount** (shared UI assets — see below), a **`/dash/assets` StaticFiles mount** (the dashboard's own CSS/JS under `src/dash/assets/`), plus a **catch-all `/{full_path}` proxy** that wraps every other request into an "envelope" dict and hands it to `router.add_request()`. The `/webui` mount is registered before the catch-all so its paths are served locally, not forwarded. `api.router`/`api.gpu_monitor`/etc. are module-level globals set by `main.py` at startup.
- **`monitor.py`**: `GPUMonitor` (NVML util/VRAM/temp/power history) and `StatusTimeline` (router status changes). Polled once/sec from a background task in `main.py`. Both keep a 2-hour rolling window. **`GPUMonitor` samples every detected NVIDIA GPU**, keeping a per-GPU history plus its enforced power limit (`.gpus` list, `.snapshot()` serializes it); `/router/gpu` returns `{gpus: [...]}` and the dashboard draws all GPUs merged — one line per GPU across four grids (util/VRAM/temp/power) sharing a zoom slider. `/router/status_timeline` still serves the timeline but the dashboard no longer renders it.
- **`translate/`**: self-contained translation feature (service + SPA + `languages.csv` + `prompt.txt`). Builds a 3-chunk message array designed for llama.cpp prompt caching, then routes it through the normal request queue.
- **`chat/`**: the `/chat` SPA — a themed shell (shared `/webui` design + a tab per live instance) that **embeds llama.cpp's own web UI in an iframe**, one per running `llama-server` replica, for quick tok/s testing (the real chat client lives elsewhere, e.g. LibreChat). Tabs are derived from `/router`'s `instances` map; each iframe loads **`/instance/{port}/`** — the `instanceProxy` route (`api.py`), a **same-origin pass-through proxy** that forwards straight to `http://127.0.0.1:{port}/…` and **bypasses `router.addRequest`/the RWLock entirely** (it streams, so SSE tok/s tick live). Two constraints drive this shape, and both are load-bearing:
  - **Same-origin (not the raw instance port).** An earlier version pointed the iframe directly at `window.location.hostname:{port}`. That only works on the LAN: over Tailscale the instance port must be independently reachable, and behind a reverse proxy (Caddy → `:11434`) the raw port isn't served at all — and an https page loading an http instance is a mixed-content block. Routing through the router's own port keeps everything on one origin/scheme.
  - **Off the scheduler/lock.** The embedded UI polls continuously (`/props`, etc.). If that traffic re-enters the router's catch-all it becomes per-GPU `AsyncRWLock` *readers* on the instance's GPUs, and since readers starve writers, the exclusive lock `unloadModel` needs on those GPUs never drains → that GPU stuck `serving`, can't unload. `instanceProxy` never calls `addRequest`, so UI traffic takes no reader slots. Only the load/unload chips (explicit `/models/load` · `/models/unload` POST routes, one-shot writers) touch the router.
  - llama.cpp's UI uses **relative** asset paths (`./_app/*`, which resolve under the `/instance/{port}/` prefix) but **absolute** API paths (`/v1/*`, `/props`). To keep the latter pinned to the replica instead of escaping to the router root, `instanceProxy` injects a small **fetch/EventSource/XHR shim** (`_INSTANCE_SHIM`) into the proxied index HTML that rewrites same-origin absolute paths to `/instance/{port}/…`. The HTML is buffered for injection; all other responses stream through raw.

  Needs ≥1 loaded instance; the shell shows a "load a model first" state until then. Chat history is intentionally disposable (llama.cpp keeps it client-side/per-origin).
- **`webui/`**: shared design language for the `/dash` and `/translate` SPAs (served under `/webui`): `theme.css` (fonts + seasonal palette + pill navbar + card/popover styles), `theme.js` (`window.LRTheme` — the shunka-shuutou palette, localStorage state under `llama-router-ui`, and the "Aa" appearance popover: theme swatches, light/dark/auto mode, text size), and `fonts/` (Google Sans Flex + Maple Mono woff2). daisyUI's `--fallback-*` color vars are aliased onto the palette in `theme.css`, so one theme source recolors every component; charts read the active palette via `LRTheme.colors()`.

### Request lifecycle (the core loop)

1. A request hits `api.py`, which builds an **envelope** `{path, method, body, headers, model, is_streaming}` and calls `router.add_request()`, getting back a `Future`.
2. `add_request` appends to `self.requests` and sets `self._has_requests`. A single long-lived `_scheduler()` task wakes up.
3. `_scheduler` picks a request to serve, **preferring one whose model is already resident** (cache-hit maximization). If nothing is servable it loads the head request's model (evicting as needed). It then picks the least-busy replica port (`inflight` count) and dispatches `_do_forward` / `_do_forward_streaming` as a concurrent task.
4. The forward task resolves the Future with an `httpx.Response` (non-streaming) or an `asyncio.Queue` of chunks terminated by a `None` sentinel (streaming), and records token counts to the SQLite history DB.

### Scheduling model — read the README before touching this

- **One `llama-server` process per loaded model replica.** Loading a model spawns `num_instance` processes, each pinned to that model's GPUs and hosting only that model. Evicting = `SIGTERM` (then `SIGKILL`) those processes, which frees VRAM unconditionally. The router owns all placement; llama.cpp's own `models-max` is irrelevant here.
- **Residency is tracked PER GPU**, capped at `MAX_MODELS_PER_GPU` (default 1). Models on disjoint GPUs coexist; loading a model only evicts residents on the GPUs it actually needs. See `_plan_evictions`.
- **`gpus` (in `config.json`'s `LLM` section) is the single source of truth for both scheduler accounting and physical placement.** The router masks each process with `CUDA_VISIBLE_DEVICES`/`HIP_VISIBLE_DEVICES` = that model's `gpus`. Absent → `[0]`; `-1`/`"all"` → every GPU. **Do NOT add a `device` key to presets.ini** — masking renumbers devices from 0 inside each process, so absolute `device = CUDAn` ids won't resolve, and masking (vs `--device`) is deliberate to keep ggml's per-device context/buffers off co-resident models' GPUs.
- **Eviction**: `lru` (default) or `fifo`. Residents with queued demand are evicted last. Anti-starvation: a head-of-queue request whose model isn't loaded and has waited past `QUEUE_FORCE_LOAD_TIMEOUT` (300s) force-loads instead of being starved by newer cache-hit requests.
- **Crash recovery**: `_reap_dead()` drops bookkeeping for exited processes; the model auto-reloads on its next request.
- **Load-failure fast-fail**: a model whose worker dies during load (typically CUDA OOM from too large a `ctx-size`) must not wedge the GPU in `swapping`. llama-server's multi-model supervisor stays healthy and only flags the failure inside `/models` (`"failed": true` / `"exit_code"`); `_load_into` polls for that (and for a whole-process exit, and a 4xx/5xx from the load POST) and fails fast with `_last_load_fatal` set, so `_spawn_replica` doesn't retry the same wall and `load_model` releases the per-GPU exclusive lock immediately. The error propagates to the triggering request (502 with the reason) instead of hanging until `LOAD_POLL_TIMEOUT`.
- **Concurrency control is PER GPU**: one `AsyncRWLock` per GPU (`_gpu_locks`). A generation holds the *shared* (reader) lock of every GPU its model spans; a load/unload holds the *exclusive* (writer) lock of every GPU it mutates and waits for that GPU's in-flight generation to drain. Multi-GPU work acquires locks in ascending GPU order, so overlapping acquisitions can't deadlock. Readers still starve writers **per GPU**, so a swap on one GPU never blocks generation on another. See `_acquire_shared`/`_acquire_exclusive` and `_load_affected_gpus` (a load also locks the full span of any resident it may evict).
- **Status is PER GPU**, computed on demand (`_gpu_status`): a GPU is `serving` while a replica on it has an in-flight request, `swapping` while a model is loading/unloading on it (the two are merged), else `idle`. `self.status` holds only the router-wide lifecycle state (`inactive`/`starting`/`stopping`/`error`); `overall_status()` aggregates the per-GPU running states for global consumers, and `/router` returns both `status` and a `gpu_status` map. The scheduler **defers** loading a queued request's model while that model's GPUs are busy serving or swapping (rather than evicting/competing) — it retries when a port release or swap completion re-wakes it, with `QUEUE_FORCE_LOAD_TIMEOUT` as the anti-starvation escape.

Because `gpus` is duplicated meaning across accounting + masking, changes to residency/placement usually touch `_resolve_gpus`, `_plan_evictions`, `load_model`, and `_start_instance` together.

## Nix ↔ config mapping

`module.nix` generates both config files from one `services.llama-router.models` attrset: `num_instance` and `gpus` go into the router JSON (`LLM` section), everything else on a model becomes its `presets.ini` section; `presetGlobals` becomes the `[*]` section. If you add a new router-only per-model field, it must be stripped in `mkPreset` (`removeAttrs`) so it doesn't leak into the INI. Keep `config.json`/`presets.ini` schema, the NixOS module options, and the README table in sync when changing configuration.

## Conventions

- Log lines are prefixed tags like `[ROUTER]`, `[LOAD/UNLOAD SUCCESS]`, `[main]`, printed with `flush=True` (they land in journald/docker logs). Match this style.
- The router binds `0.0.0.0` and spawned `llama-server`s have **no auth** — this is meant to sit behind a reverse proxy / scoped firewall (the NixOS module defaults `host` to `127.0.0.1`).
