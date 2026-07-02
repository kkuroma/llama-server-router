# llama-router

llama.cpp router with web dashboard and scheduler. Spawns and supervises
`llama-server` instances on demand, routes OpenAI-compatible API requests to
them, and serves a monitoring dashboard (GPU/VRAM timeline, request history)
plus a translation UI.

- Router API on `:11434` (OpenAI-compatible, model hot-swapping)
- Dashboard at `/dash` — status timeline, GPU utilization, request history
- Multi-GPU aware scheduling with per-GPU residency caps and per-model GPU pinning

## Scheduling model

Each model is pinned to a set of GPU ids. The router keeps at most
`MAX_MODELS_PER_GPU` models resident **per GPU** (not globally): models pinned
to disjoint GPUs stay in memory together, and loading a model only evicts
residents on the GPUs it actually needs.

Example with `MAX_MODELS_PER_GPU = 1`: models A and B pinned to GPUs `[0, 1]`,
C and D pinned to `[2]`. A and C can be resident simultaneously. Requesting B
evicts only A (GPUs 0/1); C stays loaded. Requesting D evicts only C.

- **No `gpus` field** → the model counts against GPU 0 only. On a single-GPU
  host this reproduces plain global behavior exactly.
- **`gpus = "all"` or `-1`** → the model counts against every GPU (it will
  evict on all of them as needed).
- **Eviction policy**: `lru` (default, evicts the model whose last request is
  oldest) or `fifo` (evicts the earliest-loaded model).
- **Anti-starvation**: if the head-of-queue request needs a model that isn't
  loaded and has waited longer than `QUEUE_FORCE_LOAD_TIMEOUT` (default 300 s),
  the router force-loads it instead of serving newer cache-hit requests forever.
- GPU count is autodetected via NVML; override with `ROUTER.NUM_GPUS` (falls
  back to highest pinned id + 1 when NVML is unavailable).

**Important — `models-max`:** llama-server enforces its own per-instance
resident cap (`models-max` in the presets `[*]` section). It must be at least
the maximum number of models that can be co-resident across all GPUs
(`MAX_MODELS_PER_GPU × GPU count`), otherwise llama-server evicts behind the
router's back. On a single GPU with `MAX_MODELS_PER_GPU = 1`, `models-max = 1`
is correct.

## GPU pinning

Pinning has two halves that must agree:

1. **Scheduler accounting** — the `gpus` field on the model's entry in the
   router config (`config.json` `LLM` section, or the `gpus` attribute in the
   NixOS module).
2. **Physical placement** — the `device` key in the model's presets.ini
   section (llama.cpp device names: `device = CUDA0,CUDA1`).

The NixOS module derives both from the single `gpus` attribute. Docker/manual
users set both by hand — see `examples/config.json` + `examples/presets.ini`.
Unpinned models default to GPU 0 in the scheduler; on multi-GPU hosts also set
`device = CUDA0` explicitly, otherwise llama.cpp's default will shard the
model across all GPUs while the router accounts it to GPU 0.

## Usage as a NixOS module

Add the flake input (it follows your nixpkgs):

```nix
{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    llama-router = {
      url = "git+https://git.kuroma.dev/kkuroma/llama-router";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };
}
```

Import `llama-router.nixosModules.default` into your host and configure:

```nix
{
  services.llama-router = {
    enable = true;
    port = 11434;

    # scheduler
    maxModelsPerGpu = 1;          # residency cap PER GPU
    evictionPolicy = "lru";       # or "fifo"
    queueForceLoadTimeout = 300;  # seconds before a starved request forces a load
    # gpuCount = 4;               # optional; autodetected via NVML otherwise

    # llama.cpp settings applied to every preset (the "[*]" section)
    presetGlobals = {
      jinja = true;
      fa = true;
      ngl = 99;
      models-max = 1; # must be >= max co-resident models (see Scheduling model)
    };

    # each model becomes a presets.ini section; num_instance and gpus are
    # router-only (gpus also derives the llama-server `device` key)
    models = {
      "Qwen3-4B" = {
        num_instance = 1;
        gpus = [ 0 1 ];  # omit = GPU 0; "all" or -1 = every GPU
        model = "/data/llm-models/Qwen3-4B-Q8_0.gguf";
        c = 65536;
        b = 4096;
        parallel = 4;
      };
    };
  };
}
```

The router binds `127.0.0.1` by default (`services.llama-router.host`) — put a
reverse proxy in front of it or scope your firewall accordingly before exposing
it wider; spawned `llama-server` instances have no auth.

Use `services.llama-router.llamaCpp` to supply a CUDA/ROCm build of llama.cpp,
e.g. `pkgs.llama-cpp.override { cudaSupport = true; }`. For non-CUDA backends
set `gpuDevicePrefix` (default `"CUDA"`) so derived device names match.

## Usage with Docker

The image layers the router on top of `ghcr.io/ggml-org/llama.cpp:server-cuda`
(which provides `llama-server` at `/app/llama-server`).

```
mkdir -p models configs webui
cp examples/config.json examples/presets.ini configs/
# drop your .gguf files into models/, edit the configs to match
docker compose up -d --build
```

`docker-compose.yml` expects three mounts: `./models` (ggufs, read-only),
`./configs` (`config.json` + `presets.ini`, read-only) and `./webui`
(request-history SQLite, read-write), and reserves all NVIDIA GPUs. To expose
only some GPUs to the container, change the device reservation (`count: 2` or
`device_ids: ["0","1"]`) — GPU ids inside the container are then renumbered
from 0, and `gpus`/`device` pins refer to those container-local ids.

## Running directly

```
nix run git+https://git.kuroma.dev/kkuroma/llama-router
```

Configuration is passed via environment variables:

| Variable | Default | Purpose |
|----------|---------|---------|
| `ROUTER_CONFIG_PATH` | `/configs/config.json` | Router config (models, scheduler timings, ports) |
| `LLAMA_PRESETS_PATH` | `/configs/presets.ini` | llama.cpp presets INI |
| `ROUTER_HOST` | `0.0.0.0` | API bind address |
| `HISTORY_DB_PATH` | `/webui/monitor/history.db` | SQLite request history |

Scheduler settings live in the `ROUTER` section of `config.json`:
`MAX_MODELS_PER_GPU` (default 1), `EVICTION_POLICY` (`lru`/`fifo`),
`QUEUE_FORCE_LOAD_TIMEOUT` (seconds, default 300), `NUM_GPUS` (optional
override), plus the health-check/load/unload timings shown in
`examples/config.json`.
