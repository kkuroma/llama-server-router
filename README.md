# LLaMa router

A router that sits in front of your `llama.cpp`'s server that handles seamless model switching of many small GPUs for homelabs of a few users. It spawns and supervises `llama-server` processes on demand, routes an OpenAI-compaible traffic to the right one, and hot swaps model per GPU with VRAM constraint. Pure python, no build or compilation. Ships as a Nix flake, a NixOS module, and a Docker image.

## What it does

- **It schedules your requests so nothing starves:** Overload it and requests queue and drain, **never a 429 error**. The scheduler batches queued requests by resident model to swap as little as possible, spreads a hot model across `num_instance` replicas by least-busy routing, and force-loads any request that has waited too long so nothing gets starved out.
- **It rarely, if ever, crashes silently:** Dead replicas are reaped and reload on their next request, and a load that runs out of VRAM fails fast with a reason instead of wedging the GPU in a half-dead state.
- **Explicit, declarative GPU placement:** give each model a list of GPU ids, then the router manages eviction and device masking. No `CUDA_VISIBLE_DEVICES` needed to be set by hand. Models on disjoint GPUs stay resident and serve at the same time.
- **Ships a beautiful dashboard:** `/dash` (per-GPU util/VRAM timeline, request history), `/chat` (embedded llama.cpp UI per replica for tok/s testing), `/translate`.

## What it does better than LLaMa swap

[llama-swap](https://github.com/mostlygeek/llama-swap) is another tool is aimed at a similar space of locally hosted LLMs on consumer hardware with VRAM constraint. It's gained more traction, but here's a quick comparison between the both of them:

|                         | llama-swap                       | llama-router                                   |
| ----------------------- | -------------------------------- | ---------------------------------------------- |
| Overload behavior       | reject (429)                     | queue and drain, no drops                      |
| Multi-model concurrency | manual `groups` matrix           | automatic, per-GPU residency                   |
| GPU placement           | manual `CUDA_VISIBLE_DEVICES`    | automatic masking from `gpus`                  |
| VRAM / GPU accounting   | none (you avoid oversubscribing) | tracked per GPU, targeted eviction             |
| Scheduling granularity  | whole-server                     | per GPU (a swap on GPU 0 does not block GPU 1) |
| Replicas per model      | 1                                | `num_instance`, least-busy routing             |

`llama-router` only speaks `llama.cpp`, which is intentional. `vLLM` does not play well with hot-swapping and uses a messier environment, and we've decided against using it in a few-user homelab situation. If you want vLLM, whisper, and a single Go binary, use `llama-swap`. If you own a pile of small GPUs and want one endpoint that figure out where requests go without needing to manually manage VRAM states and never drops a request, use `llama-server`. Proven by months of stable use as "just an OpenAI compatible provider" you don't need to worry again. Perfectly compatible with LiteLLM, Librechat, OpenCode, or any local LLM tool.

## Benchmarks

`scripts/` holds standalone benchmarks (stdlib only, no venv needed) that write to `outputs/benchmark-<task>.json`:

- **`benchmark-concurrency.py`** hammers a router with N short requests through a bounded worker pool and reports the success rate and latency distribution. It only needs the host and N (`--host localhost:11434 -n 100`). On a 3-GPU box serving three swapping models (gemma-4-26b on `[0,2]`, gemma-4-12b on `[1]`, a qwen-27b on `[0,1,2]`), N=100 concurrent requests completed at a **100% success rate**: the router queued and drained every request across the swaps without dropping one.
- **`benchmark-throughput.py`** measures prompt-processing and token-generation tok/s per model at concurrency 1 and each model's parallel-slot count.

## Scheduling model

**One llama-server process per loaded model replica.** Loading a model spawns `num_instance` llama-server processes (each hosting exactly that model, pinned to its GPUs); evicting kills them, which frees VRAM unconditionally. The router owns all placement decisions — llama-server's own `models-max` is irrelevant in this design and can be omitted.

Each model is pinned to a set of GPU ids. The router keeps at most `MAX_MODELS_PER_GPU` models resident **per GPU** (not globally): models pinned to disjoint GPUs stay in memory together, and loading a model only evicts residents on the GPUs it actually needs.

Example with `MAX_MODELS_PER_GPU = 1`: models A and B pinned to GPUs `[0, 1]`, C and D pinned to `[2]`. A and C can be resident simultaneously (two llama-server processes). Requesting B kills only A's process (GPUs 0/1); C's process is untouched. Requesting D evicts only C.

- **No `gpus` field** → the model counts against GPU 0 only. On a single-GPU host this reproduces plain global behavior exactly.
- **`gpus = "all"` or `-1`** → the model counts against every GPU (it will evict on all of them as needed).
- **Eviction policy**: `lru` (default, evicts the model whose last request is oldest) or `fifo` (evicts the earliest-loaded model). Either way, residents with requests still waiting in the queue are only evicted when there is no other candidate on that GPU.
- **Anti-starvation**: if the head-of-queue request needs a model that isn't loaded and has waited longer than `QUEUE_FORCE_LOAD_TIMEOUT` (default 300 s), the router force-loads it instead of serving newer cache-hit requests forever.
- **Crash recovery**: dead replicas are reaped automatically; the model simply reloads on its next request.
- **`num_instance > 1`** spawns that many replica processes of the model (requests balance across them by in-flight count). Each replica is a full copy of the weights — VRAM scales linearly.
- GPU count is autodetected via NVML; override with `ROUTER.NUM_GPUS` (falls back to highest pinned id + 1 when NVML is unavailable).
- Small overhead note: co-resident models are separate processes, so each pays its own CUDA context (~a few hundred MB per process per GPU it touches).

## GPU pinning

The `gpus` field on the model's entry (`config.json` `LLM` section, or the `gpus` attribute in the NixOS module) is the **single source of truth** for both halves of pinning:

1. **Scheduler accounting** — residency/eviction are tracked per GPU in `gpus`.
2. **Physical placement** — the router spawns each llama-server with `CUDA_VISIBLE_DEVICES` set to that model's `gpus`, so the process only ever touches those devices.

You do **not** set a `device` key in presets.ini. Masking with `CUDA_VISIBLE_DEVICES` (rather than llama.cpp's `--device`) is deliberate: ggml initializes a CUDA context and reserves buffers on _every visible_ device, so a model pinned via `--device` alone still holds hundreds of MB / a couple GB on the GPUs it isn't computing on — enough to OOM a co-resident model on a tight box. Masking keeps that overhead off other models' GPUs. Because the visible devices are renumbered from 0 inside each process, an absolute `device = CUDA2` would not even resolve; omit it. Unpinned models default to GPU 0 (masked to GPU 0), so on multi-GPU hosts they no longer silently shard across all GPUs.

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
    };

    # each model becomes a presets.ini section; num_instance and gpus are
    # router-only (gpus masks the process via CUDA_VISIBLE_DEVICES; no device key)
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

The router binds `127.0.0.1` by default (`services.llama-router.host`) — put a reverse proxy in front of it or scope your firewall accordingly before exposing it wider; spawned `llama-server` instances have no auth.

Use `services.llama-router.llamaCpp` to supply a CUDA/ROCm build of llama.cpp, e.g. `pkgs.llama-cpp.override { cudaSupport = true; }`. GPU masking sets both `CUDA_VISIBLE_DEVICES` and `HIP_VISIBLE_DEVICES`, so CUDA and ROCm both work with no extra configuration.

## Usage with Docker

The image layers the router on top of `ghcr.io/ggml-org/llama.cpp:server-cuda` (which provides `llama-server` at `/app/llama-server`).

`docker-compose.yml` is a **host-agnostic base**: it defines how to build/run the router (port 11434, env, healthcheck, reserves all GPUs) but declares **no mounts** — where the weights/configs live is the deployer's decision. The router reads three paths inside the container: `/models` (ggufs, ro), `/configs` (`config.json` + `presets.ini`, ro) and `/webui` (request-history SQLite, rw). Supply them one of two ways:

**Standalone** — one router on one host:

```
cp docker-compose.override.yml.example docker-compose.override.yml
mkdir -p models configs webui
cp examples/config.json examples/presets.ini configs/
# drop your .gguf files into models/, edit the configs to match
docker compose up -d --build
```

`docker compose` auto-merges the override, which adds `./models`, `./configs`, `./webui`. The override is gitignored, so `git pull` stays clean.

**Orchestrated** — this router as one service in a larger stack. A parent compose `include:`s this file and patches in the host mounts (and can point them at shared directories outside this repo, e.g. a central weights dir):

```yaml
# ../docker-compose.yml   (run `docker compose up` from the parent dir)
include:
  - llama-router/docker-compose.yml
services:
  llama-router:
    volumes:
      - ../clanker-weights:/models:ro
      - ./config/llama-router:/configs:ro
      - ./webui:/webui:rw
```

`include` resolves this file's `build:` context relative to `llama-router/`, while the parent's added `volumes:` resolve relative to the parent — so upstream `git pull`s here never touch the host wiring.

To expose only some GPUs, override the device reservation (`count: 2` or `device_ids: ["0","1"]`) from the deployer file — GPU ids inside the container renumber from 0, and each model's `gpus` pin refers to those container-local ids.

## Running directly

```
nix run git+https://git.kuroma.dev/kkuroma/llama-router
```

Configuration is passed via environment variables:

| Variable             | Default                     | Purpose                                          |
| -------------------- | --------------------------- | ------------------------------------------------ |
| `ROUTER_CONFIG_PATH` | `/configs/config.json`      | Router config (models, scheduler timings, ports) |
| `LLAMA_PRESETS_PATH` | `/configs/presets.ini`      | llama.cpp presets INI                            |
| `ROUTER_HOST`        | `0.0.0.0`                   | API bind address                                 |
| `HISTORY_DB_PATH`    | `/webui/monitor/history.db` | SQLite request history                           |

Scheduler settings live in the `ROUTER` section of `config.json`: `MAX_MODELS_PER_GPU` (default 1), `EVICTION_POLICY` (`lru`/`fifo`), `QUEUE_FORCE_LOAD_TIMEOUT` (seconds, default 300), `NUM_GPUS` (optional override), plus the health-check/load/unload timings shown in `examples/config.json`.
