# llama-router

llama.cpp router with web dashboard and scheduler. Spawns and supervises
`llama-server` instances on demand, routes OpenAI-compatible API requests to
them, and serves a monitoring dashboard (GPU/VRAM timeline, request history)
plus a translation UI.

- Router API on `:11434` (OpenAI-compatible, model hot-swapping)
- Dashboard at `/` — status timeline, GPU utilization, request history

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

    # llama.cpp settings applied to every preset (the "[*]" section)
    presetGlobals = {
      jinja = true;
      fa = true;
      ngl = 99;
    };

    # each model becomes a presets.ini section; num_instance is router-only
    models = {
      "Qwen3-4B" = {
        num_instance = 1;
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
e.g. `pkgs.llama-cpp.override { cudaSupport = true; }`.

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
