# tests

The parts every case shares: the throwaway router, the fixtures that own it, and
the measurements taken from its replies. `harness.py` generates a `config.json`
and a `presets.ini` from `../configs.toml`, runs `src/main.py` as a subprocess in
its own session, and times each streamed token so concurrency can be read back
without trusting the router's own bookkeeping. `conftest.py` turns that into one
session scoped router plus the preflight that refuses a busy card.

```
core/
├── harness.py    config generation, RouterProcess, stream timing, the slot probe
└── conftest.py   the cfg, preflight and router fixtures, re-exported by ../conftest.py
```

Nothing here imports `src`, so a case only ever sees the router's HTTP surface.
The slot probe reads `/slots?model=<id>` on the replica port, because the spawned
process is llama.cpp's own router and its per model views are keyed by name.
