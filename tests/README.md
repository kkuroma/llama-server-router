# llama-router

Integration tests that run a real router against real weights, since the behavior
they check is scheduling and concurrency and neither survives a mock. Each test
spawns `src/main.py` as a subprocess on its own port, drives it over HTTP like any
client, and reads concurrency off token arrival times. Everything the tests point
at lives in `configs.toml`: the router's port, its scheduler settings, the two
model directories and the size of every case.

Every request sends the same essay from `prompt.txt`, about 14k tokens, and asks
for a 2000 word continuation of it. Only the closing line differs per request, so
the long prefix is shared exactly the way a fleet of agents on one system prompt
shares theirs, and both the prefill and the generation are large enough that a
scheduler cannot hide behind short requests.

```
tests/
├── configs.toml                ports, llama-server path, model dirs, case sizes
├── prompt.txt                  the shared 9.4k word essay every request sends
├── conftest.py                 hands pytest the fixtures in core
├── core/                       the router subprocess, the fixtures, the measurements
├── test_request_order.py       a queued second model may not be overtaken
└── test_concurrent_request.py  users on one model have to generate at the same time
```

Run everything with `./run_tests.sh` from the repo root, which enters the nix dev
shell and runs the `gpu` marked tests with output unbuffered. A bare `pytest`
collects them and filters them out, so nothing touches the GPU by accident. Single
cases take the usual selectors, for example
`python -m pytest tests -m gpu -s -k u4-p2`.

The tests want the whole card to themselves: they refuse to start when GPU 0 has
less free memory than `min_free_vram_mib`, and a model the deployed router loads
mid-run will make the swaps fail. The router log of a failed run is printed at the
end of the session.
