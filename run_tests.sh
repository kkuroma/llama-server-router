#!/usr/bin/env bash
# Runs the GPU integration tests against a throwaway router, inside the nix dev shell
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

have_deps() {
    python - <<'PY'
import importlib.util, sys
sys.exit(any(importlib.util.find_spec(m) is None
             for m in ("fastapi", "uvicorn", "httpx", "aiosqlite", "pytest")))
PY
}

# A direnv shell from before pytest joined the dev shell still reports itself as one,
# so the deps decide whether to re-enter it rather than any env var
if ! have_deps && [[ -z "${LLAMA_ROUTER_TEST_SHELL:-}" ]] && command -v nix >/dev/null 2>&1; then
    export LLAMA_ROUTER_TEST_SHELL=1
    exec nix develop "$ROOT" --command bash "$ROOT/run_tests.sh" "$@"
fi

if ! have_deps; then
    echo "missing test dependencies, run this inside 'nix develop' or install" \
         "fastapi uvicorn httpx aiosqlite pytest" >&2
    exit 1
fi

nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv || true

# -s keeps the per request tables visible while the models load and swap
exec python -m pytest tests -m gpu -s -v "$@"
