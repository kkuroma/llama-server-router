import asyncio
import os
import signal
import sys

import uvicorn

import api
from router import LLMRouter

# path, move to global env later
ROUTER_CONFIG_PATH  = "/configs/config.json"
LLAMA_PRESETS_PATH  = "/configs/presets.ini"

async def _shutdown(router: LLMRouter):
    """Best-effort cleanup: stop all child processes."""
    print("[main] shutting down router ...", flush=True)
    try:
        await router.stop()
    except Exception as exc:
        print(f"[main] error during router.stop(): {exc}", flush=True)
    print("[main] all instances stopped.", flush=True)


def _print_status(router: LLMRouter):
    api_port = router.router_config.get("API-port", 8000)
    print(f"[main] LLM Router listening on port {api_port}", flush=True)
    print(f"[main] status : {router.status.value}", flush=True)
    print(f"[main] instances: {sorted(router.processes.keys())}", flush=True)


async def main():
    router = LLMRouter(ROUTER_CONFIG_PATH, LLAMA_PRESETS_PATH)
    await router.start()
    _print_status(router)
    api.router = router
    loop = asyncio.get_event_loop()

    async def _signal_handler():
        print("\n[main] received shutdown signal", flush=True)
        await _shutdown(router)
        loop.stop()

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, lambda: asyncio.ensure_future(_signal_handler()))

    api_port = router.router_config.get("API-port", 8000)
    config = uvicorn.Config(app=api.app, host="0.0.0.0", port=api_port, log_level="info")
    server = uvicorn.Server(config)

    try:
        await server.serve()
    except Exception as exc:
        print(f"[main] uvicorn exited with error: {exc}", flush=True)
    finally:
        if router.processes:
            await _shutdown(router)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
