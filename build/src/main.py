import asyncio
import os
import signal
import sys

import uvicorn

import api
from router import LLMRouter
from monitor import GPUMonitor, StatusTimeline
from translate.translate import TranslationService

ROUTER_CONFIG_PATH  = os.environ.get("ROUTER_CONFIG_PATH", "/configs/config.json")
LLAMA_PRESETS_PATH  = os.environ.get("LLAMA_PRESETS_PATH", "/configs/presets.ini")

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
    await router.init_history_db()
    _print_status(router)
    api.router = router

    # GPU + status monitoring
    try:
        gpu_monitor = GPUMonitor()
        api.gpu_monitor = gpu_monitor
        print("[main] GPU monitor initialized", flush=True)
    except Exception as exc:
        gpu_monitor = None
        print(f"[main] GPU monitor unavailable: {exc}", flush=True)

    status_timeline = StatusTimeline()
    api.status_timeline = status_timeline

    try:
        translation_service = TranslationService()
        api.translation_service = translation_service
        print("[main] Translation service initialized", flush=True)
    except Exception as exc:
        print(f"[main] Translation service unavailable: {exc}", flush=True)

    async def _monitor_loop():
        while True:
            if gpu_monitor:
                await asyncio.to_thread(gpu_monitor.poll)
            status_timeline.record(router.status.value)
            await asyncio.sleep(1.0)

    asyncio.create_task(_monitor_loop())

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
