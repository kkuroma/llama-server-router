"""
Checks the embedding pool's process supervision against a stand-in llama-server

An autostart model must come back after its process dies and stay down after an unload
The test needs no GPU, so it runs in the default `pytest` collection
"""

import asyncio
import os
import signal
import socket
import stat
from pathlib import Path

import embedding
from embedding import EmbeddingPool

FAKE = Path(__file__).parent / "core" / "fake_embed_server.py"


def free_port() -> int:
    """
    A port nothing is listening on right now

    Returns:
        The port number
    """
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def make_pool(autostart: bool) -> EmbeddingPool:
    """
    A pool of one fixed-port model and one that takes its port from the base

    Args:
        autostart (bool): Whether the fixed-port model starts with the pool

    Returns:
        The pool, not yet started
    """
    FAKE.chmod(FAKE.stat().st_mode | stat.S_IXUSR)
    models = {
        "fixed": {"model": "/nonexistent.gguf", "port": free_port(), "autostart": autostart},
        "lazy": {"model": "/nonexistent.gguf"},
    }
    return EmbeddingPool(models, str(FAKE), free_port(), lambda _m, _raw: [0], health_timeout=10.0, kill_timeout=1.0)


def test_ports_and_routing():
    """A model with no port takes the base port, and a forward loads it on demand."""

    async def run():
        pool = make_pool(autostart=False)
        assert pool.ports["fixed"] != pool.ports["lazy"]
        try:
            resp = await pool.forward("lazy", "POST", "/v1/embeddings", b'{"input": ["a", "b", "c"]}', {"Content-Type": "application/json"})
            assert resp.status_code == 200 and len(resp.json()["data"]) == 3
            assert pool.is_loaded("lazy") and not pool.is_loaded("fixed")
            state = {s["id"]: s for s in pool.report()}
            assert state["lazy"]["status"] == "loaded"
            assert state["lazy"]["requests"] == 1 and state["lazy"]["inputs"] == 3
            assert state["fixed"]["status"] == "unloaded"
        finally:
            await pool.stop()
        assert not pool.processes

    asyncio.run(run())


def test_supervisor_restarts_and_unload_holds():
    """A killed autostart process is replaced, and an unloaded one is left down."""

    async def run():
        embedding.SUPERVISE_INTERVAL = 0.2
        pool = make_pool(autostart=True)
        try:
            await pool.start()
            first = pool.processes["fixed"].pid
            os.kill(first, signal.SIGKILL)
            for _ in range(100):
                await asyncio.sleep(0.1)
                if pool.is_loaded("fixed") and pool.processes["fixed"].pid != first:
                    break
            assert pool.is_loaded("fixed") and pool.processes["fixed"].pid != first

            await pool.unload("fixed")
            await asyncio.sleep(1.0)
            assert not pool.is_loaded("fixed")
            assert {s["id"]: s["status"] for s in pool.report()}["fixed"] == "unloaded"

            await pool.load("fixed")
            second = pool.processes["fixed"].pid
            await pool.restart("fixed")
            assert pool.is_loaded("fixed") and pool.processes["fixed"].pid != second
        finally:
            await pool.stop()

    asyncio.run(run())


def test_failed_load_reports_the_reason():
    """A process that exits before it is healthy surfaces as failed with its exit code."""

    async def run():
        pool = make_pool(autostart=False)
        pool.executable = "/bin/false"
        try:
            await pool.load("lazy")
        except RuntimeError as exc:
            assert "exited" in str(exc)
        else:
            raise AssertionError("load of a process that exits at once succeeded")
        state = {s["id"]: s for s in pool.report()}["lazy"]
        assert state["status"] == "failed" and "exited" in (state["error"] or "")

    asyncio.run(run())
