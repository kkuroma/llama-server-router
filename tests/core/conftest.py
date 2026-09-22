"""
Boots one throwaway router for the whole session and refuses to run on a busy card

Both test modules share the router, so a model stays resident across tests
The heavy tests carry the gpu marker, which a bare pytest run filters out
"""

import socket
import subprocess
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from core.harness import RouterProcess, llama_server, load_config, weights_path


def _free_vram_mib() -> int | None:
    """
    Reads the free memory of GPU 0 through nvidia-smi

    Returns:
        The free MiB, or None when nvidia-smi is missing or unreadable.
    """
    try:
        out = subprocess.run(
            ["nvidia-smi", "--id=0", "--query-gpu=memory.free",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15.0, check=True,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    try:
        return int(out.strip().splitlines()[0])
    except (IndexError, ValueError):
        return None


def _port_free(host: str, port: int) -> bool:
    """
    Checks that nothing already listens on the router's test port

    Args:
        host: The address the router will bind.
        port: The port the router will bind.

    Returns:
        True when the port is free.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        return probe.connect_ex((host, port)) != 0


@pytest.fixture(scope="session")
def cfg() -> dict[str, Any]:
    """The parsed test configuration."""
    return load_config()


@pytest.fixture(scope="session")
def preflight(cfg: dict[str, Any]) -> None:
    """Fails before any model loads when the binary, the weights, the port or the VRAM are missing."""
    exe = Path(llama_server(cfg))
    if not exe.exists():
        pytest.fail(f"llama-server not found at {exe}, set LLAMA_SERVER or edit test_config.toml")
    for index in (0, 1):
        path = weights_path(cfg, index)
        if not path.exists():
            pytest.fail(f"model_{index} weights not found at {path}")
    host, port = str(cfg["Router"]["host"]), int(cfg["Router"]["api_port"])
    if not _port_free(host, port):
        pytest.fail(f"something already listens on {host}:{port}")
    needed = int(cfg["Router"]["min_free_vram_mib"])
    free = _free_vram_mib()
    if free is None:
        pytest.fail("nvidia-smi gave no free memory reading, so this host has no usable GPU")
    if free < needed:
        pytest.fail(f"only {free} MiB free on GPU 0, the swaps need {needed} MiB")


@pytest.fixture(scope="session")
def router(
    cfg: dict[str, Any], preflight: None, tmp_path_factory: pytest.TempPathFactory
) -> Iterator[RouterProcess]:
    """Runs one router subprocess for the session and reports where its log landed."""
    process = RouterProcess(cfg, tmp_path_factory.mktemp("router"))
    process.start()
    try:
        yield process
    finally:
        print(f"\n[tests] router log: {process.log_path}")
        process.stop()
