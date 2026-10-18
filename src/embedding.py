import asyncio
import os
import subprocess
import time
from collections import deque
from collections.abc import Callable
from typing import NotRequired, TypedDict

import httpx


class EmbeddingCfg(TypedDict):
    """One embedding model's entry in the router config's EMBEDDING section."""
    model: str
    args: NotRequired[list[str]]
    gpus: NotRequired[list[int] | int | str]
    port: NotRequired[int]
    autostart: NotRequired[bool]


class EmbeddingState(TypedDict):
    """What /router/embeddings reports for one embedding model."""
    id: str
    status: str
    port: int
    internal_port: int
    pid: int | None
    gpus: list[int]
    autostart: bool
    started_at: float | None
    error: str | None
    requests: int
    inputs: int
    ms_per_input: float | None


# Recent requests kept per model for the latency the tab shows
RECENT_REQUESTS = 50
# Seconds between checks for an autostart model whose process died
SUPERVISE_INTERVAL = 5.0
# Seconds before the supervisor retries a model whose last load failed
RETRY_AFTER_FAILURE = 60.0


class EmbeddingPool:
    """
    One llama-server --embedding process per loaded embedding model, beside the LLMs

    An embedding process takes no part in the LLM scheduler: it holds no per-GPU
    lock, does not count against MAX_MODELS_PER_GPU, and an LLM load never evicts
    it, so an embedding model and an LLM share a GPU. Its requests skip the LLM
    queue and go straight to its process.
    """

    def __init__(
        self,
        models: dict[str, EmbeddingCfg],
        executable: str,
        base_port: int,
        resolve_gpus: Callable[[str, list[int] | int | str | None], list[int]],
        health_timeout: float = 60.0,
        kill_timeout: float = 5.0,
    ):
        """
        Holds the configured models and assigns each its ports up front

        Every process listens on an internal port from base_port. A model that names a
        `port` gets that one as well, but the router listens on it rather than the process,
        so a call there loads the model when it is down instead of finding nothing.

        Args:
            models (dict)           : The EMBEDDING section, model id to its settings
            executable (str)        : Path to the llama-server binary
            base_port (int)         : First internal port for the processes
            resolve_gpus (callable) : The router's (model_id, raw) -> sorted GPU ids normalizer
            health_timeout (float)  : Seconds a process has to answer /health after spawning
            kill_timeout (float)    : Seconds to wait after SIGTERM before SIGKILL
        """
        self.models = models
        self.executable = executable
        self.health_timeout = health_timeout
        self.kill_timeout = kill_timeout
        self.gpus: dict[str, list[int]] = {m: resolve_gpus(m, c.get("gpus")) for m, c in models.items()}
        self.public: dict[int, str] = {int(p): m for m, c in models.items() if (p := c.get("port"))}
        self.ports: dict[str, int] = {}
        taken = set(self.public)
        nxt = base_port
        for mid in models:
            while nxt in taken:
                nxt += 1
            self.ports[mid] = nxt
            taken.add(nxt)
        self.processes: dict[str, subprocess.Popen[bytes]] = {}
        self.started_at: dict[str, float] = {}
        self.errors: dict[str, str] = {}
        self.failed_at: dict[str, float] = {}
        self.loading: set[str] = set()
        # An unload stops an autostart model from being brought back by the supervisor
        self.held_down: set[str] = set()
        self.counts: dict[str, list[int]] = {m: [0, 0] for m in models}
        self.recent: dict[str, deque[tuple[int, float]]] = {m: deque(maxlen=RECENT_REQUESTS) for m in models}
        self._locks: dict[str, asyncio.Lock] = {m: asyncio.Lock() for m in models}
        self._supervisor: asyncio.Task[None] | None = None

    def __contains__(self, model_id: object) -> bool:
        return model_id in self.models

    def public_model(self, port: int | None) -> str | None:
        """
        The model whose public port a request arrived on, None for the router's own port

        Args:
            port (int | None): The local port the request came in on

        Returns:
            The model id, or None
        """
        return self.public.get(port) if port is not None else None

    def public_port(self, model_id: str) -> int | None:
        """The port the router listens on for this model, when it names one."""
        return next((p for p, m in self.public.items() if m == model_id), None)

    def is_loaded(self, model_id: str) -> bool:
        """True while the model's process is alive."""
        proc = self.processes.get(model_id)
        return proc is not None and proc.poll() is None

    def is_ready(self, model_id: str) -> bool:
        """True once the model's process is alive and has answered /health."""
        return self.is_loaded(model_id) and model_id in self.started_at and model_id not in self.loading

    async def start(self):
        """Loads every autostart model and begins supervising them."""
        for mid, mcfg in self.models.items():
            if mcfg.get("autostart"):
                try:
                    await self.load(mid)
                except RuntimeError as exc:
                    print(f"[EMBED] autostart of {mid} failed: {exc}", flush=True)
        if self._supervisor is None:
            self._supervisor = asyncio.create_task(self._supervise())

    async def stop(self):
        """Stops supervising and kills every embedding process."""
        if self._supervisor is not None:
            self._supervisor.cancel()
            try:
                await self._supervisor
            except asyncio.CancelledError:
                pass
            self._supervisor = None
        await asyncio.gather(*[self._kill(m) for m in list(self.processes)], return_exceptions=True)

    async def _supervise(self):
        """Brings back any autostart model whose process died, unless it was unloaded."""
        while True:
            await asyncio.sleep(SUPERVISE_INTERVAL)
            for mid, mcfg in self.models.items():
                if not mcfg.get("autostart") or mid in self.held_down or mid in self.loading:
                    continue
                if self.is_loaded(mid):
                    continue
                if time.monotonic() - self.failed_at.get(mid, -RETRY_AFTER_FAILURE) < RETRY_AFTER_FAILURE:
                    continue
                proc = self.processes.get(mid)
                if proc is not None:
                    print(f"[EMBED] {mid} exited (code {proc.returncode}), restarting", flush=True)
                try:
                    await self.load(mid)
                except RuntimeError as exc:
                    print(f"[EMBED] restart of {mid} failed: {exc}", flush=True)

    async def load(self, model_id: str) -> bool:
        """
        Spawns the model's llama-server and waits for it to answer /health

        Args:
            model_id (str): The embedding model to load

        Returns:
            True once the process is healthy, at once when it already was

        Raises:
            ValueError: If the model is not configured
            RuntimeError: If the process exits or never becomes healthy
        """
        if model_id not in self.models:
            raise ValueError(f"[EMBED] Model [{model_id}] not present in list {list(self.models)}")
        self.held_down.discard(model_id)
        async with self._locks[model_id]:
            if self.is_loaded(model_id):
                return True
            self.loading.add(model_id)
            try:
                return await self._spawn(model_id)
            except RuntimeError:
                self.failed_at[model_id] = time.monotonic()
                raise
            finally:
                self.loading.discard(model_id)

    async def _spawn(self, model_id: str) -> bool:
        """
        Starts one process, masked to the model's GPUs, and polls its /health

        Args:
            model_id (str): The embedding model to start

        Returns:
            True once healthy

        Raises:
            RuntimeError: If the process exits or never becomes healthy
        """
        mcfg = self.models[model_id]
        port = self.ports[model_id]
        mask = ",".join(str(g) for g in self.gpus[model_id])
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = mask
        env["HIP_VISIBLE_DEVICES"] = mask
        cmd = [
            self.executable, "--embedding", "-m", mcfg["model"], "--alias", model_id,
            "--host", "127.0.0.1", "--port", str(port), "--metrics", *mcfg.get("args", []),
        ]
        print(f"[EMBED] starting {model_id} at port {port} (gpus {self.gpus[model_id]})...", flush=True)
        self.errors.pop(model_id, None)
        proc = subprocess.Popen(cmd, stdout=None, stderr=None, env=env)
        self.processes[model_id] = proc
        deadline = time.monotonic() + self.health_timeout
        async with httpx.AsyncClient() as client:
            while time.monotonic() < deadline:
                if proc.poll() is not None:
                    self.errors[model_id] = f"exited (code {proc.returncode}) before becoming healthy"
                    raise RuntimeError(f"[EMBED] {model_id} {self.errors[model_id]}")
                try:
                    resp = await client.get(f"http://127.0.0.1:{port}/health", timeout=2.0)
                    if resp.status_code == 200 and resp.json().get("status") == "ok":
                        self.started_at[model_id] = time.time()
                        print(f"[EMBED] {model_id} up on port {port} (pid {proc.pid})", flush=True)
                        return True
                except (httpx.ConnectError, httpx.TimeoutException, ValueError):
                    pass
                await asyncio.sleep(1.0)
        self.errors[model_id] = f"never became healthy within {self.health_timeout}s"
        await self._kill(model_id)
        raise RuntimeError(f"[EMBED] {model_id} {self.errors[model_id]}")

    async def unload(self, model_id: str):
        """
        Kills the model's process and keeps the supervisor from bringing it back

        Args:
            model_id (str): The embedding model to unload
        """
        self.held_down.add(model_id)
        async with self._locks[model_id]:
            await self._kill(model_id)
        print(f"[EMBED] unloaded {model_id}", flush=True)

    async def restart(self, model_id: str) -> bool:
        """
        Unloads and loads the model, which is the remedy when a long-lived process slows down

        Args:
            model_id (str): The embedding model to restart

        Returns:
            True once the new process is healthy
        """
        await self.unload(model_id)
        return await self.load(model_id)

    async def _kill(self, model_id: str):
        """
        Terminates the model's process, escalating to SIGKILL, and clears its bookkeeping

        Args:
            model_id (str): The embedding model to kill
        """
        proc = self.processes.pop(model_id, None)
        self.started_at.pop(model_id, None)
        if proc is None:
            return
        if proc.poll() is None:
            proc.terminate()
            deadline = time.monotonic() + self.kill_timeout
            while proc.poll() is None and time.monotonic() < deadline:
                await asyncio.sleep(0.2)
            if proc.poll() is None:
                proc.kill()
        proc.wait()

    async def forward(self, model_id: str, method: str, path: str, body: bytes, headers: dict[str, str]) -> httpx.Response:
        """
        Sends one request to the model's process, loading it first when it is down

        Args:
            model_id (str)  : The embedding model the request names
            method (str)    : The HTTP method
            path (str)      : The path and query to forward
            body (bytes)    : The raw request body
            headers (dict)  : The request headers, host and length removed

        Returns:
            The upstream response, read in full
        """
        # a process that is up but not yet healthy is waited on through the same load lock
        if not self.is_ready(model_id):
            await self.load(model_id)
        clock = time.perf_counter()
        async with httpx.AsyncClient() as client:
            resp = await client.request(
                method, f"http://127.0.0.1:{self.ports[model_id]}{path}",
                content=body, headers=headers, timeout=600.0,
            )
        self._count(model_id, resp, time.perf_counter() - clock)
        return resp

    def _count(self, model_id: str, resp: httpx.Response, seconds: float):
        """
        Records one embedding request's input count and time for the tab

        Args:
            model_id (str)          : The model that served it
            resp (httpx.Response)   : Its response
            seconds (float)         : How long it took
        """
        if resp.status_code != 200:
            return
        try:
            data = resp.json()
        except ValueError:
            return
        rows = data.get("data") if isinstance(data, dict) else data
        if not isinstance(rows, list) or not rows:
            return
        self.counts[model_id][0] += 1
        self.counts[model_id][1] += len(rows)
        self.recent[model_id].append((len(rows), seconds))

    def record(self, model_id: str, inputs: int, seconds: float):
        """
        Records a request that reached the process directly, such as the tab's own test

        Args:
            model_id (str)  : The model that served it
            inputs (int)    : How many passages it embedded
            seconds (float) : How long it took
        """
        self.counts[model_id][0] += 1
        self.counts[model_id][1] += inputs
        self.recent[model_id].append((inputs, seconds))

    def report(self) -> list[EmbeddingState]:
        """
        Every configured embedding model with its process and its recent speed

        Returns:
            One entry per model, in config order
        """
        out: list[EmbeddingState] = []
        for mid, mcfg in self.models.items():
            proc = self.processes.get(mid)
            if mid in self.loading:
                status = "loading"
            elif self.is_loaded(mid):
                status = "loaded"
            elif mid in self.errors:
                status = "failed"
            else:
                status = "unloaded"
            recent = self.recent[mid]
            inputs = sum(n for n, _ in recent)
            out.append({
                "id": mid,
                "status": status,
                "port": self.public_port(mid) or self.ports[mid],
                "internal_port": self.ports[mid],
                "pid": proc.pid if proc is not None and proc.poll() is None else None,
                "gpus": self.gpus[mid],
                "autostart": bool(mcfg.get("autostart")),
                "started_at": self.started_at.get(mid),
                "error": self.errors.get(mid),
                "requests": self.counts[mid][0],
                "inputs": self.counts[mid][1],
                "ms_per_input": round(1000 * sum(s for _, s in recent) / inputs, 3) if inputs else None,
            })
        return out
