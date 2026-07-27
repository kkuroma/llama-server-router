import asyncio
import json
import os
import subprocess
import time
from collections.abc import Iterable
from enum import Enum
from typing import Any, NotRequired, TypedDict, cast

import httpx

import aiosqlite

HISTORY_DB_PATH = os.environ.get("HISTORY_DB_PATH", "/webui/monitor/history.db")

# All available statuses
class Status(Enum):
    INACTIVE = "inactive" # not running
    STARTING = "starting" # during start()
    IDLE     = "idle"     # running, but not serving
    SERVING  = "serving"  # running with a model loaded to GPU
    STOPPING = "stopping" # during stop()
    ERROR    = "error"    # error


class ModelCfg(TypedDict):
    """One model's entry in the router config's LLM section."""
    num_instance: int
    gpus: NotRequired[list[int] | int | str]


class RouterSettings(TypedDict, total=False):
    """Optional scheduler tunables from the config's ROUTER section."""
    HEALTH_CHECK_INTERVAL: float
    HEALTH_CHECK_TIMEOUT: float
    UNLOAD_POLL_INTERVAL: float
    UNLOAD_POLL_TIMEOUT: float
    LOAD_POLL_INTERVAL: float
    LOAD_POLL_TIMEOUT: float
    START_RETRIES: int
    GRACEFUL_KILL_TIMEOUT: float
    MAX_MODELS_PER_GPU: int
    EVICTION_POLICY: str
    QUEUE_FORCE_LOAD_TIMEOUT: float
    NUM_GPUS: int


# Keys carry hyphens, so this TypedDict uses the functional syntax.
RouterConfig = TypedDict("RouterConfig", {
    "LLM": dict[str, ModelCfg],
    "API-port": NotRequired[int],
    "LLM-base-port": int,
    "llama-server-executable": str,
    "ROUTER": NotRequired[RouterSettings],
})


class HistoryRow(TypedDict):
    """One row of request history as stored in the SQLite history table."""
    id: int
    model: str
    request_time: float
    response_time: float
    prompt_n: int
    predicted_n: int


class Envelope(TypedDict):
    """A request handed to addRequest and forwarded to a llama-server replica."""
    path: str
    method: str
    body: str | bytes
    headers: dict[str, str]
    # addRequest pops is_streaming, and the catch-all sets model only when known,
    # so both are optional.
    is_streaming: NotRequired[bool]
    model: NotRequired[str]


# Streaming futures resolve with this queue: response chunks, then an Exception
# on error, then a None sentinel. Non-streaming futures resolve with a Response.
StreamQueue = asyncio.Queue[bytes | Exception | None]
ForwardResult = httpx.Response | StreamQueue


class AsyncRWLock:
    """
    Asyncio readers-writer lock

    Multiple coroutines can hold a shared reader lock, while an exclusive writer
    lock waits for all readers to release. Generation requests are readers (they
    don't interfere); load/unload are writers (they wait for requests to drain).
    Readers intentionally starve writers to maximize LLM cache hits.
    """

    def __init__(self):
        self._readers = 0
        self._writer = False
        self._lock = asyncio.Lock()
        self._readers_ok = asyncio.Condition(self._lock) # readers condition var
        self._writer_ok = asyncio.Condition(self._lock) # writers condition var

    # Exclusive to readers

    async def acquireShared(self):
        """Acquires the lock, waiting while a writer is active."""
        async with self._lock:
            while self._writer:
                await self._readers_ok.wait()
            self._readers += 1

    async def releaseShared(self):
        """Releases a reader, notifying a waiting writer once none remain."""
        async with self._lock:
            self._readers -= 1
            if self._readers == 0:
                self._writer_ok.notify()

    # Exclusive to writers

    async def acquireExclusive(self):
        """Acquires the writer lock once no writer is active and readers finish."""
        async with self._lock:
            while self._writer or self._readers > 0:
                await self._writer_ok.wait()
            self._writer = True

    async def releaseExclusive(self):
        """Releases the writer lock and wakes waiting readers and writers."""
        async with self._lock:
            self._writer = False
            self._readers_ok.notify_all()
            self._writer_ok.notify()


async def _fetchModelStatuses(port: int) -> dict[str, str]:
    """
    Queries a single llama-server instance for the status of its models

    Args:
        port (int): The port of the llama-server instance to query

    Returns:
        A {model_id: status_value} map for every model the instance reports
    """
    async with httpx.AsyncClient() as client:
        resp = await client.get(f"http://0.0.0.0:{port}/models", timeout=5.0)
        resp.raise_for_status()
        data = resp.json()
        return {
            m.get("id"): m.get("status", {}).get("value", "unknown")
            for m in data.get("data", [])
        }


class LLMRouter:
    """
    One llama-server process per loaded model replica

    Loading a model spawns num_instance processes and loads the model into each;
    evicting SIGTERMs those processes, which frees VRAM unconditionally.
    Residency is tracked per GPU: a model pinned to gpus [0, 1] counts against
    GPU 0 and GPU 1, and at most MAX_MODELS_PER_GPU models stay resident per GPU.
    Crashed replicas are reaped and the model reloads on its next request.
    """

    def __init__(self, router_config_path: str, llama_presets_path: str):
        self.llama_presets_path = llama_presets_path

        with open(router_config_path, "r") as f:
            self.router_config: RouterConfig = cast(RouterConfig, json.load(f))
            '''
                self.router_config["LLM"] -> {model_id: {num_instance: N, gpus: [ids] | "all" | -1}}
                self.router_config["API-port"] -> int (port to expose the router)
                self.router_config["LLM-base-port"] -> int (first port for llama-server instances)
                self.router_config["llama-server-executable"] -> str (path to the llama-server binary)
                self.router_config["ROUTER"] -> reassign router values
            '''
            router_settings: RouterSettings = cast(RouterSettings, self.router_config.get("ROUTER", {}))
            self.HEALTH_CHECK_INTERVAL = router_settings.get("HEALTH_CHECK_INTERVAL", 1.0) # seconds between health polls
            self.HEALTH_CHECK_TIMEOUT = router_settings.get("HEALTH_CHECK_TIMEOUT", 30.0) # max seconds to wait for /health
            self.UNLOAD_POLL_INTERVAL = router_settings.get("UNLOAD_POLL_INTERVAL", 0.5) # seconds between polls waiting for unload to finish
            self.UNLOAD_POLL_TIMEOUT = router_settings.get("UNLOAD_POLL_TIMEOUT", 60.0) # max seconds to wait for all models to unload
            self.LOAD_POLL_INTERVAL = router_settings.get("LOAD_POLL_INTERVAL", 1.0) # seconds between polls waiting for model to load
            self.LOAD_POLL_TIMEOUT = router_settings.get("LOAD_POLL_TIMEOUT", 120.0) # max seconds to wait for a model to finish loading
            self.START_RETRIES = router_settings.get("START_RETRIES", 3) # attempts per instance on start()
            self.GRACEFUL_KILL_TIMEOUT = router_settings.get("GRACEFUL_KILL_TIMEOUT", 5.0) # seconds to wait after SIGTERM before SIGKILL
            self.MAX_MODELS_PER_GPU = int(router_settings.get("MAX_MODELS_PER_GPU", 1)) # resident-model cap PER GPU (not global)
            eviction_policy = str(router_settings.get("EVICTION_POLICY", "lru")).lower() # "lru" (last request) or "fifo" (load time)
            self.QUEUE_FORCE_LOAD_TIMEOUT = float(router_settings.get("QUEUE_FORCE_LOAD_TIMEOUT", 300.0)) # seconds a queued request may starve before its model is force-loaded

        if eviction_policy not in ("lru", "fifo"):
            print(f"[ROUTER] unknown EVICTION_POLICY {eviction_policy!r}, falling back to 'lru'", flush=True)
            eviction_policy = "lru"
        self.EVICTION_POLICY = eviction_policy

        # GPU topology: each model is pinned to a set of GPU ids ("gpus" in its
        # LLM entry; absent -> [0], -1/"all" -> every GPU). Residency accounting
        # and eviction are per GPU.
        self.num_gpus = self._detectNumGpus(router_settings.get("NUM_GPUS"))
        self.model_gpus: dict[str, list[int]] = {
            mid: self._resolveGpus(mid, mcfg.get("gpus"))
            for mid, mcfg in self.router_config["LLM"].items()
        }
        self.model_loaded_at: dict[str, float] = {} # model -> ts of last successful load
        self.model_last_used: dict[str, float] = {} # model -> ts of last dispatched request (or load)

        self.status: Status = Status.INACTIVE
        self.processes: dict[int, subprocess.Popen[bytes]] = {} # port -> Popen
        self.port_model: dict[int, str] = {} # port -> model_id hosted by that process (set once loaded)
        self.inflight: dict[int, int] = {} # port -> requests currently being forwarded
        self._last_load_error: str | None = None # reason the most recent load attempt failed (e.g. worker OOM)
        self.requests:  list[dict[str, Any]] = [] # [{request, future, is_streaming, request_time}, ...]
        self.request_lock = asyncio.Lock()
        self._load_lock = AsyncRWLock()
        self._has_requests = asyncio.Event()
        self._running = False
        self._scheduler_task: asyncio.Task[None] | None = None
        self._history_db_path = HISTORY_DB_PATH
        self._history_db_ready = False

    # History DB

    async def initHistoryDb(self):
        """
        Creates the history table and indexes, enabling WAL journaling

        Ensures the DB directory exists and marks the history store as ready
        """
        os.makedirs(os.path.dirname(self._history_db_path), exist_ok=True)
        async with aiosqlite.connect(self._history_db_path) as db:
            await db.execute("PRAGMA journal_mode=WAL")
            await db.execute("""
                CREATE TABLE IF NOT EXISTS history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model TEXT NOT NULL,
                    request_time REAL NOT NULL,
                    response_time REAL NOT NULL,
                    prompt_n INTEGER NOT NULL,
                    predicted_n INTEGER NOT NULL
                )
            """)
            await db.execute("CREATE INDEX IF NOT EXISTS idx_history_model ON history(model)")
            await db.execute("CREATE INDEX IF NOT EXISTS idx_history_time ON history(request_time)")
            await db.commit()
        self._history_db_ready = True

    async def _ensureHistoryDb(self):
        """Initializes the history DB lazily if it is not ready yet."""
        if not self._history_db_ready:
            await self.initHistoryDb()

    async def recordHistory(
        self,
        model: str,
        request_time: float,
        response_time: float,
        prompt_n: int,
        predicted_n: int,
    ):
        """
        Inserts one request-history row, swallowing any storage error

        Args:
            model (str)             : The model id that served the request
            request_time (float)    : Unix timestamp the request was enqueued
            response_time (float)   : Unix timestamp the response completed
            prompt_n (int)          : Number of prompt tokens processed
            predicted_n (int)       : Number of generated tokens
        """
        try:
            await self._ensureHistoryDb()
            async with aiosqlite.connect(self._history_db_path) as db:
                await db.execute(
                    "INSERT INTO history (model, request_time, response_time, prompt_n, predicted_n) VALUES (?, ?, ?, ?, ?)",
                    (model, request_time, response_time, prompt_n, predicted_n),
                )
                await db.commit()
        except Exception as e:
            print(f"[ROUTER] failed to record history: {e}", flush=True)

    async def getHistory(
        self,
        model: str | None = None,
        limit: int = 500,
        since: float | None = None,
        until: float | None = None,
    ) -> list[HistoryRow]:
        """
        Returns request-history rows, most recent first

        Args:
            model (str | None)  : The model to filter by, or None for all models
            limit (int)         : Max rows to return; 0 means no limit
            since (float | None): Only rows with request_time >= this unix ts, if set
            until (float | None): Only rows with request_time <= this unix ts, if set

        Returns:
            The list of history rows as dicts
        """
        await self._ensureHistoryDb()
        clauses: list[str] = []
        params: list[Any] = []
        if model:
            clauses.append("model = ?")
            params.append(model)
        if since is not None:
            clauses.append("request_time >= ?")
            params.append(since)
        if until is not None:
            clauses.append("request_time <= ?")
            params.append(until)
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        lim = " LIMIT ?" if limit else ""
        if limit:
            params.append(limit)
        async with aiosqlite.connect(self._history_db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                f"SELECT * FROM history{where} ORDER BY request_time DESC{lim}",
                params,
            )
            rows = await cursor.fetchall()
            # dict(row) is a dynamic sqlite row, so widen to object before the cast.
            return [cast(HistoryRow, cast(object, dict(row))) for row in rows]

    async def resetHistory(self):
        """Deletes all rows from the request-history table."""
        await self._ensureHistoryDb()
        async with aiosqlite.connect(self._history_db_path) as db:
            await db.execute("DELETE FROM history")
            await db.commit()

    # GPU topology / eviction planning

    def _detectNumGpus(self, configured: int | None) -> int:
        """
        Determines the GPU count from config, NVML, pins, then a fallback

        Explicit ROUTER.NUM_GPUS wins, then pynvml, then highest explicit pin + 1,
        then 1

        Args:
            configured (int | None): The explicit NUM_GPUS override, if any

        Returns:
            The number of GPUs the router should account for
        """
        if configured:
            return max(1, int(configured))
        try:
            import pynvml
            pynvml.nvmlInit()
            n = pynvml.nvmlDeviceGetCount()
            if n > 0:
                return n
        except Exception:
            pass
        max_pin = 0
        for mcfg in self.router_config["LLM"].values():
            raw = mcfg.get("gpus")
            vals = raw if isinstance(raw, list) else [raw]
            for v in vals:
                if isinstance(v, int) and v > max_pin:
                    max_pin = v
        return max_pin + 1

    def _resolveGpus(self, model_id: str, raw: list[int] | int | str | None) -> list[int]:
        """
        Normalizes a model's "gpus" field to a sorted list of valid GPU ids

        None maps to [0] (unpinned = GPU 0 only); -1 or "all" maps to every GPU.
        Out-of-range and invalid ids are logged and dropped.

        Args:
            model_id (str)                      : The model whose pins are resolved (for logging)
            raw (int | str | list | None)       : The raw "gpus" value from config

        Returns:
            The sorted list of valid GPU ids for the model
        """
        if raw is None:
            return [0]
        items: Iterable[int | str] = raw if isinstance(raw, list) else [raw]
        ids: set[int] = set()
        for v in items:
            if isinstance(v, str) and v.strip().lower() in ("all", "*"):
                return list(range(self.num_gpus))
            try:
                i = int(v)
            except (TypeError, ValueError):
                print(f"[ROUTER] {model_id}: invalid gpu id {v!r}, ignoring", flush=True)
                continue
            if i == -1:
                return list(range(self.num_gpus))
            if 0 <= i < self.num_gpus:
                ids.add(i)
            else:
                print(f"[ROUTER] {model_id}: gpu id {i} out of range (num_gpus={self.num_gpus}), ignoring", flush=True)
        if not ids:
            print(f"[ROUTER] {model_id}: no valid gpu ids, defaulting to [0]", flush=True)
            return [0]
        return sorted(ids)

    def _planEvictions(self, model_id: str, loaded: set[str]) -> set[str]:
        """
        Decides which resident models must be unloaded so model_id fits

        Per GPU that model_id is pinned to, if adding model_id would exceed
        MAX_MODELS_PER_GPU, evict the oldest residents of THAT GPU (by last
        request for "lru", by load time for "fifo"). Residents with no queued
        demand are evicted before residents that still have queued requests, and
        models on GPUs model_id doesn't touch are left alone.

        Args:
            model_id (str)      : The model being made resident
            loaded (set[str])   : The currently resident models

        Returns:
            The set of model ids that must be evicted
        """
        age = self.model_last_used if self.EVICTION_POLICY == "lru" else self.model_loaded_at
        queued = {r["request"].get("model") for r in self.requests}
        evict: set[str] = set()
        for gpu in self.model_gpus.get(model_id, [0]):
            residents = [
                m for m in loaded
                if m != model_id and m not in evict
                and gpu in self.model_gpus.get(m, [0])
            ]
            overflow = len(residents) - (self.MAX_MODELS_PER_GPU - 1)
            if overflow > 0:
                residents.sort(key=lambda m: (m in queued, age.get(m, 0.0)))
                evict.update(residents[:overflow])
        return evict

    def _forgetModel(self, model_id: str):
        """
        Drops load-time and last-used bookkeeping for a model

        Args:
            model_id (str): The model to forget
        """
        self.model_loaded_at.pop(model_id, None)
        self.model_last_used.pop(model_id, None)

    # Process pool (one llama-server process per model replica)

    def _reapDead(self):
        """
        Drops bookkeeping for replicas whose process has exited

        The model auto-reloads on its next request since it no longer counts as
        loaded
        """
        for port in list(self.processes.keys()):
            if self.processes[port].poll() is not None:
                mid = self.port_model.get(port, "<none>")
                print(f"[ROUTER] replica on port {port} (model {mid}) died, reaping", flush=True)
                del self.processes[port]
                self.port_model.pop(port, None)
                self.inflight.pop(port, None)

    def modelPorts(self, model_id: str) -> list[int]:
        """
        Returns the sorted live ports currently hosting model_id

        Args:
            model_id (str): The model to look up

        Returns:
            The sorted list of ports whose live process hosts the model
        """
        return sorted(p for p, m in self.port_model.items() if m == model_id and p in self.processes)

    def _loadedModels(self) -> set[str]:
        """Returns the set of models with at least one live replica process."""
        return {m for p, m in self.port_model.items() if p in self.processes}

    def _allocPorts(self, n: int) -> list[int]:
        """
        Picks the n lowest free ports at or above LLM-base-port

        Args:
            n (int): The number of ports to allocate

        Returns:
            The list of n free ports not currently in use
        """
        ports: list[int] = []
        used = set(self.processes.keys())
        p = self.router_config["LLM-base-port"]
        while len(ports) < n:
            if p not in used:
                ports.append(p)
                used.add(p)
            p += 1
        return ports

    def _replicaExited(self, port: int) -> int | None:
        """
        Returns the exit code if the replica process on port has terminated

        A worker that dies mid-startup or mid-load (typically CUDA OOM while
        building the KV cache / compute buffers) exits with a nonzero code. The
        health/load poll loops check this so they fail fast with a real error
        instead of polling a dead port until HEALTH_CHECK_TIMEOUT / LOAD_POLL_TIMEOUT
        (which silently wedged the router with the load lock held).

        Args:
            port (int): The port whose process to check

        Returns:
            The process exit code if it has terminated, otherwise None (running)
        """
        proc = self.processes.get(port)
        if proc is None:
            return None
        return proc.poll()

    async def _startInstance(self, port: int, model_id: str) -> int | None:
        """
        Spawns one llama-server on port, isolated to model_id's pinned GPUs

        The process is masked to its pinned GPUs via CUDA_VISIBLE_DEVICES because
        ggml reserves a context and buffers on every visible device, even ones
        the model isn't computing on. Masking keeps that overhead off co-resident
        models' GPUs, so `gpus` alone drives placement with no `device` preset
        key, and devices renumber from 0 inside the process. Waits for /health.

        Args:
            port (int)      : The port the instance should listen on
            model_id (str)  : The model whose GPU mask is applied

        Returns:
            The process pid if it became healthy, otherwise None
        """
        print(f"[ROUTER] starting instance for {model_id} at port {port}...")
        exe = self.router_config["llama-server-executable"]
        env = dict(os.environ)
        gpus = self.model_gpus.get(model_id, [0])
        mask = ",".join(str(g) for g in gpus)
        # CUDA + HIP/ROCm both honor their own *_VISIBLE_DEVICES; set both so the
        # mask is backend-agnostic (the unused one is simply ignored).
        env["CUDA_VISIBLE_DEVICES"] = mask
        env["HIP_VISIBLE_DEVICES"] = mask
        proc = subprocess.Popen(
            [exe, "--host", "0.0.0.0", "--port", str(port),
             "--models-preset", self.llama_presets_path, "--metrics"],
            stdout=None, # inherit router's fds -> journald
            stderr=None, # inherit router's fds -> journald
            env=env,
        )
        self.processes[port] = proc
        deadline = asyncio.get_event_loop().time() + self.HEALTH_CHECK_TIMEOUT
        async with httpx.AsyncClient() as client:
            while asyncio.get_event_loop().time() < deadline:
                # Fail fast if the server process died before it ever went healthy
                # (bad preset, missing model file, immediate CUDA error) instead of
                # polling a dead port for the full HEALTH_CHECK_TIMEOUT.
                code = self._replicaExited(port)
                if code is not None:
                    self._last_load_error = (f"{model_id} llama-server on port {port} exited "
                                             f"(code {code}) before becoming healthy")
                    print(f"[ROUTER] {self._last_load_error}; aborting start", flush=True)
                    del self.processes[port]
                    return None
                try:
                    resp = await client.get(f"http://0.0.0.0:{port}/health", timeout=2.0)
                    if resp.status_code == 200 and resp.json().get("status") == "ok":
                        return proc.pid
                except (httpx.ConnectError, httpx.TimeoutException):
                    pass
                await asyncio.sleep(self.HEALTH_CHECK_INTERVAL)
        self._last_load_error = f"{model_id} llama-server on port {port} never became healthy within {self.HEALTH_CHECK_TIMEOUT}s"
        print(f"[ROUTER] {self._last_load_error}; killing", flush=True)
        proc.kill()
        del self.processes[port]
        return None

    async def _loadInto(self, port: int, model_id: str) -> bool:
        """
        Loads model_id into the llama-server at port and waits until loaded

        Args:
            port (int)      : The port of the target llama-server
            model_id (str)  : The model to load into that instance

        Returns:
            True once the instance reports the model "loaded", else False
        """
        try:
            async with httpx.AsyncClient() as client:
                await client.post(f"http://0.0.0.0:{port}/models/load", json={"model": model_id}, timeout=120.0)
        except Exception as exc:
            print(f"[ROUTER] load request to port {port} failed: {exc}", flush=True)
            return False
        deadline = asyncio.get_event_loop().time() + self.LOAD_POLL_TIMEOUT
        while asyncio.get_event_loop().time() < deadline:
            # The worker loads asynchronously and exits on failure (e.g. CUDA OOM
            # building the KV/compute buffers). Detect that exit instead of
            # swallowing the ensuing connection errors and polling a dead port for
            # the full LOAD_POLL_TIMEOUT with the load lock held — the silent-OOM hang.
            code = self._replicaExited(port)
            if code is not None:
                self._last_load_error = (f"{model_id} worker on port {port} exited (code {code}) "
                                         f"during load — likely CUDA OOM / insufficient VRAM "
                                         f"(check the llama-server logs)")
                print(f"[ROUTER] {self._last_load_error}", flush=True)
                return False
            try:
                statuses = await _fetchModelStatuses(port)
                if statuses.get(model_id) == "loaded":
                    return True
            except Exception:
                pass
            await asyncio.sleep(self.LOAD_POLL_INTERVAL)
        self._last_load_error = f"timed out after {self.LOAD_POLL_TIMEOUT}s waiting for {model_id} to load on port {port}"
        print(f"[ROUTER] {self._last_load_error}", flush=True)
        return False

    async def _spawnReplica(self, port: int, model_id: str) -> bool:
        """
        Spawns and loads one replica hosting exactly model_id, with retries

        Args:
            port (int)      : The port the replica should listen on
            model_id (str)  : The model the replica hosts

        Returns:
            True if the replica came up and loaded the model, else False
        """
        for attempt in range(self.START_RETRIES):
            pid = await self._startInstance(port, model_id)
            if pid is None:
                continue
            if await self._loadInto(port, model_id):
                self.port_model[port] = model_id
                print(f"[ROUTER] replica for {model_id} up on port {port} (pid {pid})", flush=True)
                return True
            await self._killInstance(port)
        return False

    async def _killInstance(self, port: int) -> bool:
        """
        Terminates the instance at port, escalating to SIGKILL if needed

        Sends SIGTERM, waits GRACEFUL_KILL_TIMEOUT, then SIGKILLs if still alive,
        and clears the port's bookkeeping

        Args:
            port (int): The port whose instance should be killed

        Returns:
            True on success, False if no process or an error occurred
        """
        print(f"[ROUTER] killing instance at port {port}...")
        proc = self.processes.get(port)
        if proc is None:
            return False
        try:
            if proc.poll() is None: # still running -> SIGTERM, escalate to SIGKILL
                proc.terminate()
                await asyncio.sleep(self.GRACEFUL_KILL_TIMEOUT)
                if proc.poll() is None:
                    proc.kill() # SIGKILL
            proc.wait() # reap (already-exited workers, e.g. OOM, land here too)
            del self.processes[port]
            self.port_model.pop(port, None)
            self.inflight.pop(port, None)
            return True
        except Exception:
            return False

    async def start(self):
        """
        Starts the scheduler; replicas spawn on demand at first request

        No llama-server is spawned up front. Does nothing unless the router is
        inactive or errored.
        """
        if self.status not in (Status.INACTIVE, Status.ERROR):
            return
        self.status = Status.STARTING
        self._running = True
        self._scheduler_task = asyncio.create_task(self._scheduler())
        self.status = Status.IDLE
        print(f"[START SUCCESS] Router started, replicas spawn on demand")

    async def stop(self):
        """
        Hard-resets the router by killing all instances and clearing state

        Cancels the scheduler, rejects pending futures, kills every replica, and
        reinitializes the locks and events
        """
        self.status = Status.STOPPING
        # Shut down the scheduler
        self._running = False
        self._has_requests.set()
        if self._scheduler_task and not self._scheduler_task.done():
            self._scheduler_task.cancel()
            try:
                await self._scheduler_task
            except asyncio.CancelledError:
                pass
        self._scheduler_task = None
        # Reject all pending futures
        for entry in self.requests:
            fut = entry.get("future")
            if fut and not fut.done():
                fut.set_exception(RuntimeError("Router is stopping"))
        results = await asyncio.gather(
            *[self._killInstance(port) for port in list(self.processes.keys())],
            return_exceptions=True,
        )
        self.processes.clear()
        self.port_model.clear()
        self.inflight.clear()
        self.requests.clear()
        self.model_loaded_at.clear()
        self.model_last_used.clear()
        self._load_lock = AsyncRWLock()
        self.request_lock = asyncio.Lock()
        self._has_requests = asyncio.Event()
        self.status = Status.INACTIVE if all(results) else Status.ERROR
        print(f"[STOP SUCCESS] Router stopped successfully")

    async def restart(self):
        """Restarts the router by stopping and starting again."""
        await self.stop()
        await self.start()

    # Load/Unload

    async def getLoadedModels(self) -> set[str]:
        """
        Returns the set of models with at least one live replica process

        Reaps dead replicas before reporting
        """
        self._reapDead()
        return self._loadedModels()

    def _sortedPorts(self) -> list[int]:
        """Returns all active ports in sorted order."""
        return sorted(self.processes.keys())

    async def loadModel(self, model_id: str):
        """
        Makes model_id resident, evicting per-GPU overflow and spawning replicas

        On each GPU model_id is pinned to, kills the oldest residents until the
        new model fits under MAX_MODELS_PER_GPU (models on other GPUs untouched),
        then spawns num_instance replicas, topping up any that are missing.

        Args:
            model_id (str): The model to make resident

        Returns:
            True once the model is resident

        Raises:
            ValueError: If model_id is not in the configured LLM list
            RuntimeError: If no replica could be started for the model
        """
        if model_id not in self.router_config["LLM"]:
            raise ValueError(f"[LOAD/UNLOAD ERROR] Model [{model_id}] not present in list {list(self.router_config['LLM'].keys())}")

        # Acquires load lock
        await self._load_lock.acquireExclusive()
        print(f"[ROUTER] initiate loading of model: {model_id}...")
        try:
            self._reapDead()
            loaded = self._loadedModels()
            target = self.router_config["LLM"].get(model_id, {"num_instance": 1})["num_instance"]
            have = self.modelPorts(model_id)
            if model_id in loaded and len(have) >= target:
                self.model_last_used[model_id] = time.time()
                print(f"[ROUTER] models: {model_id} already present in memory")
                return True

            # 1) evict per-GPU overflow (skipped when topping up lost replicas —
            #    the model already counts against its GPUs)
            if model_id not in loaded:
                evict = self._planEvictions(model_id, loaded)
                print(f"[ROUTER] models: {loaded or '{}'} present in memory, evicting {evict or 'nothing'} "
                      f"(gpus={self.model_gpus.get(model_id, [0])}, cap={self.MAX_MODELS_PER_GPU}/gpu, policy={self.EVICTION_POLICY})")
                if evict:
                    ports_to_kill = [p for mid in evict for p in self.modelPorts(mid)]
                    await asyncio.gather(*[self._killInstance(p) for p in ports_to_kill])
                    for mid in evict:
                        self._forgetModel(mid)

            # 2) spawn missing replicas, one process per replica
            needed = target - len(have)
            if needed > 0:
                ports = self._allocPorts(needed)
                self._last_load_error = None
                results = await asyncio.gather(
                    *[self._spawnReplica(port, model_id) for port in ports],
                    return_exceptions=True,
                )
                up = sum(1 for r in results if r is True)
                if up == 0:
                    reason = self._last_load_error or "no replica became healthy"
                    raise RuntimeError(f"[LOAD/UNLOAD ERROR] Failed to start any replica for {model_id}: {reason}")
                if up < needed:
                    print(f"[ROUTER] only {up}/{needed} new replicas for {model_id} came up", flush=True)

            now = time.time()
            self.model_loaded_at[model_id] = now
            self.model_last_used[model_id] = now
            print(f"[LOAD/UNLOAD SUCCESS] Successfully loaded {model_id}")
            print(f"[LOAD CONFIRMATION] Loaded models: {self._loadedModels()}")
            return True
        finally:
            await self._load_lock.releaseExclusive()
            self._has_requests.set()

    async def unloadModel(self, model_id: str):
        """
        Unloads a model by killing all of its replica processes

        Args:
            model_id (str): The model to unload
        """
        await self._load_lock.acquireExclusive()
        print(f"[ROUTER] initiate unloading of model: {model_id}...")
        try:
            ports = self.modelPorts(model_id)
            if ports:
                await asyncio.gather(*[self._killInstance(p) for p in ports])
            self._forgetModel(model_id)
            print(f"[UNLOAD SUCCESS] Successfully unloaded {model_id}")
            print(f"[UNLOAD CONFIRMATION] Currently loaded models: {self._loadedModels()}")
        finally:
            await self._load_lock.releaseExclusive()
            self._has_requests.set()

    # Request handling

    async def addRequest(self, request: Envelope) -> "asyncio.Future[ForwardResult]":
        """
        Enqueues a request and returns a Future that resolves when it is served

        The serving port is chosen at dispatch time, after the model is resident,
        since ports are per-model. Non-streaming futures resolve with an
        httpx.Response; streaming futures resolve with an asyncio.Queue of chunks
        terminated by a None sentinel.

        Args:
            request (dict): The request envelope, with "is_streaming" popped off

        Returns:
            A Future resolving to the response or the streaming queue
        """
        future = asyncio.get_event_loop().create_future()
        is_streaming = request.pop("is_streaming", False)
        if not self._running:
            await self.start()
        async with self.request_lock:
            self.requests.append({
                "request": request,
                "future": future,
                "is_streaming": is_streaming,
                "request_time": time.time(),
            })
            self._has_requests.set()
            self.status = Status.SERVING
        return future

    async def _scheduler(self):
        """
        Continuously picks queued requests and dispatches forwarding tasks

        Maximizes cache hits by preferring requests whose model is already
        loaded. When nothing is servable it loads the head request's model, and
        a starvation guard force-loads the head model once it waits past
        QUEUE_FORCE_LOAD_TIMEOUT so cache-hit requests can't starve it forever.
        """
        while self._running:
            await self._has_requests.wait()
            if not self._running:
                break

            async with self.request_lock:
                # no requests
                if not self.requests:
                    self._has_requests.clear()
                    continue

                self.status = Status.SERVING
                self._reapDead()
                loaded = self._loadedModels()
                # Starvation guard: if the oldest queued request's model is not
                # loaded and it has waited past QUEUE_FORCE_LOAD_TIMEOUT, force
                # its load instead of serving newer cache-hit requests forever.
                chosen_idx = None
                head = self.requests[0]
                head_model = head["request"].get("model")
                starved = (
                    head_model is not None
                    and head_model not in loaded
                    and time.time() - head["request_time"] >= self.QUEUE_FORCE_LOAD_TIMEOUT
                )
                if starved:
                    print(f"[ROUTER] head-of-queue request for {head_model} waited "
                          f">{self.QUEUE_FORCE_LOAD_TIMEOUT}s, force-loading", flush=True)
                else:
                    # pick a request to serve: first request that matches the model or has no model field
                    for i, entry in enumerate(self.requests):
                        req_model = entry["request"].get("model")
                        if req_model is None or req_model in loaded:
                            chosen_idx = i
                            break
                # there's a servable request
                if chosen_idx is not None:
                    entry = self.requests.pop(chosen_idx)
                # otherwise, load the model of the first request
                else:
                    entry = self.requests.pop(0)
                    model_to_load = entry["request"].get("model")
                    try:
                        await self.loadModel(model_to_load)
                    except Exception as exc:
                        print(f"[ROUTER] failed to load {model_to_load}: {exc}", flush=True)
                        if not entry["future"].done():
                            entry["future"].set_exception(exc)
                        continue
                served_model = entry["request"].get("model")
                if served_model:
                    self.model_last_used[served_model] = time.time()
                # pick the least-busy replica of the served model
                # (model-less requests go to any live instance)
                if served_model is not None:
                    ports = self.modelPorts(served_model)
                else:
                    ports = self._sortedPorts()
                if not ports:
                    err = RuntimeError(
                        f"no llama-server replica available for model {served_model!r}"
                        if served_model is not None else
                        "no llama-server instance running; specify a model to load one"
                    )
                    if not entry["future"].done():
                        entry["future"].set_exception(err)
                    continue
                port = min(ports, key=lambda p: (self.inflight.get(p, 0), p))
                entry["port"] = port
                self.inflight[port] = self.inflight.get(port, 0) + 1

            # Dispatch forwarding as a concurrent task
            if entry["is_streaming"]:
                queue = asyncio.Queue()
                entry["future"].set_result(queue)
                asyncio.create_task(self._doForwardStreaming(entry, queue))
            else:
                asyncio.create_task(self._doForward(entry))

    def _releasePort(self, port: int):
        """
        Decrements the in-flight counter for a port, floored at zero

        Args:
            port (int): The port whose in-flight count should be released
        """
        if port in self.inflight:
            self.inflight[port] = max(0, self.inflight[port] - 1)

    async def _doForward(self, entry: dict[str, Any]):
        """
        Forwards a non-streaming request to its port and resolves its future

        Records token history from the response usage/timings on success

        Args:
            entry (dict): The dispatched request entry with its assigned "port"
        """
        await self._load_lock.acquireShared()
        try:
            port = entry["port"]
            req = entry["request"]
            path = req.get("path", "/v1/chat/completions")
            method = req.get("method", "POST").upper()
            body = req.get("body")
            headers = req.get("headers", {})
            async with httpx.AsyncClient() as client:
                resp = await client.request(
                    method,
                    f"http://0.0.0.0:{port}{path}",
                    content=body if isinstance(body, (bytes, str)) else json.dumps(body) if body else None,
                    headers=headers,
                    timeout=300.0,
                )
            entry["future"].set_result(resp)
            # Record history from usage/timings
            try:
                data = resp.json()
                timings = data.get("timings", {})
                usage = data.get("usage", {})
                prompt_n = timings.get("prompt_n", usage.get("prompt_tokens", 0))
                predicted_n = timings.get("predicted_n", usage.get("completion_tokens", 0))
                model = entry["request"].get("model", "unknown")
                if prompt_n or predicted_n:
                    await self.recordHistory(model, entry["request_time"], time.time(), int(prompt_n), int(predicted_n))
            except Exception:
                pass
        except Exception as e:
            if not entry["future"].done():
                entry["future"].set_exception(e)
        finally:
            self.status = Status.IDLE
            self._releasePort(entry["port"])
            await self._load_lock.releaseShared()

    async def _doForwardStreaming(self, entry: dict[str, Any], queue: StreamQueue):
        """
        Forwards a streaming request to its port, pushing chunks to the queue

        Puts a None sentinel when done and records token history from the last
        SSE chunk that carried timings

        Args:
            entry (dict)            : The dispatched request entry with its "port"
            queue (asyncio.Queue)   : The queue chunks are pushed onto
        """
        await self._load_lock.acquireShared()
        last_data = None
        try:
            port = entry["port"]
            req = entry["request"]
            path = req.get("path", "/v1/chat/completions")
            method = req.get("method", "POST").upper()
            body = req.get("body")
            headers = req.get("headers", {})
            async with httpx.AsyncClient() as client:
                async with client.stream(
                    method,
                    f"http://0.0.0.0:{port}{path}",
                    content=body if isinstance(body, (bytes, str)) else json.dumps(body) if body else None,
                    headers=headers,
                    timeout=300.0,
                ) as resp:
                    async for chunk in resp.aiter_bytes():
                        await queue.put(chunk)
                        # Parse SSE lines for usage/timings data
                        for line in chunk.decode("utf-8", errors="ignore").split("\n"):
                            if line.startswith("data: ") and line.strip() != "data: [DONE]":
                                try:
                                    last_data = json.loads(line[6:])
                                except (json.JSONDecodeError, ValueError):
                                    pass
        except Exception as e:
            await queue.put(e)
        finally:
            queue.put_nowait(None)
            self._releasePort(entry["port"])
            await self._load_lock.releaseShared()
            # Record history from the last SSE chunk that contained timings
            if last_data:
                self.status = Status.IDLE
                try:
                    timings = last_data.get("timings", {})
                    usage = last_data.get("usage", {})
                    prompt_n = timings.get("prompt_n", usage.get("prompt_tokens", 0))
                    predicted_n = timings.get("predicted_n", usage.get("completion_tokens", 0))
                    model = entry["request"].get("model", "unknown")
                    if prompt_n or predicted_n:
                        await self.recordHistory(model, entry["request_time"], time.time(), int(prompt_n), int(predicted_n))
                except Exception:
                    pass
