import asyncio
import configparser
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

# All available statuses. INACTIVE/STARTING/STOPPING/ERROR are router-wide
# lifecycle states; IDLE/SERVING/SWAPPING are the running states and are tracked
# PER GPU (a GPU is SERVING while a generation on it is in flight, SWAPPING while
# a model is loading/unloading on it, else IDLE). self.status holds the lifecycle
# state; per-GPU running states are computed on demand (see _gpu_status).
class Status(Enum):
    INACTIVE = "inactive" # not running
    STARTING = "starting" # during start()
    IDLE     = "idle"     # running, nothing in flight
    SERVING  = "serving"  # actively forwarding a generation
    SWAPPING = "swapping" # loading or unloading a model (the two are merged)
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
    """A request handed to add_request and forwarded to a llama-server replica."""
    path: str
    method: str
    body: str | bytes
    headers: dict[str, str]
    # add_request pops is_streaming, and the catch-all sets model only when known,
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

    async def acquire_shared(self):
        """Acquires the lock, waiting while a writer is active."""
        async with self._lock:
            while self._writer:
                await self._readers_ok.wait()
            self._readers += 1

    async def release_shared(self):
        """Releases a reader, notifying a waiting writer once none remain."""
        async with self._lock:
            self._readers -= 1
            if self._readers == 0:
                self._writer_ok.notify()

    # Exclusive to writers

    async def acquire_exclusive(self):
        """Acquires the writer lock once no writer is active and readers finish."""
        async with self._lock:
            while self._writer or self._readers > 0:
                await self._writer_ok.wait()
            self._writer = True

    async def release_exclusive(self):
        """Releases the writer lock and wakes waiting readers and writers."""
        async with self._lock:
            self._writer = False
            self._readers_ok.notify_all()
            self._writer_ok.notify()


async def _fetch_model_reports(port: int) -> dict[str, dict[str, Any]]:
    """
    Queries a single llama-server instance for the full status of its models

    Returns each model's whole status object, not just its value: llama-server's
    multi-model supervisor keeps running (and /health stays 200) even when a
    model's worker dies loading, and only signals that failure inside the status
    object as "failed": true / "exit_code": N (e.g. CUDA OOM building the KV
    cache). Callers need those fields to fail fast instead of polling to timeout.

    Args:
        port (int): The port of the llama-server instance to query

    Returns:
        A {model_id: status_object} map for every model the instance reports
    """
    async with httpx.AsyncClient() as client:
        resp = await client.get(f"http://0.0.0.0:{port}/models", timeout=5.0)
        resp.raise_for_status()
        data = resp.json()
        return {
            m.get("id"): (m.get("status") or {})
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
        self.num_gpus = self._detect_num_gpus(router_settings.get("NUM_GPUS"))
        self.model_gpus: dict[str, list[int]] = {
            mid: self._resolve_gpus(mid, mcfg.get("gpus"))
            for mid, mcfg in self.router_config["LLM"].items()
        }
        self.model_loaded_at: dict[str, float] = {} # model -> ts of last successful load
        self.model_last_used: dict[str, float] = {} # model -> ts of last dispatched request (or load)
        self._context_windows: dict[str, int] | None = None # cached per-model effective context (see model_context_windows)

        self.status: Status = Status.INACTIVE
        self.processes: dict[int, subprocess.Popen[bytes]] = {} # port -> Popen
        # Bare metadata supervisor: a --no-models-autoload llama-server on a
        # reserved port with no GPU mask, alive ONLY while zero replicas are
        # loaded, so /v1/models always has a real llama.cpp process to proxy
        # (see models_report / _reconcile_idle_supervisor). Kept out of
        # self.processes so it never counts as a loaded replica.
        self._idle_port: int = self.router_config["LLM-base-port"] - 1
        self._idle_proc: subprocess.Popen[bytes] | None = None
        self._idle_lock = asyncio.Lock()
        self.port_model: dict[int, str] = {} # port -> model_id hosted by that process (set once loaded)
        self.inflight: dict[int, int] = {} # port -> requests currently being forwarded
        self._last_load_error: str | None = None # reason the most recent load attempt failed (e.g. worker OOM)
        self._last_load_fatal: bool = False # True if that failure was deterministic (worker exited/failed) -> don't retry
        self.requests:  list[dict[str, Any]] = [] # [{request, future, is_streaming, request_time}, ...]
        self.request_lock = asyncio.Lock()
        # One RWLock per GPU (generation = shared reader, load/unload = exclusive
        # writer). GPUs are independent, so a swap on one GPU never drains work on
        # another. self._swapping_gpus marks GPUs mid load/unload for status.
        self._gpu_locks: dict[int, AsyncRWLock] = {g: AsyncRWLock() for g in range(self.num_gpus)}
        self._swapping_gpus: set[int] = set()
        self._has_requests = asyncio.Event()
        self._running = False
        self._scheduler_task: asyncio.Task[None] | None = None
        self._history_db_path = HISTORY_DB_PATH
        self._history_db_ready = False

    # History DB

    async def init_history_db(self):
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

    async def _ensure_history_db(self):
        """Initializes the history DB lazily if it is not ready yet."""
        if not self._history_db_ready:
            await self.init_history_db()

    async def record_history(
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
            await self._ensure_history_db()
            async with aiosqlite.connect(self._history_db_path) as db:
                await db.execute(
                    "INSERT INTO history (model, request_time, response_time, prompt_n, predicted_n) VALUES (?, ?, ?, ?, ?)",
                    (model, request_time, response_time, prompt_n, predicted_n),
                )
                await db.commit()
        except Exception as e:
            print(f"[ROUTER] failed to record history: {e}", flush=True)

    async def get_history(
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
        await self._ensure_history_db()
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

    async def reset_history(self):
        """Deletes all rows from the request-history table."""
        await self._ensure_history_db()
        async with aiosqlite.connect(self._history_db_path) as db:
            await db.execute("DELETE FROM history")
            await db.commit()

    # GPU topology / eviction planning

    def _detect_num_gpus(self, configured: int | None) -> int:
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

    def _resolve_gpus(self, model_id: str, raw: list[int] | int | str | None) -> list[int]:
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

    def model_context_windows(self) -> dict[str, int]:
        """
        Returns the effective per-request context window for each configured model

        Reads ctx-size and the parallel-slot count from the presets INI (each
        accepting llama.cpp's long or short spelling: "c"/"ctx-size" and
        "np"/"parallel"), falling back to the "[*]" global section, and returns
        floor(ctx / parallel) per model, since llama.cpp splits the total KV
        context "-c" evenly across the "--parallel" slots, so a single request
        sees only ctx / parallel tokens (parallel defaults to 1 when unset).
        Models whose ctx-size cannot be resolved are omitted. The presets file is
        parsed once and the result cached.

        Returns:
            A dict mapping model id to its effective per-slot context in tokens
        """
        if self._context_windows is not None:
            return self._context_windows

        windows: dict[str, int] = {}
        parser = configparser.ConfigParser(inline_comment_prefixes=("#", ";"), strict=False)
        try:
            read = parser.read(self.llama_presets_path)
        except configparser.Error as exc:
            print(f"[ROUTER] could not parse presets {self.llama_presets_path!r}: {exc}", flush=True)
            read = []

        if read:
            def _preset_value(model_id: str, keys: tuple[str, ...]) -> str | None:
                # per-model overrides the [*] global; either key spelling wins.
                for section in (model_id, "*"):
                    for key in keys:
                        if parser.has_option(section, key):
                            return parser.get(section, key)
                return None

            for model_id in self.router_config["LLM"]:
                raw_c = _preset_value(model_id, ("c", "ctx-size"))
                if raw_c is None:
                    continue
                raw_parallel = _preset_value(model_id, ("np", "parallel"))
                try:
                    ctx = int(raw_c)
                    parallel = int(raw_parallel) if raw_parallel is not None else 1
                except ValueError:
                    continue
                windows[model_id] = ctx // max(parallel, 1)

        self._context_windows = windows
        return windows

    def _plan_evictions(self, model_id: str, loaded: set[str]) -> set[str]:
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

    def _forget_model(self, model_id: str):
        """
        Drops load-time and last-used bookkeeping for a model

        Args:
            model_id (str): The model to forget
        """
        self.model_loaded_at.pop(model_id, None)
        self.model_last_used.pop(model_id, None)

    # Process pool (one llama-server process per model replica)

    def _reap_dead(self):
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

    def model_ports(self, model_id: str) -> list[int]:
        """
        Returns the sorted live ports currently hosting model_id

        Args:
            model_id (str): The model to look up

        Returns:
            The sorted list of ports whose live process hosts the model
        """
        return sorted(p for p, m in self.port_model.items() if m == model_id and p in self.processes)

    def _loaded_models(self) -> set[str]:
        """Returns the set of models with at least one live replica process."""
        return {m for p, m in self.port_model.items() if p in self.processes}

    def _alloc_ports(self, n: int) -> list[int]:
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

    def _replica_exited(self, port: int) -> int | None:
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

    # Per-GPU concurrency + status

    def _gpu_lock(self, gpu: int) -> AsyncRWLock:
        """Returns the RWLock for a GPU, creating one on demand."""
        lock = self._gpu_locks.get(gpu)
        if lock is None:
            lock = self._gpu_locks[gpu] = AsyncRWLock()
        return lock

    async def _acquire_shared(self, gpus: Iterable[int]):
        """Acquires the shared (reader) lock of every GPU, in ascending order."""
        for g in sorted(set(gpus)):
            await self._gpu_lock(g).acquire_shared()

    async def _release_shared(self, gpus: Iterable[int]):
        """Releases the shared lock of every GPU, in descending order."""
        for g in sorted(set(gpus), reverse=True):
            await self._gpu_lock(g).release_shared()

    async def _acquire_exclusive(self, gpus: Iterable[int]):
        """
        Acquires the exclusive (writer) lock of every GPU, in ascending order

        Ascending order is a global lock ordering, so two overlapping multi-GPU
        acquisitions can never deadlock against each other.
        """
        for g in sorted(set(gpus)):
            await self._gpu_lock(g).acquire_exclusive()

    async def _release_exclusive(self, gpus: Iterable[int]):
        """Releases the exclusive lock of every GPU, in descending order."""
        for g in sorted(set(gpus), reverse=True):
            await self._gpu_lock(g).release_exclusive()

    def _request_gpus(self, model_id: str | None) -> list[int]:
        """
        The GPUs a request occupies while being served

        A model's request occupies that model's pinned GPUs; a model-less request
        (routed to whatever instance is up) is treated as touching every GPU.
        """
        if model_id is None:
            return list(range(self.num_gpus))
        return self.model_gpus.get(model_id, [0])

    def _gpu_status(self, gpu: int) -> Status:
        """
        Computed running status of a single GPU: swapping > serving > idle

        SWAPPING while a load/unload touching it is in progress, else SERVING if
        any live replica pinned to it has an in-flight request, else IDLE.
        """
        if gpu in self._swapping_gpus:
            return Status.SWAPPING
        for port, count in self.inflight.items():
            if count > 0 and port in self.processes:
                model = self.port_model.get(port)
                if model is not None and gpu in self.model_gpus.get(model, [0]):
                    return Status.SERVING
        return Status.IDLE

    def _gpus_busy(self, gpus: Iterable[int]) -> bool:
        """True if any of these GPUs is currently serving or swapping."""
        return any(self._gpu_status(g) is not Status.IDLE for g in gpus)

    def gpu_statuses(self) -> dict[int, str]:
        """
        Per-GPU status map for reporting

        While the router is in a lifecycle state (inactive/starting/stopping/
        error) that state applies to every GPU; otherwise each GPU reports its
        own computed running status.
        """
        if self.status in (Status.INACTIVE, Status.STARTING, Status.STOPPING, Status.ERROR):
            return {g: self.status.value for g in range(self.num_gpus)}
        return {g: self._gpu_status(g).value for g in range(self.num_gpus)}

    def overall_status(self) -> Status:
        """
        Aggregate status for global consumers (e.g. the header badge)

        Lifecycle states pass through; while running, the busiest GPU wins
        (swapping > serving > idle).
        """
        if self.status in (Status.INACTIVE, Status.STARTING, Status.STOPPING, Status.ERROR):
            return self.status
        per = [self._gpu_status(g) for g in range(self.num_gpus)]
        if any(s is Status.SWAPPING for s in per):
            return Status.SWAPPING
        if any(s is Status.SERVING for s in per):
            return Status.SERVING
        return Status.IDLE

    def _load_affected_gpus(self, model_id: str, loaded: set[str]) -> list[int]:
        """
        The GPUs a load of model_id may mutate (and must therefore lock)

        Its own pinned GPUs, plus the full GPU span of every resident that shares
        any of those pins: evicting such a resident frees its other GPUs too, so
        their in-flight work must drain under the same exclusive lock.

        Args:
            model_id (str)      : The model being loaded
            loaded (set[str])   : The currently resident models

        Returns:
            The sorted list of GPU ids the load must hold exclusively
        """
        target = set(self.model_gpus.get(model_id, [0]))
        affected = set(target)
        for m in loaded:
            if m != model_id and target & set(self.model_gpus.get(m, [0])):
                affected.update(self.model_gpus.get(m, [0]))
        return sorted(affected)

    async def _start_instance(self, port: int, model_id: str) -> int | None:
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
                code = self._replica_exited(port)
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

    async def _load_into(self, port: int, model_id: str) -> bool:
        """
        Loads model_id into the llama-server at port and waits until loaded

        Args:
            port (int)      : The port of the target llama-server
            model_id (str)  : The model to load into that instance

        Returns:
            True once the instance reports the model "loaded", else False
        """
        self._last_load_fatal = False
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(f"http://0.0.0.0:{port}/models/load", json={"model": model_id}, timeout=self.LOAD_POLL_TIMEOUT)
        except Exception as exc:
            print(f"[ROUTER] load request to port {port} failed: {exc}", flush=True)
            return False
        # A 4xx/5xx here is a deterministic rejection (bad preset, or the worker
        # OOM'd and the supervisor reported it synchronously) — don't retry.
        if resp.status_code >= 400:
            self._last_load_error = (f"{model_id} load rejected by worker on port {port} "
                                     f"(HTTP {resp.status_code}): {resp.text[:200]}")
            self._last_load_fatal = True
            print(f"[ROUTER] {self._last_load_error}", flush=True)
            return False
        deadline = asyncio.get_event_loop().time() + self.LOAD_POLL_TIMEOUT
        while asyncio.get_event_loop().time() < deadline:
            # The worker loads asynchronously and fails on OOM (e.g. building the
            # KV/compute buffers for a huge ctx-size). Two failure shapes, both
            # fatal — fail fast instead of polling for the full LOAD_POLL_TIMEOUT
            # with the GPU exclusive lock held (the silent-OOM swapping hang):
            #   1) the whole llama-server process exits, or
            #   2) only the model worker dies; the supervisor stays healthy but
            #      marks the model "failed": true / "exit_code": N in /models.
            code = self._replica_exited(port)
            if code is not None:
                self._last_load_error = (f"{model_id} worker on port {port} exited (code {code}) "
                                         f"during load — likely CUDA OOM / insufficient VRAM "
                                         f"(check the llama-server logs)")
                self._last_load_fatal = True
                print(f"[ROUTER] {self._last_load_error}", flush=True)
                return False
            try:
                status = (await _fetch_model_reports(port)).get(model_id, {})
                if status.get("value") == "loaded":
                    return True
                if status.get("failed") or (status.get("exit_code") not in (None, 0)):
                    self._last_load_error = (f"{model_id} worker on port {port} failed to load "
                                             f"(exit_code {status.get('exit_code')}) — likely CUDA OOM / "
                                             f"insufficient VRAM for its ctx-size (check the llama-server logs)")
                    self._last_load_fatal = True
                    print(f"[ROUTER] {self._last_load_error}", flush=True)
                    return False
            except Exception:
                pass
            await asyncio.sleep(self.LOAD_POLL_INTERVAL)
        self._last_load_error = f"timed out after {self.LOAD_POLL_TIMEOUT}s waiting for {model_id} to load on port {port}"
        print(f"[ROUTER] {self._last_load_error}", flush=True)
        return False

    async def _spawn_replica(self, port: int, model_id: str) -> bool:
        """
        Spawns and loads one replica hosting exactly model_id, with retries

        Args:
            port (int)      : The port the replica should listen on
            model_id (str)  : The model the replica hosts

        Returns:
            True if the replica came up and loaded the model, else False
        """
        for attempt in range(self.START_RETRIES):
            pid = await self._start_instance(port, model_id)
            if pid is None:
                continue
            if await self._load_into(port, model_id):
                self.port_model[port] = model_id
                print(f"[ROUTER] replica for {model_id} up on port {port} (pid {pid})", flush=True)
                return True
            # A deterministic load failure (worker exited / reported "failed" /
            # load rejected — flagged by _load_into, or the whole process died)
            # will just hit the same wall on retry, so bail now. Only retry a
            # transient stall (process still alive and no hard failure flagged).
            fatal = self._last_load_fatal or self._replica_exited(port) is not None
            await self._kill_instance(port)
            if fatal:
                break
        return False

    async def _kill_instance(self, port: int) -> bool:
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
        # Nothing is loaded yet, so bring up the idle metadata supervisor so
        # /v1/models can proxy a real llama.cpp process from the first request on.
        await self._reconcile_idle_supervisor()
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
        await self._kill_idle_supervisor()
        results = await asyncio.gather(
            *[self._kill_instance(port) for port in list(self.processes.keys())],
            return_exceptions=True,
        )
        self.processes.clear()
        self.port_model.clear()
        self.inflight.clear()
        self.requests.clear()
        self.model_loaded_at.clear()
        self.model_last_used.clear()
        self._gpu_locks = {g: AsyncRWLock() for g in range(self.num_gpus)}
        self._swapping_gpus.clear()
        self.request_lock = asyncio.Lock()
        self._has_requests = asyncio.Event()
        self.status = Status.INACTIVE if all(results) else Status.ERROR
        print(f"[STOP SUCCESS] Router stopped successfully")

    async def restart(self):
        """Restarts the router by stopping and starting again."""
        await self.stop()
        await self.start()

    # Load/Unload

    async def get_loaded_models(self) -> set[str]:
        """
        Returns the set of models with at least one live replica process

        Reaps dead replicas before reporting
        """
        self._reap_dead()
        return self._loaded_models()

    def _sorted_ports(self) -> list[int]:
        """Returns all active ports in sorted order."""
        return sorted(self.processes.keys())

    async def load_model(self, model_id: str):
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

        # Acquire the exclusive lock of every GPU this load may mutate. A
        # concurrent load can add a co-resident between planning and locking,
        # widening the affected set, so re-derive after acquiring and retry with
        # the wider set (locks are always taken in ascending GPU order).
        self._reap_dead()
        affected = self._load_affected_gpus(model_id, self._loaded_models())
        while True:
            await self._acquire_exclusive(affected)
            self._reap_dead()
            wider = self._load_affected_gpus(model_id, self._loaded_models())
            if set(wider) <= set(affected):
                break
            await self._release_exclusive(affected)
            affected = wider
        self._swapping_gpus.update(affected)
        print(f"[ROUTER] initiate loading of model: {model_id} (gpus {affected})...")
        try:
            self._reap_dead()
            loaded = self._loaded_models()
            target = self.router_config["LLM"].get(model_id, {"num_instance": 1})["num_instance"]
            have = self.model_ports(model_id)
            if model_id in loaded and len(have) >= target:
                self.model_last_used[model_id] = time.time()
                print(f"[ROUTER] models: {model_id} already present in memory")
                return True

            # 1) evict per-GPU overflow (skipped when topping up lost replicas —
            #    the model already counts against its GPUs)
            if model_id not in loaded:
                evict = self._plan_evictions(model_id, loaded)
                print(f"[ROUTER] models: {loaded or '{}'} present in memory, evicting {evict or 'nothing'} "
                      f"(gpus={self.model_gpus.get(model_id, [0])}, cap={self.MAX_MODELS_PER_GPU}/gpu, policy={self.EVICTION_POLICY})")
                if evict:
                    ports_to_kill = [p for mid in evict for p in self.model_ports(mid)]
                    await asyncio.gather(*[self._kill_instance(p) for p in ports_to_kill])
                    for mid in evict:
                        self._forget_model(mid)

            # 2) spawn missing replicas, one process per replica
            needed = target - len(have)
            if needed > 0:
                ports = self._alloc_ports(needed)
                self._last_load_error = None
                results = await asyncio.gather(
                    *[self._spawn_replica(port, model_id) for port in ports],
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
            print(f"[LOAD CONFIRMATION] Loaded models: {self._loaded_models()}")
            return True
        finally:
            self._swapping_gpus.difference_update(affected)
            await self._release_exclusive(affected)
            self._has_requests.set()
            await self._reconcile_idle_supervisor()

    async def unload_model(self, model_id: str):
        """
        Unloads a model by killing all of its replica processes

        Args:
            model_id (str): The model to unload
        """
        gpus = self.model_gpus.get(model_id, [0])
        await self._acquire_exclusive(gpus)
        self._swapping_gpus.update(gpus)
        print(f"[ROUTER] initiate unloading of model: {model_id} (gpus {gpus})...")
        try:
            ports = self.model_ports(model_id)
            if ports:
                await asyncio.gather(*[self._kill_instance(p) for p in ports])
            self._forget_model(model_id)
            print(f"[UNLOAD SUCCESS] Successfully unloaded {model_id}")
            print(f"[UNLOAD CONFIRMATION] Currently loaded models: {self._loaded_models()}")
        finally:
            self._swapping_gpus.difference_update(gpus)
            await self._release_exclusive(gpus)
            self._has_requests.set()
            await self._reconcile_idle_supervisor()

    # Idle metadata supervisor

    async def _start_idle_supervisor(self) -> bool:
        """
        Spawns the bare metadata supervisor (no model, no GPU) on the reserved port

        A llama-server started with --no-models-autoload and an empty GPU mask
        loads no weights and holds no VRAM, but still serves /v1/models and /props
        for the whole preset. It exists only while the router has zero replicas
        loaded, so /v1/models can proxy a real llama.cpp process (models_report)
        instead of the router fabricating the listing. It is killed the moment a
        real model loads (see _reconcile_idle_supervisor).

        Returns:
            True once the supervisor answers /v1/models, otherwise False
        """
        if self._idle_proc is not None and self._idle_proc.poll() is None:
            return True
        port = self._idle_port
        exe = self.router_config["llama-server-executable"]
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = "" # no visible GPUs -> pure metadata daemon
        env["HIP_VISIBLE_DEVICES"] = ""
        print(f"[ROUTER] starting idle metadata supervisor at port {port} (no model, no gpu)...", flush=True)
        proc = subprocess.Popen(
            [exe, "--host", "0.0.0.0", "--port", str(port),
             "--models-preset", self.llama_presets_path, "--no-models-autoload", "--metrics"],
            stdout=None, # inherit router's fds -> journald
            stderr=None,
            env=env,
        )
        self._idle_proc = proc
        deadline = asyncio.get_event_loop().time() + self.HEALTH_CHECK_TIMEOUT
        async with httpx.AsyncClient() as client:
            while asyncio.get_event_loop().time() < deadline:
                if proc.poll() is not None:
                    print(f"[ROUTER] idle supervisor exited (code {proc.returncode}) before ready", flush=True)
                    self._idle_proc = None
                    return False
                try:
                    # /v1/models is served by a router-role server regardless of
                    # whether any model is loaded, so it doubles as the readiness probe.
                    resp = await client.get(f"http://127.0.0.1:{port}/v1/models", timeout=2.0)
                    if resp.status_code == 200:
                        print(f"[ROUTER] idle metadata supervisor ready at port {port}", flush=True)
                        return True
                except (httpx.ConnectError, httpx.TimeoutException):
                    pass
                await asyncio.sleep(self.HEALTH_CHECK_INTERVAL)
        print(f"[ROUTER] idle supervisor never became ready within {self.HEALTH_CHECK_TIMEOUT}s; killing", flush=True)
        proc.kill()
        self._idle_proc = None
        return False

    async def _kill_idle_supervisor(self) -> None:
        """Stops the idle metadata supervisor if running (SIGTERM, then SIGKILL)."""
        proc = self._idle_proc
        if proc is None:
            return
        self._idle_proc = None
        try:
            if proc.poll() is None:
                print(f"[ROUTER] stopping idle metadata supervisor at port {self._idle_port}...", flush=True)
                proc.terminate()
                await asyncio.sleep(self.GRACEFUL_KILL_TIMEOUT)
                if proc.poll() is None:
                    proc.kill()
            proc.wait()
        except Exception:
            pass

    async def _reconcile_idle_supervisor(self) -> None:
        """
        Keeps the idle metadata supervisor alive iff no real replica is loaded

        Runs the supervisor while the router has zero replicas (so /v1/models can
        still proxy a real llama.cpp process) and kills it as soon as any real
        model is resident, since a live replica is a better proxy source and a
        second supervisor would be redundant. Idempotent and serialized by
        _idle_lock so overlapping load/unload transitions can't race it.
        """
        async with self._idle_lock:
            self._reap_dead()
            try:
                if self._loaded_models():
                    await self._kill_idle_supervisor()
                else:
                    await self._start_idle_supervisor()
            except Exception as exc:
                # The idle supervisor is best-effort (models_report falls back to a
                # synthesized listing); never let it break a load/unload/start.
                print(f"[ROUTER] idle supervisor reconcile failed: {exc}", flush=True)

    def _proxy_port(self) -> int | None:
        """
        Picks a live llama.cpp supervisor port to proxy metadata from

        Prefers a real loaded replica (it also carries rich per-model meta), and
        falls back to the idle metadata supervisor. Returns None if neither is up.

        Returns:
            A port hosting a live llama.cpp server, or None
        """
        for port in sorted(self.processes):
            if self.processes[port].poll() is None:
                return port
        if self._idle_proc is not None and self._idle_proc.poll() is None:
            return self._idle_port
        return None

    async def models_report(self) -> dict[str, Any]:
        """
        Builds the /v1/models listing, proxying a live llama.cpp server when possible

        Proxies /v1/models from a live supervisor (a loaded replica, else the idle
        metadata supervisor) so the listing carries llama.cpp's real fields
        (modalities, source, and per-model meta for whatever is loaded), then
        overlays our per-request context_length (c / parallel) since llama.cpp
        does not report that. Falls back to a synthesized listing (ids +
        context_length) when no supervisor is reachable.

        Returns:
            An OpenAI-style {"object": "list", "data": [...]} dict
        """
        windows = self.model_context_windows()
        configured = list(self.router_config["LLM"])
        port = self._proxy_port()
        if port is not None:
            try:
                async with httpx.AsyncClient() as client:
                    resp = await client.get(f"http://127.0.0.1:{port}/v1/models", timeout=5.0)
                if resp.status_code == 200:
                    upstream = {row["id"]: row for row in resp.json().get("data", []) if "id" in row}
                    data = []
                    for mid in configured:
                        row = dict(upstream.get(mid, {"id": mid, "object": "model"}))
                        ctx = windows.get(mid)
                        if ctx is not None:
                            # per-request usable context; llama.cpp omits this and
                            # its meta.n_ctx (when loaded) is the total -c pool.
                            row["context_length"] = ctx
                        data.append(row)
                    return {"object": "list", "data": data}
            except (httpx.HTTPError, ValueError) as exc:
                print(f"[ROUTER] models_report proxy from port {port} failed: {exc}; synthesizing", flush=True)
        # Fallback: synthesize from config + presets (no live supervisor).
        created = int(time.time())
        data = []
        for mid in configured:
            entry: dict[str, Any] = {"id": mid, "object": "model", "created": created, "owned_by": "llama-router"}
            ctx = windows.get(mid)
            if ctx is not None:
                entry["context_length"] = ctx
                entry["meta"] = {"n_ctx_train": ctx, "n_ctx": ctx}
            data.append(entry)
        return {"object": "list", "data": data}

    # Request handling

    async def add_request(self, request: Envelope) -> "asyncio.Future[ForwardResult]":
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

                # Clear the wake flag BEFORE reading GPU busy-state, so a port
                # release (or swap completion) that races in after our read still
                # re-sets it and re-wakes us — no lost wakeup when we defer below.
                self._has_requests.clear()
                self._reap_dead()
                loaded = self._loaded_models()
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
                if not starved:
                    # pick a request to serve: first request that matches the model or has no model field
                    for i, entry in enumerate(self.requests):
                        req_model = entry["request"].get("model")
                        if req_model is None or req_model in loaded:
                            chosen_idx = i
                            break
                # there's a servable (cache-hit) request: serve it as a concurrent
                # reader on its GPUs, no loading required
                if chosen_idx is not None:
                    entry = self.requests.pop(chosen_idx)
                # otherwise the head model must be loaded. Unless it's starved,
                # defer while its GPUs are busy serving or mid-swap: we don't want
                # to evict/compete with in-flight work there. A port release or
                # swap completion re-wakes the scheduler to retry the load.
                else:
                    head_gpus = self._request_gpus(head_model)
                    if not starved and self._gpus_busy(head_gpus):
                        print(f"[ROUTER] deferring load of {head_model}: gpus {head_gpus} busy", flush=True)
                        continue
                    if starved:
                        print(f"[ROUTER] head-of-queue request for {head_model} waited "
                              f">{self.QUEUE_FORCE_LOAD_TIMEOUT}s, force-loading", flush=True)
                    entry = self.requests.pop(0)
                    model_to_load = entry["request"].get("model")
                    try:
                        await self.load_model(model_to_load)
                    except Exception as exc:
                        print(f"[ROUTER] failed to load {model_to_load}: {exc}", flush=True)
                        if not entry["future"].done():
                            entry["future"].set_exception(exc)
                        self._has_requests.set()  # re-check the remaining queue
                        continue
                # More requests may remain: re-arm so the next iteration runs
                # immediately (pipelining) instead of blocking on the wake flag.
                if self.requests:
                    self._has_requests.set()
                served_model = entry["request"].get("model")
                if served_model:
                    self.model_last_used[served_model] = time.time()
                # pick the least-busy replica of the served model
                # (model-less requests go to any live instance)
                if served_model is not None:
                    ports = self.model_ports(served_model)
                else:
                    ports = self._sorted_ports()
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
                # The GPUs this forward holds as a shared reader = the hosting
                # replica's model pins (tighter than the request's declared model
                # for model-less requests).
                entry["gpus"] = self.model_gpus.get(self.port_model.get(port, ""), list(range(self.num_gpus)))
                self.inflight[port] = self.inflight.get(port, 0) + 1

            # Dispatch forwarding as a concurrent task
            if entry["is_streaming"]:
                queue = asyncio.Queue()
                entry["future"].set_result(queue)
                asyncio.create_task(self._do_forward_streaming(entry, queue))
            else:
                asyncio.create_task(self._do_forward(entry))

    def _release_port(self, port: int):
        """
        Decrements the in-flight counter for a port, floored at zero

        Args:
            port (int): The port whose in-flight count should be released
        """
        if port in self.inflight:
            self.inflight[port] = max(0, self.inflight[port] - 1)
        # A freed GPU may unblock a load the scheduler deferred; re-wake it.
        self._has_requests.set()

    async def _do_forward(self, entry: dict[str, Any]):
        """
        Forwards a non-streaming request to its port and resolves its future

        Records token history from the response usage/timings on success

        Args:
            entry (dict): The dispatched request entry with its assigned "port"
        """
        gpus = entry.get("gpus", list(range(self.num_gpus)))
        await self._acquire_shared(gpus)
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
                    await self.record_history(model, entry["request_time"], time.time(), int(prompt_n), int(predicted_n))
            except Exception:
                pass
        except Exception as e:
            if not entry["future"].done():
                entry["future"].set_exception(e)
        finally:
            self._release_port(entry["port"])
            await self._release_shared(gpus)

    async def _do_forward_streaming(self, entry: dict[str, Any], queue: StreamQueue):
        """
        Forwards a streaming request to its port, pushing chunks to the queue

        Puts a None sentinel when done and records token history from the last
        SSE chunk that carried timings

        Args:
            entry (dict)            : The dispatched request entry with its "port"
            queue (asyncio.Queue)   : The queue chunks are pushed onto
        """
        gpus = entry.get("gpus", list(range(self.num_gpus)))
        await self._acquire_shared(gpus)
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
            self._release_port(entry["port"])
            await self._release_shared(gpus)
            # Record history from the last SSE chunk that contained timings
            if last_data:
                try:
                    timings = last_data.get("timings", {})
                    usage = last_data.get("usage", {})
                    prompt_n = timings.get("prompt_n", usage.get("prompt_tokens", 0))
                    predicted_n = timings.get("predicted_n", usage.get("completion_tokens", 0))
                    model = entry["request"].get("model", "unknown")
                    if prompt_n or predicted_n:
                        await self.record_history(model, entry["request_time"], time.time(), int(prompt_n), int(predicted_n))
                except Exception:
                    pass
