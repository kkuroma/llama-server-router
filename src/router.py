import asyncio
import json
import os
import subprocess
import time
from enum import Enum

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


class AsyncRWLock:
    """
    Asyncio readers-writer lock implementation
    Idea:
        1) multiple coroutines can hold a shared reader() lock
        2) only one exclusive (writer) lock needs to wait for readers to hold the lock
    Usage:
        1) LLM generation requests are readers; they don't interfere with each other
        2) LLM load/unload requests are writers; you need to wait for all requests to that model to finish
    Caveats:
        1) Readers will starve the writer
        2) Doesn't matter in our case since we want to maximize LLM cache hit
    """

    def __init__(self):
        self._readers = 0
        self._writer = False
        self._lock = asyncio.Lock()
        self._readers_ok = asyncio.Condition(self._lock) # readers condition var
        self._writer_ok = asyncio.Condition(self._lock) # writers condition var

    # Exclusive to readers

    async def acquire_shared(self):
        '''Acquire the lock if the writer is not active'''
        async with self._lock:
            while self._writer:
                await self._readers_ok.wait()
            self._readers += 1

    async def release_shared(self):
        '''Release the lock if the all readers are done'''
        async with self._lock:
            self._readers -= 1
            if self._readers == 0:
                self._writer_ok.notify()

    # Exclusive to writers

    async def acquire_exclusive(self):
        '''Acquire the lock if the writer is active AND readers have finished'''
        async with self._lock:
            while self._writer or self._readers > 0:
                await self._writer_ok.wait()
            self._writer = True

    async def release_exclusive(self):
        '''Release the lock if the writer is not active'''
        async with self._lock:
            self._writer = False
            self._readers_ok.notify_all()
            self._writer_ok.notify()

async def _fetch_model_statuses(port: int) -> dict[str, str]:
    """Query a single llama-server instance and return {model_id: status_value}
    for every model reported by that instance."""
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
    One llama-server process per loaded model replica:
        load  = spawn num_instance processes and load the model into each
        evict = SIGTERM those processes (VRAM is freed unconditionally)
    Residency is tracked per GPU: a model pinned to gpus [0, 1] counts against
    GPU 0 and GPU 1; at most MAX_MODELS_PER_GPU models stay resident per GPU.
    Crashed replicas are reaped and the model is simply reloaded on the next
    request for it (auto-reload on demand).
    """

    def __init__(self, router_config_path: str, llama_presets_path: str):
        self.llama_presets_path = llama_presets_path

        with open(router_config_path, "r") as f:
            self.router_config = json.load(f)
            '''
                self.router_config["LLM"] -> {model_id: {num_instance: N, gpus: [ids] | "all" | -1}}
                self.router_config["API-port"] -> int (port to expose the router)
                self.router_config["LLM-base-port"] -> int (first port for llama-server instances)
                self.router_config["llama-server-executable"] -> str (path to the llama-server binary)
                self.router_config["ROUTER"] -> reassign router values
            '''
            router_settings = self.router_config.get("ROUTER", {})
            self.HEALTH_CHECK_INTERVAL = router_settings.get("HEALTH_CHECK_INTERVAL", 1.0) # seconds between health polls
            self.HEALTH_CHECK_TIMEOUT = router_settings.get("HEALTH_CHECK_TIMEOUT", 30.0) # max seconds to wait for /health
            self.UNLOAD_POLL_INTERVAL = router_settings.get("UNLOAD_POLL_INTERVAL", 0.5) # seconds between polls waiting for unload to finish
            self.UNLOAD_POLL_TIMEOUT = router_settings.get("UNLOAD_POLL_TIMEOUT", 60.0) # max seconds to wait for all models to unload
            self.LOAD_POLL_INTERVAL = router_settings.get("LOAD_POLL_INTERVAL", 1.0) # seconds between polls waiting for model to load
            self.LOAD_POLL_TIMEOUT = router_settings.get("LOAD_POLL_TIMEOUT", 120.0) # max seconds to wait for a model to finish loading
            self.START_RETRIES = router_settings.get("START_RETRIES", 3) # attempts per instance on start()
            self.GRACEFUL_KILL_TIMEOUT = router_settings.get("GRACEFUL_KILL_TIMEOUT", 5.0) # seconds to wait after SIGTERM before SIGKILL
            self.MAX_MODELS_PER_GPU = int(router_settings.get("MAX_MODELS_PER_GPU", 1)) # resident-model cap PER GPU (not global)
            self.EVICTION_POLICY = str(router_settings.get("EVICTION_POLICY", "lru")).lower() # "lru" (last request) or "fifo" (load time)
            self.QUEUE_FORCE_LOAD_TIMEOUT = float(router_settings.get("QUEUE_FORCE_LOAD_TIMEOUT", 300.0)) # seconds a queued request may starve before its model is force-loaded

        if self.EVICTION_POLICY not in ("lru", "fifo"):
            print(f"[ROUTER] unknown EVICTION_POLICY {self.EVICTION_POLICY!r}, falling back to 'lru'", flush=True)
            self.EVICTION_POLICY = "lru"

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

        self.status: Status = Status.INACTIVE
        self.processes: dict[int, subprocess.Popen] = {} # port -> Popen
        self.port_model: dict[int, str] = {} # port -> model_id hosted by that process (set once loaded)
        self.inflight: dict[int, int] = {} # port -> requests currently being forwarded
        self.requests:  list[dict] = [] # [{request, future, is_streaming, request_time}, ...]
        self.request_lock = asyncio.Lock()
        self._load_lock = AsyncRWLock()
        self._has_requests = asyncio.Event()
        self._running = False
        self._scheduler_task: asyncio.Task | None = None
        self._history_db_path = HISTORY_DB_PATH
        self._history_db_ready = False

    # History DB

    async def init_history_db(self):
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
        if not self._history_db_ready:
            await self.init_history_db()

    async def record_history(self, model: str, request_time: float, response_time: float, prompt_n: int, predicted_n: int):
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

    async def get_history(self, model: str | None = None, limit: int = 500) -> list[dict]:
        '''Most recent rows first; limit=0 means no limit'''
        await self._ensure_history_db()
        lim = " LIMIT ?" if limit else ""
        async with aiosqlite.connect(self._history_db_path) as db:
            db.row_factory = aiosqlite.Row
            if model:
                cursor = await db.execute(
                    f"SELECT * FROM history WHERE model = ? ORDER BY request_time DESC{lim}",
                    (model, limit) if limit else (model,))
            else:
                cursor = await db.execute(
                    f"SELECT * FROM history ORDER BY request_time DESC{lim}",
                    (limit,) if limit else ())
            rows = await cursor.fetchall()
            return [dict(row) for row in rows]

    async def reset_history(self):
        await self._ensure_history_db()
        async with aiosqlite.connect(self._history_db_path) as db:
            await db.execute("DELETE FROM history")
            await db.commit()

    # GPU topology / eviction planning

    def _detect_num_gpus(self, configured) -> int:
        '''
            GPU count: explicit ROUTER.NUM_GPUS wins, then pynvml,
            then highest explicit pin + 1, then 1.
        '''
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

    def _resolve_gpus(self, model_id: str, raw) -> list[int]:
        '''
            Normalize a model's "gpus" field to a sorted list of valid GPU ids.
            None -> [0] (unpinned = GPU 0 only); -1 or "all" -> every GPU.
        '''
        if raw is None:
            return [0]
        if not isinstance(raw, list):
            raw = [raw]
        ids: set[int] = set()
        for v in raw:
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

    def _plan_evictions(self, model_id: str, loaded: set[str]) -> set[str]:
        '''
            Decide which resident models must be unloaded so *model_id* fits.
            Per GPU that model_id is pinned to: if residents would exceed
            MAX_MODELS_PER_GPU with model_id added, evict the oldest residents
            of THAT GPU (by last request for "lru", by load time for "fifo").
            Residents with no queued demand are always evicted before residents
            that still have requests waiting in the queue.
            Models on GPUs model_id doesn't touch are left alone.
        '''
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
        self.model_loaded_at.pop(model_id, None)
        self.model_last_used.pop(model_id, None)

    # Process pool (one llama-server process per model replica)

    def _reap_dead(self):
        '''Drop bookkeeping for replicas whose process has exited. The model
        auto-reloads on its next request (it no longer counts as loaded).'''
        for port in list(self.processes.keys()):
            if self.processes[port].poll() is not None:
                mid = self.port_model.get(port, "<none>")
                print(f"[ROUTER] replica on port {port} (model {mid}) died, reaping", flush=True)
                del self.processes[port]
                self.port_model.pop(port, None)
                self.inflight.pop(port, None)

    def _model_ports(self, model_id: str) -> list[int]:
        return sorted(p for p, m in self.port_model.items() if m == model_id and p in self.processes)

    def _loaded_models(self) -> set[str]:
        return {m for p, m in self.port_model.items() if p in self.processes}

    def _alloc_ports(self, n: int) -> list[int]:
        '''Pick the n lowest ports >= LLM-base-port not currently in use'''
        ports: list[int] = []
        used = set(self.processes.keys())
        p = self.router_config["LLM-base-port"]
        while len(ports) < n:
            if p not in used:
                ports.append(p)
                used.add(p)
            p += 1
        return ports

    async def _start_instance(self, port: int) -> int | None:
        '''
            Spawn one llama-server on *port*
            Return its PID on success, None on failure
        '''
        print(f"[ROUTER] starting instance at port {port}...")
        exe = self.router_config["llama-server-executable"]
        proc = subprocess.Popen(
            [exe, "--host", "0.0.0.0", "--port", str(port),
             "--models-preset", self.llama_presets_path, "--metrics"],
            stdout=None, # inherit router's fds -> journald
            stderr=None, # inherit router's fds -> journald
        )
        self.processes[port] = proc
        deadline = asyncio.get_event_loop().time() + self.HEALTH_CHECK_TIMEOUT
        async with httpx.AsyncClient() as client:
            while asyncio.get_event_loop().time() < deadline:
                try:
                    resp = await client.get(f"http://0.0.0.0:{port}/health", timeout=2.0)
                    if resp.status_code == 200 and resp.json().get("status") == "ok":
                        return proc.pid
                except (httpx.ConnectError, httpx.TimeoutException):
                    pass
                await asyncio.sleep(self.HEALTH_CHECK_INTERVAL)
        proc.kill()
        del self.processes[port]
        return None

    async def _load_into(self, port: int, model_id: str) -> bool:
        '''
            Load *model_id* into the llama-server at *port* and wait until it
            reports "loaded". Returns True on success.
        '''
        try:
            async with httpx.AsyncClient() as client:
                await client.post(f"http://0.0.0.0:{port}/models/load", json={"model": model_id}, timeout=120.0)
        except Exception as exc:
            print(f"[ROUTER] load request to port {port} failed: {exc}", flush=True)
            return False
        deadline = asyncio.get_event_loop().time() + self.LOAD_POLL_TIMEOUT
        while asyncio.get_event_loop().time() < deadline:
            try:
                statuses = await _fetch_model_statuses(port)
                if statuses.get(model_id) == "loaded":
                    return True
            except Exception:
                pass
            await asyncio.sleep(self.LOAD_POLL_INTERVAL)
        print(f"[ROUTER] timed out waiting for {model_id} to load on port {port}", flush=True)
        return False

    async def _spawn_replica(self, port: int, model_id: str) -> bool:
        '''
            Spawn a llama-server on *port* hosting exactly *model_id*.
        '''
        for attempt in range(self.START_RETRIES):
            pid = await self._start_instance(port)
            if pid is None:
                continue
            if await self._load_into(port, model_id):
                self.port_model[port] = model_id
                print(f"[ROUTER] replica for {model_id} up on port {port} (pid {pid})", flush=True)
                return True
            await self._kill_instance(port)
        return False

    async def _kill_instance(self, port: int) -> bool:
        '''
            "Gracefully" kills an instance at *port*
            By "Gracefully", I meant SIGKILLing it in case of disobedience
            Returns True on success and False otherwise
        '''
        print(f"[ROUTER] killing instance at port {port}...")
        proc = self.processes.get(port)
        if proc is None:
            return False
        try:
            proc.terminate() # SIGTERM
            await asyncio.sleep(self.GRACEFUL_KILL_TIMEOUT)
            if proc.poll() is None: # still running?
                proc.kill() # SIGKILL
            proc.wait()
            del self.processes[port]
            self.port_model.pop(port, None)
            self.inflight.pop(port, None)
            return True
        except Exception:
            return False

    async def start(self):
        '''
            Start the scheduler. No llama-server is spawned up front — replicas
            are spawned on demand when a model is first requested.
        '''
        if self.status not in (Status.INACTIVE, Status.ERROR):
            return
        self.status = Status.STARTING
        self._running = True
        self._scheduler_task = asyncio.create_task(self._scheduler())
        self.status = Status.IDLE
        print(f"[START SUCCESS] Router started, replicas spawn on demand")

    async def stop(self):
        '''
            Hard thanos resets the router by killing all instances and resetting states
        '''
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
            *[self._kill_instance(port) for port in list(self.processes.keys())],
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
        '''
            Restarts the router by stopping and starting again
        '''
        await self.stop()
        await self.start()

    # Load/Unload

    async def get_loaded_models(self) -> set[str]:
        '''
            Set of models with at least one live replica process
        '''
        self._reap_dead()
        return self._loaded_models()

    def _sorted_ports(self) -> list[int]:
        '''
            Returns a list of all active ports
        '''
        return sorted(self.processes.keys())

    async def load_model(self, model_id: str):
        '''
            Make *model_id* resident by the following algorithm
                1) Evicts per-GPU overflow: on each GPU model_id is pinned to,
                   kill the oldest residents' processes until the new model fits
                   under MAX_MODELS_PER_GPU. Models on other GPUs are untouched.
                2) Spawns num_instance llama-server processes, each hosting
                   exactly model_id (tops up missing replicas if some are alive)
        '''
        if model_id not in self.router_config["LLM"]:
            raise ValueError(f"[LOAD/UNLOAD ERROR] Model [{model_id}] not present in list {list(self.router_config['LLM'].keys())}")

        # Acquires load lock
        await self._load_lock.acquire_exclusive()
        print(f"[ROUTER] initiate loading of model: {model_id}...")
        try:
            self._reap_dead()
            loaded = self._loaded_models()
            target = self.router_config["LLM"].get(model_id, {"num_instance": 1})["num_instance"]
            have = self._model_ports(model_id)
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
                    ports_to_kill = [p for mid in evict for p in self._model_ports(mid)]
                    await asyncio.gather(*[self._kill_instance(p) for p in ports_to_kill])
                    for mid in evict:
                        self._forget_model(mid)

            # 2) spawn missing replicas, one process per replica
            needed = target - len(have)
            if needed > 0:
                ports = self._alloc_ports(needed)
                results = await asyncio.gather(
                    *[self._spawn_replica(port, model_id) for port in ports],
                    return_exceptions=True,
                )
                up = sum(1 for r in results if r is True)
                if up == 0:
                    raise RuntimeError(f"[LOAD/UNLOAD ERROR] Failed to start any replica for {model_id}")
                if up < needed:
                    print(f"[ROUTER] only {up}/{needed} new replicas for {model_id} came up", flush=True)

            now = time.time()
            self.model_loaded_at[model_id] = now
            self.model_last_used[model_id] = now
            print(f"[LOAD/UNLOAD SUCCESS] Successfully loaded {model_id}")
            print(f"[LOAD CONFIRMATION] Loaded models: {self._loaded_models()}")
            return True
        finally:
            await self._load_lock.release_exclusive()
            self._has_requests.set()

    async def unload_model(self, model_id: str):
        '''
            Unload a model by killing all of its replica processes
        '''
        await self._load_lock.acquire_exclusive()
        print(f"[ROUTER] initiate unloading of model: {model_id}...")
        try:
            ports = self._model_ports(model_id)
            if ports:
                await asyncio.gather(*[self._kill_instance(p) for p in ports])
            self._forget_model(model_id)
            print(f"[UNLOAD SUCCESS] Successfully unloaded {model_id}")
            print(f"[UNLOAD CONFIRMATION] Currently loaded models: {self._loaded_models()}")
        finally:
            await self._load_lock.release_exclusive()
            self._has_requests.set()

    # Request handling

    async def add_request(self, request: dict) -> asyncio.Future:
        '''
            Enqueue a request and return a Future that resolves when the request is processed.
            The serving port is picked at dispatch time (after the model is
            resident), since ports are per-model now.
            Non-streaming: future resolves with httpx.Response
            Streaming: future resolves with asyncio.Queue (chunks terminated by None sentinel)
        '''
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
        '''
            Background scheduler that continuously picks requests from the queue and dispatches forwarding tasks
            Maximizes cache hits by preferring requests whose model is already loaded
            When nothing is servable, loads the first request's model
        '''
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
                        await self.load_model(model_to_load)
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
                    ports = self._model_ports(served_model)
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
                self.inflight[port] = self.inflight.get(port, 0) + 1

            # Dispatch forwarding as a concurrent task
            if entry["is_streaming"]:
                queue = asyncio.Queue()
                entry["future"].set_result(queue)
                asyncio.create_task(self._do_forward_streaming(entry, queue))
            else:
                asyncio.create_task(self._do_forward(entry))

    def _release_port(self, port: int):
        if port in self.inflight:
            self.inflight[port] = max(0, self.inflight[port] - 1)

    async def _do_forward(self, entry):
        '''
            Forward a non-streaming request to the assigned port and resolve its future.
        '''
        await self._load_lock.acquire_shared()
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
            self.status = Status.IDLE
            self._release_port(entry["port"])
            await self._load_lock.release_shared()

    async def _do_forward_streaming(self, entry, queue: asyncio.Queue):
        '''
            Forward a streaming request to the assigned port, pushing chunks to the queue.
            Puts None as sentinel when done.
        '''
        await self._load_lock.acquire_shared()
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
            await self._load_lock.release_shared()
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
                        await self.record_history(model, entry["request_time"], time.time(), int(prompt_n), int(predicted_n))
                except Exception:
                    pass
