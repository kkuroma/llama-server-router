import asyncio
import json
import subprocess
import sys
from enum import Enum
from collections import Counter
import httpx
import time

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

async def _fetch_loaded_models(port: int) -> set[str]:
    """Query a single llama-server instance and return the set of model IDs
    with status exactly 'loaded'"""
    async with httpx.AsyncClient() as client:
        resp = await client.get(f"http://0.0.0.0:{port}/models", timeout=5.0)
        resp.raise_for_status()
        data = resp.json()
        return {
            m.get("id")
            for m in data.get("data", [])
            if m.get("status", {}).get("value") == "loaded"
        }

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
    def __init__(self, router_config_path: str, llama_presets_path: str):
        self.llama_presets_path = llama_presets_path

        with open(router_config_path, "r") as f:
            self.router_config = json.load(f)
            '''
                self.router_config["LLM"] -> {model_id: {num_instances: N}}
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

        self.status: Status = Status.INACTIVE
        self.processes: dict[int, subprocess.Popen] = {} # port -> Popen
        self.requests:  list[dict] = [] # [{port, request, future, is_streaming}, ...]
        self.request_lock = asyncio.Lock()
        self._load_lock = AsyncRWLock()
        self._has_requests = asyncio.Event()
        self._running = False
        self._scheduler_task: asyncio.Task | None = None

    async def _start_instance(self, port: int) -> int | None:
        '''
            Spawn one llama-server on *port*
            Return its PID on success, None on failure
        '''
        print(f"[ROUTER] starting instance at port {port}...")
        exe = self.router_config["llama-server-executable"]
        proc = subprocess.Popen(
            [exe, "--host", "0.0.0.0", "--port", str(port),
             "--models-preset", self.llama_presets_path],
            stdout=subprocess.DEVNULL, # sys.stdout,
            stderr=subprocess.DEVNULL, # sys.stderr,
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

    async def start(self):
        '''
            Start the default number of instances in parallel
            Won't start if either lock is currently held exclusively
        '''
        if self.status not in (Status.INACTIVE, Status.ERROR):
            return

        self.status = Status.STARTING
        num_instance = 1 # always start with 1 instance
        base_port = self.router_config["LLM-base-port"]
        async def _try_start(port: int):
            for attempt in range(self.START_RETRIES):
                pid = await self._start_instance(port)
                if pid is not None:
                    return True
            return False

        results = await asyncio.gather(
            *[_try_start(base_port + i) for i in range(num_instance)],
            return_exceptions=True,
        )

        if all(results):
            self.status = Status.IDLE
            self._running = True
            self._scheduler_task = asyncio.create_task(self._scheduler())
            print(f"[START SUCCESS] Router started successfully, models available at port {list(self.processes.keys())}")
        else:
            self.status = Status.ERROR
            print("[START ERROR] Failed to start router")

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
            return True
        except Exception:
            return False

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
        self.requests.clear()
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
            Get the set of all loaded models across all live instances
        '''
        if not self.processes:
            return set()
        results = await asyncio.gather(
            *[_fetch_loaded_models(port) for port in self.processes],
            return_exceptions=True,
        )
        models: set[str] = set()
        for r in results:
            if isinstance(r, set):
                models |= r
        return models

    def _sorted_ports(self) -> list[int]:
        '''
            Returns a list of all active ports
        '''
        return sorted(self.processes.keys())

    async def _all_instances_report_status(self, model_id: str, target_status: str) -> bool:
        '''
            Returns True only if *model_id* has exactly *target_status* on every live instance.
        '''
        if not self.processes:
            return False
        results = await asyncio.gather(
            *[_fetch_model_statuses(port) for port in self.processes],
            return_exceptions=True,
        )
        for r in results:
            if isinstance(r, Exception):
                return False
            if r.get(model_id) != target_status:
                return False
        return True

    async def load_model(self, model_id: str):
        '''
            Swap all instances' loaded model to *model_id* by the following algorithm
                1) Unloads everything currently loaded (/models/unload)
                2) adjusts instance count to math model_id's num_instance config
                3) Loads *model_id* to all instances (/model/load) then wait for success
        '''
        if model_id not in self.router_config["LLM"]:
            raise ValueError(f"[LOAD/UNLOAD ERROR] Model [{model_id}] not present in list {list(self.router_config['LLM'].keys())}")

        # Acquires load lock
        await self._load_lock.acquire_exclusive()
        print(f"[ROUTER] initiate loading of model: {model_id}...")
        try:
            # 1) unload all currently loaded models
            loaded = await self.get_loaded_models()
            if model_id in loaded:
                print(f"[ROUTER] models: {model_id} already present in memory")
                return True
            print(f"[ROUTER] models: {loaded} present in memory, unloading...")
            if loaded:
                async with httpx.AsyncClient() as client:
                    for port in self._sorted_ports():
                        for mid in loaded:
                            await client.post(f"http://0.0.0.0:{port}/models/unload", json={"model": mid}, timeout=30.0)
                # poll until every model reports exactly "unloaded" on all instances
                deadline = asyncio.get_event_loop().time() + self.UNLOAD_POLL_TIMEOUT
                while asyncio.get_event_loop().time() < deadline:
                    statuses = await asyncio.gather(
                        *[self._all_instances_report_status(mid, "unloaded") for mid in loaded]
                    )
                    if all(statuses):
                        break
                    await asyncio.sleep(self.UNLOAD_POLL_INTERVAL)
                else:
                    raise RuntimeError("[LOAD/UNLOAD ERROR] Timed out waiting for models to unload")
                    return False

            # 2) adjust instance count
            target = self.router_config["LLM"].get(model_id, {"num_instance": 1})["num_instance"]
            current_ports = self._sorted_ports()
            if len(current_ports) > target: # kill the highest-numbered ports first
                to_kill = current_ports[target:]
                await asyncio.gather(*[self._kill_instance(p) for p in to_kill])
            elif len(current_ports) < target: # spawn new ports numerically above the current max
                base = (max(current_ports) + 1) if current_ports else self.router_config["LLM-base-port"]
                needed = target - len(current_ports)
                await asyncio.gather(*[self._start_instance(base + i) for i in range(needed)])

            # 3) load model on all instances
            loaded = await self.get_loaded_models()
            print(f"[ROUTER] {len(loaded)} models present in memory, loading...")
            async with httpx.AsyncClient() as client:
                load_tasks = []
                for port in self._sorted_ports():
                    load_tasks.append(
                        client.post(f"http://0.0.0.0:{port}/models/load", json={"model": model_id}, timeout=120.0)
                    )
                    print(f"loading {model_id}")
                results = await asyncio.gather(*load_tasks, return_exceptions=True)
            for r in results:
                if isinstance(r, Exception):
                    raise RuntimeError(f"[LOAD/UNLOAD ERROR] Failed to load model on an instance: {r}")
            # poll until model reports exactly "loaded" on all instances
            deadline = asyncio.get_event_loop().time() + self.LOAD_POLL_TIMEOUT
            while asyncio.get_event_loop().time() < deadline:
                if await self._all_instances_report_status(model_id, "loaded"):
                    print(f"[LOAD/UNLOAD SUCCESS] Successfully loaded {model_id}")
                    loaded = await self.get_loaded_models()
                    print(f"[LOAD CONFIRMATION] Loaded models: {loaded}")
                    return True
                await asyncio.sleep(self.LOAD_POLL_INTERVAL)
            raise RuntimeError(f"[LOAD/UNLOAD ERROR] Timed out waiting for {model_id} to report as loaded")
            return False
        finally:
            await self._load_lock.release_exclusive()
            self._has_requests.set()

    async def unload_model(self, model_id: str):
        '''
            Unload a specific model from all instances
        '''
        await self._load_lock.acquire_exclusive()
        print(f"[ROUTER] initiate unloading of model: {model_id}...")
        try:
            async with httpx.AsyncClient() as client:
                for port in self._sorted_ports():
                    await client.post(f"http://0.0.0.0:{port}/models/unload", json={"model": model_id}, timeout=30.0)
            # poll until model reports exactly "unloaded" on all instances
            deadline = asyncio.get_event_loop().time() + self.UNLOAD_POLL_TIMEOUT
            while asyncio.get_event_loop().time() < deadline:
                if await self._all_instances_report_status(model_id, "unloaded"):
                    print(f"[UNLOAD SUCCESS] Successfully unloaded {model_id}")
                    loaded = await self.get_loaded_models()
                    print(f"[UNLOAD CONFIRMATION] Currently loaded models: {loaded}")
                    break
                await asyncio.sleep(self.UNLOAD_POLL_INTERVAL)
            else:
                raise RuntimeError(f"[UNLOAD ERROR] Timed out waiting for {model_id} to unload")
        finally:
            await self._load_lock.release_exclusive()
            self._has_requests.set()

    # Request handling

    async def add_request(self, request: dict) -> asyncio.Future:
        '''
            Enqueue a request and return a Future that resolves when the request is processed.
            Non-streaming: future resolves with httpx.Response
            Streaming: future resolves with asyncio.Queue (chunks terminated by None sentinel)
        '''
        future = asyncio.get_event_loop().create_future()
        is_streaming = request.pop("is_streaming", False)
        async with self.request_lock:
            ports = self._sorted_ports()
            if not ports:
                await self.start()
                ports = self._sorted_ports()
            counts = Counter(r["port"] for r in self.requests)
            port = min(ports, key=lambda p: (counts.get(p, 0), p))
            self.requests.append({
                "port": port,
                "request": request,
                "future": future,
                "is_streaming": is_streaming,
            })
            self._has_requests.set()
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
                    self.status = Status.IDLE
                    continue

                self.status = Status.SERVING
                loaded = await self.get_loaded_models()
                # pick a request to serve: first request that matches the model or has no model field
                chosen_idx = None
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
                    await self.load_model(model_to_load)

            # Dispatch forwarding as a concurrent task
            if entry["is_streaming"]:
                queue = asyncio.Queue()
                entry["future"].set_result(queue)
                asyncio.create_task(self._do_forward_streaming(entry, queue))
            else:
                asyncio.create_task(self._do_forward(entry))

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
        except Exception as e:
            if not entry["future"].done():
                entry["future"].set_exception(e)
        finally:
            await self._load_lock.release_shared()

    async def _do_forward_streaming(self, entry, queue: asyncio.Queue):
        '''
            Forward a streaming request to the assigned port, pushing chunks to the queue.
            Puts None as sentinel when done.
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
                async with client.stream(
                    method,
                    f"http://0.0.0.0:{port}{path}",
                    content=body if isinstance(body, (bytes, str)) else json.dumps(body) if body else None,
                    headers=headers,
                    timeout=300.0,
                ) as resp:
                    async for chunk in resp.aiter_bytes():
                        await queue.put(chunk)
        except Exception as e:
            await queue.put(e)
        finally:
            queue.put_nowait(None)
            await self._load_lock.release_shared()
