"""
Spawns a throwaway router over real weights and measures how its requests overlap

Nothing here imports src, so a test sees exactly what an API client sees
Overlap comes from token arrival times rather than from the router's bookkeeping
"""

import asyncio
import json
import os
import signal
import subprocess
import sys
import time
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, BinaryIO

import httpx

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "tests" / "configs.toml"


def load_config() -> dict[str, Any]:
    """
    Reads the test configuration, honoring a LLAMA_ROUTER_TEST_CONFIG override

    Returns:
        The parsed configs.toml.
    """
    path = Path(os.environ.get("LLAMA_ROUTER_TEST_CONFIG", DEFAULT_CONFIG))
    with open(path, "rb") as handle:
        return tomllib.load(handle)


def model_id(cfg: dict[str, Any], index: int, slots: int) -> str:
    """
    Names the generated preset serving one model at one slot count

    Args:
        cfg: The parsed test configuration.
        index: 0 or 1, selecting model_0 or model_1.
        slots: The llama.cpp server slot count that preset carries.

    Returns:
        The model id the generated router config exposes for that pair.
    """
    return f"{cfg['Models'][f'model_{index}']['id']}-p{slots}"


def weights_path(cfg: dict[str, Any], index: int) -> Path:
    """
    Resolves one model's gguf from its directory and file name

    Args:
        cfg: The parsed test configuration.
        index: 0 or 1, selecting model_0 or model_1.

    Returns:
        The full path to that model's weights.
    """
    model = cfg["Models"][f"model_{index}"]
    return Path(model["dir"]) / model["weights"]


def load_prompt(cfg: dict[str, Any]) -> str:
    """
    Reads the shared prompt every case sends

    Args:
        cfg: The parsed test configuration.

    Returns:
        The prompt text, resolved next to the config file.
    """
    path = Path(os.environ.get("LLAMA_ROUTER_TEST_CONFIG", DEFAULT_CONFIG)).parent
    return (path / cfg["Prompt"]["file"]).read_text()


def variant_prompt(cfg: dict[str, Any], prompt: str, index: int) -> str:
    """
    Appends the per request tail that keeps two requests from being identical

    The tail goes last so the whole essay stays a shared prefix, which is what a
    fleet of agents on one system prompt actually looks like.

    Args:
        cfg: The parsed test configuration.
        prompt: The shared prompt text.
        index: The requesting caller's index.

    Returns:
        The prompt with its variant tail.
    """
    return f"{prompt}\n\n{cfg['Prompt']['variant'].format(index=index)}\n"


def llama_server(cfg: dict[str, Any]) -> str:
    """
    Resolves the llama-server binary, letting LLAMA_SERVER override the config

    Args:
        cfg: The parsed test configuration.

    Returns:
        The path the generated router config will spawn.
    """
    return os.environ.get("LLAMA_SERVER", str(cfg["Router"]["llama_server"]))


def _ini_value(value: bool | int | float | str) -> str:
    """
    Renders one preset value the way llama.cpp's INI parser expects it

    Args:
        value: A bool, number or string from the test configuration.

    Returns:
        The string written to the right of the equals sign.
    """
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def write_configs(cfg: dict[str, Any], work_dir: Path) -> tuple[Path, Path]:
    """
    Writes the router JSON and the llama.cpp presets INI the test router boots from

    Each model is emitted once per slot count, so changing the slot count is a
    model swap and never an edit to a file a live process already read.

    Args:
        cfg: The parsed test configuration.
        work_dir: The directory both generated files land in.

    Returns:
        The (config.json, presets.ini) paths.
    """
    slot_counts = sorted(set(int(n) for n in cfg["Models"]["slot_counts"]))
    llm: dict[str, Any] = {}
    sections: list[tuple[str, dict[str, Any]]] = []
    for index in (0, 1):
        model = cfg["Models"][f"model_{index}"]
        for slots in slot_counts:
            name = model_id(cfg, index, slots)
            llm[name] = {"num_instance": 1, "gpus": list(model["gpus"])}
            sections.append((name, {
                "model": str(weights_path(cfg, index)),
                "c": int(cfg["Models"]["per_slot_ctx"]) * slots,
                "b": int(model["b"]),
                "ub": int(model["ub"]),
                "parallel": slots,
            }))

    router_config = {
        "LLM": llm,
        "ROUTER": cfg["Router"]["settings"],
        "API-port": int(cfg["Router"]["api_port"]),
        "LLM-base-port": int(cfg["Router"]["llm_base_port"]),
        "llama-server-executable": llama_server(cfg),
    }
    config_path = work_dir / "config.json"
    config_path.write_text(json.dumps(router_config, indent=2) + "\n")

    lines = ["[*]"]
    lines += [f"{key} = {_ini_value(value)}" for key, value in cfg["Presets"].items()]
    for name, section in sections:
        lines += ["", f"[{name}]"]
        lines += [f"{key} = {_ini_value(value)}" for key, value in section.items()]
    presets_path = work_dir / "presets.ini"
    presets_path.write_text("\n".join(lines) + "\n")
    return config_path, presets_path


class RouterProcess:
    """
    Runs src/main.py as a subprocess and tears down its whole process group

    The router is spawned in its own session, so the group SIGKILL after the
    graceful stop also reaps any llama-server it failed to clean up.
    """

    def __init__(self, cfg: dict[str, Any], work_dir: Path) -> None:
        self.cfg = cfg
        self.work_dir = work_dir
        self.host = str(cfg["Router"]["host"])
        self.port = int(cfg["Router"]["api_port"])
        self.log_path = work_dir / "router.log"
        self._proc: subprocess.Popen[bytes] | None = None
        self._log: BinaryIO | None = None

    @property
    def base_url(self) -> str:
        """The router's API root."""
        return f"http://{self.host}:{self.port}"

    def start(self) -> None:
        """
        Generates the config pair, spawns the router and waits for its API

        Raises:
            RuntimeError: If the router exits early or never answers /router.
        """
        config_path, presets_path = write_configs(self.cfg, self.work_dir)
        env = dict(os.environ)
        env["ROUTER_CONFIG_PATH"] = str(config_path)
        env["LLAMA_PRESETS_PATH"] = str(presets_path)
        env["ROUTER_HOST"] = self.host
        env["HISTORY_DB_PATH"] = str(self.work_dir / "monitor" / "history.db")
        self._log = open(self.log_path, "wb")
        self._proc = subprocess.Popen(
            [sys.executable, str(REPO_ROOT / "src" / "main.py")],
            cwd=str(REPO_ROOT),
            env=env,
            stdout=self._log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        self._wait_ready()

    def _wait_ready(self) -> None:
        """
        Polls /router until it answers or the startup budget runs out

        Raises:
            RuntimeError: If the router exits early or never becomes reachable.
        """
        deadline = time.monotonic() + float(self.cfg["Router"]["startup_timeout"])
        with httpx.Client() as client:
            while time.monotonic() < deadline:
                if self._proc is not None and self._proc.poll() is not None:
                    raise RuntimeError(f"router exited during startup\n{self.tail_log()}")
                try:
                    if client.get(f"{self.base_url}/router", timeout=2.0).status_code == 200:
                        return
                except httpx.HTTPError:
                    pass
                time.sleep(0.25)
        raise RuntimeError(f"router never answered {self.base_url}/router\n{self.tail_log()}")

    def stop(self) -> None:
        """Stops the router with SIGTERM, then SIGKILLs whatever is left of its group."""
        if self._proc is None:
            return
        try:
            pgid = os.getpgid(self._proc.pid)
        except ProcessLookupError:
            pgid = None
        if self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=float(self.cfg["Router"]["shutdown_timeout"]))
            except subprocess.TimeoutExpired:
                pass
        if pgid is not None:
            try:
                os.killpg(pgid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        self._proc.wait()
        if self._log is not None:
            self._log.close()

    def tail_log(self, lines: int = 40) -> str:
        """
        Returns the tail of the router's captured output

        Args:
            lines: How many trailing lines to include.

        Returns:
            The tail, prefixed with the log path.
        """
        try:
            text = self.log_path.read_text(errors="replace").splitlines()
        except OSError:
            return f"(no log at {self.log_path})"
        return "\n".join([f"--- {self.log_path} ---", *text[-lines:]])


@dataclass
class StreamRecord:
    """One streamed completion, timestamped at every token that arrived."""

    index: int
    model: str
    prompt: str = ""
    status: int = 0
    served_model: str = ""
    finish_reason: str = ""
    canceled: bool = False
    text: str = ""
    error: str = ""
    t_send: float = 0.0
    t_end: float = 0.0
    chunk_times: list[float] = field(default_factory=list)
    prompt_n: int = 0
    predicted_n: int = 0
    cache_n: int = 0

    @property
    def t_first(self) -> float:
        """When the first token arrived, or 0.0 if none did."""
        return self.chunk_times[0] if self.chunk_times else 0.0

    @property
    def t_last(self) -> float:
        """When the last token arrived, or 0.0 if none did."""
        return self.chunk_times[-1] if self.chunk_times else 0.0

    @property
    def produced(self) -> int:
        """Generated tokens, from the reported timings or from the chunks seen."""
        return self.predicted_n or len(self.chunk_times)

    @property
    def gen_seconds(self) -> float:
        """Seconds between the first and the last token."""
        return max(self.t_last - self.t_first, 1e-9)

    @property
    def tokens_per_second(self) -> float:
        """Decode rate over the generation window, excluding prompt processing."""
        return self.produced / self.gen_seconds

    @property
    def wall_seconds(self) -> float:
        """Seconds from sending the request to the end of its stream."""
        return max(self.t_end - self.t_send, 1e-9)

    @property
    def ok(self) -> bool:
        """True when the request returned 200 and produced at least two tokens."""
        return self.status == 200 and not self.error and len(self.chunk_times) >= 2



def _read_counts(record: StreamRecord, data: dict[str, Any]) -> None:
    """
    Copies llama.cpp's token counts off one SSE chunk when it carries them

    Args:
        record: The record being filled.
        data: One parsed SSE payload.
    """
    timings = data.get("timings") or {}
    usage = data.get("usage") or {}
    if not timings and not usage:
        return
    cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
    record.cache_n = int(timings.get("cache_n", cached) or 0)
    record.prompt_n = int(
        timings.get("prompt_n", max(int(usage.get("prompt_tokens", 0)) - record.cache_n, 0)) or 0
    )
    record.predicted_n = int(timings.get("predicted_n", usage.get("completion_tokens", 0)) or 0)


def _consume_line(record: StreamRecord, line: str) -> None:
    """
    Timestamps one SSE line and folds its content into the record

    Args:
        record: The record being filled.
        line: One raw line of the event stream.
    """
    if not line.startswith("data:"):
        return
    payload = line[5:].strip()
    if not payload or payload == "[DONE]":
        return
    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        return
    record.served_model = data.get("model") or record.served_model
    choices = data.get("choices") or []
    record.finish_reason = (choices[0].get("finish_reason") if choices else None) or record.finish_reason
    delta = choices[0].get("delta") or {} if choices else {}
    content = delta.get("content") or delta.get("reasoning_content") or ""
    if content:
        record.chunk_times.append(time.monotonic())
        record.text += content
    _read_counts(record, data)


async def stream_chat(
    base_url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    index: int = 0,
    timeout: float = 900.0,
    force_length: bool = True,
    stop_after: float | None = None,
) -> StreamRecord:
    """
    Streams one chat completion through the router and timestamps every token

    Args:
        base_url: The router's API root.
        model: The model id to request.
        prompt: The user message.
        max_tokens: The generation cap.
        index: The caller's label for this request, kept in the record.
        timeout: Seconds before the client gives up on the stream.
        force_length: Sends ignore_eos so every request generates max_tokens.
        stop_after: Seconds after sending to drop the connection mid stream.

    Returns:
        The record, with an error field set instead of raising on failure.
    """
    body: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
        "temperature": 0.7,
    }
    if force_length:
        body["ignore_eos"] = True
    record = StreamRecord(index=index, model=model, prompt=prompt)
    record.t_send = time.monotonic()
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            async with client.stream(
                "POST", f"{base_url}/v1/chat/completions", json=body
            ) as resp:
                record.status = resp.status_code
                if resp.status_code != 200:
                    record.error = (await resp.aread()).decode(errors="replace")[:400]
                else:
                    async for line in resp.aiter_lines():
                        _consume_line(record, line)
                        if stop_after is not None and time.monotonic() - record.t_send >= stop_after:
                            # Leaving the block unread closes the socket, which is the disconnect
                            record.canceled = True
                            break
    except (httpx.HTTPError, asyncio.TimeoutError) as exc:
        record.error = f"{type(exc).__name__}: {exc}"
    record.t_end = time.monotonic()
    return record


async def router_state(base_url: str) -> dict[str, Any]:
    """
    Reads the router's own view of its status and live instances

    Args:
        base_url: The router's API root.

    Returns:
        The parsed /router body.
    """
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.get(f"{base_url}/router")
        resp.raise_for_status()
        return resp.json()


async def load_model(base_url: str, model: str, timeout: float) -> None:
    """
    Loads one model and waits for the router to finish the swap

    Args:
        base_url: The router's API root.
        model: The model id to make resident.
        timeout: Seconds allowed for the load.

    Raises:
        RuntimeError: If the router reports the load failed.
    """
    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(f"{base_url}/models/load", json={"model": model})
    if resp.status_code != 200 or not resp.json().get("success"):
        raise RuntimeError(f"loading {model} failed: HTTP {resp.status_code} {resp.text[:400]}")


async def unload_model(base_url: str, model: str, timeout: float) -> None:
    """
    Unloads one model, ignoring a model that is not resident

    Args:
        base_url: The router's API root.
        model: The model id to evict.
        timeout: Seconds allowed for the unload.
    """
    async with httpx.AsyncClient(timeout=timeout) as client:
        await client.post(f"{base_url}/models/unload", json={"model": model})


async def replica_port(base_url: str, model: str) -> int | None:
    """
    Finds the port of a live replica hosting one model

    Args:
        base_url: The router's API root.
        model: The model id to look for.

    Returns:
        The replica's port, or None when the model has none.
    """
    state = await router_state(base_url)
    ports = [int(p) for p, mid in state.get("instances", {}).items() if mid == model]
    return min(ports) if ports else None


def _busy(slot: dict[str, Any]) -> bool:
    """
    Decides whether one slot from llama.cpp's /slots is generating

    Args:
        slot: One entry of the /slots array.

    Returns:
        True while the slot holds a task.
    """
    if "is_processing" in slot:
        return bool(slot["is_processing"])
    return int(slot.get("state", 0)) != 0


async def slots_snapshot(
    client: httpx.AsyncClient, port: int, model: str
) -> list[dict[str, Any]] | None:
    """
    Reads one replica's slot array, which is keyed by model on this llama.cpp build

    The spawned process is llama.cpp's own router, so /slots without a model name
    is a 400 and the reply describes the child that actually holds the weights.

    Args:
        client: The client used for the probe.
        port: The replica's port.
        model: The model id whose slots to read.

    Returns:
        The slot entries, or None when the endpoint does not answer.
    """
    try:
        resp = await client.get(f"http://127.0.0.1:{port}/slots", params={"model": model})
    except httpx.HTTPError:
        return None
    if resp.status_code != 200:
        return None
    try:
        slots = resp.json()
    except json.JSONDecodeError:
        return None
    return slots if isinstance(slots, list) and slots else None


async def slot_count(port: int, model: str) -> int | None:
    """
    Reads how many server slots one replica actually came up with

    Args:
        port: The replica's port.
        model: The model id whose slots to count.

    Returns:
        The slot count, or None when the replica does not report them.
    """
    async with httpx.AsyncClient(timeout=5.0) as client:
        slots = await slots_snapshot(client, port, model)
    return len(slots) if slots is not None else None


class SlotMonitor:
    """
    Samples a replica's slot occupancy for the life of an async block

    A router that serializes leaves one slot busy while the rest sit idle, so
    this separates a stalled queue from a replica that only has one slot.
    """

    def __init__(self, port: int | None, model: str, interval: float = 0.1) -> None:
        self.port = port
        self.model = model
        self.interval = interval
        self.available = False
        self.max_busy = 0
        self.total_slots = 0
        self.samples = 0
        self.history: list[tuple[float, int]] = []
        self._task: asyncio.Task[None] | None = None

    async def _run(self) -> None:
        """Polls the replica until the block it guards exits."""
        if self.port is None:
            return
        async with httpx.AsyncClient() as client:
            while True:
                slots = await slots_snapshot(client, self.port, self.model)
                if slots is not None:
                    busy = sum(1 for slot in slots if _busy(slot))
                    self.available = True
                    self.samples += 1
                    self.total_slots = len(slots)
                    self.max_busy = max(self.max_busy, busy)
                    self.history.append((time.monotonic(), busy))
                await asyncio.sleep(self.interval)

    async def __aenter__(self) -> "SlotMonitor":
        """Starts the sampling task."""
        self._task = asyncio.create_task(self._run())
        return self

    async def __aexit__(self, *exc: object) -> None:
        """Stops the sampling task."""
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass


def busy_drop_after(history: list[tuple[float, int]], stamp: float, slots: int) -> float:
    """
    Reports how long after stamp the replica first had a free slot

    Args:
        history: The slot monitor's samples.
        stamp: The moment the connection was dropped.
        slots: The replica's slot count.

    Returns:
        Seconds until the first sample below full occupancy, or -1.0 if there was none.
    """
    for when, busy in history:
        if when >= stamp and busy < slots:
            return when - stamp
    return -1.0


def overlap_series(records: list[StreamRecord], step: float) -> list[int]:
    """
    Counts how many streams were emitting tokens at each sampled instant

    A stream counts from its first token to its last one, so a request waiting
    in a queue contributes nothing.

    Args:
        records: The streams to measure.
        step: The sampling period in seconds.

    Returns:
        One count per sample, empty when no stream produced two tokens.
    """
    spans = [(r.t_first, r.t_last) for r in records if len(r.chunk_times) >= 2]
    if not spans:
        return []
    start = min(span[0] for span in spans)
    end = max(span[1] for span in spans)
    series: list[int] = []
    now = start
    while now <= end:
        series.append(sum(1 for lo, hi in spans if lo <= now <= hi))
        now += step
    return series


def max_overlap(records: list[StreamRecord], step: float) -> int:
    """
    Reports the largest number of streams that ever generated at the same time

    Args:
        records: The streams to measure.
        step: The sampling period in seconds.

    Returns:
        The peak concurrency, 0 when nothing generated.
    """
    return max(overlap_series(records, step), default=0)


def mean_overlap(records: list[StreamRecord], step: float) -> float:
    """
    Reports the average concurrency over the window where anything generated

    Args:
        records: The streams to measure.
        step: The sampling period in seconds.

    Returns:
        The mean count over active samples, 0.0 when nothing generated.
    """
    active = [count for count in overlap_series(records, step) if count]
    return sum(active) / len(active) if active else 0.0


def format_records(records: list[StreamRecord], origin: float) -> str:
    """
    Renders the per request timing table the tests print

    Args:
        records: The streams to tabulate.
        origin: The time all columns are measured from.

    Returns:
        A table with one row per request.
    """
    head = (f"{'req':>3}  {'model':<16} {'sent':>7} {'first':>7} {'last':>7} "
            f"{'tok':>5} {'tok/s':>7}  {'finish':<10}")
    rows = [head, "-" * len(head)]
    for record in records:
        rows.append(
            f"{record.index:>3}  {record.model:<16} "
            f"{record.t_send - origin:>7.2f} "
            f"{(record.t_first - origin) if record.chunk_times else float('nan'):>7.2f} "
            f"{(record.t_last - origin) if record.chunk_times else float('nan'):>7.2f} "
            f"{record.produced:>5} {record.tokens_per_second:>7.1f}  "
            f"{'canceled' if record.canceled else (record.finish_reason or 'none'):<10}"
        )
        if record.error:
            rows.append(f"     error: {record.error}")
    return "\n".join(rows)
