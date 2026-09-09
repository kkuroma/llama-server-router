#!/usr/bin/env python3
"""
Staggered mixed-model load that forces the router to swap

Where benchmark-concurrency.py fires everything at once (so the scheduler can
batch queued same-model requests and swap rarely), this one staggers the
requests so they arrive spread out across different models. That denies the
scheduler its batching and makes it actually swap, which is what we want to
watch: how it holds up when swaps are frequent rather than coalesced.

A monitor thread polls /router and counts each GPU entering the swapping state,
so the report shows swaps performed alongside success rate and latency.

Only the host and N are needed. Outputs to outputs/benchmark-concurrent-swap.json.
"""

import os
import ssl
import json
import time
import socket
import argparse
import statistics
import threading
from collections import Counter
from urllib.error import URLError, HTTPError
from urllib.request import Request, urlopen
from concurrent.futures import ThreadPoolExecutor

PROMPT = "Hello, how are you?"
MAX_TOKENS = 32
HERE = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(os.path.dirname(HERE), "outputs")
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "benchmark-concurrent-swap.json")


def normalize_base(host: str) -> str:
    """
    Normalizes a user-supplied host into a base URL

    Args:
        host (str): The raw host, host:port or full URL from the command line

    Returns:
        The cleaned base URL with a scheme and no trailing slash
    """
    if "://" not in host:
        host = "http://" + host
    return host.rstrip("/")


def http_json(url: str, payload: dict, timeout: int) -> dict:
    """
    Performs an HTTP request and decodes the JSON response

    Args:
        url (str)       : The absolute URL to request
        payload (dict)  : The JSON body to post, or an empty dict for a GET
        timeout (int)   : The socket timeout in seconds

    Returns:
        The decoded JSON response as a dict

    Raises:
        urllib.error.URLError: If the request fails at the transport layer
    """
    body = json.dumps(payload).encode() if payload else None
    req = Request(url, data=body, headers={"Content-Type": "application/json"})
    ctx = ssl._create_unverified_context() if url.startswith("https") else None
    with urlopen(req, timeout=timeout, context=ctx) as resp:
        return json.loads(resp.read())


def discover_models(base_url: str) -> list:
    """
    Finds the models to hit, preferring the ones already resident

    Args:
        base_url (str): The router base URL to query

    Returns:
        The sorted list of model ids to distribute requests across

    Raises:
        RuntimeError: If no models can be discovered
    """
    try:
        data = http_json(f"{base_url}/router", {}, 30)
        loaded = sorted(set(data.get("instances", {}).values()))
        if loaded:
            return loaded
    except (URLError, TimeoutError, OSError, ValueError):
        pass
    try:
        data = http_json(f"{base_url}/v1/models", {}, 30)
        return sorted(row["id"] for row in data.get("data", []))
    except (URLError, TimeoutError, OSError, ValueError) as exc:
        raise RuntimeError(f"could not discover models from {base_url}: {exc}")


def classify_error(exc: Exception) -> str:
    """
    Maps an exception to a short, groupable error label

    Args:
        exc (Exception): The exception raised while sending a request

    Returns:
        A short string label for the failure category
    """
    if isinstance(exc, HTTPError):
        return f"http_{exc.code}"
    if isinstance(exc, (TimeoutError, socket.timeout)):
        return "timeout"
    if isinstance(exc, URLError):
        reason = getattr(exc, "reason", exc)
        if isinstance(reason, (TimeoutError, socket.timeout)):
            return "timeout"
        return f"conn_{type(reason).__name__}"
    return type(exc).__name__


def one_request(base_url: str, model: str, timeout: int) -> dict:
    """
    Sends one short completion and records its outcome and latency

    Never raises: any failure is caught and returned as a labelled result

    Args:
        base_url (str)  : The router base URL to send to
        model (str)     : The model id to route the request to
        timeout (int)   : The per-request socket timeout in seconds

    Returns:
        A result dict with ok, model, latency and either token count or error
    """
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": MAX_TOKENS,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    start = time.monotonic()
    try:
        data = http_json(f"{base_url}/v1/chat/completions", payload, timeout)
        latency = time.monotonic() - start
        choices = data.get("choices")
        if not isinstance(choices, list) or not choices:
            return {"ok": False, "model": model, "latency": latency, "error": "no_choices"}
        return {"ok": True, "model": model, "latency": latency}
    except (HTTPError, URLError, TimeoutError, socket.timeout, OSError, ValueError) as exc:
        return {"ok": False, "model": model, "latency": time.monotonic() - start, "error": classify_error(exc)}
    except Exception as exc:
        return {"ok": False, "model": model, "latency": time.monotonic() - start, "error": type(exc).__name__}


class SwapMonitor:
    """
    Polls /router in the background and counts GPUs entering the swapping state

    A rising edge into "swapping" on any GPU is counted as one swap event, giving
    an approximate measure of how much swapping the staggered load caused.
    """

    def __init__(self, base_url: str, interval: float = 0.3) -> None:
        """
        Args:
            base_url (str)  : The router base URL to poll
            interval (float): Seconds between polls
        """
        self.base_url = base_url
        self.interval = interval
        self.swaps = 0
        self._stop = threading.Event()
        self._prev: dict = {}
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def _loop(self) -> None:
        """Polls gpu_status and tallies rising edges into swapping until stopped."""
        while not self._stop.is_set():
            try:
                data = http_json(f"{self.base_url}/router", {}, 5)
                for gpu, state in data.get("gpu_status", {}).items():
                    if state == "swapping" and self._prev.get(gpu) != "swapping":
                        self.swaps += 1
                    self._prev[gpu] = state
            except (URLError, TimeoutError, OSError, ValueError):
                pass
            self._stop.wait(self.interval)

    def start(self) -> None:
        """Starts the monitor thread."""
        self._thread.start()

    def stop(self) -> int:
        """Stops the monitor and returns the total swap events observed."""
        self._stop.set()
        self._thread.join(timeout=2)
        return self.swaps


def run_staggered(base_url: str, models: list, n: int, stagger: float, concurrency: int, timeout: int) -> dict:
    """
    Dispatches n requests spaced by `stagger` seconds and collects every outcome

    Round robins the requests across the models and submits them one every
    `stagger` seconds, so arrivals are spread out and the scheduler must swap
    rather than batch. A monitor counts the swaps meanwhile.

    Args:
        base_url (str)      : The router base URL
        models (list)       : The model ids to alternate across
        n (int)             : The total number of requests to send
        stagger (float)     : Seconds between successive dispatches
        concurrency (int)   : The maximum number of in-flight requests
        timeout (int)       : The per-request socket timeout in seconds

    Returns:
        A dict of the summarized results including the swap count
    """
    monitor = SwapMonitor(base_url)
    monitor.start()
    results = []
    lock = threading.Lock()
    done = 0

    def record(future_result: dict) -> None:
        nonlocal done
        with lock:
            results.append(future_result)
            done += 1

    start = time.monotonic()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = []
        for i in range(n):
            model = models[i % len(models)]
            fut = pool.submit(one_request, base_url, model, timeout)
            fut.add_done_callback(lambda f: record(f.result()))
            futures.append(fut)
            print(f"\r  dispatched {i + 1}/{n}, completed {done}", end="", flush=True)
            if i < n - 1:
                time.sleep(stagger)
        for fut in futures:
            fut.result()
    print()
    wall = time.monotonic() - start
    swaps = monitor.stop()
    return summarize(results, n, wall, swaps, concurrency, stagger)


def summarize(results: list, n: int, wall: float, swaps: int, concurrency: int, stagger: float) -> dict:
    """
    Reduces the raw outcomes into headline metrics plus the swap count

    Args:
        results (list)      : The per-request result dicts
        n (int)             : The total number of requests attempted
        wall (float)        : The wall-clock seconds the run took
        swaps (int)         : The number of swap events observed
        concurrency (int)   : The worker pool size
        stagger (float)     : The dispatch spacing used

    Returns:
        A dict of the summarized run metrics
    """
    oks = [r for r in results if r["ok"]]
    fails = [r for r in results if not r["ok"]]
    latencies = sorted(r["latency"] for r in oks)
    per_model = Counter(r["model"] for r in results)

    def pct(p: float) -> float:
        if not latencies:
            return 0.0
        return round(latencies[min(len(latencies) - 1, int(round(p / 100 * (len(latencies) - 1))))], 3)

    return {
        "requests": n,
        "stagger_seconds": stagger,
        "concurrency": concurrency,
        "swaps_observed": swaps,
        "succeeded": len(oks),
        "failed": len(fails),
        "success_rate": round(len(oks) / n, 4) if n else 0.0,
        "wall_seconds": round(wall, 2),
        "latency_seconds": {
            "mean": round(statistics.mean(latencies), 3) if latencies else 0.0,
            "p50": pct(50),
            "p90": pct(90),
            "p99": pct(99),
            "max": latencies[-1] if latencies else 0.0,
        },
        "errors": dict(Counter(r["error"] for r in fails)),
        "per_model": dict(per_model),
    }


def save_json(path: str, data: dict) -> None:
    """
    Writes a dict to disk as indented JSON, creating the parent directory

    Args:
        path (str)  : The destination file path
        data (dict) : The structure to serialize

    Returns:
        None
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as handle:
        json.dump(data, handle, indent=2)


def parse_args() -> argparse.Namespace:
    """
    Parses the command-line flags for the staggered swap run

    Args:
        (none)

    Returns:
        The populated namespace
    """
    parser = argparse.ArgumentParser(description="Staggered mixed-model load that forces the router to swap")
    parser.add_argument("--host", default="localhost:11434", help="Router host, host:port or URL (default localhost:11434)")
    parser.add_argument("-n", "--n", type=int, default=100, help="Total requests to send (default 100)")
    parser.add_argument("--stagger", type=float, default=0.5, help="Seconds between successive dispatches (default 0.5)")
    parser.add_argument("--concurrency", type=int, default=64, help="Max in-flight requests (default 64)")
    parser.add_argument("--timeout", type=int, default=600, help="Per-request timeout in seconds (default 600)")
    parser.add_argument("--models", default=None, help="Comma-separated model ids to alternate across, overriding discovery")
    return parser.parse_args()


def main() -> None:
    """
    Entry point that runs one staggered swap benchmark and records it

    Args:
        (none)

    Returns:
        None
    """
    args = parse_args()
    base_url = normalize_base(args.host)
    models = [m.strip() for m in args.models.split(",") if m.strip()] if args.models else discover_models(base_url)
    print(f"staggering N={args.n} every {args.stagger}s across {len(models)} model(s): {models}")
    summary = run_staggered(base_url, models, args.n, args.stagger, args.concurrency, args.timeout)
    lat = summary["latency_seconds"]
    print(f"  success : {summary['succeeded']}/{summary['requests']} ({summary['success_rate'] * 100:.2f}%)")
    print(f"  swaps   : {summary['swaps_observed']} observed over {summary['wall_seconds']}s")
    print(f"  latency : mean {lat['mean']}s p50 {lat['p50']}s p90 {lat['p90']}s p99 {lat['p99']}s max {lat['max']}s")
    if summary["errors"]:
        print(f"  errors  : {summary['errors']}")
    save_json(OUTPUT_PATH, {
        "host": base_url,
        "models": models,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "summary": summary,
    })
    print(f"wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
