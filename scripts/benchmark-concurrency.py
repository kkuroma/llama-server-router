#!/usr/bin/env python3
"""
Overload benchmark: fires N short requests at a router through a bounded worker
pool and reports how many completed and how long each took

The router's differentiator is that it queues rather than rejects, so a flood of
concurrent requests should come back with a near-100% success rate while latency
grows. This script proves that: it discovers the currently loaded models, round
robins a trivial "Hello, how are you?" across them, and accounts for every single
request (success, http error, timeout or transport failure) so nothing is lost.

Only the router host and N are needed; everything else defaults. Results are keyed
by N in outputs/benchmark-concurrency.json, so running 100, 1000 and 10000 in turn
builds one file with all three.
"""

import os
import ssl
import json
import time
import socket
import argparse
import statistics
from collections import Counter
from urllib.error import URLError, HTTPError
from urllib.request import Request, urlopen
from concurrent.futures import ThreadPoolExecutor, as_completed

PROMPT = "Hello, how are you?"
MAX_TOKENS = 32
DEFAULT_CONCURRENCY = 256
DEFAULT_TIMEOUT = 300
HERE = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(os.path.dirname(HERE), "outputs")
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "benchmark-concurrency.json")


def normalize_base(host: str) -> str:
    """
    Normalizes a user-supplied host into a base URL

    Prepends http:// when no scheme is present and strips any trailing slash, so
    both "10.10.30.29:11434" and "http://host:11434/" resolve to the same base

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

    Sends a GET when payload is empty, otherwise POSTs it as a JSON body; TLS
    verification is disabled so a self-signed https instance still works

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
    Finds the models to hammer, preferring the ones already resident

    Reads the router's /router endpoint and takes the distinct models behind its
    live ports; when that endpoint is absent (a non-router server) it falls back
    to the configured /v1/models listing

    Args:
        base_url (str): The router base URL to query

    Returns:
        The sorted list of model ids to distribute requests across

    Raises:
        RuntimeError: If no models can be discovered from either endpoint
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

    Turns http errors into http_<code>, timeouts into timeout, and everything else
    into a compact transport or parse label, so the summary can tally failures

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
    if isinstance(exc, ValueError):
        return "bad_json"
    return type(exc).__name__


def one_request(base_url: str, model: str, timeout: int) -> dict:
    """
    Sends one short completion and records its outcome and latency

    Never raises: any failure is caught and returned as a labelled result, so a
    single bad request can never abort the run or go uncounted

    Args:
        base_url (str)  : The router base URL to send to
        model (str)     : The model id to route the request to
        timeout (int)   : The per-request socket timeout in seconds

    Returns:
        A result dict with ok, model, latency and either the token count or error
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
        tokens = data.get("usage", {}).get("completion_tokens", 0)
        return {"ok": True, "model": model, "latency": latency, "tokens": tokens}
    except (HTTPError, URLError, TimeoutError, socket.timeout, OSError, ValueError) as exc:
        return {"ok": False, "model": model, "latency": time.monotonic() - start, "error": classify_error(exc)}
    except Exception as exc:
        # Last-resort guard: the run must account for every request, so even an
        # unforeseen error becomes a counted failure rather than a lost one.
        return {"ok": False, "model": model, "latency": time.monotonic() - start, "error": type(exc).__name__}


def run_load(base_url: str, models: list, n: int, concurrency: int, timeout: int) -> dict:
    """
    Fires n requests through a bounded pool and collects every outcome

    Submits n tasks round robined across the models; a pool of `concurrency`
    workers drains them back to back, printing live progress, so the wall time
    reflects genuine sustained overload

    Args:
        base_url (str)      : The router base URL to hammer
        models (list)       : The model ids to distribute requests across
        n (int)             : The total number of requests to send
        concurrency (int)   : The maximum number of in-flight requests
        timeout (int)       : The per-request socket timeout in seconds

    Returns:
        A dict of the summarized results plus the raw per-request outcomes
    """
    workers = min(concurrency, n)
    results = []
    done = 0
    ok = 0
    start = time.monotonic()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(one_request, base_url, models[i % len(models)], timeout) for i in range(n)]
        for future in as_completed(futures):
            row = future.result()
            results.append(row)
            done += 1
            ok += 1 if row["ok"] else 0
            if done % max(1, n // 100) == 0 or done == n:
                rate = done / max(time.monotonic() - start, 1e-9)
                print(f"\r  {done}/{n} done, {ok} ok, {rate:6.1f} req/s", end="", flush=True)
    print()
    wall = time.monotonic() - start
    return summarize(results, n, wall, workers)


def percentiles(values: list, points: list) -> dict:
    """
    Computes the requested percentiles of a list of numbers

    Uses nearest-rank on the sorted values, returning 0.0 for each point when the
    list is empty so the summary never crashes on an all-failed run

    Args:
        values (list)   : The numbers to summarize (e.g. latencies)
        points (list)   : The percentile points to compute, each in [0, 100]

    Returns:
        A dict mapping each point label (e.g. "p90") to its value, rounded
    """
    if not values:
        return {f"p{p}": 0.0 for p in points}
    ordered = sorted(values)
    out = {}
    for p in points:
        rank = min(len(ordered) - 1, int(round(p / 100 * (len(ordered) - 1))))
        out[f"p{p}"] = round(ordered[rank], 3)
    return out


def summarize(results: list, n: int, wall: float, workers: int) -> dict:
    """
    Reduces the raw per-request outcomes into headline metrics

    Splits successes from failures, tallies the failure labels, and computes the
    latency distribution and per-model counts, so the JSON captures both the
    success rate and where any losses happened

    Args:
        results (list)  : The per-request result dicts from the run
        n (int)         : The total number of requests attempted
        wall (float)    : The wall-clock seconds the whole run took
        workers (int)   : The number of concurrent workers used

    Returns:
        A dict of the summarized run metrics
    """
    oks = [r for r in results if r["ok"]]
    fails = [r for r in results if not r["ok"]]
    latencies = [r["latency"] for r in oks]
    per_model = Counter(r["model"] for r in results)
    per_model_ok = Counter(r["model"] for r in oks)
    return {
        "requests": n,
        "concurrency": workers,
        "succeeded": len(oks),
        "failed": len(fails),
        "success_rate": round(len(oks) / n, 4) if n else 0.0,
        "wall_seconds": round(wall, 2),
        "throughput_rps": round(n / wall, 2) if wall else 0.0,
        "latency_seconds": {
            "mean": round(statistics.mean(latencies), 3) if latencies else 0.0,
            **percentiles(latencies, [50, 90, 99]),
            "max": round(max(latencies), 3) if latencies else 0.0,
        },
        "errors": dict(Counter(r["error"] for r in fails)),
        "per_model": {m: {"sent": per_model[m], "ok": per_model_ok[m]} for m in sorted(per_model)},
    }


def load_json(path: str) -> dict:
    """
    Loads the results file, returning a fresh skeleton when it is absent

    Args:
        path (str): The file path to read

    Returns:
        The decoded dict, or a {"meta": {}, "runs": {}} skeleton
    """
    if not os.path.exists(path):
        return {"meta": {}, "runs": {}}
    with open(path) as handle:
        return json.load(handle)


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


def print_summary(summary: dict) -> None:
    """
    Prints a human-readable recap of one run to stdout

    Args:
        summary (dict): The summarized run metrics from summarize

    Returns:
        None
    """
    lat = summary["latency_seconds"]
    print(f"  success  : {summary['succeeded']}/{summary['requests']} ({summary['success_rate'] * 100:.2f}%)")
    print(f"  wall     : {summary['wall_seconds']}s at {summary['throughput_rps']} req/s")
    print(f"  latency  : mean {lat['mean']}s, p50 {lat['p50']}s, p90 {lat['p90']}s, p99 {lat['p99']}s, max {lat['max']}s")
    if summary["errors"]:
        print(f"  errors   : {summary['errors']}")
    print(f"  per model: {summary['per_model']}")


def parse_args() -> argparse.Namespace:
    """
    Parses the command-line flags controlling the overload run

    Only the host and N matter for a basic run; concurrency and timeout expose the
    load shape for tuning

    Args:
        (none)

    Returns:
        The populated namespace holding host, n, concurrency and timeout
    """
    parser = argparse.ArgumentParser(description="Overload a router and measure concurrent request success rate")
    parser.add_argument("--host", default="localhost:11434", help="Router host, host:port or URL (default localhost:11434)")
    parser.add_argument("-n", "--n", type=int, default=100, help="Total requests to send (default 100)")
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY, help=f"Max in-flight requests (default {DEFAULT_CONCURRENCY})")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT, help=f"Per-request timeout in seconds (default {DEFAULT_TIMEOUT})")
    parser.add_argument("--models", default=None, help="Comma-separated model ids to hammer, overriding auto-discovery of loaded models")
    return parser.parse_args()


def main() -> None:
    """
    Entry point that runs one overload benchmark and records it

    Discovers the loaded models, fires N requests at the router, prints the recap
    and stores the summary under its N key in the shared output file

    Args:
        (none)

    Returns:
        None
    """
    args = parse_args()
    base_url = normalize_base(args.host)
    models = [m.strip() for m in args.models.split(",") if m.strip()] if args.models else discover_models(base_url)
    print(f"hammering {base_url} with N={args.n} across {len(models)} model(s): {models}")
    summary = run_load(base_url, models, args.n, args.concurrency, args.timeout)
    print_summary(summary)
    data = load_json(OUTPUT_PATH)
    data["meta"] = {
        "host": base_url,
        "prompt": PROMPT,
        "max_tokens": MAX_TOKENS,
        "models": models,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    data["runs"][str(args.n)] = summary
    save_json(OUTPUT_PATH, data)
    print(f"wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
