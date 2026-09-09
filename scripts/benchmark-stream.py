#!/usr/bin/env python3
"""
One heavy streaming user vs several light users on the other GPUs

Simulates the real case: one user runs long generations on the all-GPU model
(big prompt, many response tokens) while other users send short requests to the
models pinned to the remaining GPUs. It measures how long those other users wait
for their first token while a long generation is already in flight.

The point of interest is exactly the contention the operator asked about: a
request wants a model whose GPUs are busy generating a long response, so its load
has to wait for that generation to drain (or for the starve timeout to force it).
Run it with QUEUE_FORCE_LOAD_TIMEOUT set to 300 to see that bound.

Outputs to outputs/benchmark-stream.json.
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

HERE = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(os.path.dirname(HERE), "outputs")
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "benchmark-stream.json")
SHORT_PROMPT = "Hello, how are you?"


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


def big_prompt(words: int) -> str:
    """
    Builds a filler prompt of roughly the requested word count

    Repeats a fixed sentence so the heavy user sends a large, cache-cold prompt
    without needing any external text

    Args:
        words (int): The approximate number of words the prompt should contain

    Returns:
        The filler prompt string
    """
    sentence = "The quick brown fox jumps over the lazy dog and then keeps running. "
    per = len(sentence.split())
    return (sentence * (words // per + 1)).strip()


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


def stream_request(base_url: str, model: str, prompt: str, max_tokens: int, timeout: int) -> dict:
    """
    Sends one streaming completion and times the first token and the whole run

    Reads the SSE stream, marking the first content delta as time-to-first-token
    and counting deltas as an approximate token count. Never raises: failures come
    back as a labelled result so the run always accounts for every request.

    Args:
        base_url (str)  : The router base URL to send to
        model (str)     : The model id to route the request to
        prompt (str)    : The user message content to process
        max_tokens (int): The response token budget
        timeout (int)   : The socket timeout in seconds (waiting for first bytes)

    Returns:
        A result dict with ok, model, ttft, total, tokens or an error label
    """
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": True,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    body = json.dumps(payload).encode()
    req = Request(f"{base_url}/v1/chat/completions", data=body, headers={"Content-Type": "application/json"})
    ctx = ssl._create_unverified_context() if base_url.startswith("https") else None
    start = time.monotonic()
    ttft = None
    tokens = 0
    try:
        with urlopen(req, timeout=timeout, context=ctx) as resp:
            for raw in resp:
                line = raw.decode("utf-8", "ignore").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    obj = json.loads(data)
                except ValueError:
                    continue
                delta = obj.get("choices", [{}])[0].get("delta", {}).get("content")
                if delta:
                    if ttft is None:
                        ttft = time.monotonic() - start
                    tokens += 1
        total = time.monotonic() - start
        if ttft is None:
            return {"ok": False, "model": model, "error": "no_tokens", "total": total}
        return {"ok": True, "model": model, "ttft": round(ttft, 3), "total": round(total, 3), "tokens": tokens}
    except (HTTPError, URLError, TimeoutError, socket.timeout, OSError, ValueError) as exc:
        return {"ok": False, "model": model, "error": classify_error(exc), "total": round(time.monotonic() - start, 3)}
    except Exception as exc:
        return {"ok": False, "model": model, "error": type(exc).__name__, "total": round(time.monotonic() - start, 3)}


def hog_loop(stop: threading.Event, base_url: str, model: str, prompt: str, max_tokens: int, timeout: int, out: list) -> None:
    """
    Keeps one heavy streaming request in flight until the run stops

    Sends the big-prompt request back to back so the all-GPU model stays resident
    and busy for the whole run, which is the pressure the light users push against

    Args:
        stop (threading.Event)  : Set when the run duration elapses
        base_url (str)          : The router base URL
        model (str)             : The heavy model id (the all-GPU one)
        prompt (str)            : The big filler prompt
        max_tokens (int)        : The heavy response token budget
        timeout (int)           : The socket timeout in seconds
        out (list)              : The list to append per-request results to

    Returns:
        None
    """
    while not stop.is_set():
        out.append(stream_request(base_url, model, prompt, max_tokens, timeout))


def other_worker(stop: threading.Event, base_url: str, models: list, timeout: int, out: list, lock: threading.Lock, counter: list) -> None:
    """
    Sends short streaming requests to the other GPUs' models until the run stops

    Round robins across the light models by a shared counter so all of them get
    hit evenly, recording each request's time-to-first-token

    Args:
        stop (threading.Event)  : Set when the run duration elapses
        base_url (str)          : The router base URL
        models (list)           : The light model ids (the 1 and 2 GPU models)
        timeout (int)           : The socket timeout in seconds
        out (list)              : The shared list to append results to
        lock (threading.Lock)   : Guards the round-robin counter and the out list
        counter (list)          : A single-element mutable round-robin counter

    Returns:
        None
    """
    while not stop.is_set():
        with lock:
            idx = counter[0]
            counter[0] += 1
        model = models[idx % len(models)]
        result = stream_request(base_url, model, SHORT_PROMPT, 32, timeout)
        with lock:
            out.append(result)


def summarize_light(results: list) -> dict:
    """
    Reduces the light users' results into first-token latency metrics

    Args:
        results (list): The per-request result dicts from the light workers

    Returns:
        A dict of success, first-token latency distribution, errors and per-model counts
    """
    oks = [r for r in results if r["ok"]]
    fails = [r for r in results if not r["ok"]]
    ttfts = sorted(r["ttft"] for r in oks)
    per_model = Counter(r["model"] for r in results)
    per_model_ok = Counter(r["model"] for r in oks)

    def pct(p: float) -> float:
        if not ttfts:
            return 0.0
        return round(ttfts[min(len(ttfts) - 1, int(round(p / 100 * (len(ttfts) - 1))))], 3)

    return {
        "requests": len(results),
        "succeeded": len(oks),
        "failed": len(fails),
        "success_rate": round(len(oks) / len(results), 4) if results else 0.0,
        "ttft_seconds": {
            "mean": round(statistics.mean(ttfts), 3) if ttfts else 0.0,
            "p50": pct(50),
            "p90": pct(90),
            "p99": pct(99),
            "max": ttfts[-1] if ttfts else 0.0,
        },
        "errors": dict(Counter(r["error"] for r in fails)),
        "per_model": {m: {"sent": per_model[m], "ok": per_model_ok[m]} for m in sorted(per_model)},
    }


def summarize_hog(results: list) -> dict:
    """
    Reduces the heavy user's results into throughput metrics

    Args:
        results (list): The per-request result dicts from the heavy loop

    Returns:
        A dict of completed count, mean generation time and mean tokens per second
    """
    oks = [r for r in results if r["ok"]]
    tps = [r["tokens"] / r["total"] for r in oks if r["total"] > 0]
    return {
        "completed": len(oks),
        "failed": len(results) - len(oks),
        "mean_total_seconds": round(statistics.mean([r["total"] for r in oks]), 2) if oks else 0.0,
        "mean_tokens": round(statistics.mean([r["tokens"] for r in oks]), 1) if oks else 0.0,
        "mean_tps": round(statistics.mean(tps), 2) if tps else 0.0,
    }


def run(base_url: str, hog_model: str, other_models: list, duration: int, others: int, words: int, hog_max_tokens: int, timeout: int) -> dict:
    """
    Runs the heavy-vs-light contention scenario for a fixed duration

    Starts one heavy streaming loop and `others` light workers, lets them run for
    `duration` seconds, then stops and summarizes both sides

    Args:
        base_url (str)      : The router base URL
        hog_model (str)     : The all-GPU model the heavy user streams
        other_models (list) : The light models on the remaining GPUs
        duration (int)      : How many seconds to run
        others (int)        : How many concurrent light users to simulate
        words (int)         : The heavy prompt word count
        hog_max_tokens (int): The heavy response token budget
        timeout (int)       : The per-request socket timeout in seconds

    Returns:
        A dict with the light and heavy summaries plus run parameters
    """
    stop = threading.Event()
    hog_out: list = []
    light_out: list = []
    lock = threading.Lock()
    counter = [0]
    prompt = big_prompt(words)

    threads = [threading.Thread(target=hog_loop, args=(stop, base_url, hog_model, prompt, hog_max_tokens, timeout, hog_out), daemon=True)]
    for _ in range(others):
        threads.append(threading.Thread(target=other_worker, args=(stop, base_url, other_models, timeout, light_out, lock, counter), daemon=True))
    for t in threads:
        t.start()

    start = time.monotonic()
    while time.monotonic() - start < duration:
        with lock:
            done = len(light_out)
            ok = sum(1 for r in light_out if r["ok"])
        elapsed = time.monotonic() - start
        print(f"\r  {int(elapsed)}/{duration}s, light {ok}/{done} served, heavy {len(hog_out)} done", end="", flush=True)
        time.sleep(1)
    stop.set()
    print("\n  draining in-flight requests ...")
    for t in threads:
        t.join(timeout=timeout + 5)

    return {
        "params": {
            "hog_model": hog_model,
            "other_models": other_models,
            "duration": duration,
            "other_users": others,
            "hog_prompt_words": words,
            "hog_max_tokens": hog_max_tokens,
        },
        "light_users": summarize_light(light_out),
        "heavy_user": summarize_hog(hog_out),
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
    Parses the command-line flags for the heavy-vs-light run

    Args:
        (none)

    Returns:
        The populated namespace
    """
    parser = argparse.ArgumentParser(description="One heavy streaming user vs several light users on the other GPUs")
    parser.add_argument("--host", default="localhost:11434", help="Router host, host:port or URL (default localhost:11434)")
    parser.add_argument("--hog-model", default="wordslop-qwen-3-6-27b", help="The all-GPU model the heavy user streams")
    parser.add_argument("--other-models", default="gemma-4-26b-a4b,gemma-4-12b", help="Comma-separated light models on the other GPUs")
    parser.add_argument("--duration", type=int, default=360, help="Run duration in seconds (default 360, past the 300s starve timeout)")
    parser.add_argument("--others", type=int, default=4, help="Concurrent light users (default 4)")
    parser.add_argument("--words", type=int, default=10000, help="Heavy prompt word count (default 10000)")
    parser.add_argument("--hog-max-tokens", type=int, default=2048, help="Heavy response token budget (default 2048)")
    parser.add_argument("--timeout", type=int, default=600, help="Per-request socket timeout in seconds (default 600)")
    return parser.parse_args()


def main() -> None:
    """
    Entry point that runs the contention scenario and records it

    Args:
        (none)

    Returns:
        None
    """
    args = parse_args()
    base_url = normalize_base(args.host)
    other_models = [m.strip() for m in args.other_models.split(",") if m.strip()]
    print(f"heavy user on {args.hog_model} ({args.words} words, {args.hog_max_tokens} tok) "
          f"vs {args.others} light users on {other_models}, {args.duration}s")
    result = run(base_url, args.hog_model, other_models, args.duration, args.others, args.words, args.hog_max_tokens, args.timeout)
    light = result["light_users"]
    heavy = result["heavy_user"]
    ttft = light["ttft_seconds"]
    print(f"  light : {light['succeeded']}/{light['requests']} served ({light['success_rate'] * 100:.1f}%), "
          f"first-token mean {ttft['mean']}s p50 {ttft['p50']}s p90 {ttft['p90']}s p99 {ttft['p99']}s max {ttft['max']}s")
    if light["errors"]:
        print(f"  light errors: {light['errors']}")
    print(f"  light per model: {light['per_model']}")
    print(f"  heavy : {heavy['completed']} done, {heavy['mean_tokens']} tok in {heavy['mean_total_seconds']}s ({heavy['mean_tps']} tok/s)")
    data = {
        "host": base_url,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "result": result,
    }
    save_json(OUTPUT_PATH, data)
    print(f"wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
