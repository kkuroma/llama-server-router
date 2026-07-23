"""
Benchmarks prompt-processing (PP) and token-generation (TG) speed for every model
served by a llama.cpp / llama-router instance, at concurrency 1 and the model's np

Each model's parallel-slot count (np) is read from the router /models listing, and the
shared prompt.txt is halved until it fits the model's context. Every run sends the
prompt with cache_prompt=false and thinking disabled, then reads llama.cpp's
server-side `timings`; at concurrency > 1 the per-stream rates are summed into an
aggregate throughput. Results are cached per model in the output benchmark.json
(skipped on re-run) and rendered to benchmark.md as one row per model@concurrency
"""

import os
import ssl
import json
import argparse
import statistics
from datetime import datetime, timezone
from urllib.parse import quote
from urllib.request import Request, urlopen
from concurrent.futures import ThreadPoolExecutor

TG_TOKENS = 1024
LOAD_TIMEOUT = 900
BENCH_TIMEOUT = 900
HERE = os.path.dirname(os.path.abspath(__file__))
PROMPT_PATH = os.path.join(HERE, "prompt.txt")


def now() -> str:
    """
    Returns the current UTC time as an ISO-8601 string

    Formats the wall clock in UTC with a trailing Z for use as a timestamp

    Args:
        (none)

    Returns:
        The current timestamp string, e.g. 2026-07-23T06:11:19Z
    """
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def httpJson(url: str, payload: dict, timeout: int) -> dict:
    """
    Performs an HTTP request and decodes the JSON response

    Sends a GET when payload is empty, otherwise POSTs it as a JSON body; TLS
    verification is disabled so self-signed https instances still work

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


def listModels(instance_url: str) -> list:
    """
    Fetches the model ids advertised by an instance

    Queries GET /v1/models and pulls the id field from each entry

    Args:
        instance_url (str): The instance base URL to query

    Returns:
        The list of model id strings served by the instance
    """
    data = httpJson(f"{instance_url}/v1/models", {}, 30)
    return [row["id"] for row in data["data"]]


def readPrompt() -> str:
    """
    Reads the shared benchmark prompt from prompt.txt beside this script

    Args:
        (none)

    Returns:
        The prompt text reused for every run

    Raises:
        FileNotFoundError: If prompt.txt is missing from the script directory
    """
    with open(PROMPT_PATH) as handle:
        return handle.read()


def callModel(instance_url: str, model: str, prompt: str, max_tokens: int, timeout: int) -> dict:
    """
    Sends one non-streaming chat completion and returns the parsed response

    Disables thinking, forces the full token budget (ignore_eos) and a full prompt
    reprocess (cache_prompt=false), giving clean PP/TG samples

    Args:
        instance_url (str)  : The instance base URL to send to
        model (str)         : The model id to route the request to
        prompt (str)        : The user message content to process
        max_tokens (int)    : The number of tokens to generate
        timeout (int)       : The per-request socket timeout in seconds

    Returns:
        The decoded JSON response, including the llama.cpp `timings` block
    """
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "ignore_eos": True,
        "cache_prompt": False,
        "chat_template_kwargs": {"enable_thinking": False},
        "stream": False,
    }
    return httpJson(f"{instance_url}/v1/chat/completions", payload, timeout)


def loadModel(instance_url: str, model: str) -> None:
    """
    Forces the instance to load a model and blocks until it is resident

    Issues a tiny warmup completion whose return implies the weights finished
    loading onto the GPUs, so later timings exclude load cost

    Args:
        instance_url (str)  : The instance base URL to send to
        model (str)         : The model id to load

    Returns:
        None
    """
    callModel(instance_url, model, "ok", 1, LOAD_TIMEOUT)


def npFromArgs(args: list) -> int:
    """
    Reads the parallel-slot count from a llama-server argument list

    Scans for the --parallel/-np flag and parses the value that follows it,
    defaulting to 1 (llama-server's default) when the flag is absent

    Args:
        args (list): The llama-server launch arguments as a list of strings

    Returns:
        The number of parallel decode slots the flag requests, or 1
    """
    for flag in ("--parallel", "-np", "--np"):
        if flag in args:
            try:
                return int(args[args.index(flag) + 1])
            except (IndexError, ValueError):
                pass
    return 1


def getNp(instance_url: str, model: str) -> int:
    """
    Reads a model's parallel-slot count (np) without loading the model

    Prefers the router /models listing, reading --parallel from the model's
    launch args so a tight-GPU model is never spawned twice; falls back to a
    cold /props?model=<id> query for total_slots, then to 1

    Args:
        instance_url (str)  : The instance base URL to query
        model (str)         : The model id to inspect

    Returns:
        The number of parallel decode slots the model was launched with
    """
    try:
        data = httpJson(f"{instance_url}/models", {}, 30)
        for row in data.get("data", []):
            if row.get("id") == model:
                return npFromArgs(row.get("status", {}).get("args", []))
    except Exception:
        pass
    try:
        data = httpJson(f"{instance_url}/props?model={quote(model, safe='')}", {}, 30)
        slots = data.get("total_slots")
        return slots if isinstance(slots, int) and slots > 0 else 1
    except Exception:
        return 1


def fitPrompt(instance_url: str, model: str, prompt: str) -> tuple:
    """
    Halves the prompt until a probe request fits the model's context

    Sends a small generation and, on any failure, halves the text and retries, so
    an oversized prompt shrinks to the largest power-of-two fraction that fits

    Args:
        instance_url (str)  : The instance base URL to send to
        model (str)         : The loaded model id to probe
        prompt (str)        : The full candidate prompt text

    Returns:
        A tuple of the working prompt and its measured token count

    Raises:
        Exception: If even a minimal prompt fails to process
    """
    current = prompt
    while True:
        try:
            data = callModel(instance_url, model, current, 8, BENCH_TIMEOUT)
            return current, data["timings"]["prompt_n"]
        except Exception:
            if len(current) < 256:
                raise
            current = current[:len(current) // 2]


def runConcurrent(instance_url: str, model: str, prompt: str, concurrency: int, n_runs: int) -> dict:
    """
    Measures aggregate PP/TG throughput at one concurrency level

    Repeats n_runs experiments; each fires `concurrency` simultaneous requests and
    sums their server-side rates, so the totals reflect batched throughput

    Args:
        instance_url (str)  : The instance base URL to send to
        model (str)         : The loaded model id to benchmark
        prompt (str)        : The fitted prompt text to process each request
        concurrency (int)   : The number of simultaneous requests per experiment
        n_runs (int)        : The number of experiments to run

    Returns:
        A dict of mean/std aggregate PP and TG throughput plus raw per-run totals
    """
    pp_agg = []
    tg_agg = []
    for _ in range(n_runs):
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = [pool.submit(callModel, instance_url, model, prompt, TG_TOKENS, BENCH_TIMEOUT) for _ in range(concurrency)]
            timings = [future.result()["timings"] for future in futures]
        pp_agg.append(round(sum(row["prompt_per_second"] for row in timings), 2))
        tg_agg.append(round(sum(row["predicted_per_second"] for row in timings), 2))
    return {
        "pp_tps_mean": round(statistics.mean(pp_agg), 2),
        "pp_tps_std": round(statistics.pstdev(pp_agg), 2),
        "tg_tps_mean": round(statistics.mean(tg_agg), 2),
        "tg_tps_std": round(statistics.pstdev(tg_agg), 2),
        "pp_agg_runs": pp_agg,
        "tg_agg_runs": tg_agg,
    }


def benchOne(instance_url: str, model: str, prompt: str, n_runs: int) -> dict:
    """
    Benchmarks one model at concurrency 1 and its np

    Reads the slot count while the model is still cold, loads it, halves the
    prompt to fit its context, then measures throughput at each concurrency level

    Args:
        instance_url (str)  : The instance base URL to send to
        model (str)         : The model id to benchmark
        prompt (str)        : The full candidate prompt text
        n_runs (int)        : The number of experiments per concurrency level

    Returns:
        A dict holding the slot count, fitted prompt size and per-level stats
    """
    np_count = getNp(instance_url, model)
    loadModel(instance_url, model)
    fitted, prompt_n = fitPrompt(instance_url, model, prompt)
    levels = sorted({1, np_count})
    concurrency = {
        str(level): runConcurrent(instance_url, model, fitted, level, n_runs) for level in levels
    }
    return {
        "np": np_count,
        "prompt_tokens": prompt_n,
        "concurrency": concurrency,
        "benchmarked_at": now(),
    }


def loadJson(path: str) -> dict:
    """
    Loads a JSON file, returning a fresh skeleton when it is absent

    Args:
        path (str): The file path to read

    Returns:
        The decoded dict, or a {"meta": {}, "results": {}} skeleton
    """
    if not os.path.exists(path):
        return {"meta": {}, "results": {}}
    with open(path) as handle:
        return json.load(handle)


def saveJson(path: str, data: dict) -> None:
    """
    Writes a dict to disk as indented JSON

    Args:
        path (str)  : The destination file path
        data (dict) : The structure to serialise

    Returns:
        None
    """
    with open(path, "w") as handle:
        json.dump(data, handle, indent=2)


def writeReport(data: dict, path: str) -> None:
    """
    Renders the benchmark results to a Markdown table

    Emits a heading with the run parameters, then one row per model@concurrency;
    PP/TG are aggregate tok/s across that row's stream count, errors show a note

    Args:
        data (dict) : The full benchmark structure (meta + results)
        path (str)  : The destination Markdown path

    Returns:
        None
    """
    meta = data["meta"]
    head = (
        f"Instance `{meta.get('instance_url')}` · {meta.get('tg_tokens')} gen tokens · "
        f"{meta.get('n_runs')} runs/model · thinking {meta.get('thinking')} · "
        f"PP/TG = aggregate tok/s across the row's streams · generated {meta.get('generated_at')}"
    )
    lines = [
        "# llama-router model benchmark",
        "",
        head,
        "",
        "| Run | PP (tok/s) | TG (tok/s) | Prompt tokens |",
        "| --- | ---------- | ---------- | ------------- |",
    ]
    for model, row in data["results"].items():
        if "error" in row:
            lines.append(f"| {model} | — | — | error: {row['error'][:60]} |")
            continue
        for level in sorted(row["concurrency"], key=int):
            cur = row["concurrency"][level]
            lines.append(
                f"| {model}@{level} | {cur['pp_tps_mean']} ± {cur['pp_tps_std']} | "
                f"{cur['tg_tps_mean']} ± {cur['tg_tps_std']} | {row['prompt_tokens']} |"
            )
    with open(path, "w") as handle:
        handle.write("\n".join(lines) + "\n")


def runAll(instance_url: str, n_runs: int, output_dir: str) -> None:
    """
    Benchmarks every advertised model, skipping ones already recorded

    Loads prior results and the shared prompt, iterates the live model list, runs
    benchOne for missing models (recording errors instead of crashing) while
    saving after each, then regenerates the report into output_dir

    Args:
        instance_url (str)  : The instance base URL to benchmark
        n_runs (int)        : The number of experiments per concurrency level
        output_dir (str)    : The directory to write benchmark.json/.md into

    Returns:
        None
    """
    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, "benchmark.json")
    md_path = os.path.join(output_dir, "benchmark.md")
    prompt = readPrompt()
    data = loadJson(json_path)
    data["meta"] = {
        "instance_url": instance_url,
        "tg_tokens": TG_TOKENS,
        "n_runs": n_runs,
        "thinking": "disabled",
        "generated_at": now(),
    }
    for model in listModels(instance_url):
        if model in data["results"]:
            print(f"skip {model} (already benchmarked)")
            continue
        print(f"benchmarking {model} ...")
        try:
            data["results"][model] = benchOne(instance_url, model, prompt, n_runs)
            row = data["results"][model]
            summary = ", ".join(f"@{lv}:{row['concurrency'][lv]['tg_tps_mean']}" for lv in sorted(row["concurrency"], key=int))
            print(f"  np={row['np']} ctx={row['prompt_tokens']} TG[{summary}] tok/s")
        except Exception as exc:
            data["results"][model] = {"error": str(exc)[:200], "benchmarked_at": now()}
            print(f"  error: {exc}")
        saveJson(json_path, data)
    saveJson(json_path, data)
    writeReport(data, md_path)
    print(f"wrote {json_path} and {md_path}")


def parseArgs() -> argparse.Namespace:
    """
    Parses the command-line flags controlling the benchmark run

    Defines the required instance URL, the output directory, and the run count

    Args:
        (none)

    Returns:
        The populated namespace holding instance_url, output_dir and n_runs
    """
    parser = argparse.ArgumentParser(description="Benchmark PP/TG speed of llama.cpp models")
    parser.add_argument("--instance-url", required=True, help="Instance base URL, e.g. http://10.10.30.29:11434")
    parser.add_argument("--output-dir", default="./", help="Directory for benchmark.json/.md (default ./)")
    parser.add_argument("--n-runs", type=int, default=10, help="Experiments per concurrency level (default 10)")
    return parser.parse_args()


def main() -> None:
    """
    Entry point that parses flags and runs the benchmark

    Args:
        (none)

    Returns:
        None
    """
    args = parseArgs()
    runAll(args.instance_url.rstrip("/"), args.n_runs, args.output_dir)


if __name__ == "__main__":
    main()
