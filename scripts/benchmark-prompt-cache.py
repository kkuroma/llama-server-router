"""
Measures how much of a prompt llama.cpp reuses instead of reprocessing, per request

Reuse is read from the server side timings as cache_n, so no reuse means cache_n = 0
The capacity pass revisits long conversations, which is what --cache-ram has to hold
"""

import os
import ssl
import json
import time
import argparse
from datetime import datetime, timezone
from urllib.request import Request, urlopen

REQUEST_TIMEOUT = 900
GEN_TOKENS = 8
FILLER_LINE = "Record {i:04d} of series {tag}: the operator logged a routine check with no fault raised."
HERE = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(os.path.dirname(HERE), "outputs")


def now() -> str:
    """
    Returns the current UTC time as an ISO-8601 string

    Args:
        (none)

    Returns:
        The current timestamp string, e.g. 2026-07-23T06:11:19Z
    """
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def httpJson(url: str, payload: dict, timeout: int = REQUEST_TIMEOUT) -> dict:
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


def filler(tag: str, lines: int, offset: int = 0) -> str:
    """
    Builds a deterministic block of prose that no two tags share a prefix of

    Args:
        tag (str)   : A label mixed into every line so two blocks tokenize differently
        lines (int) : How many lines the block holds
        offset (int): The first record number, so two blocks can differ only at the front

    Returns:
        The block as one string
    """
    return "\n".join(FILLER_LINE.format(i=i, tag=tag) for i in range(offset, offset + lines))


def conversation(tag: str, turns: int, lines: int) -> list:
    """
    Builds a chat whose first turn carries the bulk of the prompt

    Args:
        tag (str)   : The series label handed to filler
        turns (int) : How many user turns the chat holds
        lines (int) : How many filler lines the first turn carries

    Returns:
        The message list, ending on a user turn
    """
    messages = [
        {"role": "system", "content": "You are a terse log analyst. Answer in under ten words."},
        {"role": "user", "content": f"{filler(tag, lines)}\n\nHow many records are in series {tag}?"},
    ]
    for i in range(1, turns):
        messages.append({"role": "assistant", "content": f"Reply {i}."})
        messages.append({"role": "user", "content": f"Follow-up {i}: name the series again."})
    return messages


def runCase(url: str, model: str, messages: list, cachePrompt: bool = True) -> dict:
    """
    Sends one non-streaming completion and returns its prompt-side timings

    Args:
        url (str)           : The instance base URL
        model (str)         : The model id to request
        messages (list)     : The chat messages to send
        cachePrompt (bool)  : Passed through to llama.cpp as cache_prompt

    Returns:
        A dict of cache_n, prompt_n, prompt_ms, prompt_tps and the measured wall time
    """
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": GEN_TOKENS,
        "temperature": 0.0,
        "cache_prompt": cachePrompt,
    }
    started = time.perf_counter()
    reply = httpJson(f"{url}/v1/chat/completions", payload)
    timings = reply.get("timings", {})
    return {
        "cache_n": timings.get("cache_n", 0),
        "prompt_n": timings.get("prompt_n", 0),
        "prompt_ms": round(timings.get("prompt_ms", 0.0), 1),
        "prompt_tps": round(timings.get("prompt_per_second", 0.0), 1),
        "wall_s": round(time.perf_counter() - started, 2),
    }


def runReuse(url: str, model: str, lines: int) -> list:
    """
    Runs the turn-by-turn cases a chat client produces against one model

    Reloading the model between runs is what clears the cache, so the caller does that
    rather than this pass

    Args:
        url (str)   : The instance base URL
        model (str) : The model id to request
        lines (int) : Filler lines per conversation

    Returns:
        One result row per case, in the order they were sent
    """
    cases = [
        ("cold", lambda: runCase(url, model, conversation("alpha", 1, lines))),
        ("repeat", lambda: runCase(url, model, conversation("alpha", 1, lines))),
        ("continuation", lambda: runCase(url, model, conversation("alpha", 2, lines))),
        ("other-conversation", lambda: runCase(url, model, conversation("beta", 1, lines))),
        ("return-to-first", lambda: runCase(url, model, conversation("alpha", 3, lines))),
        ("cache-prompt-off", lambda: runCase(url, model, conversation("alpha", 1, lines), False)),
    ]
    rows = []
    for name, run in cases:
        row = run()
        row["case"] = name
        rows.append(row)
        print(f"[reuse] {name}: cache_n={row['cache_n']} prompt_n={row['prompt_n']} "
              f"prompt_ms={row['prompt_ms']}", flush=True)
    return rows


def runCapacity(url: str, model: str, lines: int, conversations: int) -> list:
    """
    Fills the host-side prompt cache with long conversations, then revisits each one

    A conversation whose state no longer fits reports cache_n near zero on the revisit
    and reprocesses its whole prompt, which is the signal that --cache-ram is too small

    Args:
        url (str)           : The instance base URL
        model (str)         : The model id to request
        lines (int)         : Filler lines per conversation
        conversations (int) : How many distinct conversations to hold at once

    Returns:
        One result row per fill and per revisit
    """
    tags = [f"series{i}" for i in range(conversations)]
    rows = []
    for phase, turns in (("fill", 1), ("revisit", 2)):
        for tag in tags:
            row = runCase(url, model, conversation(tag, turns, lines))
            row["case"] = f"{phase}-{tag}"
            rows.append(row)
            print(f"[capacity] {row['case']}: cache_n={row['cache_n']} "
                  f"prompt_n={row['prompt_n']} prompt_ms={row['prompt_ms']}", flush=True)
    return rows


def writeReport(data: dict, path: str) -> None:
    """
    Renders the results to a Markdown table, one row per case

    Args:
        data (dict) : The full result structure (meta + reuse + capacity)
        path (str)  : The destination Markdown path

    Returns:
        None
    """
    meta = data["meta"]
    lines = [
        "# llama-router prompt cache benchmark",
        "",
        f"Instance `{meta['instance_url']}` · model `{meta['model']}` · "
        f"{meta['filler_lines']} filler lines · {meta['conversations']} conversations · "
        f"generated {meta['generated_at']}",
        "",
    ]
    for section in ("reuse", "capacity"):
        if not data.get(section):
            continue
        lines += [
            f"## {section}",
            "",
            "| Case | Reused | Processed | Prompt ms | Prompt tok/s |",
            "| ---- | ------ | --------- | --------- | ------------ |",
        ]
        for row in data[section]:
            lines.append(
                f"| {row['case']} | {row['cache_n']} | {row['prompt_n']} | "
                f"{row['prompt_ms']} | {row['prompt_tps']} |"
            )
        lines.append("")
    with open(path, "w") as handle:
        handle.write("\n".join(lines) + "\n")


def runAll(url: str, model: str, lines: int, conversations: int, outputDir: str) -> None:
    """
    Reloads the model to clear its cache, runs both passes and writes the results

    Args:
        url (str)           : The instance base URL
        model (str)         : The model id to request
        lines (int)         : Filler lines per conversation
        conversations (int) : How many conversations the capacity pass holds
        outputDir (str)     : Directory the JSON and Markdown are written to

    Returns:
        None
    """
    os.makedirs(outputDir, exist_ok=True)
    httpJson(f"{url}/models/unload", {"model": model})
    httpJson(f"{url}/models/load", {"model": model})

    data = {
        "meta": {
            "instance_url": url,
            "model": model,
            "filler_lines": lines,
            "conversations": conversations,
            "generated_at": now(),
        },
        "reuse": runReuse(url, model, lines),
    }
    httpJson(f"{url}/models/unload", {"model": model})
    httpJson(f"{url}/models/load", {"model": model})
    data["capacity"] = runCapacity(url, model, lines, conversations)

    jsonPath = os.path.join(outputDir, "benchmark-prompt-cache.json")
    mdPath = os.path.join(outputDir, "benchmark-prompt-cache.md")
    with open(jsonPath, "w") as handle:
        json.dump(data, handle, indent=2)
    writeReport(data, mdPath)
    print(f"wrote {jsonPath} and {mdPath}")


def parseArgs() -> argparse.Namespace:
    """
    Parses the command-line flags controlling the run

    Args:
        (none)

    Returns:
        The populated namespace holding instance_url, model, sizes and output_dir
    """
    parser = argparse.ArgumentParser(description="Measure llama.cpp prompt cache reuse")
    parser.add_argument("--instance-url", required=True, help="Router base URL, e.g. http://127.0.0.1:11434")
    parser.add_argument("--model", required=True, help="Model id to benchmark")
    parser.add_argument("--filler-lines", type=int, default=900, help="Filler lines per conversation (default 900)")
    parser.add_argument("--conversations", type=int, default=6, help="Conversations held at once (default 6)")
    parser.add_argument("--output-dir", default=OUTPUT_DIR, help="Directory for the results (default ../outputs)")
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
    runAll(args.instance_url.rstrip("/"), args.model, args.filler_lines, args.conversations, args.output_dir)


if __name__ == "__main__":
    main()
