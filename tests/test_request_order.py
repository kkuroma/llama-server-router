"""
Checks that a queued request for a second model is served in the order it arrived

Three requests arrive a tenth of a second apart, for model 0, model 1 and model 0
The router may not let the later model 0 request overtake the queued model 1 one
"""

import asyncio
import json
import time

import httpx
import pytest

from core.harness import (
    RouterProcess,
    StreamRecord,
    format_records,
    load_config,
    load_model,
    load_prompt,
    model_id,
    router_state,
    stream_chat,
    variant_prompt,
)

CFG = load_config()
ORDER = CFG["Order"]
PROMPT = load_prompt(CFG)

# The expanded sequence only runs once the small one it extends has passed
_SMALL_CASE_PASSED = False


async def _poll_residency(
    base_url: str, timeline: list[tuple[float, list[str]]], stop: asyncio.Event
) -> None:
    """
    Records which models the router reports resident, until the run ends

    Args:
        base_url: The router's API root.
        timeline: The list each sample is appended to.
        stop: The event that ends the polling.
    """
    interval = float(ORDER["poll_interval"])
    while not stop.is_set():
        try:
            state = await router_state(base_url)
            timeline.append((time.monotonic(), sorted(set(state.get("instances", {}).values()))))
        except (httpx.HTTPError, json.JSONDecodeError):
            pass
        try:
            await asyncio.wait_for(stop.wait(), timeout=interval)
        except asyncio.TimeoutError:
            pass


def _resident_runs(timeline: list[tuple[float, list[str]]]) -> list[str]:
    """
    Reduces the residency samples to the order models actually occupied the GPU

    Args:
        timeline: The (time, resident models) samples.

    Returns:
        One model id per occupancy run, with repeats and empty samples dropped.
    """
    runs: list[str] = []
    for _, models in timeline:
        if len(models) != 1:
            continue
        if not runs or runs[-1] != models[0]:
            runs.append(models[0])
    return runs


def _resident_during(
    timeline: list[tuple[float, list[str]]], start: float, end: float
) -> set[str]:
    """
    Collects every model that was resident while one request was producing tokens

    Args:
        timeline: The (time, resident models) samples.
        start: The first token's timestamp.
        end: The last token's timestamp.

    Returns:
        The union of the models resident over that window.
    """
    seen: set[str] = set()
    for stamp, models in timeline:
        if start <= stamp <= end:
            seen.update(models)
    return seen


async def _run_sequence(
    router: RouterProcess, sequence: list[int]
) -> tuple[list[StreamRecord], list[tuple[float, list[str]]]]:
    """
    Fires one request per sequence entry, gap seconds apart, and watches the swaps

    The first model is loaded up front, so the gaps between arrivals mean what
    they say instead of being swallowed by a cold load. Every request sends the
    same long essay and differs only in its closing line.

    Args:
        router: The running router under test.
        sequence: Model indices, one per request, in arrival order.

    Returns:
        The records in arrival order, and the residency samples taken alongside.
    """
    slots = int(ORDER["slots"])
    models = [model_id(CFG, index, slots) for index in sequence]
    await load_model(router.base_url, models[0], float(CFG["Router"]["load_timeout"]))

    timeline: list[tuple[float, list[str]]] = []
    stop = asyncio.Event()
    watcher = asyncio.create_task(_poll_residency(router.base_url, timeline, stop))

    async def fire(index: int) -> StreamRecord:
        """Sends request index after its share of the arrival gap."""
        await asyncio.sleep(index * float(ORDER["gap"]))
        return await stream_chat(
            router.base_url,
            models[index],
            variant_prompt(CFG, PROMPT, index),
            int(ORDER["max_tokens"]),
            index=index,
        )

    records = await asyncio.gather(*[fire(i) for i in range(len(sequence))])
    stop.set()
    await watcher
    return list(records), timeline


def _check_sequence(
    records: list[StreamRecord], timeline: list[tuple[float, list[str]]], sequence: list[int]
) -> None:
    """
    Asserts the requests were served in arrival order, each by the model it asked for

    Args:
        records: The records in arrival order.
        timeline: The residency samples taken during the run.
        sequence: Model indices, one per request, in arrival order.
    """
    slots = int(ORDER["slots"])
    models = [model_id(CFG, index, slots) for index in sequence]
    origin = min(record.t_send for record in records)
    print()
    print(format_records(records, origin))
    print(f"residency: {' -> '.join(_resident_runs(timeline))}")

    for record in records:
        assert record.ok, f"request {record.index} failed: HTTP {record.status} {record.error}"
        assert record.produced >= int(ORDER["min_tokens"]), (
            f"request {record.index} generated only {record.produced} tokens"
        )

    for record in records:
        assert record.served_model == record.model, (
            f"request {record.index} asked for {record.model} and was answered by "
            f"{record.served_model or 'an unnamed model'}"
        )
        served = _resident_during(timeline, record.t_first, record.t_last)
        assert served == {record.model}, (
            f"request {record.index} generated while {served or 'nothing'} was resident"
        )

    finish_order = [record.index for record in sorted(records, key=lambda r: r.t_end)]
    assert finish_order == list(range(len(records))), (
        f"requests finished {finish_order}, which reorders the arrival sequence {models}"
    )

    for i in range(1, len(records)):
        if models[i] == models[i - 1]:
            continue
        assert records[i].t_first > records[i - 1].t_last, (
            f"request {i} ({models[i]}) started generating "
            f"{records[i - 1].t_last - records[i].t_first:.2f}s before request {i - 1} "
            f"({models[i - 1]}) finished, so it overtook the queue"
        )

    force_load = float(CFG["Router"]["settings"]["QUEUE_FORCE_LOAD_TIMEOUT"])
    for record in records:
        waited = record.t_first - record.t_send
        assert waited < force_load, (
            f"request {record.index} waited {waited:.1f}s, so only the "
            f"{force_load}s starvation guard rescued it"
        )


@pytest.mark.gpu
@pytest.mark.slow
def test_request_order(router: RouterProcess) -> None:
    """Serves model 0, model 1 and model 0 again, and expects that order back."""
    global _SMALL_CASE_PASSED
    sequence = [int(index) for index in ORDER["sequence"]]
    records, timeline = asyncio.run(_run_sequence(router, sequence))
    _check_sequence(records, timeline, sequence)
    _SMALL_CASE_PASSED = True


@pytest.mark.gpu
@pytest.mark.slow
def test_request_order_expanded(router: RouterProcess) -> None:
    """Repeats the alternation twice more, so every swap has to happen in turn."""
    if not _SMALL_CASE_PASSED:
        pytest.skip("the three request case has to pass before the longer one runs")
    sequence = [int(index) for index in ORDER["expanded_sequence"]]
    records, timeline = asyncio.run(_run_sequence(router, sequence))
    _check_sequence(records, timeline, sequence)
