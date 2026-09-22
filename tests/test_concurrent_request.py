"""
Checks that concurrent users really do generate at the same time on one model

Every user streams, so concurrency is read off token arrival times and off the
replica's own slot occupancy, which separates a router stall from a busy replica
"""

import asyncio
from typing import Any

import pytest

from core.harness import (
    RouterProcess,
    SlotMonitor,
    StreamRecord,
    format_records,
    load_config,
    load_model,
    load_prompt,
    max_overlap,
    mean_overlap,
    model_id,
    replica_port,
    slot_count,
    stream_chat,
    variant_prompt,
)

CFG = load_config()
CONCURRENCY = CFG["Concurrency"]
STAGES: list[dict[str, Any]] = CONCURRENCY["stages"]
PROMPT = load_prompt(CFG)

# A stage only runs while every smaller one has passed
_FAILED = False


async def _run_stage(router: RouterProcess, users: int, slots: int) -> dict[str, Any]:
    """
    Loads the model at one slot count, measures it alone, then under load

    The single request baseline runs against the same loaded replica, so the
    speedup it anchors is batching alone and not a difference in model state.

    Args:
        router: The running router under test.
        users: How many users fire at once.
        slots: The llama.cpp server slot count the model preset carries.

    Returns:
        The stage measurements, keyed for the assertions and the printed table.
    """
    base_url = router.base_url
    model = model_id(CFG, 0, slots)
    await load_model(base_url, model, float(CFG["Router"]["load_timeout"]))
    port = await replica_port(base_url, model)
    reported_slots = await slot_count(port, model) if port is not None else None

    max_tokens = int(CONCURRENCY["max_tokens"])
    baseline = await stream_chat(
        base_url, model, variant_prompt(CFG, PROMPT, -1), max_tokens, index=-1,
    )

    async def user(index: int) -> StreamRecord:
        """Streams one user's request, differing from its neighbors only in the last line."""
        return await stream_chat(
            base_url, model, variant_prompt(CFG, PROMPT, index), max_tokens, index=index
        )

    async with SlotMonitor(port, model) as monitor:
        records = list(await asyncio.gather(*[user(i) for i in range(users)]))

    return {
        "model": model,
        "port": port,
        "reported_slots": reported_slots,
        "baseline": baseline,
        "records": records,
        "monitor": monitor,
    }


def _check_stage(stage: dict[str, Any], users: int, slots: int, min_speedup: float) -> None:
    """
    Asserts the stage generated min(users, slots) streams at once at full rate

    The speedup is wall clock against the same users run one at a time, because a
    long shared prompt puts prefill between the waves and dilutes a token rate.

    Args:
        stage: The measurements from one stage.
        users: How many users fired at once.
        slots: The slot count the model preset carries.
        min_speedup: How much faster than serial the batch has to finish.
    """
    step = float(CONCURRENCY["overlap_step"])
    min_tokens = int(CONCURRENCY["min_tokens"])
    records: list[StreamRecord] = stage["records"]
    baseline: StreamRecord = stage["baseline"]
    monitor: SlotMonitor = stage["monitor"]
    expected = min(users, slots)

    origin = min(record.t_send for record in records)
    peak = max_overlap(records, step)
    mean = mean_overlap(records, step)
    produced = sum(record.produced for record in records)
    batch_wall = max(record.t_end for record in records) - min(
        record.t_send for record in records
    )
    serial_wall = users * baseline.wall_seconds
    speedup = serial_wall / batch_wall if batch_wall > 0 else 0.0
    window = max(record.t_last for record in records) - min(
        record.t_first for record in records if record.chunk_times
    )
    aggregate = produced / window if window > 0 else 0.0

    print()
    print(format_records(records, origin))
    print(f"model {stage['model']} on port {stage['port']}, "
          f"slots reported {stage['reported_slots']}, wanted {slots}")
    print(f"overlap peak {peak}, mean {mean:.2f}, wanted {expected}")
    print(f"slot probe: available {monitor.available}, peak busy slots {monitor.max_busy} "
          f"of {monitor.total_slots}, samples {monitor.samples}")
    print(f"throughput: baseline {baseline.tokens_per_second:.1f} tok/s, "
          f"aggregate {aggregate:.1f} tok/s over {window:.1f}s")
    print(f"wall clock: {batch_wall:.1f}s for {users} users against {serial_wall:.1f}s "
          f"one at a time, speedup {speedup:.2f}, wanted {min_speedup}")

    assert baseline.ok, f"the single request baseline failed: HTTP {baseline.status} {baseline.error}"
    assert baseline.produced >= min_tokens, (
        f"the baseline generated only {baseline.produced} tokens"
    )
    for record in records:
        assert record.ok, f"user {record.index} failed: HTTP {record.status} {record.error}"
        assert record.produced >= min_tokens, (
            f"user {record.index} generated only {record.produced} tokens, "
            f"finish reason {record.finish_reason or 'none'}"
        )

    if stage["reported_slots"] is not None:
        assert stage["reported_slots"] == slots, (
            f"the replica came up with {stage['reported_slots']} slots, "
            f"so the preset's parallel = {slots} never reached llama.cpp"
        )

    assert peak >= expected, (
        f"at most {peak} of {users} users generated at the same time, wanted {expected}"
    )
    assert mean >= float(CONCURRENCY["mean_overlap_factor"]) * expected, (
        f"mean concurrency was {mean:.2f} against {expected} slots, "
        f"so the users mostly took turns"
    )

    if monitor.available:
        assert monitor.max_busy >= expected, (
            f"the replica never had more than {monitor.max_busy} of {monitor.total_slots} "
            f"slots busy at once, wanted {expected}"
        )

    assert speedup >= min_speedup, (
        f"{users} users took {batch_wall:.1f}s against {serial_wall:.1f}s one at a time, "
        f"a speedup of {speedup:.2f} where {min_speedup} was wanted, so the slots did not batch"
    )

    started = sorted(
        (record.t_first for record in records if record.chunk_times)
    )[:expected]
    spread = started[-1] - started[0]
    assert spread <= float(CONCURRENCY["first_token_spread"]), (
        f"the first {expected} users started {spread:.1f}s apart, so they were let in one by one"
    )


@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.parametrize(
    "stage_cfg",
    STAGES,
    ids=[f"u{stage['users']}-p{stage['slots']}" for stage in STAGES],
)
def test_concurrent_users(router: RouterProcess, stage_cfg: dict[str, Any]) -> None:
    """Fires users users at a model with slots slots and expects them to share it."""
    global _FAILED
    if _FAILED:
        pytest.skip("a smaller stage already failed, so the larger one proves nothing")
    users, slots = int(stage_cfg["users"]), int(stage_cfg["slots"])
    try:
        stage = asyncio.run(_run_stage(router, users, slots))
        _check_stage(stage, users, slots, float(stage_cfg["min_speedup"]))
    except BaseException:
        _FAILED = True
        raise
