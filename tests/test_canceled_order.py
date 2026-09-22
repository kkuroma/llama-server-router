"""
Checks that dropping a streaming client frees its slot instead of generating into nothing

Two long requests fill both slots, one is disconnected, and two trivial requests
sent behind it have to finish before the request that was never interrupted
"""

import asyncio
from typing import Any

import pytest

from core.harness import (
    RouterProcess,
    SlotMonitor,
    StreamRecord,
    busy_drop_after,
    format_records,
    load_config,
    load_model,
    load_prompt,
    model_id,
    replica_port,
    slot_count,
    stream_chat,
    variant_prompt,
)

CFG = load_config()
CANCEL = CFG["Cancel"]
PROMPT = load_prompt(CFG)


async def _run_cancel(router: RouterProcess) -> dict[str, Any]:
    """
    Fills both slots, drops one client, then sends two short requests behind it

    The short requests run in the same task as the dropped one, so they are sent
    strictly after the disconnect rather than racing it.

    Args:
        router: The running router under test.

    Returns:
        The four records, the slot monitor, and the moment of the disconnect.
    """
    base_url = router.base_url
    slots = int(CANCEL["slots"])
    model = model_id(CFG, 0, slots)
    await load_model(base_url, model, float(CFG["Router"]["load_timeout"]))
    port = await replica_port(base_url, model)
    reported_slots = await slot_count(port, model) if port is not None else None

    dropped_at: list[float] = []

    async def cancel_then_short() -> list[StreamRecord]:
        """Sends the request that gets dropped, then the two short ones in turn."""
        first = await stream_chat(
            base_url, model, variant_prompt(CFG, PROMPT, 1), int(CANCEL["max_tokens"]),
            index=1, stop_after=float(CANCEL["kill_after"]),
        )
        dropped_at.append(first.t_end)
        short = []
        for index in (3, 4):
            short.append(await stream_chat(
                base_url, model, str(CANCEL["short_prompt"]),
                int(CANCEL["short_max_tokens"]), index=index, force_length=False,
            ))
        return [first, *short]

    async def survivor() -> StreamRecord:
        """Sends the request that is never interrupted."""
        return await stream_chat(
            base_url, model, variant_prompt(CFG, PROMPT, 2), int(CANCEL["max_tokens"]), index=2,
        )

    async with SlotMonitor(port, model) as monitor:
        dropped_group, second = await asyncio.gather(cancel_then_short(), survivor())

    first, third, fourth = dropped_group
    return {
        "model": model,
        "port": port,
        "reported_slots": reported_slots,
        "records": [first, second, third, fourth],
        "monitor": monitor,
        "dropped_at": dropped_at[0],
    }


@pytest.mark.gpu
@pytest.mark.slow
def test_canceled_order(router: RouterProcess) -> None:
    """Drops one of two streams and expects the queue behind it to move immediately."""
    stage = asyncio.run(_run_cancel(router))
    slots = int(CANCEL["slots"])
    records: list[StreamRecord] = stage["records"]
    first, second, third, fourth = records
    monitor: SlotMonitor = stage["monitor"]
    dropped_at: float = stage["dropped_at"]

    origin = min(record.t_send for record in records)
    freed = busy_drop_after(monitor.history, dropped_at, slots)
    freed_note = f"a slot freed {freed:.1f}s later" if freed >= 0 else "no slot ever freed"
    print()
    print(format_records(records, origin))
    print(f"model {stage['model']} on port {stage['port']}, "
          f"slots reported {stage['reported_slots']}, wanted {slots}")
    print(f"request 1 dropped at {dropped_at - origin:.1f}s, {freed_note}")
    print(f"slot probe: available {monitor.available}, peak busy {monitor.max_busy} "
          f"of {monitor.total_slots}, samples {monitor.samples}")

    assert first.canceled, "request 1 was never dropped, so the test proves nothing"
    assert second.ok, f"request 2 failed: HTTP {second.status} {second.error}"
    assert second.produced >= int(CANCEL["min_tokens"]), (
        f"request 2 generated only {second.produced} tokens, "
        f"finish reason {second.finish_reason or 'none'}"
    )
    for record in (third, fourth):
        assert record.ok, f"request {record.index} failed: HTTP {record.status} {record.error}"

    assert third.t_end < second.t_end, (
        f"request 3 finished {third.t_end - second.t_end:.1f}s after request 2, so the "
        f"slot held by the dropped request 1 was never released"
    )
    assert fourth.t_end < second.t_end, (
        f"request 4 finished {fourth.t_end - second.t_end:.1f}s after request 2, so the "
        f"slot held by the dropped request 1 was never released"
    )
