"""Real two-GPU Qwen3 verification, including GPU->DRAM->GPU reclaim cycles."""

import time
import torch

from .runtime import tensor_device


def validate(engine):
    tokenizer = engine.consumer.tokenizer
    checks = []
    for text in [
        "What is 2 + 2? Reply only with the number.",
        "Translate into Chinese: The cache is working correctly.",
        "The access code is SILVER-731. What is the code? Reply with only the code.",
    ]:
        ids = engine.consumer.ids(messages=[dict(role="user", content=text)])
        reference = engine.consumer.reference(ids, 16)
        result = engine.drain([engine.submit(ids, 16)])[0]
        assert result["token_ids"] == reference, (text, result["text"], reference)
        checks.append(dict(prompt=text, reference=reference, result=result))
    assert checks[0]["result"]["text"].strip() == "4"

    burst_ids = [
        engine.consumer.ids(
            messages=[
                dict(
                    role="user",
                    content=f"Worker {i}. "
                    + ("This is a document about safely managing memory. " * 35)
                    + "List the integers from 1 to 20, separated by commas.",
                )
            ]
        )
        for i in range(4)
    ]
    burst_ref = [engine.consumer.reference(ids, 12) for ids in burst_ids]
    requests = [engine.submit(ids, 12) for ids in burst_ids]
    burst = engine.drain(requests)
    assert [r["token_ids"] for r in burst] == burst_ref
    assert any(e["event"] == "preempt" for e in engine.trace.snapshot())
    assert any(e["event"] == "resume" for e in engine.trace.snapshot())
    assert engine.store.bytes == 0 and not engine.client.status()["allocations"]

    # Prefix KV is stored as one whole Aqua tensor, not as SwiftCache LSC blocks.
    head = tokenizer.encode(
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nThe access code is ALPHA-42.\n"
    )
    paragraph = tokenizer.encode(
        "The memory manager must preserve every key and value.\n"
    )
    prefix = (head + paragraph * 200)[:512]
    tail = tokenizer.encode(
        "\nWhat is the access code? Reply only with the code.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )
    full = prefix + tail
    reference = engine.consumer.reference(full, 12)
    key = engine.cache_prefix(prefix, "validation-prefix")
    rt = engine.prefixes[key][1]
    assert (
        rt.active_device == tensor_device.GPU and rt.to_torch_tensor().device.index == 1
    )
    warm = engine.drain([engine.submit(full, 12, prefix_id=key)])[0]
    assert warm["cached_tokens"] == 512 and warm["token_ids"] == reference

    probe = torch.arange(4096, dtype=torch.float32, device="cuda:0").reshape(16, 256)
    stored = engine.store.put(probe, "byte-integrity")
    assert stored.active_device == tensor_device.GPU
    assert torch.equal(engine.store.restore(stored), probe)
    cycles = []
    for cycle in range(3):
        engine.producer.request_reclaim()
        assert not engine.client.reclaim_status(engine.producer.id)["can_reclaim"]
        engine.store.respond()
        assert engine.client.reclaim_status(engine.producer.id)["can_reclaim"]
        assert all(t.active_device == tensor_device.DRAM for t in engine.store.handles)
        assert torch.equal(engine.store.restore(stored), probe)
        engine.producer.poll(engine.producer.high)
        state = engine.client.status()
        assert state["providers"][str(engine.producer.id)]["capacity"] == 0
        assert not state["allocations"]
        # A fresh tensor must fall back to real pinned DRAM while the offer is gone.
        fallback = engine.store.put(probe + 1, "no-offer")
        assert fallback.active_device == tensor_device.DRAM
        assert (
            fallback.to_torch_tensor().device.type == "cpu"
            and fallback.to_torch_tensor().is_pinned()
        )
        assert torch.equal(engine.store.restore(fallback), probe + 1)
        engine.store.close(fallback)
        engine.producer.poll(0)
        engine.store.respond()
        assert all(t.active_device == tensor_device.GPU for t in engine.store.handles)
        assert torch.equal(engine.store.restore(stored), probe)
        replay = engine.drain([engine.submit(full, 12, prefix_id=key)])[0]
        assert replay["token_ids"] == reference and replay["cached_tokens"] == 512
        cycles.append(
            dict(cycle=cycle, replay=replay, coordinator=engine.client.status())
        )
    engine.store.close(stored)

    # Real requests on the second Qwen model drive the upstream high/low policy.
    producer_results = []
    if engine.producer_model:
        ids = engine.producer_model.ids(
            messages=[
                dict(role="user", content="What is 2 + 2? Reply only with the number.")
            ]
        )
        producer_reference = engine.producer_model.reference(ids, 8)
        checkpoint = len(engine.trace.snapshot())
        queued = [engine.submit(ids, 8, role="producer") for _ in range(10)]
        consumer = engine.submit(full, 12, prefix_id=key)
        combined = engine.drain(queued + [consumer])
        producer_results = combined[:-1]
        assert all(r["token_ids"] == producer_reference for r in producer_results)
        assert combined[-1]["token_ids"] == reference
        events = engine.trace.snapshot()[checkpoint:]
        assert any(e["event"] == "producer_reclaimed" for e in events)
        assert any(e["event"] == "producer_offered" for e in events)
        assert any(e["event"] == "restore" and e["source"] == "DRAM" for e in events)
        assert any(
            e["event"] == "scheduler_mode" and e["mode"] == "fcfs" for e in events
        )
        reclaimed = min(
            e["time_ns"] for e in events if e["event"] == "producer_reclaimed"
        )
        forwarded = min(
            e["time_ns"]
            for e in events
            if e["event"] == "forward" and e["role"] == "producer"
        )
        assert reclaimed < forwarded

    # Cancellation of a paged request must release its Aqua allocation.
    pending = [engine.submit(ids, 16) for ids in burst_ids[:3]]
    for _ in range(3):
        engine.tick()
    pending[0].cancelled = True
    remaining = engine.drain(pending[1:])
    assert pending[0].future.done() and pending[0].future.exception() is not None
    assert len(engine.store.handles) == len(engine.prefixes)
    assert not engine.jobs and engine.resident is None
    engine.tick()
    return dict(
        summary=dict(
            real_model="Qwen3-8B",
            models_loaded=2 if engine.producer_model else 1,
            reference_cases=len(checks),
            burst_requests=len(burst),
            reclaim_cycles=len(cycles),
            producer_requests=len(producer_results),
            prefix_cached_tokens=512,
            gpu_dram_roundtrip_exact=True,
            cpu_fallback=True,
            cancellation_released=True,
            p2p=bool(torch.cuda.can_device_access_peer(0, 1)),
        ),
        semantic=checks,
        burst=burst,
        prefix=warm,
        reclaims=cycles,
        producer=producer_results,
        cancel_survivors=remaining,
        events=engine.trace.snapshot(),
        ledger=engine.client.status(),
    )
