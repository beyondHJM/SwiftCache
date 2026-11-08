"""Differential scheduling and reuse of the existing elastic safety assertions."""

import gc
import random
import weakref

import pytest

from swiftcache.engine import _native_scheduler as native
from swiftcache.engine.native_scheduler import enable_native
from swiftcache.engine.block_manager.block_manager import BlockManagerBase
from swiftcache.engine.sequence import Sequence
from swiftcache.sampling_params import SamplingParams
from bench_scheduler import scheduler
import test_elastic as elastic


@pytest.mark.parametrize("count", [1, 255, 256, 257, 102400])
def test_native_hash_is_byte_identical_to_python(count):
    tokens = random.Random(42).choices(range(151936), k=count)
    for prefix in [-1, 0, 123, (1 << 64) - 1]:
        assert native.BlockManager.compute_hash(
            tokens, None if prefix == -1 else prefix
        ) == BlockManagerBase.compute_hash(tokens, prefix)


@pytest.mark.parametrize(
    "name",
    [
        "test_active_master_reclaim_and_duplicate_messages",
        "test_worker_shrink_invalidates_cached_data_before_handoff",
        "test_shrink_protects_worker_highwater",
        "test_exact_block_prefix_and_stale_hash_eviction",
        "test_random_handoffs_do_not_overlap_or_leak",
        "test_draining_live_block_cannot_reenter_prefix_cache",
        "test_invalid_resize_fails_before_mutating_ownership",
    ],
)
def test_existing_elastic_protocol_on_native_metadata(monkeypatch, name):
    original = elastic.pair

    def pair(*args, **kwargs):
        master, worker, incoming, outgoing = original(*args, **kwargs)
        return enable_native(master), enable_native(worker), incoming, outgoing

    monkeypatch.setattr(elastic, "pair", pair)
    getattr(elastic, name)()


@pytest.mark.parametrize("backend", ["python", "cpp"])
def test_long_live_shared_prefix_and_cancellation(backend):
    s = scheduler(448, backend)
    tokens = [100 + i % 999 for i in range(102400)]
    first = Sequence(tokens, SamplingParams(max_tokens=8))
    s.add(first)
    seqs, kind = s.schedule()
    assert seqs == [first] and kind == "prefill"
    suffix = tokens.copy()
    suffix[101376] = 55555
    second = Sequence(suffix, SamplingParams(max_tokens=8))
    s.add(second)
    seqs, kind = s.schedule()
    assert seqs == [second] and second.num_cached_tokens == 101376
    assert len(s.block_manager.used_block_ids) == 404
    s.cancel(first)
    s.cancel(second)
    s.block_manager.assert_invariants()
    assert s.is_finished() and not s.block_manager.used_block_ids
    ref = weakref.ref(second)
    del second, seqs
    gc.collect()
    assert ref() is None


def test_python_cpp_queue_preemption_decode_and_stops_match():
    engines = [scheduler(12, kind) for kind in ["python", "cpp"]]
    rng = random.Random(20261108)
    requests = [{}, {}]
    for step in range(300):
        if step % 3 == 0:
            count = rng.choice([1, 255, 256, 257, 511, 512, 513, 1024])
            tokens = [100 + (i % 300) for i in range(count)]
            tokens[-1] = 10000 + step
            max_tokens = rng.choice([1, 2, 4, 8])
            for j, engine in enumerate(engines):
                seq = Sequence(
                    tokens, SamplingParams(max_tokens=max_tokens), seq_id=step
                )
                requests[j][step] = seq
                engine.add(seq)
        a, ka = engines[0].schedule()
        b, kb = engines[1].schedule()
        assert ka == kb and [s.seq_id for s in a] == [s.seq_id for s in b]
        assert [(s.num_cached_tokens, s.block_table) for s in a] == [
            (s.num_cached_tokens, s.block_table) for s in b
        ]
        ids = [rng.randrange(1000) for _ in a]
        if step % 11 == 0 and a:
            a[0].force_stop = b[0].force_stop = True
        for engine, seqs in zip(engines, [a, b]):
            engine.postprocess(seqs, ids)
            engine.block_manager.assert_invariants()
        if step % 17 == 0:
            for j, engine in enumerate(engines):
                pending = engine.waiting
                if pending:
                    engine.cancel(pending[0])
        for seq_id in requests[0]:
            x, y = requests[0][seq_id], requests[1][seq_id]
            assert x.token_ids == y.token_ids and x.status == y.status
    for engine in engines:
        for seq in list(engine.waiting) + list(engine.running):
            engine.cancel(seq)
        assert engine.is_finished() and not engine.block_manager.used_block_ids


def test_native_content_check_protects_hash_collision():
    s = scheduler(8, "cpp")
    bm = s.block_manager
    tokens = [123] * 512
    block = bm._allocate_block(bm.free_block_ids[0])
    h = bm.compute_hash(tokens[:256])
    bm.register_hash(block, h, [456] * 256)
    block.ref_count = 0
    bm._deallocate_block(block.block_id)
    seq = Sequence(tokens, SamplingParams(max_tokens=1))
    s.add(seq)
    seqs, _ = s.schedule()
    assert seq.num_cached_tokens == 0
    s.postprocess(seqs, [7])
    bm.assert_invariants()


def test_cached_native_plan_cannot_be_reused_across_allocators():
    first = scheduler(8, "cpp")
    second = scheduler(8, "cpp")
    seq = Sequence([100] * 512, SamplingParams(max_tokens=1))
    # Equal mutation counts give equal epochs, but different cached contents.
    for engine, value in [(first, 200), (second, 100)]:
        seed = Sequence([value] * 256, SamplingParams(max_tokens=1))
        engine.add(seed)
        batch, _ = engine.schedule()
        engine.postprocess(batch, [7])
    assert first.block_manager.allocation_plan(seq)["cached_tokens"] == 0
    second.add(seq)
    batch, _ = second.schedule()
    assert seq.num_cached_tokens == 256
    with pytest.raises(RuntimeError, match="different native allocator"):
        first.block_manager.deallocate(seq)
    second.postprocess(batch, [7])
    for engine in [first, second]:
        engine.block_manager.assert_invariants()
