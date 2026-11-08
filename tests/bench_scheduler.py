"""CPU-only metadata benchmark of the REAL Python scheduler, no model inference.

--source selects an archived source tree so baseline and final code use the same
driver. GPU/transport are omitted; prefix/hash/free-list/scheduler are real.
"""

import argparse
from collections import deque
import json
from pathlib import Path
import statistics
import sys
import time
from types import SimpleNamespace

PROFILE = None
BACKEND = "python"


def summary(values):
    values = sorted(values)
    return dict(
        mean=statistics.mean(values),
        p50=statistics.median(values),
        p95=values[min(len(values) - 1, int(0.95 * len(values)))],
        max=max(values),
    )


def scheduler(pool, backend=None):
    from swiftcache.engine.scheduler import Scheduler
    from swiftcache.engine.block_manager.block_manager import (
        BlockManagerBase,
        MasterBlockManager,
    )
    from swiftcache.engine.block_manager.block import Block
    from swiftcache.engine.block_manager.free_blocks_ids import MultiFreeBlockIds

    cfg = SimpleNamespace(
        role="master",
        rank=0,
        kvcache_block_size=256,
        initial_reserved_master=[0],
        max_num_seqs=8,
        max_num_batched_tokens=1048576,
        num_kvcache_blocks=pool,
        eos=-123,
    )
    bm = object.__new__(MasterBlockManager)
    BlockManagerBase.__init__(bm, cfg)
    bm.blocks = [Block(i) for i in range(pool)]
    bm.free_block_ids = MultiFreeBlockIds(bm.blocks, [0, pool], 0, cfg, [1])
    bm.pending = {}
    bm.zmq_server = SimpleNamespace(recv_all_dict_nonblock=lambda: [])
    if (backend or BACKEND) == "cpp":
        from swiftcache.engine.native_scheduler import Scheduler as NativeScheduler

        return NativeScheduler.from_block_manager(bm, cfg)
    obj = object.__new__(Scheduler)
    obj.cnt = 0
    obj.config = cfg
    obj.role = "master"
    obj.max_num_seqs = 8
    obj.max_num_batched_tokens = 1048576
    obj.eos = -123
    obj.block_manager = bm
    obj.waiting = deque()
    obj.running = deque()
    return obj


def run_case(total, uncached, pool, samples, rounds, warmups):
    from swiftcache.engine.sequence import Sequence
    from swiftcache.sampling_params import SamplingParams
    import numpy as np

    all_rounds = []
    prefix_len = (total - uncached) // 256 * 256
    uncached = total - prefix_len
    # Stable, nonconstant real token IDs; a new first suffix token breaks its hash chain.
    base = np.random.default_rng(1234).integers(100, 50000, total).tolist()
    serialized = np.asarray(base, dtype=np.int64).tobytes()
    for run in range(rounds):
        s = scheduler(pool)
        if prefix_len:
            seed = Sequence(
                base[:prefix_len], SamplingParams(temperature=0, max_tokens=1)
            )
            s.add(seed)
            seqs, kind = s.schedule()
            assert kind == "prefill"
            s.postprocess(seqs, [7])
        measured = dict(
            construct_ms=[],
            enqueue_ms=[],
            cleanup_ms=[],
            lifecycle_ms=[],
            schedule_wall_ms=[],
            schedule_cpu_ms=[],
            prefix_allocate_ms=[],
            postprocess_ms=[],
        )
        subtotal = [0.0]
        for name in ["can_allocate", "allocate"]:
            original = getattr(s.block_manager, name)

            def wrapped(*a, _original=original, **kw):
                start = time.perf_counter_ns()
                try:
                    return _original(*a, **kw)
                finally:
                    subtotal[0] += (time.perf_counter_ns() - start) / 1e6

            setattr(s.block_manager, name, wrapped)
        for i in range(warmups + samples):
            # Independent Python ints mimic separate tokenizer/JSON requests.
            # Reusing base's int objects makes list equality unrealistically cheap.
            tokens = np.frombuffer(serialized, dtype=np.int64).tolist()
            tokens[prefix_len] = (i + 60000) % 150000
            if PROFILE is not None and i >= warmups:
                PROFILE.enable()
            start = time.perf_counter_ns()
            seq = Sequence(tokens, SamplingParams(temperature=0, max_tokens=1))
            construct = (time.perf_counter_ns() - start) / 1e6
            start = time.perf_counter_ns()
            s.add(seq)
            enqueue = (time.perf_counter_ns() - start) / 1e6
            subtotal[0] = 0
            wall = time.perf_counter_ns()
            cpu = time.thread_time_ns()
            seqs, kind = s.schedule()
            cpu_ms = (time.thread_time_ns() - cpu) / 1e6
            wall_ms = (time.perf_counter_ns() - wall) / 1e6
            assert kind == "prefill" and seqs == [seq]
            assert seq.num_cached_tokens == prefix_len, (
                total,
                uncached,
                seq.num_cached_tokens,
                prefix_len,
            )
            start = time.perf_counter_ns()
            s.postprocess(seqs, [7])
            post = (time.perf_counter_ns() - start) / 1e6
            # Do not charge destruction of the previous request to the next
            # schedule() assignment. Keep retirement visible as its own stage.
            start = time.perf_counter_ns()
            seqs = None
            seq = None
            tokens = None
            cleanup = (time.perf_counter_ns() - start) / 1e6
            if PROFILE is not None and i >= warmups:
                PROFILE.disable()
            if i >= warmups:
                for key, value in [
                    ("construct_ms", construct),
                    ("schedule_wall_ms", wall_ms),
                    ("schedule_cpu_ms", cpu_ms),
                    (
                        "prefix_allocate_ms",
                        s.last_metrics["prefix_match_allocate_ms"]
                        if BACKEND == "cpp"
                        else subtotal[0],
                    ),
                    ("postprocess_ms", post),
                    ("enqueue_ms", enqueue),
                    ("cleanup_ms", cleanup),
                    ("lifecycle_ms", construct + enqueue + wall_ms + post + cleanup),
                ]:
                    measured[key].append(value)
        s.block_manager.assert_invariants()
        all_rounds.append(
            dict(raw=measured, summary={k: summary(v) for k, v in measured.items()})
        )
    return dict(
        total_tokens=total,
        cached_tokens=prefix_len,
        uncached_tokens=uncached,
        pool_blocks=pool,
        samples_per_round=samples,
        rounds=rounds,
        warmups=warmups,
        measurements=all_rounds,
        median_round_p50_ms=statistics.median(
            r["summary"]["schedule_wall_ms"]["p50"] for r in all_rounds
        ),
    )


def admission_probe():
    from swiftcache.engine.sequence import Sequence
    from swiftcache.sampling_params import SamplingParams

    s = scheduler(448)
    tokens = [100 + (i % 999) for i in range(102400)]
    first = Sequence(tokens, SamplingParams(temperature=0, max_tokens=8))
    s.add(first)
    seqs, kind = s.schedule()
    assert kind == "prefill"
    second_tokens = tokens.copy()
    second_tokens[101376] = 55555
    second = Sequence(second_tokens, SamplingParams(temperature=0, max_tokens=8))
    s.add(second)
    seqs, kind = s.schedule()
    result = dict(
        second_admitted=kind == "prefill" and second in seqs,
        free_blocks=len(s.block_manager.free_block_ids),
        second_total_blocks=400,
        shared_live_prefix_blocks=396,
        new_blocks_needed=4,
    )
    return result


def queue_probe():
    """First scheduling call for real queues; remaining requests are drained."""
    import numpy as np
    from swiftcache.engine.sequence import Sequence
    from swiftcache.sampling_params import SamplingParams

    base = np.random.default_rng(1234).integers(100, 50000, 102400)
    serialized = base.tobytes()
    output = []
    for count in [1, 2, 4, 8, 32]:
        s = scheduler(512)
        seed = Sequence(base[:101376].tolist(), SamplingParams(max_tokens=1))
        s.add(seed)
        seqs, _ = s.schedule()
        s.postprocess(seqs, [7])
        times, sizes = [], []
        for iteration in range(25):
            for i in range(count):
                tokens = np.frombuffer(serialized, dtype=np.int64).tolist()
                tokens[101376] = 60000 + iteration * count + i
                s.add(Sequence(tokens, SamplingParams(max_tokens=1)))
            start = time.perf_counter_ns()
            seqs, kind = s.schedule()
            elapsed = (time.perf_counter_ns() - start) / 1e6
            assert kind == "prefill" and seqs
            assert all(seq.num_cached_tokens == 101376 for seq in seqs)
            if iteration >= 5:
                times.append(elapsed)
                sizes.append(len(seqs))
            s.postprocess(seqs, [7] * len(seqs))
            while not s.is_finished():
                seqs, kind = s.schedule()
                assert seqs and kind == "prefill"
                s.postprocess(seqs, [7] * len(seqs))
            seqs = None
            tokens = None
        s.block_manager.assert_invariants()
        output.append(
            dict(
                waiting_requests=count,
                previous_request_retirement_outside_clock=True,
                samples=20,
                warmups=5,
                first_call_ms=summary(times),
                observed_first_batch_sizes=sizes,
            )
        )
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--backend", choices=["python", "cpp"], default="python")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--suite",
        choices=["core", "diagnostic", "edge", "profile", "supplemental"],
        default="core",
    )
    parser.add_argument("--profile-output", type=Path)
    args = parser.parse_args()
    sys.path.insert(0, str(args.source.resolve()))
    global PROFILE, BACKEND
    BACKEND = args.backend
    if args.profile_output:
        import cProfile

        PROFILE = cProfile.Profile()
    cases = []
    if args.suite == "core":
        matrix = [(t, 1024, 512, 1000, 5, 100) for t in [51200, 76800, 102400]]
    elif args.suite == "diagnostic":
        matrix = [(102400, 1024, pool, 200, 3, 100) for pool in [8192, 32768]]
        matrix += [(t, t, 512, 30, 3, 10) for t in [51200, 102400]]
        matrix += [(102400, u, 512, 200, 3, 100) for u in [1, 256, 4096]]
        matrix += [(100000, 1000, 512, 200, 3, 100)]
    elif args.suite == "edge":
        matrix = [(102401, 1, 512, 200, 3, 100)]
    elif args.suite == "supplemental":
        matrix = [(50000, 1000, 512, 200, 3, 100), (102400, 51200, 512, 200, 3, 100)]
    else:
        matrix = [(102400, 1024, 512, 100, 1, 100)]
    for parameters in matrix:
        result = run_case(*parameters)
        cases.append(result)
        print(
            json.dumps({k: v for k, v in result.items() if k != "measurements"}),
            flush=True,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(
                dict(
                    cpu_metadata_only=True,
                    independent_token_objects=True,
                    request_retirement_measured_separately=True,
                    source=str(args.source),
                    backend=BACKEND,
                    cases=cases,
                    admission=admission_probe(),
                ),
                indent=2,
            )
        )
    if args.suite == "supplemental":
        result = json.loads(args.output.read_text())
        result["queue_probe"] = queue_probe()
        args.output.write_text(json.dumps(result, indent=2))
    if PROFILE is not None:
        PROFILE.dump_stats(str(args.profile_output))
        rows = []
        for entry in PROFILE.getstats():
            code = entry.code
            rows.append(
                dict(
                    function=code if isinstance(code, str) else code.co_name,
                    file="" if isinstance(code, str) else code.co_filename,
                    line=0 if isinstance(code, str) else code.co_firstlineno,
                    calls=entry.callcount,
                    self_ms=entry.inlinetime * 1000,
                    cumulative_ms=entry.totaltime * 1000,
                )
            )
        args.profile_output.with_suffix(".json").write_text(
            json.dumps(
                sorted(rows, key=lambda x: x["cumulative_ms"], reverse=True), indent=2
            )
        )


if __name__ == "__main__":
    main()
