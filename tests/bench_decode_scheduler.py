"""CPU decode scheduling; prefill/ingestion are outside this separate metric."""

import argparse
import json
from pathlib import Path
import time
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bench_scheduler import scheduler, summary
from swiftcache.engine.sequence import Sequence
from swiftcache.sampling_params import SamplingParams


def run(backend, count):
    s = scheduler(512, backend)
    prefix = [100 + i % 999 for i in range(101376)]
    seed = Sequence(prefix, SamplingParams(max_tokens=1))
    s.add(seed)
    batch, _ = s.schedule()
    s.postprocess(batch, [7])
    schedule_times, post_times, rollover = [], [], []
    for repetition in range(25):
        requests = []
        for index in range(count):
            suffix = [1000 + repetition * count + index] + [999] * 1023
            seq = Sequence(prefix + suffix, SamplingParams(max_tokens=65))
            requests.append(seq)
            s.add(seq)
        batch, kind = s.schedule()
        assert kind == "prefill" and len(batch) == count
        assert all(seq.num_cached_tokens == 101376 for seq in batch)
        s.postprocess(batch, [7] * count)
        for step in range(64):
            start = time.perf_counter_ns()
            batch, kind = s.schedule()
            elapsed = (time.perf_counter_ns() - start) / 1e6
            assert kind == "decoding" and len(batch) == count
            start = time.perf_counter_ns()
            s.postprocess(batch, [7] * count)
            post = (time.perf_counter_ns() - start) / 1e6
            if repetition >= 5:
                schedule_times.append(elapsed)
                post_times.append(post)
                if step == 0:
                    rollover.append(elapsed)
        assert s.is_finished()
        s.block_manager.assert_invariants()
    return dict(
        concurrent_sequences=count,
        total_input_tokens=102400,
        cached_tokens=101376,
        measured_requests=20 * count,
        decode_batches=len(schedule_times),
        schedule_ms=summary(schedule_times),
        postprocess_ms=summary(post_times),
        block_rollover_schedule_ms=summary(rollover),
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    rows = []
    for backend in ["python", "cpp"]:
        for count in [1, 2, 4, 8]:
            record = dict(backend=backend, **run(backend, count))
            rows.append(record)
            print(json.dumps(record), flush=True)
            args.output.write_text(
                json.dumps(dict(cpu_metadata_only=True, cases=rows), indent=2)
            )


if __name__ == "__main__":
    main()
