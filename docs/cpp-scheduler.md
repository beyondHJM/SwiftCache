# C++ scheduler

## Build and select

The new backend is an opt-in C++17/pybind11 extension for the elastic runtime.
It does not use the old `c_scheduler.cpython-310-*.so` or the pre-elastic C++
coordinator under `csrc/scheduler.cpp`.

```bash
cd /root/hujianmin/SwiftCache
bash scripts/build_cpp_scheduler.sh
.venv/bin/python -m server.starter --config configs/two_qwen8b.yaml --scheduler-backend cpp
```

Rebuild with the service's Python after changing Python/ABI or native source.
The build requires a C++17 compiler and installs `requirements-native.txt` into
the selected Python (`SWIFTCACHE_PYTHON`, default `.venv/bin/python`). It reuses
the repository's BSD-licensed xxHash header. No CUDA compiler or libtorch C++ ABI
is needed for this extension. The existing CUDA model dependencies remain
necessary for inference.

The Python backend is retained for comparisons:

```bash
.venv/bin/python -m server.starter --config configs/two_qwen8b.yaml --scheduler-backend python
```

YAML supports `models[].engine.scheduler_backend: cpp` or `python`. The CLI
overrides every instance. A missing extension fails with a build instruction;
there is no silent fallback. Without an explicit engine setting, the old
`USE_CSCHEDULER=1` environment switch now selects the new backend. Explicit
configuration takes precedence. `/cache` and internal request metrics report
the backend. The supported model path is text, TP=1 and eager, as in the existing
elastic service.

## Architecture and semantics

| Responsibility | Implementation |
| --- | --- |
| Waiting/running queues, batching, preemption, decode scheduling, completion/cancellation | C++ `NativeScheduler` |
| Contiguous token storage, chained XXH64, complete content verification | C++ `NativeSequence`/`NativeBlocks` |
| Prefix plan cache and allocator epoch, active shared blocks, references | C++ |
| Owned/free/draining state and eviction candidates | C++, one authoritative data set |
| Resize messages, acknowledgments/replay, demand window, GPU quiescence | Existing Python coordinator |
| Model-facing Sequence, OpenAI requests, GPU inference | Existing Python/model runner |

The adapter initializes native metadata after the normal empty-manager handshake
and exposes native block/group views to the same Python coordinator. It does not
maintain independently updated Python/native block maps. Only one call per
candidate sequence crosses into the Python admission gate, so worker growth and
pending resize still use the existing protocol. Token/block loops stay in C++.
Schedulers are owned by a single inference event loop; this is not a concurrent
multi-threaded allocator and native operations retain the GIL.

The free index is an ordered C++ set with the same priority fields as the Python
allocator: invalid hash first, hit count, prefix distance, then locality/ID.
It supports root selection and arbitrary cached-block removal without rebuilding
a Python heap. Hashing uses the same int64 representation and little-endian
prefix bytes as the Python reference, followed by full token comparison. The
last block is recomputed even when it is full. A hash alone never proves a hit.

Admission charges only blocks that actually need to be acquired and counts
uncached query tokens against the batch target. One oversized cold request can
run alone, matching the Python reference. Ownership changes invalidate plans;
plans also carry allocator identity. Draining active blocks retain their live
contents but cannot appear in new prefix matches. The existing coordinator
quiesces CUDA before transferring ownership. Finishing/cancelling releases all
references, and native token state does not retain its Python facade in a cycle.

Python token IDs enter native storage once on admission through a checked list
conversion. Decode headroom is reserved at ingestion. During decoding, the
Python block table is updated only when it changes, avoiding a 400-ID conversion
on every token at 100K context. Sampling/top_p, stop handling and model execution
remain shared between backends.

## Validation and measurements

```bash
.venv/bin/python -m pytest -q tests/test_elastic.py tests/test_serving.py tests/test_cpp_scheduler.py
mkdir -p artifacts/cpp-scheduler/python-baseline
git archive fe79070 | tar -x -C artifacts/cpp-scheduler/python-baseline
.venv/bin/python tests/bench_scheduler.py --source artifacts/cpp-scheduler/python-baseline --backend python --suite core --output artifacts/cpp-scheduler/cpu-python-core.json
.venv/bin/python tests/bench_scheduler.py --source . --backend cpp --suite core --output artifacts/cpp-scheduler/cpu-cpp-core.json
.venv/bin/python tests/bench_decode_scheduler.py --output artifacts/cpp-scheduler/decode.json
```

The metadata benchmark uses real scheduler/allocator classes. It excludes JSON,
tokenization and GPU execution, and uses independent Python integer objects on
each request. Request retirement is outside the next `schedule()` call and is
measured separately. Report both `schedule_wall_ms` and `lifecycle_ms`; the latter
includes Sequence construction, enqueue/native conversion, schedule, postprocess
and object retirement. Avoid inferring complete API throughput from a scheduler
speedup.

Other suites: `diagnostic` (pool size, cold/partial hit and suffix size), `edge`
(one uncached token), `supplemental` (nonaligned inputs and queue sizes).
The queue probe records actual admitted batch size. Large metadata pools are
CPU experiments rather than GPU KV capacity claims.

The actual-model helper is separate:

```bash
# Uses one GPU and local KV, without a worker or public HTTP gateway.
# Full mode requires approximately the same free memory as the prior local-KV
# 100K test. Do not run it on an occupied GPU without checking memory headroom.
CUDA_VISIBLE_DEVICES=0 .venv/bin/python tests/bench_native_model.py --backend cpp --output artifacts/cpp-scheduler/model-cpp.json
# Bounded-cache semantic regression against saved phase-one references:
CUDA_VISIBLE_DEVICES=0 .venv/bin/python tests/bench_native_model.py --short-only --backend cpp --output artifacts/cpp-scheduler/model-cpp-short.json
```

`--short-only` requires the saved `artifacts/fixed-results.json` and
`artifacts/boundary-results.json`. It validates actual model outputs, including
block rollover and a request close to the 8192-token capacity. It does not test
two-GPU handoff or measure 100K GPU latency. Use the unified two-model service and
the existing elastic tests for that separate validation when GPUs are available.

Build integration follows the [pybind11 build documentation](https://pybind11.readthedocs.io/en/stable/compiling.html).
The bundled hash implementation is from [xxHash](https://github.com/Cyan4973/xxHash),
with its original license retained in `csrc/include/xxhash.h`.
