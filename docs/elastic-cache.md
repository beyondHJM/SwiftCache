# Elastic cache: implementation and two-GPU verification

## Scope

The serving path uses the repository's Qwen3 model, paged attention, prefix block
allocator, layer stream cache, and GPU-to-GPU KV transport. It runs one master
and one worker, each with tensor parallelism 1. The elastic control path uses the
Python scheduler. Do not enable `USE_CSCHEDULER` for this path: the bundled C++
extension does not implement the new handoff protocol.

The target verification machine is `Cloud-A100-233`. GPUs 0 and 1 are A100 PCIe
40GB devices with peer access, without NVLink. Correctness can be verified here;
these runs do not reproduce the paper's NVLink performance numbers.

## Relationship to the paper

The implementation follows Algorithm 1's aligned growth and window-based
shrink policy. For per-K (or per-V) complete-block element counts `Em` and `Ew`,
the shared elastic unit is `lcm(Em, Ew)`. Master and worker block increments are
`lcm / Em` and `lcm / Ew`. Both models use BF16 in the verified configuration.

Worker demand observations include requested output headroom. Concurrent live
blocks and waiting requests are additional lower bounds on shrinking. A worker
can return only its unused tail, without moving referenced KV data.

Physical K and V buffers are block-major. The remote layer-first view is a
strided view of `(block, layer, token, head, dim)`, not a reinterpretation as
contiguous layer-major storage. Remote scatter respects this stride.

Changing the physical ownership boundary needs no KV tensor reallocation or
cross-layer compaction. Coordination, free-list maintenance, hash invalidation,
and draining live references are additional work; the full transaction is not
claimed to take constant time.

## Ownership protocol

Worker-owned blocks occupy a prefix of each worker buffer. The master may use
the remaining suffix. Global master block ranges remain fixed across resizing,
so unrelated workers and the master's regular cache do not change block IDs.

For worker growth:

1. The worker requests an absolute aligned boundary with a monotonically
   increasing request ID and its previous boundary.
2. The master quarantines the interval immediately, removing it from new
   allocations and prefix reuse. Running requests retain their references.
3. The master waits until the interval has no live references and GPU transfers
   have completed. It then invalidates cached hashes and acknowledges ownership.
4. Only after that acknowledgment may the worker allocate the newly owned blocks.

For worker shrink:

1. The worker waits for its GPU work, checks the tail has no references,
   invalidates its prefix hashes, and removes that tail from local allocation.
2. It requests the lower boundary. Until acknowledgment the interval is
   quarantined and not reused by the worker.
3. The master waits for outstanding transfers, updates its free list, and
   acknowledges. It can then allocate the returned interval.

Messages are idempotent within a bounded replay history. Conflicting and stale
requests fail closed. The worker retries the same transaction ID; master
heartbeats distinguish legitimate draining of a long request from a failed
control channel. Missing acknowledgments never authorize reuse of memory.

An active request can be preempted and recomputed from its preserved token
history when it cannot obtain another block. Reclamation may wait; it never
overwrites live KV to force immediate growth. Process failure requires restarting
the serving pair; transparent crash recovery is not implemented.

## Other correctness fixes needed by the serving path

- Decode rotary positions use `len(sequence) - 1`.
- Streams are created after each process selects its GPU, not during module import.
- New layer data cannot overwrite layer stream cache while the previous store
  is reading it, including prefill steps that store without loading.
- Shared control memory is written only after preceding receiver commands finish.
- GPU work finishes before remote readiness is signaled or block references are freed.
- Remote send/receive streams wait for nonblocking index copies before indexing KV.
- Fully cached, exact-block prompts recompute the last block, keeping prefill nonempty.
- Prefix hashes are invalidated on eviction and ownership transfer; hash hits
  check token contents and current ownership.
- Free IDs use global IDs consistently across remote and local groups.
- The asynchronous loop yields between decode iterations, and polls control
  messages even when no inference request is active.
- Metadata, IPC sockets, and shared memory are scoped to a unique serving run.

## Run on the target server

Tested environment: Python 3.11, PyTorch 2.6.0+cu124, FlashAttention 2.8.3,
Triton 3.2.0, Transformers 4.51.0. The `.venv` inherits CUDA packages from
`/root/miniconda3/envs/nano`; additional packages are installed inside `.venv`.

```bash
cd /root/hujianmin/SwiftCache
CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
  .venv/bin/python -u -m server.elastic_server \
  --model /root/llm-resource/Models/Qwen3-8B \
  --cache-blocks 32 --initial-blocks 1 \
  --max-model-len 8192 --window 3 --check-interval 0.1 \
  --port 18800 --dist-port 18334
```

The master listens on `127.0.0.1:18800`, and the worker on `127.0.0.1:18801`.
Use `--fixed --initial-blocks 8` for the fixed-boundary baseline. The three-second
window is used to exercise resizing quickly; the default demand window is sixty
seconds. `--cache-blocks` caps physical cache blocks per worker and single-layer
buffer blocks on the master, making small, reproducible tests possible.

```bash
curl http://127.0.0.1:18800/health
curl http://127.0.0.1:18801/cache
curl -H 'Content-Type: application/json' \
  -d '{"prompt":"What is 2 + 2?","max_tokens":16,"temperature":0}' \
  http://127.0.0.1:18800/generate
```

`/cache` checks allocator invariants and reports current ownership and recent
resize events. Both models return generated token IDs as well as decoded text.
Terminate the launcher to stop both model processes and release its IPC resources.

## Validation

```bash
cd /root/hujianmin/SwiftCache
CUDA_VISIBLE_DEVICES= .venv/bin/python -m pytest -q tests/test_elastic.py

# With the fixed pair running:
.venv/bin/python -u tests/validate_serving.py fixed

# Stop the fixed pair, then start the elastic pair:
.venv/bin/python -u tests/validate_serving.py elastic --cycles 12
.venv/bin/python -u tests/validate_boundaries.py
.venv/bin/python -u tests/validate_smoke.py

# Stop the pair before loading an independent reference model:
CUDA_VISIBLE_DEVICES=0,1 .venv/bin/python -u tests/validate_serving.py reference
```

The unit tests exercise live-block draining, duplicate/stale messages, ownership
boundaries, cache hash invalidation, exact-block prefix reuse, high-water
protection, and 1000 randomized allocation/release/handoff steps.

Real-model checks compare fixed and elastic outputs, cold and warm prefix
results, and independent Transformers greedy decoding. The stress client
requires evidence of reclaim requests overlapping live master KV, successful
growth and shrink, zero live references after completion, and consistent total
capacity at both peers. Logs and JSON evidence are written under `artifacts/`.
Numerical agreement on these cases does not establish exhaustive model accuracy
or support for untested architectures, quantization, TP, or the C++ scheduler.
