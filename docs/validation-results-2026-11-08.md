# Validation results: 2026-11-08

Server: `Cloud-A100-233`; repository: `/root/hujianmin/SwiftCache`.

Base commit: `3217c43`; branch: `codex/elastic-cache-correctness`.

Both models are Qwen3-8B BF16, TP=1, on A100 PCIe 40GB GPUs 0 and 1. No NVLink is present.

| Check | Result |
| --- | --- |
| Allocator/protocol unit tests | 10 passed, including 1000 random state steps |
| Fixed serving cases | 8 model/case pairs; cold/warm outputs match |
| Elastic outputs versus fixed baseline | 8/8 exact token matches |
| Repeated resize stress | 12 cycles passed |
| Acknowledged transactions through boundary checks | 54 (27 grows, 27 shrinks) |
| Reclaim overlapping live master KV | 2 observed events |
| Decode block rollover | Both models passed 255-token input + 24 output tokens |
| Full worker reclamation | Worker owned all 32 blocks; queued master recovered after return |
| Queued master completion time in full-reclaim check | 2.810 s |
| Independent Transformers greedy reference | 11/11 exact token matches |
| Final concurrent/shared-prefix requests | 12 passed |
| Final state | Master 31 free blocks + worker 1 free block; 0 live blocks and request records |

The stress client intentionally uses `ignore_eos=True` for its long-running probes. Text after an EOS in those probes is not an ordinary conversational response. The independent reference uses the same stopping settings.

Successful-run evidence is under `artifacts/`: `fixed-results.json`, `elastic-results.json`, `boundary-results.json`, `reference-results.json`, `final-smoke-results.json`, `final-idle-state.json`, and `validation-summary.json`. The diagnostic index-race log records an earlier failure that led to adding the missing CUDA-stream dependency; it is separate from the successful stress log.

These checks establish correctness for the tested workload and configuration, not exhaustive accuracy or performance claims. The C++ scheduler, TP>1, quantized models, and transparent process-crash recovery are outside the tested scope.

See [implementation and commands](elastic-cache.md) for reproducible steps.
