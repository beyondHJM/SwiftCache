<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="./assets/sc.png">
    <img alt="SwiftCache" src="./assets/sc.png" width=55%>
  </picture>
</p>

<h3 align="center">
Efficient LLM Serving for Multi-turn Conversations with Heterogeneous KV Cache Sharing
</h3>

<!-- [![GitHub](https://img.shields.io/github/stars/beyondHJM/SwiftCache?style=social)](https://github.com/beyondHJM/SwiftCache)
[![GitHub license](https://img.shields.io/github/license/beyondHJM/SwiftCache)](https://github.com/beyondHJM/SwiftCache/blob/main/LICENSE.txt) -->

---

## Introduction

`SwiftCache` is an efficient serving framework designed for long-context, multi-turn conversations in Large Language Models (LLMs). It improves inference latency and extends maximum context length by enabling heterogeneous models to share underutilized GPU memory and NVLink bandwidth within a server.

Unlike traditional KV cache offloading approaches that rely on CPU memory or SSD through PCIe, SwiftCache transfers prefix KV caches directly between GPUs via NVLink. In addition, it introduces a **Layer Stream Cache** mechanism that keeps only the KV cache of the currently active layer in local GPU memory, significantly reducing memory pressure and supporting longer-context inference.

SwiftCache is especially effective for multi-turn conversational workloads, where historical tokens accumulate continuously and prefix cache reuse is frequent.

## Features

### Core Features

- **Heterogeneous KV Cache Sharing**  
  Allows models with high KV cache demand to borrow idle GPU memory from co-located low-demand models.

- **NVLink-Based KV Transfer**  
  Transfers prefix KV cache directly across GPUs via NVLink, avoiding slow CPU/SSD offloading through PCIe.

- **Layer Stream Cache**  
  Keeps only the KV cache of the currently active layer in local GPU memory to reduce memory footprint and extend maximum context length.

- **Elastic Cache**  
  Dynamically reclaims and reallocates KV cache capacity between master and worker models according to workload demand.

- **Block-Major KV Layout**  
  Uses a block-major cache layout to support efficient O(1) resizing of KV allocations.

### Integrated Optimizations

- **PagedAttention**  
  Improves KV cache management and reduces memory fragmentation.

- **FlashAttention**  
  Accelerates attention computation with optimized memory access.

- **Continuous Batching**  
  Improves throughput by dynamically scheduling requests of different lengths.

## Performance Highlights

- Reduces **P99 TTFT latency by up to 69%** compared to vLLM and SGLang.
- Extends maximum context length by up to **3.98×**.
- Minimizes interference to co-located models while improving overall server utilization.
- Achieves major latency reduction on real-world multi-turn conversation workloads such as ShareGPT and L-Eval.

## How It Works

SwiftCache is built around two main ideas:

1. **Layer Stream Cache**  
   Only the KV cache of the currently active layer is kept locally. Other layers’ KV caches are stored on slave GPUs and streamed in on demand.

2. **Elastic Cache**  
   Slave models dynamically donate underused KV cache blocks to the master model. When the slave workload increases, it can reclaim these blocks in constant time.

Together, these two mechanisms allow SwiftCache to:

- reduce local GPU memory consumption,
- exploit unused NVLink bandwidth,
- support longer context lengths,
- and lower end-to-end TTFT latency.


## Quick start

The commands below use the `codex/cpp-scheduler` branch, which includes unified
serving, the elastic ownership protocol and the optional C++ scheduler.

### 1. Prepare the environment

Use Linux, a CUDA-enabled NVIDIA driver and a base Python environment with
compatible **PyTorch, FlashAttention and Triton** already installed. The tested
stack is Python 3.11, PyTorch 2.6.0+cu124, FlashAttention 2.8.3, Triton 3.2.0 and
Transformers 4.51.0. The C++ scheduler also requires a C++17 compiler (tested
with GCC 11.4).

The validated deployment runs **one complete Qwen3-8B on each of two GPUs**:
one master and one worker, TP=1 and eager execution. Each GPU must have enough
free memory for its model, KV cache and activations. The validation server uses
two A100 PCIe 40GB GPUs with peer access; NVLink is not required for correctness.
PCIe measurements are not a reproduction of the paper's NVLink performance.

```bash
git clone --branch codex/cpp-scheduler https://github.com/beyondHJM/SwiftCache.git
cd SwiftCache

# Replace this with the Python executable in your existing CUDA environment.
export SWIFTCACHE_BASE_PYTHON=/path/to/cuda-env/bin/python
bash scripts/setup_serving_env.sh
```

The setup script creates `.venv` with access to the base environment's GPU
packages, checks those imports and installs the serving/client dependencies.
It does not install or replace the CUDA/PyTorch stack. If the active `python3`
already provides that stack, the environment-variable override can be omitted.

### 2. Build the C++ scheduler

```bash
bash scripts/build_cpp_scheduler.sh
```

This installs the native build dependencies and builds
`swiftcache.engine._native_scheduler` for `.venv`'s Python. Rebuild after changing
the Python version or native source. The older `c_scheduler.cpython-310-*.so`
is not used by this backend. You can skip this step when using the Python
scheduler.

### 3. Start both model services

Replace `/path/to/Qwen3-8B` with a local checkpoint directory containing its
weights, model configuration and tokenizer. On the validation server, the path
is `/root/llm-resource/Models/Qwen3-8B`.

```bash
.venv/bin/python -m server.starter \
  --config configs/two_qwen8b.yaml \
  --model /path/to/Qwen3-8B \
  --gpus 0,1 \
  --scheduler-backend cpp
```

`server/starter.py` is the canonical entry point and launches both model
processes plus one public gateway. `--model` applies the same checkpoint to all
instances; `--gpus` supplies one GPU per instance. YAML can assign model paths,
aliases and engine settings separately. Use physical GPU indices with `--gpus`;
the launcher sets `CUDA_VISIBLE_DEVICES` and rejects a conflicting existing value.

For the Python backend, use the same command with `--scheduler-backend python`.
Python remains the default when no backend is configured. A requested C++ backend
fails with a build instruction if the extension is unavailable; it does not
silently fall back to Python.

Wait for the launcher's `SWIFTCACHE_READY` message, or check:

```bash
curl http://127.0.0.1:18000/ready
curl http://127.0.0.1:18000/v1/models
```

The default aliases are `qwen3-8b-master` and `qwen3-8b-worker`. Both are selected
through the same public base URL, **`http://127.0.0.1:18000/v1`**. The example
reserves internal ports 18001/18002 and distributed port 18334. Stop the service
with `Ctrl+C` or `SIGTERM` sent to the launcher; it supervises its children and
cleans up its run-specific IPC resources.

The example's 32-block cache cap is a bounded validation configuration, not an
automatic maximum-memory configuration. Runtime context limits and usable KV
capacity are checked independently; inspect `/cache` for the effective limits.

## OpenAI-compatible clients

### Command-line client

```bash
.venv/bin/python -m server.client \
  --base-url http://127.0.0.1:18000/v1 \
  --model qwen3-8b-master \
  --prompt 'What is 2 + 2? Answer briefly.' \
  --max-tokens 32 \
  --stream
```

Use `--model qwen3-8b-worker` to route to the other model. Omit `--stream` for a
single completed response. `example.py` invokes the same client.

### Official Python SDK

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:18000/v1", api_key="local")
response = client.chat.completions.create(
    model="qwen3-8b-master",
    messages=[{"role": "user", "content": "What is 2 + 2?"}],
    temperature=0,
    max_completion_tokens=32,
)
print(response.choices[0].message.content)

stream = client.chat.completions.create(
    model="qwen3-8b-master",
    messages=[{"role": "user", "content": "Briefly explain prefix caching."}],
    temperature=0,
    max_tokens=64,
    stream=True,
    stream_options={"include_usage": True},
)
for chunk in stream:
    if chunk.choices:
        print(chunk.choices[0].delta.content or "", end="", flush=True)
print()
```

`local` is a placeholder key when service authentication is disabled. These
requests go to SwiftCache, not to OpenAI-hosted models. The same endpoint works
with `AsyncOpenAI` and the tested LangChain `ChatOpenAI` adapter; configure its
base URL, key and served model alias, and set `use_responses_api=False`.

Supported routes include `GET /v1/models`, text `POST /v1/chat/completions` and
`POST /v1/completions` (one text prompt or one flat token-ID list). Streaming is
incremental and supports cancellation. Sampling supports temperature/top_p;
usage reports full input and cached input separately. Specify either
`max_tokens` or `max_completion_tokens`, not both.

Tools, Responses, multimodal inputs, embeddings, structured output and multiple
candidates are outside the supported text subset. See the
[complete compatibility matrix and async/third-party examples](docs/openai-serving.md).

## Configuration

Edit YAML or use the following common CLI overrides:

| Setting | Purpose |
| --- | --- |
| `--model /path/to/model` | Override all instance checkpoint paths |
| `--gpus 0,1` | Assign one GPU to each model instance |
| `--scheduler-backend cpp` / `python` | Select the scheduling data path |
| `--host 0.0.0.0 --port 18000` | Change the public listener |
| `--internal-base-port 18001 --dist-port 18334` | Change reserved internal ports |
| `--max-model-len 131072` | Set the requested runtime context limit |
| `--ignore-model-max-length` | Explicitly bypass the original model-config cap |
| `--print-config` | Print the validated effective configuration and exit |

Defaults bind the public gateway to `127.0.0.1`. For remote clients, configure an
appropriate listener and use that server's address in `base_url`. To require a
Bearer key, set `api_key_env` in YAML to an environment-variable name (for example
`SWIFTCACHE_API_KEY`), export that variable before launch, and supply its value
as the client's `api_key`.

Each model's YAML `engine` mapping controls `scheduler_backend`, context and
batch budgets, concurrency, warmup size, cache cap and elastic timing settings.
The `--print-config` command validates model paths, aliases, GPUs and port
conflicts without loading the models. Detailed options and the ownership
protocol are documented in [serving](docs/openai-serving.md),
[C++ scheduling](docs/cpp-scheduler.md) and [elastic cache](docs/elastic-cache.md).

## Long-prefix tests and validation

`configs/long_prefix_local.yaml` and `configs/long_prefix_remote.yaml` define
separate two-Qwen3-8B configurations for local and remote KV placement:

```bash
.venv/bin/python -m server.starter \
  --config configs/long_prefix_local.yaml \
  --model /path/to/Qwen3-8B \
  --gpus 0,1 \
  --scheduler-backend cpp \
  --max-model-len 131072 \
  --ignore-model-max-length
```

Use the remote YAML for the remote-placement experiment and check available
memory before choosing either configuration. The length override extends the
runtime position cache without changing weights or RoPE scaling. It retains
physical KV capacity checks; exceeding the model's original context window is
a system execution test, not a long-context accuracy guarantee.

Install the test dependencies and run the safety/differential checks:

```bash
.venv/bin/python -m pip install -r requirements-test.txt
# Build the C++ extension first for test_cpp_scheduler.py.
.venv/bin/python -m pytest -q \
  tests/test_elastic.py tests/test_serving.py tests/test_cpp_scheduler.py

# Run while the two-model service is ready; uses the public OpenAI endpoint.
.venv/bin/python tests/validate_openai.py
```

CPU benchmarks under `tests/bench_scheduler.py` import the real scheduler and
allocator, verify observed prefix hits and record native input conversion and
object retirement separately. GPU and HTTP benchmarks measure different paths;
large CPU metadata pools do not imply the same GPU KV capacity.

The measured C++ comparison reduces 100K/+1K CPU `schedule()` p50 from 7.211 ms
to 0.190 ms; the full scheduling lifecycle drops from 9.118 ms to 1.832 ms.
The actual single-GPU local-KV engine's request median changes from 499.139 ms
to 496.769 ms, while mean and p95 do not improve in that run. Scheduler speedup
must not be presented as the same end-to-end inference speedup.

See [C++ results and reproduction commands](docs/cpp-scheduler-results-2026-11-08.md)
and [Python/OpenAI long-prefix results](docs/performance-results-2026-11-08.md).
Raw artifacts are stored under `artifacts/cpp-scheduler/` and `artifacts/phase2/`
on the validation server; they are not committed to Git. Compact metric
summaries and charts are committed under `docs/benchmarks/`.

## Baselines

The independent [Aqua Qwen3-8B baseline](baselines/aqua/README.md) lives under
`baselines/aqua/` on branch `codex/aqua-baseline`. It uses official Aqua tensors,
a compatible memory coordinator and a minimal HF/SDPA executor, with producer
reclamation to DRAM, prefix reuse and logical CFS scheduling. It does not reuse
SwiftCache's scheduler, block allocator or Layer Stream Cache.

```bash
git fetch origin
git switch codex/aqua-baseline
.venv/bin/python -m pip install -r baselines/aqua/requirements.txt
.venv/bin/python -m baselines.aqua.run --gpus 0,2 --scenario validate
```

This is a functional baseline with serialized execution, not a reproduction of
the original Aqua vLLM/FlexGen throughput. See its README for model requirements,
the minimal API, memory limits, source provenance and reproduction commands.
