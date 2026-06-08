<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="./assets/sc.png">
    <img alt="SwiftCache" src="./assets/sc.png" width=55%>
  </picture>
</p>

<h3 align="center">
Efficient LLM Serving for Multi-turn Conversations with Heterogeneous KV Cache Sharing
</h3>

[![GitHub](https://img.shields.io/github/stars/beyondHJM/SwiftCache?style=social)](https://github.com/beyondHJM/SwiftCache)
[![GitHub license](https://img.shields.io/github/license/beyondHJM/SwiftCache)](https://github.com/beyondHJM/SwiftCache/blob/main/LICENSE.txt)

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


## Installation

### Prerequisites

- Python 3.10+
- PyTorch 2.x
- CUDA 12.x
- NVIDIA GPUs with NVLink support, at least 2 GPUs
- Sufficient GPU memory for at least one master/slave model setup

### Installation Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/your-org/SwiftCache.git
   cd SwiftCache

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   pip install -e .

3. Start a server:
   ```bash
   python ./server/starter.py

4. Run a example in a new terminal:
   ```bash
   python example.py