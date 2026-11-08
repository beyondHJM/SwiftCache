# Baselines

| Baseline | Location | Model | Scope |
| --- | --- | --- | --- |
| Aqua (ASPLOS 2025) | [aqua/](aqua/README.md) | Qwen3-8B | Official Aqua tensor storage with a minimal HF executor, elastic GPU/DRAM migration, prefix reuse and logical CFS planning |

The Aqua directory is an independent execution path. Its results are a functional
baseline; the serialized Hugging Face executor is not the original optimized
vLLM/FlexGen implementation or a reproduction of the paper's throughput.
