# Source provenance

Paper: **Aqua: Network-Accelerated Memory Offloading for LLMs in Scale-Up GPU
Domains**, Abhishek Vijaya Kumar, Gianni Antichi and Rachee Singh, ASPLOS 2025.
[Paper v3](https://arxiv.org/html/2407.21255v3),
[official documentation](https://aquaml.github.io/).

Pinned official repositories:

| Repository | Commit | Use here |
| --- | --- | --- |
| [aquaml/aqua](https://github.com/aquaml/aqua) | `0ed7c15c07f5387c71b8eeaec48e4921c79d18d2` | Vendored Aqua tensor, manager, policy interfaces and upstream policy/client references |
| [aquaml/coordinator](https://github.com/aquaml/coordinator) | `222d9a94c7e9682fc1fde52f3ab37e9ba7f27418` | Protocol reference; coordinator here is independently implemented |
| [aquaml/vllm](https://github.com/aquaml/vllm) | `a81d9e6d776f7fba09deca92734b4ccf591ecd50` | CFS reference; the old serving engine is not built or imported |

`vendor/official_aqua/` retains Aqua's original AGPL-3.0 license. Its Python
sources namespace import paths to avoid global names such as `core` and `utils`,
and normalize trailing whitespace/final newlines; empty `__init__.py` files
provide package namespaces. Original source hashes and commits are recorded in
[upstream.lock.json](upstream.lock.json).
The original library documentation remains available in the linked repository.

Executable adaptations are outside `vendor/`:

- `runtime.py`: bounded HTTP requests, explicit GPU identity, safe reclamation
  acknowledgment, release accounting, CPU-only expansion candidates and direct
  KV writes into Aqua's selected backing tensor.
- `coordinator.py`: one locked ledger for offers, mappings and allocations;
  reclaim blocks new grants until acknowledgments and capacity changes finish.
- `engine.py`: Qwen3's native HF cache adapter and serialized logical CFS plan.

The new baseline implementation is provided under the accompanying AGPL-3.0
license, retaining the upstream license in the vendor directory as well.
