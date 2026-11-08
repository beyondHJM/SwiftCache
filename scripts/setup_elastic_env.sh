#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
BASE_PYTHON=${SWIFTCACHE_BASE_PYTHON:-/root/miniconda3/envs/nano/bin/python}
"$BASE_PYTHON" -m venv --system-site-packages .venv
.venv/bin/python -m pip install transformers==4.51.0 pyzmq pytest fastapi uvicorn pillow sentencepiece xxhash accelerate
.venv/bin/python - <<'PY'
import sys,torch,flash_attn,triton,transformers
print('Python:',sys.version)
print('PyTorch:',torch.__version__,'CUDA:',torch.version.cuda)
print('FlashAttention:',flash_attn.__version__)
print('Triton:',triton.__version__)
print('Transformers:',transformers.__version__)
PY
