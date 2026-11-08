#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
BASE_PYTHON=${SWIFTCACHE_BASE_PYTHON:-python3}
"$BASE_PYTHON" -c 'import torch, flash_attn, triton; print("CUDA base:", torch.__version__, flash_attn.__version__, triton.__version__)'
"$BASE_PYTHON" -m venv --system-site-packages .venv
.venv/bin/python -m pip install -r requirements-serving.txt
.venv/bin/python - <<'PY'
import sys,torch,flash_attn,triton,transformers,openai
print('Python:',sys.version)
print('PyTorch:',torch.__version__,'CUDA:',torch.version.cuda)
print('FlashAttention:',flash_attn.__version__)
print('Triton:',triton.__version__)
print('Transformers:',transformers.__version__)
print('OpenAI SDK:',openai.__version__)
PY
