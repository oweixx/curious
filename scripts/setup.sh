#!/usr/bin/env bash
set -euo pipefail
TASK_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$TASK_ROOT"
uv python install 3.11.13
uv sync --locked --python 3.11.13
# Verify installation on CPU; setup never initializes a GPU.
CUDA_VISIBLE_DEVICES='' uv run --locked python -c 'import torch; import flame_uv; print("PyTorch:", torch.__version__, "CUDA wheel:", torch.version.cuda)'
if [ -f FLAME2023/flame2023.pkl ]; then
  bash scripts/prepare_flame.sh
fi
