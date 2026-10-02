#!/usr/bin/env bash
set -euo pipefail
TASK_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$TASK_ROOT"
MODEL_SOURCE="${1:-FLAME2023/flame2023.pkl}"
if [ ! -f "$MODEL_SOURCE" ]; then
  echo "Missing model: $MODEL_SOURCE. Place the official model here or pass its path." >&2
  exit 2
fi
if [ ! -x .venv-legacy/bin/python ]; then
  uv python install 3.10.18
  uv venv .venv-legacy --python 3.10.18
fi
uv pip install --python .venv-legacy/bin/python 'pip==24.3.1' 'setuptools==69.5.1' 'wheel==0.45.1'
uv pip install --python .venv-legacy/bin/python --no-build-isolation \
  'numpy==1.23.5' 'scipy==1.10.1' 'chumpy==0.70'
CUDA_VISIBLE_DEVICES='' .venv-legacy/bin/python scripts/convert_flame.py \
  --input "$MODEL_SOURCE" --output assets/flame/flame2023_canonical.npz
# Public matching template from the FLAME/DECA author's repository, pinned commit.
CUDA_VISIBLE_DEVICES='' .venv/bin/python scripts/fetch_uv_template.py
CUDA_VISIBLE_DEVICES='' uv run --locked flame-uv --check-assets
