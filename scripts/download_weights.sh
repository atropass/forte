#!/usr/bin/env bash
# Download all model weights once (needs internet). Afterwards the service runs fully offline.
#   GPU server:  scripts/download_weights.sh            -> ./weights (mounted into the backend container)
#   Mac (MLX):   scripts/download_weights.sh --mac      -> ~/.cache/huggingface
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ "${1:-}" == "--mac" ]]; then
  MODELS=(mlx-community/Qwen3-VL-4B-Instruct-4bit PekingU/rtdetr_v2_r18vd)
else
  export HF_HOME="$PWD/weights"
  MODELS=(Qwen/Qwen3-VL-4B-Instruct PekingU/rtdetr_v2_r18vd)
fi
HF=$(command -v hf || command -v huggingface-cli || echo "")
if [[ -z "$HF" ]]; then pip install -q "huggingface_hub[cli]"; HF=$(command -v hf || command -v huggingface-cli); fi
for m in "${MODELS[@]}"; do echo "== $m"; "$HF" download "$m" >/dev/null; done
echo "weights ready${HF_HOME:+ in $HF_HOME}"
