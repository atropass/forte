#!/usr/bin/env bash
# One-time setup of a vLLM GPU pod (image runpod/pytorch:*-cu128*-ubuntu2404, /workspace = local or network volume).
# Installs supervisor + nginx, a Python env with the exact versions of the production vLLM server, the Qwen3-VL-4B
# weights and the RT-DETRv2 weights (for the /cv/ copy of the CV service) into /workspace/hf_home.
# Afterwards: copy the code (GPU_ACCESS.md), write /workspace/secrets.env (VLLM_API_KEY=..., CV_API_KEY=...),
# run scripts/setup_gpu_prod.sh.
#   nohup bash setup_vllm_pod.sh > /workspace/setup.log 2>&1 &
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq --no-install-recommends supervisor nginx curl procps >/dev/null
mkdir -p /workspace/logs /workspace/forte /workspace/hf_home /workspace/.cache/vllm /workspace/.cache/torch_extensions
[ -x /workspace/venv/bin/python ] || python3 -m venv /workspace/venv
/workspace/venv/bin/pip install -q --upgrade pip
# versions of the production vLLM server (2026-10-18); vllm pins its own torch, the rest must match it
/workspace/venv/bin/pip install -q vllm==0.31.0 torch==2.13.0 torchvision==0.28.0 torchaudio==2.11.0 \
  transformers==5.17.0 tokenizers==0.23.3 safetensors==0.8.0 huggingface_hub==1.33.0 \
  flashinfer-python==0.7.0.post1 triton==3.7.1 timm==1.0.30 numpy==2.3.5 pillow==13.0.0 \
  fastapi==0.136.3 starlette==1.8.0 uvicorn==0.54.0 pydantic==2.14.1 pydantic_core==2.50.1 \
  python-multipart==0.0.32 PyYAML==6.0.3 httpx==0.28.1
HF_HOME=/workspace/hf_home /workspace/venv/bin/python - <<'PY'
from huggingface_hub import snapshot_download as d
for repo in ("Qwen/Qwen3-VL-4B-Instruct", "PekingU/rtdetr_v2_r18vd"):
    print(repo, d(repo))
PY
/workspace/venv/bin/python -c "import torch, vllm; assert torch.cuda.is_available(), 'no CUDA'; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'vllm', vllm.__version__)"
echo SETUP_DONE
