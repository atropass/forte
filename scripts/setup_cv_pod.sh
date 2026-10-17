#!/usr/bin/env bash
# One-time setup of a fresh CV GPU pod (image runpod/pytorch:*-cu128*-ubuntu2404, persistent volume at /workspace).
# Installs supervisor, a Python env with the pinned CV stack, and the RT-DETRv2 weights in /workspace/hf_home.
# Afterwards: copy the code (see GPU_ACCESS.md), write /workspace/secrets.env (CV_API_KEY=...), run scripts/start_cv.sh.
#   nohup bash setup_cv_pod.sh > /workspace/setup.log 2>&1 &
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq --no-install-recommends supervisor nginx curl procps >/dev/null
mkdir -p /workspace/logs /workspace/forte /workspace/hf_home
[ -x /workspace/venv/bin/python ] || python3 -m venv /workspace/venv
/workspace/venv/bin/pip install -q --upgrade pip
# same versions as the production CV service (results verified identical); torch wheel is the CUDA 13.0 build
/workspace/venv/bin/pip install -q torch==2.14.1 torchvision==0.29.1 transformers==5.20.1 tokenizers==0.23.3 \
  safetensors==0.8.0 huggingface_hub==2.2.0 timm==1.0.30 numpy==2.5.4 pillow==13.0.0 fastapi==0.143.3 \
  starlette==1.8.0 uvicorn==0.54.0 pydantic==2.14.1 pydantic_core==2.50.1 python-multipart==0.0.32 PyYAML==6.0.3 \
  httpx==0.28.1
HF_HOME=/workspace/hf_home /workspace/venv/bin/python -c \
  "from huggingface_hub import snapshot_download as d; print(d('PekingU/rtdetr_v2_r18vd'))"
/workspace/venv/bin/python -c "import torch; assert torch.cuda.is_available(), 'no CUDA'; print('torch', torch.__version__, 'cuda', torch.version.cuda)"
echo SETUP_DONE
