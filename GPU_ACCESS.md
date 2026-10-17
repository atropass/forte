# GPU Server & vLLM Access Guide

This guide details the deployment, architecture, endpoints, and team access for the parallel GPU pods running on RunPod.

---

## 1. Parallel Cluster Overview

Two pods are deployed and running simultaneously:

| Feature | Pod 1: Fast Compliance VLM | Pod 2: High-Capacity Reasoning VLM |
|---|---|---|
| **Pod ID** | `tw6jhre43eeunm` | `7pd5og5br2oe9c` |
| **GPU** | 1x **NVIDIA GeForce RTX 3090** (24 GB) | 1x **NVIDIA A100-SXM4-80GB** (80 GB) |
| **Cloud Tier** | Community Cloud | Community Cloud |
| **Hourly Cost** | **$0.22 / hour** | **$1.39 / hour** |
| **Active Model** | **`Qwen/Qwen3-VL-4B-Instruct`** (or 2B via switcher) | **`Qwen/Qwen3.8-27B`** (Apache-2.0, BF16 native) |
| **Base URL** | `https://tw6jhre43eeunm-8000.proxy.runpod.net/v1` | `https://7pd5og5br2oe9c-8000.proxy.runpod.net/v1` |
| **Bearer Token** | `fv-prod-secret-forte-2026` | `fv-prod-secret-forte-2026` |
| **Direct SSH** | `ssh -i ~/.ssh/id_ed25519 -p 42346 root@80.15.7.37` | `ssh -i ~/.ssh/id_ed25519 -p 20618 root@216.249.100.66` |
| **Primary Use** | Fast video compliance pipeline & dev testing | Heavy reasoning, deep visual verification, complex tasks |

---

## 2. Public API Endpoints (OpenAI-Compatible)

Both servers implement the standard OpenAI REST specification over secure HTTPS:

### Pod 1 — `Qwen3-VL-4B` / `Qwen2-VL-2B`:
- **Base URL**: `https://tw6jhre43eeunm-8000.proxy.runpod.net/v1`
- **Health Check**: `https://tw6jhre43eeunm-8000.proxy.runpod.net/health`
- **Model List**: `https://tw6jhre43eeunm-8000.proxy.runpod.net/v1/models`
- **Completions**: `https://tw6jhre43eeunm-8000.proxy.runpod.net/v1/chat/completions`

### Pod 2 — `Qwen3.8-27B` (A100 80GB):
- **Base URL**: `https://7pd5og5br2oe9c-8000.proxy.runpod.net/v1`
- **Health Check**: `https://7pd5og5br2oe9c-8000.proxy.runpod.net/health`
- **Model List**: `https://7pd5og5br2oe9c-8000.proxy.runpod.net/v1/models`
- **Completions**: `https://7pd5og5br2oe9c-8000.proxy.runpod.net/v1/chat/completions`

---

## 3. Verified `curl` Examples

### Query Pod 1 (`Qwen3-VL-4B-Instruct`):
```bash
curl -s https://tw6jhre43eeunm-8000.proxy.runpod.net/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer fv-prod-secret-forte-2026" \
  -d '{
    "model": "Qwen/Qwen3-VL-4B-Instruct",
    "messages": [
      {"role": "user", "content": "Explain bank safe compliance in one sentence."}
    ],
    "max_tokens": 50
  }'
```

### Query Pod 2 (`Qwen3.8-27B` on A100):
```bash
curl -s https://7pd5og5br2oe9c-8000.proxy.runpod.net/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer fv-prod-secret-forte-2026" \
  -d '{
    "model": "Qwen/Qwen3.8-27B",
    "messages": [
      {"role": "user", "content": "What are your primary multimodal capabilities?"}
    ],
    "max_tokens": 60
  }'
```

---

## 4. Team Integration (Python SDK)

Using `sdk/client.py`:

```python
from sdk.client import VLLMClient

# Connect to Pod 1 (Fast compliance checks)
client_fast = VLLMClient(
    base_url="https://tw6jhre43eeunm-8000.proxy.runpod.net/v1",
    model="Qwen/Qwen3-VL-4B-Instruct",
    api_key="fv-prod-secret-forte-2026"
)

# Connect to Pod 2 (Deep reasoning 27B model)
client_heavy = VLLMClient(
    base_url="https://7pd5og5br2oe9c-8000.proxy.runpod.net/v1",
    model="Qwen/Qwen3.8-27B",
    api_key="fv-prod-secret-forte-2026"
)

print("Pod 1:", client_fast.chat("Is the safe door open?"))
print("Pod 2:", client_heavy.chat("Perform legal reasoning on article 24 clause 3."))
```

---

## 5. Model Switching on Pod 1

On Pod 1, switch between lightweight dev model and production accuracy model:

```bash
# Switch to lightweight dev model (Qwen2-VL-2B, ~4.5 GB VRAM, 150+ tok/s):
ssh -i ~/.ssh/id_ed25519 -p 42346 root@80.15.7.37 "/workspace/forte/scripts/switch_model.sh dev"

# Switch back to production model (Qwen3-VL-4B, ~8.5 GB VRAM):
ssh -i ~/.ssh/id_ed25519 -p 42346 root@80.15.7.37 "/workspace/forte/scripts/switch_model.sh prod"
```

---

## 6. Process Monitoring

### Pod 1 (RTX 3090):
```bash
ssh -i ~/.ssh/id_ed25519 -p 42346 root@80.15.7.37 "supervisorctl -c /workspace/forte/configs/supervisord.conf status; nvidia-smi"
```

### Pod 2 (A100 80GB):
```bash
ssh -i ~/.ssh/id_ed25519 -p 20618 root@216.249.100.66 "supervisorctl -c /workspace/forte/configs/supervisord.conf status; nvidia-smi"
```
