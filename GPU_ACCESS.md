# GPU Server & vLLM Access Guide

This guide details the deployment, architecture, model switching, and team access for the vLLM service running on RunPod.

---

## 1. Server Specification

- **Provider**: RunPod
- **Pod ID**: `hwghc0pbdxkkfo`
- **GPU**: 1x NVIDIA GeForce RTX 3090 (24 GB VRAM)
- **Hourly Cost**: $0.50 / hour
- **Persistent Volume**: 50 GB mounted at `/workspace`
- **Default Active Model (Dev Testing)**: `Qwen/Qwen2-VL-2B-Instruct` (Fast, lightweight 2.2B VLM)
- **Production Model**: `Qwen/Qwen3-VL-4B-Instruct` (Available via 1-click model switcher)
- **Process Supervisor**: `supervisord` (auto-restarts vLLM and Nginx on crash or OOM)

---

## 2. Public API Endpoints (OpenAI-Compatible)

The server exposes standard OpenAI-compatible endpoints directly over HTTPS:

- **Base URL**: `https://hwghc0pbdxkkfo-8000.proxy.runpod.net/v1`
- **Bearer Token**: `fv-prod-secret-forte-2026`
- **Health Check**: `https://hwghc0pbdxkkfo-8000.proxy.runpod.net/health`
- **Prometheus Metrics**: `https://hwghc0pbdxkkfo-8000.proxy.runpod.net/metrics`
- **Model List**: `https://hwghc0pbdxkkfo-8000.proxy.runpod.net/v1/models`
- **Chat / Vision**: `https://hwghc0pbdxkkfo-8000.proxy.runpod.net/v1/chat/completions`

---

## 3. Switching Models (Dev vs Prod)

The pod includes an automated model switcher (`scripts/switch_model.sh`):

### Switch to Lightweight Dev Model (Default, Ultra-Fast):
```bash
ssh -i ~/.ssh/id_ed25519 -p 40063 root@213.192.2.117 "/workspace/forte/scripts/switch_model.sh dev"
```
* **Model**: `Qwen/Qwen2-VL-2B-Instruct`
* **VRAM**: ~4.5 GB, 230+ tokens/sec throughput, instant testing.

### Switch to Production Accuracy Model:
```bash
ssh -i ~/.ssh/id_ed25519 -p 40063 root@213.192.2.117 "/workspace/forte/scripts/switch_model.sh prod"
```
* **Model**: `Qwen/Qwen3-VL-4B-Instruct`
* **VRAM**: ~8.5 GB, full legal/banking compliance prompt fidelity.

---

## 4. Team Integration (For 3 Developers)

### A. Python Client (`sdk/client.py`)

```python
from sdk.client import VLLMClient

client = VLLMClient(
    base_url="https://hwghc0pbdxkkfo-8000.proxy.runpod.net/v1",
    api_key="fv-prod-secret-forte-2026",
    model="Qwen/Qwen2-VL-2B-Instruct"  # or "Qwen/Qwen3-VL-4B-Instruct"
)

# Text Query
print(client.chat("Explain cash drawer audit rules in 2 sentences."))

# Vision Query (local image file or remote URL)
print(client.analyze_image("What objects do you see on the desk?", "path/to/frame.jpg"))
```

### B. TypeScript / Next.js Client (`sdk/client.ts`)

```typescript
import { VLLMClient } from "@/sdk/client";

const client = new VLLMClient({
  baseUrl: "https://hwghc0pbdxkkfo-8000.proxy.runpod.net/v1",
  apiKey: "fv-prod-secret-forte-2026",
  model: "Qwen/Qwen2-VL-2B-Instruct",
});

const answer = await client.chat("Is the cashier desk unattended?");
```

### C. Direct `curl`

```bash
curl -s https://hwghc0pbdxkkfo-8000.proxy.runpod.net/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer fv-prod-secret-forte-2026" \
  -d '{
    "model": "Qwen/Qwen2-VL-2B-Instruct",
    "messages": [
      {"role": "user", "content": "Hello!"}
    ],
    "max_tokens": 50
  }'
```

---

## 5. Concurrency Benchmark (3 Concurrent Users)

Tested with `scripts/bench_vllm_prod.py`:

```
User     Status     Latency (s)    Tokens   Tok/s   
-----------------------------------------------------------------
User 1   success    1.10           87       78.9    
User 2   success    1.10           87       78.8    
User 3   success    1.08           84       77.6    
-----------------------------------------------------------------
Total Wall Time:          1.11 s
Total Tokens Generated:   258
Aggregate Throughput:     232.4 tokens/sec
Average Per-User Latency: 1.09 s
```

---

## 6. SSH Direct Access

Connect directly:
```bash
ssh -i ~/.ssh/id_ed25519 -p 40063 root@213.192.2.117
```

Inside the pod:
- **Check Status**: `supervisorctl -c /workspace/forte/configs/supervisord.conf status`
- **Monitor Logs**: `tail -f /workspace/logs/vllm.log`
- **GPU Stats**: `nvidia-smi`
