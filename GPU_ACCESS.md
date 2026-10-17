# GPU Production Service & vLLM Access Guide

This guide details the production deployment, architecture, and team access for the **Qwen3-VL-4B-Instruct** vision-language model running on RunPod.

---

## 1. Production Architecture Overview

The server runs a fully supervised, hardened production stack:

```
[ Developer 1 ]   [ Developer 2 ]   [ Developer 3 ]
       │                 │                 │
       └────────────┬────┴─────────────────┘
                    ▼
     https://h2mcn28299gso1-8000.proxy.runpod.net
                    │
         ┌──────────▼──────────┐
         │ Nginx Gateway (:8000)│
         │ - Bearer Token Auth │
         │ - Rate Limiting     │
         │ - Direct Streaming  │
         └──────────┬──────────┘
                    │ 127.0.0.1:8001
         ┌──────────▼──────────┐
         │ vLLM Engine (:8001) │
         │ - Chunked Prefill   │
         │ - Prefix Caching    │
         │ - PagedAttention    │
         │ - Supervisord Auto-R│
         └─────────────────────┘
```

### Server Specifications
- **Provider**: RunPod (GPU Cluster)
- **Pod ID**: `h2mcn28299gso1`
- **GPU**: 1x NVIDIA GeForce RTX 3090 (24 GB VRAM)
- **Cost**: $0.22 / hour (Dev/Staging) — scale to RTX 4090 / L40S Secure Cloud ($0.89/hr) with the same configs
- **Process Supervisor**: `supervisord` (auto-restarts vLLM and Nginx on crash or OOM)
- **Model**: `Qwen/Qwen3-VL-4B-Instruct` (FlashAttention-2 + Torch Compile)

---

## 2. Endpoints & Authentication

All API requests require the production Bearer token:

- **API Base URL**: `https://h2mcn28299gso1-8000.proxy.runpod.net/v1`
- **Bearer Token**: `fv-prod-secret-forte-2026`
- **Health Check**: `https://h2mcn28299gso1-8000.proxy.runpod.net/health` (unauthenticated for uptime monitors)
- **Prometheus Metrics**: `https://h2mcn28299gso1-8000.proxy.runpod.net/metrics`

---

## 3. Team Usage (For the 3 Developers)

### A. Python Client (`sdk/client.py`)

A client with retry logic, base64 vision encoding, and token authentication is provided in `sdk/client.py`:

```python
from sdk.client import VLLMClient

client = VLLMClient(
    base_url="https://h2mcn28299gso1-8000.proxy.runpod.net/v1",
    api_key="fv-prod-secret-forte-2026"
)

# 1. Text Query
answer = client.chat("Summarize the cashier compliance checklist.")
print(answer)

# 2. Vision / Image Query (local image or remote URL)
verdict = client.analyze_image(
    prompt="Is there an unattended visitor at the desk?",
    image_path_or_url="path/to/frame.jpg"
)
print(verdict)
```

### B. TypeScript / Next.js Client (`sdk/client.ts`)

```typescript
import { VLLMClient } from "@/sdk/client";

const client = new VLLMClient({
  baseUrl: process.env.NEXT_PUBLIC_VLLM_URL,
  apiKey: process.env.VLLM_API_KEY,
});

const response = await client.chat("Is the cash safe door closed?");
```

### C. Direct `curl`

```bash
curl -s https://h2mcn28299gso1-8000.proxy.runpod.net/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer fv-prod-secret-forte-2026" \
  -d '{
    "model": "Qwen/Qwen3-VL-4B-Instruct",
    "messages": [
      {"role": "user", "content": "Explain bank cashier desk security."}
    ],
    "max_tokens": 100
  }'
```

---

## 4. Concurrency Benchmark Results

Tested with `scripts/bench_vllm_prod.py` simulating 3 concurrent developers sending simultaneous requests:

```
User     Status     Latency (s)    Tokens   Tok/s   
-----------------------------------------------------------------
User 1   success    2.15           101      47.0    
User 2   success    2.07           107      51.8    
User 3   success    1.93           112      58.1    
-----------------------------------------------------------------
Total Wall Time:          2.15 s
Total Tokens Generated:   320
Aggregate Throughput:     148.8 tokens/sec
Average Per-User Latency: 2.05 s
```

---

## 5. SSH Access & Process Maintenance

Connect via direct SSH:
```bash
ssh -i ~/.ssh/id_ed25519 -p 8655 root@64.119.209.250
```

Inside the pod:
- **Check Supervisor Status**:
  ```bash
  supervisorctl -c /workspace/forte/configs/supervisord.conf status
  ```
- **Restart vLLM Service**:
  ```bash
  supervisorctl -c /workspace/forte/configs/supervisord.conf restart vllm
  ```
- **Inspect Live Logs**:
  ```bash
  tail -f /workspace/logs/vllm.log
  tail -f /workspace/logs/nginx_access.log
  ```
- **Re-run Automated Setup**:
  ```bash
  /workspace/forte/scripts/setup_gpu_prod.sh
  ```
