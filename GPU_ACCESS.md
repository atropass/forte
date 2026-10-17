# GPU Server & vLLM Access Guide

This guide details how to access the GPU pod running the **Qwen3-VL-4B-Instruct** model on RunPod.

---

## 1. Server Specification

- **Provider**: RunPod (Community Cloud)
- **Pod ID**: `h2mcn28299gso1`
- **GPU**: 1x NVIDIA GeForce RTX 3090 (24 GB VRAM)
- **Hourly Cost**: $0.22 / hour
- **Persistent Volume**: 50 GB mounted at `/workspace`
- **Model Hosted**: `Qwen/Qwen3-VL-4B-Instruct` (served with vLLM)

---

## 2. Public API Access (OpenAI-Compatible)

The server exposes standard OpenAI-compatible endpoints directly over HTTPS:

- **Base URL**: `https://h2mcn28299gso1-8000.proxy.runpod.net/v1`
- **Health Check**: `https://h2mcn28299gso1-8000.proxy.runpod.net/health`
- **Model List**: `https://h2mcn28299gso1-8000.proxy.runpod.net/v1/models`
- **Chat/Vision Endpoint**: `https://h2mcn28299gso1-8000.proxy.runpod.net/v1/chat/completions`

### Example: Text Query (`curl`)

```bash
curl -s https://h2mcn28299gso1-8000.proxy.runpod.net/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-VL-4B-Instruct",
    "messages": [
      {"role": "user", "content": "Hello, what model are you?"}
    ],
    "max_tokens": 50
  }'
```

### Example: Multimodal / Vision Query (`curl`)

```bash
curl -s https://h2mcn28299gso1-8000.proxy.runpod.net/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-VL-4B-Instruct",
    "messages": [
      {
        "role": "user",
        "content": [
          {"type": "text", "text": "Describe what is visible in this image."},
          {
            "type": "image_url",
            "image_url": {
              "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/640px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
            }
          }
        ]
      }
    ],
    "max_tokens": 100
  }'
```

### Example: Python OpenAI Client

```python
from openai import OpenAI

client = OpenAI(
    base_url="https://h2mcn28299gso1-8000.proxy.runpod.net/v1",
    api_key="empty",  # vLLM runs unauthenticated by default
)

response = client.chat.completions.create(
    model="Qwen/Qwen3-VL-4B-Instruct",
    messages=[
        {"role": "user", "content": "Explain cash compliance rules in one sentence."}
    ],
)
print(response.choices[0].message.content)
```

---

## 3. SSH Shell Access

Your registered SSH key (`~/.ssh/id_ed25519`) has authorized access to the pod.

### Direct SSH (Recommended)

```bash
ssh -i ~/.ssh/id_ed25519 -p 8655 root@64.119.209.250
```

### RunPod Proxy SSH (Alternative)

```bash
ssh -i ~/.ssh/id_ed25519 h2mcn28299gso1-64411c0f@ssh.runpod.io
```

---

## 4. Server Management & Maintenance

Once connected via SSH:

- **Check GPU Status**:
  ```bash
  nvidia-smi
  ```

- **Live Service Logs**:
  ```bash
  tail -f /workspace/vllm.log
  ```

- **Environment & Paths**:
  - Python Virtual Environment: `/workspace/venv`
  - Model Weights Cache: `/workspace/hf_home`
  - Service Log: `/workspace/vllm.log`
  - Repository Code: `/workspace/forte`

- **Restarting vLLM**:
  If the server process ever needs a restart:
  ```bash
  pkill -f vllm
  export HF_HOME=/workspace/hf_home
  setsid /workspace/venv/bin/python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-VL-4B-Instruct \
    --host 0.0.0.0 \
    --port 8000 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.85 \
    --trust-remote-code > /workspace/vllm.log 2>&1 < /dev/null &
  ```
