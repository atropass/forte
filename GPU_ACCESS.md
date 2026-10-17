# GPU Server & vLLM Access Guide

This guide details the deployment, architecture, endpoints, and team access for the parallel GPU cluster running on RunPod.

---

## 1. Multi-Server Cluster Overview

Three pods are deployed and running simultaneously:

| Feature | Pod 1: Fast Compliance VLM | Pod 2: High-Capacity Reasoning VLM | Pod 3: Dedicated Vision Microservice |
|---|---|---|---|
| **Pod ID** | `tw6jhre43eeunm` | `7pd5og5br2oe9c` | `ydfg26qe73blrr` |
| **GPU** | 1x **NVIDIA GeForce RTX 3090** (24 GB) | 1x **NVIDIA A100-SXM4-80GB** (80 GB) | 1x **NVIDIA L4** (24 GB, Ada Lovelace) |
| **Cloud Tier** | Community Cloud | Community Cloud | Secure Cloud (`EU-RO-1`) |
| **Hourly Cost** | **$0.22 / hour** | **$1.39 / hour** | **$0.59 / hour** |
| **Active Model** | **`Qwen/Qwen3-VL-4B-Instruct`** (or 2B via switcher) | **`Qwen/Qwen3.8-27B`** (Apache-2.0, BF16 native) | **`RT-DETRv2-r18vd`** + ByteTrack Tracking |
| **Base URL** | `https://tw6jhre43eeunm-8000.proxy.runpod.net/v1` | `https://7pd5og5br2oe9c-8000.proxy.runpod.net/v1` | `https://ydfg26qe73blrr-8000.proxy.runpod.net` |
| **Bearer Token** | `fv-prod-secret-forte-2026` | `fv-prod-secret-forte-2026` | `fv-prod-secret-forte-2026` |
| **Direct SSH** | `ssh -i ~/.ssh/id_ed25519 -p 42346 root@80.15.7.37` | `ssh -i ~/.ssh/id_ed25519 -p 20618 root@216.249.100.66` | `ssh -i ~/.ssh/id_ed25519 -p 13636 root@213.173.105.19` |
| **Primary Use** | Fast video compliance checks & dev testing | Heavy reasoning, deep visual verification, complex tasks | Real-time object detection (35ms), person tracking & trajectories |

---

## 2. Public API Endpoints

### Pod 1 — `Qwen3-VL-4B` / `Qwen2-VL-2B` (OpenAI-Compatible):
- **Base URL**: `https://tw6jhre43eeunm-8000.proxy.runpod.net/v1`
- **Health Check**: `https://tw6jhre43eeunm-8000.proxy.runpod.net/health`
- **Model List**: `https://tw6jhre43eeunm-8000.proxy.runpod.net/v1/models`
- **Completions**: `https://tw6jhre43eeunm-8000.proxy.runpod.net/v1/chat/completions`

### Pod 2 — `Qwen3.8-27B` (A100 80GB, OpenAI-Compatible):
- **Base URL**: `https://7pd5og5br2oe9c-8000.proxy.runpod.net/v1`
- **Health Check**: `https://7pd5og5br2oe9c-8000.proxy.runpod.net/health`
- **Model List**: `https://7pd5og5br2oe9c-8000.proxy.runpod.net/v1/models`
- **Completions**: `https://7pd5og5br2oe9c-8000.proxy.runpod.net/v1/chat/completions`

### Pod 3 — Dedicated Computer Vision Microservice (`forte-cv-vision-pod`):
- **Base URL**: `https://ydfg26qe73blrr-8000.proxy.runpod.net`
- **Service Info**: `GET /`
- **Health Check**: `GET /health` or `GET /cv/health`
- **Interactive Swagger Docs**: `GET /docs`
- **OpenAPI Schema**: `GET /openapi.json`
- **Object Detection**: `POST /detect` or `POST /cv/detect`
- **Multi-Object Tracking**: `POST /track` or `POST /cv/track`

---

## 3. Computer Vision API Instructions

The CV service on Pod 3 runs real-time **RT-DETRv2** with half-precision (FP16) on the NVIDIA L4 GPU (~35–100ms per frame). Supported COCO object classes: `person`, `laptop`, `bottle`, `cup`, `cell phone`, `suitcase`, `book`, `handbag`, `backpack`.

### 3.1 Object Detection (`POST /detect`)

The endpoint accepts input in **three formats**:
1. Multipart file upload (`file=@image.jpg`)
2. JSON with Base64-encoded image (`{"image_base64": "..."}`)
3. JSON with remote image URL (`{"url": "https://..."}`)

Optional parameters:
- `threshold` (float, e.g. `0.3`): minimum confidence score
- `classes` (list of str or comma-separated): filter specific classes (e.g. `["person", "laptop"]`)

#### Option A: Multipart File Upload
```bash
curl -s -X POST https://ydfg26qe73blrr-8000.proxy.runpod.net/detect \
  -H "Authorization: Bearer fv-prod-secret-forte-2026" \
  -F "file=@/path/to/frame.jpg"
```

#### Option B: Remote Image URL
```bash
curl -s -X POST https://ydfg26qe73blrr-8000.proxy.runpod.net/detect \
  -H "Authorization: Bearer fv-prod-secret-forte-2026" \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://images.unsplash.com/photo-1544005313-94ddf0286df2?w=640&q=80",
    "threshold": 0.4
  }'
```

#### Option C: Base64 JSON
```bash
IMAGE_B64=$(base64 -w 0 /path/to/frame.jpg)
curl -s -X POST https://ydfg26qe73blrr-8000.proxy.runpod.net/detect \
  -H "Authorization: Bearer fv-prod-secret-forte-2026" \
  -H "Content-Type: application/json" \
  -d "{\"image_base64\": \"$IMAGE_B64\"}"
```

**Response Format**:
```json
{
  "elapsed_ms": 35.04,
  "detections": [
    {
      "label": "person",
      "confidence": 0.9678,
      "box": [0.62, 120.56, 637.5, 948.0]
    }
  ]
}
```
*Note: `box` coordinates are `[x1, y1, x2, y2]` in image pixels.*

---

### 3.2 Multi-Object Tracking (`POST /track`)

Tracks bounding box associations across sequential video frames using ByteTrack-style greedy IoU matching (<0.1 ms latency).

```bash
curl -s -X POST https://ydfg26qe73blrr-8000.proxy.runpod.net/track \
  -H "Authorization: Bearer fv-prod-secret-forte-2026" \
  -H "Content-Type: application/json" \
  -d '{
    "frames": [
      {
        "frame_index": 0,
        "person_boxes": [[100, 100, 200, 200]]
      },
      {
        "frame_index": 1,
        "person_boxes": [[105, 102, 204, 202]]
      },
      {
        "frame_index": 2,
        "person_boxes": [[110, 105, 208, 205]]
      }
    ],
    "iou_threshold": 0.25,
    "max_misses": 4
  }'
```

**Response Format**:
```json
{
  "elapsed_ms": 0.09,
  "total_tracks": 1,
  "tracks": [
    {
      "track_id": 0,
      "first_frame": 0,
      "last_frame": 2,
      "trajectory": {
        "0": [100.0, 100.0, 200.0, 200.0],
        "1": [105.0, 102.0, 204.0, 202.0],
        "2": [110.0, 105.0, 208.0, 205.0]
      }
    }
  ]
}
```

---

## 4. VLM `curl` Examples

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

## 5. Team Python SDK Integration

Using `sdk/client.py`:

```python
from sdk.client import VLLMClient, CVClient

# 1. Computer Vision Service (Pod 3: NVIDIA L4)
cv = CVClient(
    base_url="https://ydfg26qe73blrr-8000.proxy.runpod.net",
    api_key="fv-prod-secret-forte-2026"
)

# Object Detection (local file, URL, or base64)
result = cv.detect("frame_001.jpg", threshold=0.35)
print(f"Inference: {result['elapsed_ms']} ms | Detections: {len(result['detections'])}")

# Multi-frame IoU Tracking
tracks = cv.track(frames=[
    {"frame_index": 0, "person_boxes": [[100, 100, 200, 200]]},
    {"frame_index": 1, "person_boxes": [[105, 102, 204, 202]]}
])
print("Tracks found:", tracks["total_tracks"])

# 2. Fast VLM Compliance (Pod 1: RTX 3090)
vlm_fast = VLLMClient(
    base_url="https://tw6jhre43eeunm-8000.proxy.runpod.net/v1",
    model="Qwen/Qwen3-VL-4B-Instruct",
    api_key="fv-prod-secret-forte-2026"
)
print("VLM Fast:", vlm_fast.chat("Is the safe door open?"))

# 3. High-Capacity Reasoning (Pod 2: A100 80GB)
vlm_heavy = VLLMClient(
    base_url="https://7pd5og5br2oe9c-8000.proxy.runpod.net/v1",
    model="Qwen/Qwen3.8-27B",
    api_key="fv-prod-secret-forte-2026"
)
print("VLM Heavy:", vlm_heavy.chat("Perform legal reasoning on article 24 clause 3."))
```

---

## 6. Model Switching on Pod 1

On Pod 1, switch between lightweight dev model and production accuracy model:

```bash
# Switch to lightweight dev model (Qwen2-VL-2B, ~4.5 GB VRAM, 150+ tok/s):
ssh -i ~/.ssh/id_ed25519 -p 42346 root@80.15.7.37 "/workspace/forte/scripts/switch_model.sh dev"

# Switch back to production model (Qwen3-VL-4B, ~8.5 GB VRAM):
ssh -i ~/.ssh/id_ed25519 -p 42346 root@80.15.7.37 "/workspace/forte/scripts/switch_model.sh prod"
```

---

## 7. Process Monitoring & Diagnostics

### Pod 1 (RTX 3090 - Fast VLM):
```bash
ssh -i ~/.ssh/id_ed25519 -p 42346 root@80.15.7.37 "supervisorctl -c /workspace/forte/configs/supervisord.conf status; nvidia-smi"
```

### Pod 2 (A100 80GB - Heavy VLM):
```bash
ssh -i ~/.ssh/id_ed25519 -p 20618 root@216.249.100.66 "supervisorctl -c /workspace/forte/configs/supervisord.conf status; nvidia-smi"
```

### Pod 3 (NVIDIA L4 - Dedicated CV Microservice):
```bash
ssh -i ~/.ssh/id_ed25519 -p 13636 root@213.173.105.19 "supervisorctl -c /workspace/forte/configs/supervisord.cv.conf status; nvidia-smi"
```
