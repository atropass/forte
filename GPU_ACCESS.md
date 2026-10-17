# GPU servers (vLLM + CV service)

The analysis workers call two GPU services over HTTPS. Both run on team-controlled GPU pods (RunPod today; any
GPU host with Docker/NVIDIA works). **No URLs, tokens or SSH endpoints are kept in this repository**: they live in
the team password vault and in the deployment secrets (`VLM_API_BASE`, `VLM_API_KEY`, `DET_API_BASE`, `DET_API_KEY`).

| | VLM server | CV service |
|---|---|---|
| RunPod pod (name) | `forte-vllm-3090` | `forte-cv-3070` |
| Model | `Qwen/Qwen3-VL-4B-Instruct` (Apache-2.0) via vLLM, OpenAI-compatible API | `RT-DETRv2-r18vd` (Apache-2.0) + IoU tracking |
| GPU | 1× RTX 3090 24 GB, Community Cloud, $0.22/h, power limit **330 W** | 1× RTX 3070 8 GB, Community Cloud, **$0.13/h** (240 W) |
| Storage | `/workspace` 50 GB local volume (venv + weights in `/workspace/hf_home`) | `/workspace` 20 GB local volume (venv + weights in `/workspace/hf_home`) |
| Software | vLLM 0.31.0, torch 2.13.0 (CUDA 13.0 build), transformers 5.17.0 (`scripts/setup_vllm_pod.sh`) | torch 2.14.1 (CUDA 13.0 build), transformers 5.20.1 (`scripts/setup_cv_pod.sh`) |
| Gateway | nginx `:8000` → vLLM `127.0.0.1:8002` (`configs/nginx.vllm.conf`) | nginx `:8000` → uvicorn `127.0.0.1:8005` (`configs/nginx.cv.conf`) |
| Processes | supervisord, `configs/supervisord.conf` | supervisord, `configs/supervisord.cv.conf` |
| Auth | `Authorization: Bearer $VLLM_API_KEY` (enforced by vLLM) | `Authorization: Bearer $CV_API_KEY` (enforced by the service) |
| Health | `GET /health` | `GET /health` |
| Service env | `VLM_API_BASE=https://<vlm-host>/v1`, `VLM_API_KEY` | `DET_API_BASE=https://<cv-host>`, `DET_API_KEY` |

Both gateways are server-to-server only: no CORS, request bodies limited to 20 MB, `/metrics` and API docs not
exposed, GET/POST only. The VLM pod also runs a copy of the CV service at `/cv/` (`DET_API_BASE=https://<vlm-host>/cv`),
usable as a zero-cost fallback if the CV pod is down.

Clients must send their own `User-Agent` header: RunPod's proxy (Cloudflare) rejects Python's default `urllib` agent
with 403 "error code: 1010" (`httpx`, `curl` and the analysis service are fine). Both software stacks are CUDA 13.0
builds: create pods with min CUDA **13.0**.

History:
- VLM: until 2026-10-18 on `forte-vllm-dev-3090`, an RTX 3090 whose host capped power at 150 W (stock card: 350 W).
  Replaced by `forte-vllm-3090` (330 W limit, same versions and model); the same video (X-4) now finishes in 45 s
  instead of 62 s with an identical result.
- CV: the CV service ran on an L4 (Secure Cloud, $0.59/h) until 2026-10-18. It was replaced by the RTX 3070 after
measuring ~4–5 detector frames per video (L4 idle at 0 % GPU). On the same 24 frames the 3070 matched every detection the
pipeline uses (score ≥ 0.4: 181/181, max score difference 0.009).

### CV service batching
Concurrent `/detect` requests are batched into one forward pass (`service/cv_service.py`, `Batcher`):

| Env (supervisord / secrets.env) | Default | Meaning |
|---|---|---|
| `CV_MAX_BATCH` | 32 | frames per forward pass |
| `CV_BATCH_WAIT_MS` | 5 | how long the first request waits for others (20 ms measured no better) |
| `CV_INFER_WORKERS` | 2 | batches in flight (CPU pre-processing of one overlaps the GPU pass of another; measured +20 % at 8 concurrent requests, no gain at 32) |
| `CV_THRESHOLD_FLOOR` | `DET_THRESHOLD` (0.35) | score floor the model runs at; each request keeps `score > its threshold` |

Measured on the RTX 3070, 256 frames: no batching 46–52 frames/s at any concurrency; batching 79 frames/s at 32
concurrent requests (+62 %, latency −40 %). Past that point the CPU-side pre-processing (JPEG decode, resize,
normalise), not the GPU, is the limit. `GET /health` shows `batching.requests / batches / largest_batch`.

## New vLLM pod from scratch
1. RunPod: Community RTX 3090 (or any 24 GB GPU), image `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`, min CUDA
   13.0, 30 GB container disk, 50 GB **local** volume at `/workspace`, ports `8000/http`, `22/tcp`, SSH key from the
   account. Check `nvidia-smi --query-gpu=power.limit --format=csv`: some Community hosts cap a 3090 at 150 W; keep the
   pod only at ≥ 300 W.
2. `scp scripts/setup_vllm_pod.sh root@<pod>:/workspace/ && ssh root@<pod> 'nohup bash /workspace/setup_vllm_pod.sh > /workspace/setup.log 2>&1 &'`
   (exact production versions + Qwen3-VL-4B + RT-DETRv2 weights, ~5 min; ends with `SETUP_DONE`).
3. Copy the code: `tar -cf - configs/supervisord.conf configs/nginx.vllm.conf scripts/setup_gpu_prod.sh scripts/pod_watchdog.sh scripts/switch_model.sh service/__init__.py service/cv_service.py pipeline/__init__.py pipeline/perception.py | ssh root@<pod> 'cd /workspace/forte && tar --no-same-owner -xf - && chmod +x scripts/*.sh'`.
4. `/workspace/secrets.env` with new `VLLM_API_KEY` and `CV_API_KEY` (umask 077, written via stdin), then
   `nohup /workspace/forte/scripts/setup_gpu_prod.sh > /workspace/logs/setup_gpu_prod.log 2>&1 &` (healthy in ~2.5 min).
5. Check from outside: `/health` 200; `/v1/models` with the key 200, without 401; `/metrics` 403; one image request;
   `/cv/health`. Then set `VLM_API_BASE` / `VLM_API_KEY` on the analysis service and restart `api` + `worker`.

## New CV pod from scratch
1. RunPod: GPU pod, image `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`, min CUDA 13.0, 20 GB container disk,
   20 GB **local** volume at `/workspace` (a network volume makes cold starts take minutes), ports `8000/http`,
   `22/tcp`, SSH key from the account.
2. `scp scripts/setup_cv_pod.sh root@<pod>:/workspace/ && ssh root@<pod> 'nohup bash /workspace/setup_cv_pod.sh > /workspace/setup.log 2>&1 &'`
   (pinned CV stack + weights, ~5 min; ends with `SETUP_DONE`).
3. Copy the code: `tar -cf - service/__init__.py service/cv_service.py pipeline/__init__.py pipeline/perception.py configs/supervisord.cv.conf configs/nginx.cv.conf scripts/start_cv.sh scripts/pod_watchdog.sh | ssh root@<pod> 'cd /workspace/forte && tar --no-same-owner -xf -'`.
4. `/workspace/secrets.env` with a new `CV_API_KEY` (umask 077), then `scripts/start_cv.sh` (healthy in ~5 s).
5. Check from outside (`/health`, 401 without/with a wrong key), then set `DET_API_BASE` / `DET_API_KEY` on the
   analysis service and restart `api` + `worker`.

## Secure Cloud standby (planned, blocked)
Network volume + pod template per service in EU-RO-1 (RTX 4000 Ada $0.28/h for vLLM with `--max-num-seqs 16`,
RTX 2000 Ada $0.24/h for CV), booted by `scripts/standby_boot.sh vllm|cv`. Templates cannot carry a network volume:
the emergency pod is created from the template **with** the volume attached. Not built yet: RunPod requires an
account balance of at least $5 to create network volumes.

## Secrets on a pod

Each pod keeps its keys in `/workspace/secrets.env` (owner root, `chmod 600`, outside the code directory). On a
network volume (Secure standby) `chmod` has no effect; those pods have only the root user.

```bash
# VLM pod
VLLM_API_KEY=<random, >= 32 chars>
CV_API_KEY=<random, >= 32 chars>     # only if the CV service also runs on this pod
# CV pod
CV_API_KEY=<random, >= 32 chars>
```

Generate keys with `python3 -c "import secrets; print(secrets.token_urlsafe(32))"`. vLLM reads `VLLM_API_KEY`
from the environment, so the key never appears on a command line (`ps`). Use different keys for the two pods.

## Redeploy / rotate keys

The pods have no git checkout: `/workspace/forte` is updated by streaming the relevant files from a local clone
(`--no-same-owner`: the CV pod's network filesystem cannot change owners).

```bash
# from a local clone; VLM_SSH / CV_SSH = "ssh -i <key> -p <port> root@<host>" (team vault)
FILES="configs scripts service/cv_service.py pipeline/perception.py GPU_ACCESS.md sdk/client.py sdk/client.ts"
tar -cf - $FILES | $VLM_SSH 'cd /workspace/forte && tar --no-same-owner -xf - && chmod +x scripts/*.sh'
tar -cf - $FILES | $CV_SSH  'cd /workspace/forte && tar --no-same-owner -xf - && chmod +x scripts/*.sh'

# on the VLM pod: new VLLM_API_KEY (and CV_API_KEY) in /workspace/secrets.env, then
nohup /workspace/forte/scripts/setup_gpu_prod.sh > /workspace/logs/setup_gpu_prod.log 2>&1 &
# stops the old stack, waits until it is gone, starts vLLM + gateway + watchdog, waits for /health (model load: 2-5 min)

# on the CV pod: new CV_API_KEY in /workspace/secrets.env, then
/workspace/forte/scripts/start_cv.sh             # restarts the CV service + gateway (cold start ~2-3 min)
```

The watchdog (`scripts/pod_watchdog.sh`) never interrupts a starting service; it restarts vLLM / the CV service only
after they were healthy and then failed 3 checks in a row, or did not come up within 15 / 10 minutes.

Then update `VLM_API_KEY` / `DET_API_KEY` in the service's `.env` (or secret store) and restart the workers:

```bash
docker compose up -d api worker
```

Check from the service host:

```bash
curl -s -o /dev/null -w '%{http_code}\n' -H "Authorization: Bearer $VLM_API_KEY" "$VLM_API_BASE/models"   # 200
curl -s -o /dev/null -w '%{http_code}\n' "$VLM_API_BASE/models"                                           # 401
curl -s "$DET_API_BASE/health"                                                                             # {"status":"ok","ready":true,...}
```

## CV service API

`POST /detect` (also `/cv/detect`) accepts one image as multipart `file`, a raw `image/jpeg|png|webp` body, or JSON
`{"image_base64": "...", "threshold": 0.4, "classes": ["person"]}`. Remote URLs are not accepted.

```bash
curl -s -X POST "$DET_API_BASE/detect" -H "Authorization: Bearer $DET_API_KEY" -F "file=@frame.jpg"
# {"elapsed_ms": 35.0, "detections": [{"label": "person", "confidence": 0.97, "box": [x1, y1, x2, y2]}]}
```

`POST /track` (also `/cv/track`): IoU tracking over per-frame person boxes,
`{"frames": [{"frame_index": 0, "person_boxes": [[x1, y1, x2, y2]]}], "iou_threshold": 0.25, "max_misses": 4}`.

## Model switching on the VLM pod

```bash
scripts/switch_model.sh dev    # Qwen2-VL-2B, development only
scripts/switch_model.sh prod   # Qwen3-VL-4B, the model the thresholds are calibrated for
```

`/readyz` of the analysis service reports a warning when the served model differs from the calibrated one.

## Diagnostics on a pod

```bash
supervisorctl -c /workspace/forte/configs/supervisord.conf status      # VLM pod
supervisorctl -c /workspace/forte/configs/supervisord.cv.conf status   # CV pod
tail -f /workspace/logs/vllm_err.log /workspace/logs/cv_service_err.log
nvidia-smi
```
