# Deploying in a bank's infrastructure

The service needs only standard components; nothing is tied to a cloud vendor.

| Component | Default (docker-compose.yml) | In a bank |
|---|---|---|
| API | `forte-vision` image, role `api` (stateless, port 8000) | Deployment behind the internal load balancer / ingress with TLS |
| Workers | same image, role `worker` (`WORKER_COUNT` processes per container) | Deployment, scaled by replicas |
| Queue + state | Redis 7 with AOF | managed Redis / Sentinel (`REDIS_URL`, `rediss://` supported) |
| Object storage | MinIO | Ceph RGW / any S3 API (`S3_ENDPOINT`, `S3_BUCKET`, `S3_ACCESS_KEY`, `S3_SECRET_KEY`, `S3_SSE`) |
| VLM | vLLM + Qwen3-VL-4B on a GPU server | on-prem GPU server (24 GB+), `VLM_API_BASE`, `VLM_API_KEY` |
| Detector | CV service (RT-DETRv2) on a GPU server | same or another GPU server, `DET_API_BASE`, `DET_API_KEY` |

## Network
* Inbound: only the API (HTTPS via the bank's proxy). Redis, S3 and the GPU servers stay internal.
* Outbound from API/workers: Redis, S3, the two GPU endpoints. Nothing else (no internet, no telemetry).
* GPU servers: model weights are copied once (`scripts/download_weights.sh` on a machine with access, then moved);
  they run with `HF_HUB_OFFLINE=1`.

## Configuration
Everything is environment variables (see `.env.example`); secrets come from the platform's secret store.
Minimum: `REDIS_URL`, `STORAGE_BACKEND=s3` + `S3_*`, `PUBLIC_ORIGIN`, `VLM_API_BASE`/`VLM_API_KEY`,
`DET_API_BASE`/`DET_API_KEY`, `METRICS_TOKEN`.

## Kubernetes
Reference manifests: `deploy/k8s/` (kustomize). Probes: liveness `GET /healthz`, readiness `GET /readyz`; worker
liveness `python -m service.worker_main check`; `terminationGracePeriodSeconds: 330` for workers so running analyses
finish during rollouts. Pods run as uid 10001 with a read-only root filesystem (`/tmp` as `emptyDir`).

```bash
kubectl create namespace forte-vision
kubectl -n forte-vision create secret generic forte-vision --from-env-file=prod.env
kubectl apply -k deploy/k8s
kubectl -n forte-vision exec deploy/forte-api -- python -m service.admin create-api-key --name jury
```

## Integration
Any system can call the public API (`POST /api/v1/analyze`, `GET /api/v1/jobs/{id}`) with an API key; see the
SDKs in `sdk/`. Results follow one JSON schema; failures return `status: failed` with a stable error code.
Prometheus scrapes `/metrics`; logs are JSON lines with `request_id` / `job_id` for the bank's log platform.

## Sizing
One RTX 3090 / L4-class GPU per model handles the per-video load of the pilot; throughput is bounded by the VLM
server. Measure with `scripts/loadtest.py` and size `WORKER_COUNT` so per-video processing stays under the limit.
For higher availability run two GPU servers per model behind the bank's load balancer.
