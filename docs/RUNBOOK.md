# Runbook

Two deployments, same commands with a different prefix:

| | Public (Fly.io) | Docker Compose (any host) |
|---|---|---|
| URL | https://forte-vision.fly.dev | `http://<host>:8080` (frontend nginx) |
| Admin CLI | `fly ssh console -a forte-vision -C "entrypoint admin …"` | `docker compose run --rm init admin …` |
| Logs | `fly logs -a forte-vision` | `docker compose logs -f api worker` |
| Restart | `fly machine restart -a forte-vision` | `docker compose restart api worker` |
| Redis data | `/data/redis` on the `forte_data` volume | `redis-data` volume |

## Health at a glance
```bash
curl -s https://<host>/healthz                 # api process alive (through the frontend nginx)
curl -s https://<host>/readyz | jq             # redis, storage, workers, GPU upstreams, queue depth, calibration
# /metrics is not exposed publicly (nginx returns 404); scrape the api service on the internal network:
docker compose exec api python -c "import os,urllib.request as u; print(u.urlopen(u.Request('http://127.0.0.1:8000/metrics', headers={'Authorization': 'Bearer ' + os.environ['METRICS_TOKEN']})).read().decode())" | grep '^fv_'
```
Key metrics: `fv_queue_depth{queue}`, `fv_jobs_active`, `fv_workers`, `fv_upstream_up{service}`,
`fv_jobs{status,source}`, `fv_job_errors{code}`, `fv_job_duration_seconds`, `fv_job_queue_wait_seconds`.

Uptime monitoring: point an external monitor at `/readyz?strict=1` (fails when no worker is alive or a GPU server is
unreachable) and alert after 2 consecutive failures.

## Before / during the jury evaluation
1. Freeze: every push to `main` deploys to Fly (CI) — do not push to `main` during the evaluation (Compose: pin
   `IMAGE_TAG` to the tested `sha-…`).
1. Fly machine size: at least `fly scale vm shared-cpu-4x --memory 4096 -a forte-vision`. On `shared-cpu-1x` / 1 GB
   the CPU (video decoding) is the bottleneck: 5 simultaneous videos took up to 234 s each while the GPUs idled.
   Then `make loadtest URL=https://forte-vision.fly.dev KEY=<jury key> BURST=5`: every video < 300 s.
2. `curl /readyz?strict=1` → 200, `calibration.model_match` true, no warnings.
3. One end-to-end check: `python sdk/client.py`-style call or `make eval-api URL=… KEY=… ` with one video.
4. Watch `fv_queue_depth` and `fv_job_errors` during the run.

## A GPU server is down (`fv_upstream_up{service} == 0`, jobs fail with `inference_unavailable`)
* Jobs wait up to `INFRA_WAIT_S` for the server, retry once, then fail with `inference_unavailable` (never a fake verdict).
* On the GPU server: `supervisorctl -c /workspace/forte/configs/supervisord.conf status`, logs in `/workspace/logs/`.
  The watchdog restarts a hung vLLM / CV service automatically after 3 failed health checks.
* Pod gone: create a new pod and set it up with GPU_ACCESS.md → "New vLLM pod from scratch" / "New CV pod from
  scratch" (pods have no git checkout; ~10 min), then update `VLM_API_BASE`/`VLM_API_KEY` or
  `DET_API_BASE`/`DET_API_KEY` (Fly: `fly secrets set …`, restarts the machine; Compose: `.env` +
  `docker compose up -d api worker`).
* CV pod down, quick fallback: the vLLM pod runs a copy of the CV service at `/cv/`
  (`DET_API_BASE=https://<vlm-pod>-8000.proxy.runpod.net/cv` with that pod's `CV_API_KEY`).

## Queue is backing up (`fv_queue_depth` grows, queue wait > 60 s)
* Check that workers are alive (`/readyz` → `workers`). Restart: `docker compose restart worker` / `fly machine restart`.
* Check where the time goes: if the GPU servers are idle (`nvidia-smi` on the pods) while jobs queue, the service host's
  CPU is the bottleneck (video decoding) — scale the host up, not `WORKER_COUNT`.
* Add capacity: raise `WORKER_COUNT` (watch per-job processing time: the single GPU is shared; past a point more
  parallel jobs make each one slower — measure with `make loadtest`), or `docker compose up -d --scale worker=2`.
* Jury jobs (API keys) are always taken before web-UI jobs.

## A job is stuck in "processing"
It cannot stay there: the job watchdog fails it at `JOB_TIMEOUT_S`, and the reaper re-queues or fails jobs whose
worker stopped heart-beating (within ~30–45 s). To inspect: `GET /api/v1/jobs/<id>/progress`, worker logs
(`docker compose logs worker | grep <job_id>`), stored log `jobs/<id>/log.txt`.

## Keys and accounts
```bash
admin create-api-key --name jury          # prefix as in the table above
admin revoke-api-key jury
admin create-user <user>                  # password printed once
admin reset-password <user>
```
On Fly the jury key can also be the `API_KEY` secret (`fly secrets set API_KEY=… -a forte-vision`).
Rotate GPU tokens: see GPU_ACCESS.md → "Redeploy / rotate keys".

## Redis
* Persistence: AOF (`appendfsync everysec`) on the `redis-data` volume.
* Lost Redis data = lost job records, accounts and API keys (results stay in storage). Recreate accounts/keys with the
  admin CLI. For HA in a bank deployment use the bank's managed Redis (Sentinel/cluster) via `REDIS_URL`.

## Storage
* `docker compose exec api python -m service.admin init-storage` checks access (creates the bucket if missing).
* Retention: `python -m service.admin purge --older-than-days 30` (cron) or an S3 lifecycle rule on `jobs/`.

## Deploys
* Workers finish running jobs on SIGTERM (`stop_grace_period: 330s`).
* Fly: every push to `main` deploys after CI; by hand `fly deploy --remote-only`. Rollback: `fly releases -a
  forte-vision --image` → `fly deploy -a forte-vision --image <previous image>`.
* Compose: set `IMAGE_TAG=sha-<commit>` in `.env`, `docker compose pull && docker compose up -d --no-build`.
  Rollback: the previous `IMAGE_TAG`, same commands.
