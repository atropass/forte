# Runbook

Commands assume the compose stack: `frontend` (nginx, public :8080) → `api` → Redis, plus `worker`; files on the `data` volume.
Admin CLI: `docker compose exec api python -m service.admin …`.

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
1. Freeze the server: pin `IMAGE_TAG` in `.env` to the tested `sha-…` and do not run `docker compose pull`.
2. `curl /readyz?strict=1` → 200, `calibration.model_match` true, no warnings.
3. One end-to-end check: `python sdk/client.py`-style call or `make eval-api URL=… KEY=… ` with one video.
4. Watch `fv_queue_depth` and `fv_job_errors` during the run.

## A GPU server is down (`fv_upstream_up{service} == 0`, jobs fail with `inference_unavailable`)
* Jobs wait up to `INFRA_WAIT_S` for the server, retry once, then fail with `inference_unavailable` (never a fake verdict).
* On the GPU server: `supervisorctl -c /workspace/forte/configs/supervisord.conf status`, logs in `/workspace/logs/`.
  The watchdog restarts a hung vLLM / CV service automatically after 3 failed health checks.
* Pod gone: start a new one, `git pull`, recreate `/workspace/secrets.env`, run `scripts/setup_gpu_prod.sh` /
  `scripts/start_cv.sh`, update `VLM_API_BASE` / `DET_API_BASE` secrets, restart workers.

## Queue is backing up (`fv_queue_depth` grows, queue wait > 60 s)
* Check that workers are alive (`/readyz` → `workers`). Restart: `docker compose restart worker`.
* Add capacity: raise `WORKER_COUNT` (watch per-job processing time: the single GPU is shared; past a point more
  parallel jobs make each one slower — measure with `make loadtest`), or `docker compose up -d --scale worker=2`.
* Jury jobs (API keys) are always taken before web-UI jobs.

## A job is stuck in "processing"
It cannot stay there: the job watchdog fails it at `JOB_TIMEOUT_S`, and the reaper re-queues or fails jobs whose
worker stopped heart-beating (within ~30–45 s). To inspect: `GET /api/v1/jobs/<id>/progress`, worker logs
(`docker compose logs worker | grep <job_id>`), stored log `jobs/<id>/log.txt`.

## Keys and accounts
```bash
docker compose exec api python -m service.admin create-api-key --name jury
docker compose exec api python -m service.admin revoke-api-key jury
docker compose exec api python -m service.admin reset-password <user>
```
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
* Deploy: set `IMAGE_TAG=sha-<commit>` in `.env`, `docker compose pull && docker compose up -d --no-build`.
* Rollback: the previous `IMAGE_TAG`, same commands.
