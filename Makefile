# Settings come from .env (copy .env.example). Main flows:
#   make setup        once: Python venv + UI dependencies
#   make redis        local Redis in Docker (needed by `make run`)
#   make run          API + UI on http://localhost:8000 and the workers, VLM = deployed Qwen (from .env)
#   make up           production stack in Docker (api + worker + Redis + MinIO): docker-compose.yml
#   make test         tests without models / Redis / GPU
-include .env
export VLM_BACKEND VLM_MODEL VLM_API_BASE VLM_API_KEY VLM_CONCURRENCY DET_API_BASE DET_API_KEY DET_CONCURRENCY \
       WORKER_COUNT JOB_TIMEOUT_S MAX_UPLOAD_MB API_KEYS METRICS_TOKEN

PY ?= .venv/bin/python
PORT ?= 8000
DEV_ENV = APP_ENV=development COOKIE_SECURE=0 LOG_FORMAT=text REDIS_URL=redis://127.0.0.1:6379/0 \
          STORAGE_BACKEND=fs DATA_DIR=$(CURDIR)/data_service CORS_ORIGINS=http://localhost:3000 \
          PUBLIC_ORIGIN=http://localhost:$(PORT),http://localhost:3000 HF_HUB_OFFLINE=1

.PHONY: help setup ui redis run api worker dev up down admin check-vlm calibrate eval eval-report eval-api loadtest audit feasibility bench test deploy
help:             ## list commands
	@grep -E '^[a-z-]+:.*##' Makefile | sed 's/:.*##/\t/'

setup:            ## once: Python venv + UI dependencies
	test -d .venv || python3 -m venv .venv
	$(PY) -m pip install -q -r requirements-dev.txt
	cd forte-vision-ai-development && pnpm install

ui:               ## build the static UI (forte-vision-ai-development/out)
	cd forte-vision-ai-development && pnpm typecheck && NEXT_TELEMETRY_DISABLED=1 pnpm build

redis:            ## local Redis for development (Docker)
	docker run -d --rm --name forte-redis -p 127.0.0.1:6379:6379 redis:7.4-alpine >/dev/null && echo "redis on 127.0.0.1:6379"

api:              ## API + UI only (http://localhost:$(PORT))
	$(DEV_ENV) $(PY) -m uvicorn service.app:app --host 127.0.0.1 --port $(PORT)

worker:           ## workers only (WORKER_COUNT from .env)
	$(DEV_ENV) $(PY) -m service.worker_main

run: ui           ## API + UI + workers locally (needs `make redis`)
	@echo "UI: http://localhost:$(PORT)   accounts: make admin ARGS='create-user <name>'   VLM: $(VLM_BACKEND) $(VLM_MODEL)"
	$(DEV_ENV) $(PY) -m service.worker_main & \
	  trap 'kill %1' EXIT; $(DEV_ENV) $(PY) -m uvicorn service.app:app --host 127.0.0.1 --port $(PORT)

admin:            ## admin CLI locally: make admin ARGS="create-api-key --name jury"
	$(DEV_ENV) $(PY) -m service.admin $(ARGS)

dev:              ## UI hot-reload on :3000 against the API on :$(PORT)
	cd forte-vision-ai-development && NEXT_PUBLIC_API_BASE=http://localhost:$(PORT) pnpm dev

up:               ## production stack in Docker (docker-compose.yml)
	docker compose up -d --build

down:             ## stop the Docker stack (volumes are kept)
	docker compose down

check-vlm:        ## smoke test of the VLM backend (reachability, model, latency)
	$(PY) scripts/check_vlm.py

calibrate:        ## tune thresholds on the training videos with the deployed model -> pipeline/thresholds.yaml
	$(PY) scripts/calibrate.py

eval:             ## run the analyzer on all training videos (VLM from .env) -> reports/eval.md
	$(PY) -m pipeline.eval

eval-report:      ## recompute eval metrics from saved results
	$(PY) -m pipeline.eval --reuse

eval-api:         ## evaluate a running service through its public API: make eval-api URL=https://host KEY=fv_...
	$(PY) scripts/eval_remote.py --url $(or $(URL),http://localhost:$(PORT)) --key $(KEY)

loadtest:         ## burst test: make loadtest URL=https://host KEY=fv_... BURST=10
	$(PY) scripts/loadtest.py --url $(URL) --key $(KEY) --videos open/ --burst $(or $(BURST),10) --unique --out reports/loadtest.md

test:             ## tests without models (API contract, security, queue, CV service, logic)
	$(PY) -m pytest tests -q

audit:            ## Phase 1: labels.csv + audit report
	$(PY) scripts/audit.py

feasibility:      ## Phase 2: per-question VLM accuracy
	$(PY) scripts/feasibility.py --model $(VLM_MODEL) --backend $(VLM_BACKEND)

bench:            ## time budget for a 145 s video
	$(PY) scripts/bench.py

deploy:           ## deploy committed HEAD to Fly.io
	scripts/deploy_fly.sh
