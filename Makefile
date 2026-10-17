# Settings come from .env (copy .env.example). Main flow on this machine:
#   make setup        once: Python venv + UI dependencies
#   make check-vlm    is the deployed Qwen reachable?
#   make run          app (UI + API) on http://localhost:8000, VLM = deployed Qwen
-include .env
export

PY ?= .venv/bin/python
PORT ?= 8000
export HF_HUB_OFFLINE ?= 1
export CORS_ORIGINS ?= http://localhost:3000

.PHONY: help setup ui run dev check-vlm calibrate eval eval-report eval-api audit feasibility bench test weights serve
help:             ## list commands
	@grep -E '^[a-z-]+:.*##' Makefile | sed 's/:.*##/\t/'

setup:            ## once: Python venv + UI dependencies (+ detector weights)
	test -d .venv || python3 -m venv .venv
	$(PY) -m pip install -q -r requirements-mac.txt
	cd forte-vision-ai-development && pnpm install
	HF_HUB_OFFLINE=0 $(PY) -c "from huggingface_hub import snapshot_download as d; d('PekingU/rtdetr_v2_r18vd')"

ui:               ## build the static UI (forte-vision-ai-development/out)
	cd forte-vision-ai-development && NEXT_TELEMETRY_DISABLED=1 pnpm build

run: ui           ## UI + API on http://localhost:$(PORT); VLM from .env (deployed Qwen)
	@echo "UI: http://localhost:$(PORT)   API key: $(API_KEY)   VLM: $(VLM_BACKEND) $(VLM_MODEL)"
	$(PY) -m uvicorn service.app:app --host 0.0.0.0 --port $(PORT)

dev:              ## UI hot-reload on :3000 against the API on :$(PORT) (run `make run` in another terminal)
	cd forte-vision-ai-development && NEXT_PUBLIC_API_BASE=http://localhost:$(PORT) pnpm dev

check-vlm:        ## smoke test of the VLM backend (reachability, model, latency)
	$(PY) scripts/check_vlm.py

calibrate:        ## tune thresholds on the training videos with the deployed model -> pipeline/thresholds.yaml
	$(PY) scripts/calibrate.py

eval:             ## run the analyzer on all training videos (VLM from .env) -> reports/eval.md
	$(PY) -m pipeline.eval

eval-report:      ## recompute eval metrics from saved results
	$(PY) -m pipeline.eval --reuse

eval-api:         ## evaluate a running app through its public API: make eval-api URL=http://host:8000
	$(PY) scripts/eval_remote.py --url $(or $(URL),http://localhost:$(PORT)) --key $(API_KEY)

test:             ## tests without models (logic, API contract, remote backend)
	$(PY) -m pytest tests -q

audit:            ## Phase 1: labels.csv + audit report
	$(PY) scripts/audit.py

feasibility:      ## Phase 2: per-question VLM accuracy
	$(PY) scripts/feasibility.py --model $(VLM_MODEL) --backend $(VLM_BACKEND)

bench:            ## time budget for a 145 s video
	$(PY) scripts/bench.py

weights:          ## GPU-server Docker deployment: download weights to ./weights
	scripts/download_weights.sh

serve:            ## GPU-server Docker deployment (in-container vLLM)
	docker compose up -d --build
