PY ?= .venv/bin/python
export HF_HUB_OFFLINE ?= 1

.PHONY: audit feasibility bench eval eval-report ui serve serve-local test weights
audit:            ## Phase 1: labels.csv + audit
	$(PY) scripts/audit.py
feasibility:      ## Phase 2: per-question VLM accuracy
	$(PY) scripts/feasibility.py --model $${VLM_MODEL:-mlx-community/Qwen3-VL-4B-Instruct-4bit} --backend $${VLM_BACKEND:-mlx}
bench:            ## latency budget for a 145 s video (re-run on the GPU server)
	$(PY) scripts/bench.py
eval:             ## Phase 4: run all training videos, write reports/eval.md
	$(PY) -m pipeline.eval
eval-report:      ## recompute metrics from saved results
	$(PY) -m pipeline.eval --reuse
test:             ## API contract tests (stub analyzer, no models)
	$(PY) -m pytest tests -q
weights:          ## download weights once (GPU server layout)
	scripts/download_weights.sh
ui:               ## build the static UI (forte-vision-ai-development/out)
	cd forte-vision-ai-development && pnpm install && NEXT_TELEMETRY_DISABLED=1 pnpm build
serve-local: ui   ## Mac/dev: backend (MLX) + UI on http://localhost:8000, key "dev-key" unless API_KEY is set
	API_KEY=$${API_KEY:-dev-key} VLM_BACKEND=$${VLM_BACKEND:-mlx} VLM_MODEL=$${VLM_MODEL:-mlx-community/Qwen3-VL-4B-Instruct-4bit} \
	$(PY) -m uvicorn service.app:app --host 0.0.0.0 --port 8000
serve:            ## GPU server: docker compose (backend vLLM + nginx UI)
	docker compose up -d --build
