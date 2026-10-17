PY ?= .venv/bin/python

.PHONY: audit feasibility bench eval serve
audit:
	$(PY) scripts/audit.py
feasibility:
	$(PY) scripts/feasibility.py --model $${VLM_MODEL:-mlx-community/Qwen3-VL-8B-Instruct-4bit} --backend $${VLM_BACKEND:-mlx}
eval:
	$(PY) -m pipeline.eval
serve:
	docker compose up --build
bench:
	$(PY) scripts/bench.py
