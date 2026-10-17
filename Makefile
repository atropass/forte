PY ?= .venv/bin/python

.PHONY: audit feasibility eval serve
audit:
	$(PY) scripts/audit.py
feasibility:
	$(PY) scripts/feasibility.py
eval:
	$(PY) -m pipeline.eval
serve:
	docker compose up --build
