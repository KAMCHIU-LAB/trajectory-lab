PYTHON ?= .venv/bin/python
CLI = $(PYTHON) -m trajectorylab

.PHONY: doctor check smoke study pointmaze ablation report serve portal demo
doctor:
	$(CLI) doctor

check:
	$(PYTHON) -m ruff check src tests scripts
	$(PYTHON) -m ruff format --check src tests scripts
	$(PYTHON) -m pytest -q

smoke:
	$(CLI) run --config configs/smoke.toml

study:
	$(CLI) run --config configs/pendulum.toml

pointmaze:
	$(CLI) run --config configs/pointmaze.toml

ablation:
	$(CLI) run --config configs/iql-ablation.toml

report:
	$(CLI) report --output reports/latest

serve:
	$(CLI) serve --directory reports/latest --port 8765

portal:
	$(PYTHON) scripts/build_portal.py

demo:
	$(CLI) serve --directory reports --port 8765
