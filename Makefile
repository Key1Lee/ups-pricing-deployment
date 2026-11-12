PYTHON := .venv/bin/python
BATCH_DATE ?= 2025-08-29
API_PORT ?= 8000
DOCKER_PORT ?= 8001
.PHONY: setup bootstrap train score test lint serve smoke dagster docker-up docker-down docker-smoke demo show simulate-outcomes monitor rollback

setup:
	uv sync --frozen --python 3.12

bootstrap:
	$(PYTHON) -m shipping.cli bootstrap --batch-date $(BATCH_DATE)

train:
	$(PYTHON) -m shipping.cli train

score:
	$(PYTHON) -m shipping.cli score --batch-date $(BATCH_DATE)

test:
	$(PYTHON) -m pytest -q

lint:
	.venv/bin/ruff check .

serve:
	$(PYTHON) -m uvicorn shipping.api:app --host 127.0.0.1 --port $(API_PORT)

smoke:
	$(PYTHON) -m shipping.cli smoke --url http://127.0.0.1:$(API_PORT)

dagster:
	mkdir -p .dagster
	DAGSTER_HOME="$(CURDIR)/.dagster" .venv/bin/dagster dev -m shipping.definitions --host 127.0.0.1 --port 3000

docker-up:
	$(PYTHON) -c 'from shipping.registry import load_approved; load_approved(); print("Approved trained artifact verified")'
	DOCKER_PORT=$(DOCKER_PORT) docker compose up --build -d --wait

docker-down:
	docker compose down

docker-smoke:
	$(PYTHON) -m shipping.cli smoke --url http://127.0.0.1:$(DOCKER_PORT)
	RUN_DOCKER_TESTS=1 DOCKER_API_URL=http://127.0.0.1:$(DOCKER_PORT) $(PYTHON) -m pytest tests/test_docker.py -q

demo:
	$(PYTHON) -m shipping.cli demo --batch-date $(BATCH_DATE)

show:
	$(PYTHON) -m shipping.cli show

simulate-outcomes:
	$(PYTHON) -m shipping.cli simulate-outcomes --batch-date $(BATCH_DATE)

monitor:
	$(PYTHON) -m shipping.cli monitor --batch-date $(BATCH_DATE)

rollback:
	$(PYTHON) -m shipping.cli rollback
