# J26-SE-360 developer commands — run `make` (or `make help`) to list them.
# Works on macOS / Linux / Windows (Git Bash or WSL). Python 3.11+ and Docker Desktop required.

SHELL        := /bin/bash
.DEFAULT_GOAL := help

VENV         := .venv
VENV_PY      := $(firstword $(wildcard $(VENV)/bin/python $(VENV)/Scripts/python.exe))
PY           ?= $(if $(VENV_PY),$(abspath $(VENV_PY)),python3)
COMPOSE      ?= docker compose
API          ?= http://localhost:8000
# module services present on this branch (M3 lives on its own branch until merged)
SERVICES     := $(notdir $(wildcard services/m*-*))
PY_DIRS      := contracts backend/core-api services scripts

.PHONY: help setup env up down logs ps test test-core test-m1 test-m2 test-m3 test-m4 lint fmt \
        contracts contracts-check migrate seed simulate clean

help: ## Show this help
	@echo "J26-SE-360 — common commands"
	@awk 'BEGIN{FS=":.*## "} /^[a-zA-Z0-9_-]+:.*## /{printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@echo ""
	@echo "  Variables: API=$(API)  PY=$(PY)  s=<service> (for logs)"

# ------------------------------------------------------------------------------------------- setup
setup: ## Create .venv and install contracts, core API, all module services and dev tools
	python3 -m venv $(VENV)
	@PYV=$$( [ -x $(VENV)/bin/python ] && echo $(VENV)/bin/python || echo $(VENV)/Scripts/python.exe ); \
	set -e; \
	$$PYV -m pip install --upgrade pip; \
	$$PYV -m pip install -e contracts; \
	$$PYV -m pip install -r backend/core-api/requirements.txt; \
	if [ -f backend/core-api/requirements-dev.txt ]; then $$PYV -m pip install -r backend/core-api/requirements-dev.txt; fi; \
	for s in $(SERVICES); do $$PYV -m pip install -r services/$$s/requirements.txt; done; \
	$$PYV -m pip install pytest httpx ruff pre-commit; \
	echo "Done."; if [ -d services/m3-schedule-optimizer ]; then echo "Training extras for Module 3: $$PYV -m pip install -r services/m3-schedule-optimizer/requirements-train.txt"; fi
	@$(MAKE) --no-print-directory env

env: ## Create .env from .env.example if it does not exist
	@test -f .env && echo ".env already exists" || (cp .env.example .env && echo "Created .env from .env.example")

# ------------------------------------------------------------------------------------------- docker
up: env ## Build and start the whole stack (Postgres, Redis, core API, M1–M4)
	$(COMPOSE) up -d --build
	@echo ""
	@echo "  Core API      $(API)/docs"
	@echo "  M1..M4 docs   http://localhost:8101/docs … http://localhost:8104/docs"

down: ## Stop the stack (data volumes are kept)
	$(COMPOSE) down

logs: ## Follow logs (all, or one service: make logs s=core-api)
	$(COMPOSE) logs -f --tail=100 $(s)

ps: ## Show container status and health
	$(COMPOSE) ps

migrate: ## Apply database migrations inside the running core-api container
	$(COMPOSE) exec core-api alembic upgrade head

# ------------------------------------------------------------------------------------------- quality
test: test-core test-m1 test-m2 test-m3 test-m4 contracts-check ## Run every test suite (+ contract schema check)

test-core: ## Core API tests
	cd backend/core-api && $(PY) -m pytest

test-m1: ## Module 1 tests
	@if [ -d services/m1-affective-state ]; then cd services/m1-affective-state && $(PY) -m pytest; else echo "services/m1-affective-state is not on this branch — skipped"; fi

test-m2: ## Module 2 tests
	@if [ -d services/m2-wellness-prediction ]; then cd services/m2-wellness-prediction && $(PY) -m pytest; else echo "services/m2-wellness-prediction is not on this branch — skipped"; fi

test-m3: ## Module 3 tests
	@if [ -d services/m3-schedule-optimizer ]; then cd services/m3-schedule-optimizer && $(PY) -m pytest; else echo "services/m3-schedule-optimizer is not on this branch — skipped"; fi

test-m4: ## Module 4 tests
	@if [ -d services/m4-food-recommendation ]; then cd services/m4-food-recommendation && $(PY) -m pytest; else echo "services/m4-food-recommendation is not on this branch — skipped"; fi

lint: ## Ruff lint for all Python code
	$(PY) -m ruff check $(PY_DIRS)
	$(PY) -m ruff format --check $(PY_DIRS)

fmt: ## Auto-format and auto-fix Python code
	$(PY) -m ruff format $(PY_DIRS)
	$(PY) -m ruff check --fix $(PY_DIRS)

contracts: ## Export JSON Schemas from the Pydantic contracts (run after changing contracts/)
	$(PY) contracts/export_schemas.py

contracts-check: ## Fail if contracts/schemas is out of date
	$(PY) contracts/export_schemas.py --check

# ------------------------------------------------------------------------------------------- demo data
seed: ## Create the demo user and demo tasks via the running core API
	$(PY) scripts/seed_demo.py --base-url $(API)

simulate: ## Simulate a phone: upload realistic raw sensor data, run the M1→M4 pipeline, print results
	$(PY) scripts/simulate_phone.py --base-url $(API)

# ------------------------------------------------------------------------------------------- housekeeping
clean: ## Remove Python caches (keeps .venv, .env and Docker volumes)
	find . -type d \( -name __pycache__ -o -name .pytest_cache -o -name .ruff_cache \) -prune -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
