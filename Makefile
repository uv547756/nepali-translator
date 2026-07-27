.PHONY: all setup download-models build-client run dev test lint clean

REPO_ROOT := $(shell pwd)
VENV      := $(REPO_ROOT)/.venv
PYTHON    := $(VENV)/bin/python
PIP       := $(VENV)/bin/pip

# ── Setup ──────────────────────────────────────────────────────────────────

setup:
	bash scripts/setup.sh

download-models:
	bash scripts/download_models.sh

# ── Build ──────────────────────────────────────────────────────────────────

build-client:
	cd client && npm install && npm run build

# ── Run ────────────────────────────────────────────────────────────────────

run: build-client
	$(PYTHON) -m server.main --config configs/default.yaml

run-offline:
	@if [ -z "$(INPUT)" ]; then echo "Usage: make run-offline INPUT=path/to/file.wav OUTPUT=out.wav"; exit 1; fi
	$(PYTHON) -m server.main --mode offline \
		--config configs/default.yaml \
		--input $(INPUT) \
		--output $(or $(OUTPUT),translated.wav)

dev:
	@echo "Starting dev server (uvicorn + Vite)..."
	$(PYTHON) -m server.main --config configs/default.yaml &
	cd client && npm run dev

# ── Test ───────────────────────────────────────────────────────────────────

test:
	$(VENV)/bin/pytest tests/server/ -v --asyncio-mode=auto

test-fast:
	$(VENV)/bin/pytest tests/server/test_ringbuffer.py tests/server/test_incremental.py \
		tests/server/test_config.py -v

# ── Lint ───────────────────────────────────────────────────────────────────

lint:
	$(VENV)/bin/python -m py_compile server/core/config.py
	$(VENV)/bin/python -m py_compile server/audio/ringbuffer.py
	$(VENV)/bin/python -m py_compile server/audio/vad.py
	$(VENV)/bin/python -m py_compile server/audio/noise.py
	$(VENV)/bin/python -m py_compile server/asr/engine.py
	$(VENV)/bin/python -m py_compile server/asr/whisper.py
	$(VENV)/bin/python -m py_compile server/translation/engine.py
	$(VENV)/bin/python -m py_compile server/translation/seamless.py
	$(VENV)/bin/python -m py_compile server/translation/nllb.py
	$(VENV)/bin/python -m py_compile server/tts/engine.py
	$(VENV)/bin/python -m py_compile server/tts/piper.py
	$(VENV)/bin/python -m py_compile server/core/incremental.py
	$(VENV)/bin/python -m py_compile server/core/pipeline.py
	$(VENV)/bin/python -m py_compile server/core/gpu_manager.py
	$(VENV)/bin/python -m py_compile server/core/metrics.py
	$(VENV)/bin/python -m py_compile server/core/scheduler.py
	$(VENV)/bin/python -m py_compile server/api/websocket.py
	$(VENV)/bin/python -m py_compile server/api/rest.py
	$(VENV)/bin/python -m py_compile server/main.py
	@echo "All Python files compiled successfully"
	@cd client && npx tsc --noEmit 2>/dev/null && echo "TypeScript OK" || echo "TypeScript check failed (install node_modules first)"

# ── Docker ─────────────────────────────────────────────────────────────────

docker-build:
	docker compose -f docker/docker-compose.yml build

docker-up:
	docker compose -f docker/docker-compose.yml up

docker-down:
	docker compose -f docker/docker-compose.yml down

# ── Clean ──────────────────────────────────────────────────────────────────

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	rm -rf client/dist client/node_modules
	rm -f translator.log /tmp/translator.pid
