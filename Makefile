.PHONY: install smoke test db-up db-down consultant-eval

install:
	python3 -m venv .venv && . .venv/bin/activate && pip install --upgrade pip \
		&& pip install -r requirements.txt || true

smoke:
	python -m src.ingest_factoid --faiss-out /tmp/factoid_vecs.npz

# assumes PG_DSN and Neo4j vars are set if used
smoke-all:
	python -m src.ingest_factoid --write-pg --write-neo4j --faiss-out /tmp/factoid_vecs.npz

test:
	pytest -q

consultant-eval:
	@echo "Starting consultant evaluation harness..."
	@echo "Make sure API is running: uvicorn src.api.retrieve:app --host 127.0.0.1 --port 8092"
	@echo "Running evaluation against live API + FAISS stack..."
	python tools/run_consultant_eval.py --api-url http://localhost:8092 --with-faiss

# convenience docker compose wrappers
db-up:
	docker compose up -d

db-down:
	docker compose down

# === FAISS Environment Setup ===

.PHONY: faiss-setup
faiss-setup:
	@mkdir -p artifacts
	@echo "[faiss-setup] running setup script..."
	@./scripts/setup_faiss_env.sh $(ARGS) | tee artifacts/setup_faiss_env.log
	@echo "strategy=$$(grep -m1 'Selected strategy:' artifacts/setup_faiss_env.log | awk -F': ' '{print $$2}')" 	    "date=$$(date -u +"%Y-%m-%dT%H:%M:%SZ")" >> artifacts/runtime.txt
	@echo "[faiss-setup] complete. Strategy recorded in artifacts/runtime.txt"
