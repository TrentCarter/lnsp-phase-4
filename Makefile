.PHONY: install smoke test db-up db-down

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

# convenience docker compose wrappers
db-up:
	docker compose up -d

db-down:
	docker compose down
