.PHONY: install smoke test db-up db-down consultant-eval faiss-setup ingest-10k ingest-all build-faiss api prune slo-snapshot slo-grid gating-snapshot smoketest cpesh-rotate cpesh-index graph-smoke rag-status rag-watch graphrag-track

install:
	python3 -m venv .venv && . .venv/bin/activate && pip install --upgrade pip \
		&& pip install -r requirements.txt || true

.PHONY: faiss-setup
faiss-setup:
	@mkdir -p artifacts
	@echo "[faiss-setup] running setup script..."
	@./scripts/setup_faiss_env.sh $(ARGS) | tee artifacts/setup_faiss_env.log
	@echo "strategy=$$(grep -m1 '^::strategy=' artifacts/setup_faiss_env.log | cut -d= -f2)" \t    "date=$$(date -u +"%Y-%m-%dT%H:%M:%SZ")" >> artifacts/runtime.txt
	@echo "[faiss-setup] complete. Strategy recorded in artifacts/runtime.txt"
.PHONY: lnsp-status
lnsp-status:
	@PYTHONPATH=src ./.venv/bin/python tools/lnsprag_status.py --api http://127.0.0.1:$${PORT:-8092}

.PHONY: cpesh-rotate
cpesh-rotate:
	@./.venv/bin/python scripts/cpesh_rotate.py

.PHONY: cpesh-index
cpesh-index:
	@./.venv/bin/python scripts/cpesh_index_refresh.py

.PHONY: test
test:
	./.venv/bin/pytest -m 'not slow'

.PHONY: vec-upsert
vec-upsert:
	@PYTHONPATH=src ./.venv/bin/python -m tools.vec_upsert --text "$${TEXT}" --doc-id "$${DOCID}"
# === Sprint 3 Tasks ===

ingest-10k:
	@echo "Ingesting 10k curated FactoidWiki items..."
	@./scripts/ingest_10k.sh
	@echo "✅ Ingest complete. Ready to build FAISS index."

ingest-all:
	@echo "🚀 Comprehensive VecRAG Ingestion - Processing ALL items..."
	@./scripts/ingest_comprehensive.sh $(ARGS)
	@echo "✅ Comprehensive ingest complete. Check artifacts for detailed report."

build-faiss:
	@echo "Building FAISS index with dial-plan flags..."
	@python src/faiss_index.py $(ARGS)
	@echo "✅ Index built. Check artifacts/index_meta.json for details."

api:
	@PORT=$${PORT:-8092}; echo "[api] starting on $$PORT (safe threads)"
	@OMP_NUM_THREADS=$${OMP_NUM_THREADS:-1} \
	VECLIB_MAXIMUM_THREADS=$${VECLIB_MAXIMUM_THREADS:-1} \
	OPENBLAS_NUM_THREADS=$${OPENBLAS_NUM_THREADS:-1} \
	MKL_NUM_THREADS=$${MKL_NUM_THREADS:-1} \
	FAISS_NUM_THREADS=$${FAISS_NUM_THREADS:-1} \
	KMP_DUPLICATE_LIB_OK=TRUE \
	PYTHONPATH=src ./.venv/bin/uvicorn src.api.retrieve:app --host 127.0.0.1 --port $$PORT

prune:
	@echo "Applying pruning manifest..."
	@if [ ! -f "eval/prune_manifest.json" ]; then \
		echo "❌ eval/prune_manifest.json not found. Create manifest first."; \
		exit 1; \
	fi
	@python scripts/prune_cache.py --manifest eval/prune_manifest.json
	@echo "✅ Pruning complete. Check eval/prune_report.md for details."

.PHONY: slo-grid
slo-grid:
	@API=$${API:-http://127.0.0.1:$${PORT:-8094}} QUERIES=$${QUERIES:-eval/100q.jsonl} \
	SLO_NOTES="$${SLO_NOTES:-auto}" TOPK=$${TOPK:-5} \
	PYTHONPATH=src ./.venv/bin/python tools/slo_grid.py

.PHONY: slo-snapshot
slo-snapshot:
	@curl -s http://127.0.0.1:$${PORT:-8094}/metrics/slo | tee artifacts/metrics_slo.json >/dev/null
	@echo "[slo] snapshot saved to artifacts/metrics_slo.json"

# === Composite Commands ===

# End-to-end 10k workflow
workflow-10k: ingest-10k build-faiss
	@echo "🎉 10k workflow complete!"
	@echo "   Start API: make api"
	@echo "   Run eval: make consultant-eval"
	@echo "   Take snapshot: make slo-snapshot"

.PHONY: gating-snapshot
gating-snapshot:
	@mkdir -p eval/snapshots
	@curl -sf http://127.0.0.1:$${PORT:-8092}/metrics/gating \
	  -o eval/snapshots/gating_$$(/bin/date -u +"%Y%m%dT%H%M%SZ").json
	@echo "[gating-snapshot] wrote eval/snapshots/…"

# Prune and re-index workflow
workflow-prune: prune build-faiss
	@echo "🔄 Prune and re-index complete!"
	@echo "   Take snapshot: make slo-snapshot"
	@echo "   Run eval: make consultant-eval"

smoketest:
	@bash scripts/s7_smoketest.sh

# === QC & Backfill Tools (Sprint S2) ===

.PHONY: lnsprag-validate
lnsprag-validate:
	@python3 tools/lnsprag_validator.py \
		--chunks-jsonl artifacts/cpesh_active.jsonl \
		--cpesh-jsonl artifacts/cpesh_active.jsonl \
		--tmd-default 9.1.27 \
		--min-words 120 \
		--p95-target 250

.PHONY: cpesh-backfill
cpesh-backfill:
	@python3 tools/cpesh_backfill.py \
		--chunks-jsonl artifacts/cpesh_active.jsonl \
		--out-jsonl artifacts/cpesh_backfill_windows.jsonl \
		--min-words 180 --max-words 320

.PHONY: tmd-report
tmd-report:
	@python3 tools/tmd_histogram_report.py \
		--chunks-jsonl artifacts/cpesh_active.jsonl --top-n 30

# === GraphRAG Testing (S4) ===

graph-smoke:
	@echo "[graph] health:"; curl -s http://127.0.0.1:$${PORT:-8094}/graph/health | jq .
	@echo "[graph] search (fts):"; curl -s -X POST http://127.0.0.1:$${PORT:-8094}/graph/search \
	  -H 'Content-Type: application/json' -d '{"q":"photosynthesis","top_k":5}' | jq '.[0]'
	@echo "[graph] hop (example):"; curl -s -X POST http://127.0.0.1:$${PORT:-8094}/graph/hop \
	  -H 'Content-Type: application/json' -d '{"node_id":"cpe:demo","max_hops":2,"top_k":5}' | jq .

# === Nightly Operations (S5) ===

.PHONY: cpesh-rotate-nightly
cpesh-rotate-nightly:
	@echo "Running nightly CPESH rotation and index refresh..."
	@./.venv/bin/python scripts/cpesh_rotate.py
	@./.venv/bin/python scripts/cpesh_index_refresh.py
	@echo "✅ Nightly rotation complete"

# === RAG Performance Monitoring ===

.PHONY: rag-status
rag-status:
	@PYTHONPATH=. ./.venv/bin/python tools/rag_dashboard.py

.PHONY: rag-watch
rag-watch:
	@PYTHONPATH=. ./.venv/bin/python tools/rag_dashboard.py --watch


# =============================
# Makefile (phase 2)
# Save as: Makefile
# =============================

ARTIFACTS ?= artifacts/lvm/models_phase2
RUN ?= run_500ctx_routingA
SEQS ?= data/gwom_sequences.jsonl      # chains (ordered) passing coherence gates
VECDB ?= data/vector_index.npz          # retrieval bank (post‑norm, correct dim)
LANE_FILTER ?= on

# Common hyperparams
LR ?= 1e-4
WD ?= 1e-4
BATCH ?= 64
ACCUM ?= 4
CTX ?= 500
ALPHA ?= 0.03        # InfoNCE weight (warm)
TAU ?= 0.07          # InfoNCE temperature
PATIENCE ?= 3
MODEL ?= memory_gru  # winner from Phase 1
ROUTING ?= tmd16     # 16 TMD specialists (top2)
SCHED ?= cosine
WARMUP ?= 1

.PHONY: phase2-warm phase2-softnegs phase2-hardnegs eval best monitor clean

phase2-warm:
	bash scripts/train_phase2.sh \
	  --model $(MODEL) --context $(CTX) --routing $(ROUTING) \
	  --alpha $(ALPHA) --tau $(TAU) \
	  --lr $(LR) --wd $(WD) --sched $(SCHED) --warmup $(WARMUP) \
	  --batch $(BATCH) --accum $(ACCUM) \
	  --earlystop hit5 --patience $(PATIENCE) \
	  --seqs $(SEQS) --vecdb $(VECDB) \
	  --save $(ARTIFACTS)/$(RUN)

phase2-softnegs:
	bash scripts/train_phase2.sh \
	  --model $(MODEL) --context $(CTX) --routing $(ROUTING) \
	  --alpha 0.05 --tau $(TAU) --negatives soft --neg-lo 0.60 --neg-hi 0.80 \
	  --lr $(LR) --wd $(WD) --sched $(SCHED) --warmup $(WARMUP) \
	  --batch $(BATCH) --accum $(ACCUM) \
	  --earlystop hit5 --patience $(PATIENCE) \
	  --seqs $(SEQS) --vecdb $(VECDB) \
	  --save $(ARTIFACTS)/$(RUN)_soft

phase2-hardnegs:
	bash scripts/train_phase2.sh \
	  --model $(MODEL) --context $(CTX) --routing $(ROUTING) \
	  --alpha 0.05 --tau $(TAU) --negatives hard --neg-lo 0.75 --neg-hi 0.90 \
	  --lr $(LR) --wd $(WD) --sched $(SCHED) --warmup $(WARMUP) \
	  --batch $(BATCH) --accum $(ACCUM) \
	  --earlystop hit5 --patience $(PATIENCE) \
	  --seqs $(SEQS) --vecdb $(VECDB) \
	  --save $(ARTIFACTS)/$(RUN)_hard

# Evaluate best checkpoint (must mirror training vector: delta recon + L2 norm)
# Expects best_val_hit5.pt and training_history.json in save dir
best:
	bash scripts/eval_phase2.sh \
	  --ckpt $(ARTIFACTS)/$(RUN)/best_val_hit5.pt \
	  --vecdb $(VECDB) --k 1 5 10 --lane $(LANE_FILTER)

# Evaluate an arbitrary run folder
EVAL_RUN ?= $(ARTIFACTS)/$(RUN)
eval:
	bash scripts/eval_phase2.sh \
	  --ckpt $(EVAL_RUN)/best_val_hit5.pt \
	  --vecdb $(VECDB) --k 1 5 10 --lane $(LANE_FILTER)

monitor:
	@echo "History:" && jq '.epoch, .metrics.val.hit5' $(EVAL_RUN)/training_history.json 2>/dev/null || echo "no history"

clean:
	rm -rf $(ARTIFACTS)/$(RUN) $(ARTIFACTS)/$(RUN)_soft $(ARTIFACTS)/$(RUN)_hard
