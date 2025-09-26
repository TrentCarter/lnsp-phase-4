#!/usr/bin/env bash
    # ===================================================================
    # LNSP Phase-4: 10k Curated FactoidWiki Ingest
    # ===================================================================
    # Features: idempotent, resumable, batch processing
    #
    # Usage: ./scripts/ingest_10k.sh [path/to/factoid_wiki.jsonl]
    #
    # Environment variables:
    #   BATCH_SIZE    - Batch size for processing (default: 1000)
    #   RESUME_FROM   - Resume from entry N (default: 0)
    #   NO_DOCKER     - Skip docker checks (default: 0)
    # ===================================================================

    set -euo pipefail

    ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
    cd "$ROOT_DIR"

    # Load env if present
    if [ -f .env ]; then
        set -a; source .env; set +a
    fi

    # Use curated sample if no argument provided
    INPUT_JSONL=${1:-artifacts/fw10k_chunks.jsonl}
    ART_DIR=${ART_DIR:-artifacts}
    NPZ_PATH=${NPZ_PATH:-$ART_DIR/fw10k_vectors.npz}
    BATCH_SIZE=${BATCH_SIZE:-1000}
    RESUME_FROM=${RESUME_FROM:-0}

    mkdir -p "$ART_DIR"

    # Check if curated chunks exist, create if needed
    if [ ! -f "$INPUT_JSONL" ]; then
        echo "📝 Curated chunks not found at $INPUT_JSONL"
        echo "   Creating curated 10k sample from dataset..."
        python3 scripts/data_processing/create_10k_sample.py
        if [ ! -f "$INPUT_JSONL" ]; then
            echo "❌ Failed to create curated sample"
            exit 1
        fi
    fi

    # Quick service checks (non-fatal warnings)
    if [ "${NO_DOCKER:-0}" = "1" ]; then
        echo "📝 NO_DOCKER mode - ensure PostgreSQL and Neo4j are running manually"
    fi

    if ! command -v psql >/dev/null 2>&1; then
        echo "⚠️  psql not found; Postgres write will fail unless PG_DSN uses socket/driver auth via psycopg2."
    fi
    if ! command -v cypher-shell >/dev/null 2>&1; then
        echo "⚠️  cypher-shell not found; ensure Neo4j is running and env vars are set."
    fi

    # Check if we can resume from existing vectors
    EXISTING_COUNT=0
    if [ -f "$NPZ_PATH" ]; then
        EXISTING_COUNT=$(python3 -c "import numpy as np; print(np.load('$NPZ_PATH')['vectors'].shape[0])" 2>/dev/null || echo 0)
    fi

    if [ "$RESUME_FROM" -eq 0 ] && [ "$EXISTING_COUNT" -gt 0 ]; then
        echo "📝 Found existing vectors with $EXISTING_COUNT entries"
        read -p "   Resume from existing vectors? [y/N]: " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo "   Removing existing vectors..."
            rm -f "$NPZ_PATH"
            EXISTING_COUNT=0
        fi
    fi

    # Ingest with batch processing
    echo "▶️  Ingesting ~10k items from $INPUT_JSONL"
    echo "   Batch size: $BATCH_SIZE, Resume from: $RESUME_FROM"
    python3 -m src.ingest_factoid \
        --jsonl-path "$INPUT_JSONL" \
        --num-samples 10000 \
        --batch-size "$BATCH_SIZE" \
        --resume-from "$RESUME_FROM" \
        --write-pg \
        --write-neo4j \
        --faiss-out "$NPZ_PATH"

    # Verify output
    FINAL_COUNT=$(python3 -c "import numpy as np; print(np.load('$NPZ_PATH')['vectors'].shape[0])" 2>/dev/null || echo 0)
    if [ "$FINAL_COUNT" -eq 0 ]; then
        echo "❌ No vectors generated"
        exit 1
    fi

    echo "✅ Ingest complete: $FINAL_COUNT vectors saved → $NPZ_PATH"

    # Summaries (best-effort)
    if command -v psql >/dev/null 2>&1; then
        echo "
🔎 Postgres counts:"
        psql -h "${PGHOST:-localhost}" -U "${PGUSER:-lnsp}" -d "${PGDATABASE:-lnsp}" -c "SELECT COUNT(*) AS cpe_rows FROM cpe_entry; SELECT COUNT(*) AS vec_rows FROM cpe_vectors;"
    fi

    echo "
🎉 Ready for indexing! Run: make build-faiss"
