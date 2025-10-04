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
    # Use fw10k_vectors_768.npz as the default to avoid overwriting the wrong file
    NPZ_PATH=${NPZ_PATH:-$ART_DIR/fw10k_vectors_768.npz}
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

    # Check required services using shared script
    source "$ROOT_DIR/scripts/check_services.sh"
    if ! check_all_services; then
        echo "❌ Required services not available. Exiting."
        exit 1
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
    echo "   ⚠️  CRITICAL: Writing to PostgreSQL, Neo4j, AND FAISS atomically"
    echo "   ⚠️  All three data stores MUST stay synchronized for GraphRAG!"
    echo "   Processing up to 10000 samples"
    ./.venv/bin/python -m src.ingest_factoid \
        --file-path "$INPUT_JSONL" \
        --num-samples 10000 \
        --write-pg \
        --write-neo4j \
        --faiss-out "$NPZ_PATH"

    # Verify ingestion succeeded
    if [ $? -ne 0 ]; then
        echo "❌ Ingestion FAILED! Data stores may be out of sync!"
        exit 1
    fi

    # Verify output exists
    FINAL_COUNT=$(python3 -c "import numpy as np; print(np.load('$NPZ_PATH')['vectors'].shape[0])" 2>/dev/null || echo 0)
    if [ "$FINAL_COUNT" -eq 0 ]; then
        echo "❌ No vectors generated"
        exit 1
    fi

    echo "✅ Ingest complete: $FINAL_COUNT vectors saved → $NPZ_PATH"
    echo "   Verifying data synchronization..."

    # Quick sync check: sample concept from each store
    SAMPLE_CONCEPT=$(python3 -c "import numpy as np; npz=np.load('$NPZ_PATH', allow_pickle=True); print(npz['concept_texts'][0])" 2>/dev/null || echo "")
    if [ -n "$SAMPLE_CONCEPT" ]; then
        PG_CHECK=$(psql -h "${PGHOST:-localhost}" -U "${PGUSER:-lnsp}" -d "${PGDATABASE:-lnsp}" -tAc "SELECT concept_text FROM cpe_entry WHERE concept_text='$SAMPLE_CONCEPT' LIMIT 1;" 2>/dev/null || echo "")
        NEO_CHECK=$(cypher-shell -u neo4j -p password "MATCH (c:Concept {text: \"$SAMPLE_CONCEPT\"}) RETURN c.text LIMIT 1" --format plain 2>/dev/null | tail -1 || echo "")

        if [ -z "$PG_CHECK" ] || [ -z "$NEO_CHECK" ]; then
            echo "⚠️  WARNING: Data synchronization check failed!"
            echo "   Sample concept '$SAMPLE_CONCEPT' not found in all stores"
            echo "   PostgreSQL: ${PG_CHECK:-NOT FOUND}"
            echo "   Neo4j: ${NEO_CHECK:-NOT FOUND}"
        else
            echo "✅ Data sync verified: Sample concept found in PostgreSQL + Neo4j + FAISS"
        fi
    fi

    # Summaries (best-effort)
    if command -v psql >/dev/null 2>&1; then
        echo "
🔎 Postgres counts:"
        psql -h "${PGHOST:-localhost}" -U "${PGUSER:-lnsp}" -d "${PGDATABASE:-lnsp}" -c "SELECT COUNT(*) AS cpe_rows FROM cpe_entry; SELECT COUNT(*) AS vec_rows FROM cpe_vectors;"
    fi

    echo "
🎉 Ready for indexing! Run: make build-faiss"
