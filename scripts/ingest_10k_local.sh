#!/usr/bin/env bash
# Ingest ~10k FactoidWiki items locally end-to-end (chunks → CPE → vectors → FAISS → graph)
# Local version that assumes services are running (use with bootstrap_all.sh --no-docker)

set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT_DIR"

# Load env if present
if [ -f .env ]; then
  set -a; source .env; set +a
fi

# Force NO_DOCKER mode for local execution
export NO_DOCKER=1

# Use sample data if no argument provided
INPUT_JSONL=${1:-data/datasets/factoid-wiki-large/factoid_wiki.jsonl}
ART_DIR=${ART_DIR:-artifacts}
NPZ_PATH=${NPZ_PATH:-$ART_DIR/fw10k_vectors.npz}

mkdir -p "$ART_DIR"

if [ ! -f "$INPUT_JSONL" ]; then
  echo "❌ Missing input JSONL: $INPUT_JSONL"
  echo "   Provide a file with lines: {id, contents, meta}"
  exit 1
fi

echo "🚀 Starting local 10k ingest DAG..."
echo "   Input: $INPUT_JSONL"
echo "   Output: $NPZ_PATH"

# DAG Step 1: Chunks → CPE extraction
echo "📝 Step 1/4: Extracting CPE from chunks..."
python3 -m src.ingest_factoid \
  --jsonl-path "$INPUT_JSONL" \
  --num-samples 10000 \
  --write-pg \
  --write-neo4j \
  --faiss-out "$NPZ_PATH"

# DAG Step 2: Already done - vectors saved to NPZ during CPE extraction

# DAG Step 3: Build FAISS index (if not already built by ingest_factoid)
if [ ! -f "$ART_DIR/fw10k_ivf.index" ]; then
  echo "🔍 Step 3/4: Building FAISS index..."
  python3 -c "
import numpy as np
from src.db_faiss import build_index

# Load vectors
data = np.load('$NPZ_PATH')
vectors = data['fused'] if 'fused' in data else data['vectors']

# Build and save index
index = build_index(vectors, nlist=128)
import faiss
faiss.write_index(index, '$ART_DIR/fw10k_ivf.index')

# Save metadata
import json
meta = {
    'num_vectors': len(vectors),
    'dimension': vectors.shape[1],
    'nlist': 128,
    'index_type': 'IVF128,Flat'
}
with open('$ART_DIR/faiss_meta.json', 'w') as f:
    json.dump(meta, f)

print(f'Built FAISS index with {len(vectors)} vectors')
"
fi

# DAG Step 4: Graph extraction (already done during ingest_factoid via LightRAG)

# Summaries
echo "📊 Final verification:"
if command -v psql >/dev/null 2>&1; then
  echo "🔎 Postgres counts:"
  psql -h "${PGHOST:-localhost}" -U "${PGUSER:-lnsp}" -d "${PGDATABASE:-lnsp}" -c "SELECT COUNT(*) AS cpe_rows FROM cpe_entry; SELECT COUNT(*) AS vec_rows FROM cpe_vectors;" || true
fi

if command -v cypher-shell >/dev/null 2>&1; then
  echo "🔎 Neo4j graph nodes:"
  cypher-shell -a "${NEO4J_URI:-bolt://localhost:7687}" -u "${NEO4J_USER:-neo4j}" -p "${NEO4J_PASS:-password}" -q "MATCH (n) RETURN count(n) as node_count" || true
fi

echo "✅ Local ingest DAG complete!"
echo "   Vectors: $NPZ_PATH"
echo "   Index: $ART_DIR/fw10k_ivf.index"
echo "   Meta: $ART_DIR/faiss_meta.json"
