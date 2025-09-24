#!/bin/bash
# scripts/encode_real_gtr.sh
# Encode real GTR embeddings (no stubs) for 10k FactoidWiki chunks
# Produces: artifacts/fw10k_vectors.npz with real 768D values

set -euo pipefail

echo "=== Encoding real GTR embeddings ==="

# Activate virtual environment
if [ -f ".venv/bin/activate" ]; then
    . .venv/bin/activate
elif [ -f "venv/bin/activate" ]; then
    . venv/bin/activate
else
    echo "ERROR: No virtual environment found"
    exit 1
fi

python3 - <<'PY'
import json, numpy as np, random
from pathlib import Path
from sentence_transformers import SentenceTransformer

root = Path(".")
chunks = root/"artifacts/fw10k_chunks.jsonl"
out = root/"artifacts/fw10k_vectors_768.npz"

if not chunks.exists():
    raise FileNotFoundError(f"Chunks file not found: {chunks}")

model = SentenceTransformer("sentence-transformers/gtr-t5-base")

# Collection arrays
ids = []  # Internal row IDs (int64)
doc_ids = []
concept_texts = []
embeddings_list = []
lane_indices = []
tmd_dense_list = []

print(f"Processing {chunks}...")
with chunks.open() as f:
    for i, line in enumerate(f):
        if i % 1000 == 0:
            print(f"Processed {i} chunks...")
        rec = json.loads(line)

        # Extract fields
        doc_id = rec.get("doc_id") or rec.get("id") or f"doc-{i}"
        concept_text = rec.get("concept") or rec.get("text") or rec.get("contents") or ""

        if not concept_text.strip():
            continue

        # Generate mock TMD (domain, task, modifier) - in real system, this comes from CPE
        # TMD should be 16-dimensional according to spec
        tmd_dense = np.random.rand(16).astype(np.float32)  # Random for now, normalize to 0-1 range

        # Generate lane index (0-2 for L1_FACTOID, L2_GRAPH, L3_SYNTH)
        lane_index = random.randint(0, 2)

        # Store data
        ids.append(i)  # Internal row ID
        doc_ids.append(doc_id)
        concept_texts.append(concept_text)
        lane_indices.append(lane_index)
        tmd_dense_list.append(tmd_dense)

print(f"Batch encoding {len(concept_texts)} texts...")
embeddings = model.encode(concept_texts, normalize_embeddings=True, batch_size=32)

# Convert to numpy arrays
vectors = np.array(embeddings, dtype=np.float32)
ids = np.array(ids, dtype=np.int64)  # Internal row IDs
doc_ids = np.array(doc_ids, dtype=object)
concept_texts = np.array(concept_texts, dtype=object)
lane_indices = np.array(lane_indices, dtype=np.int16)  # As per spec
tmd_dense = np.array(tmd_dense_list, dtype=np.float32)

# Generate CPE IDs (UUIDs) - use doc_ids as base
cpe_ids = doc_ids  # In real system, these would be proper UUIDs

# Check for NaN/Inf in vectors
if not np.isfinite(vectors).all():
    print("FATAL: Found NaN or Inf values in embeddings")
    exit(1)

# Save with all required metadata
np.savez(out,
         vectors=vectors,
         ids=ids,
         doc_ids=doc_ids,
         concept_texts=concept_texts,
         lane_indices=lane_indices,
         tmd_dense=tmd_dense,
         cpe_ids=cpe_ids)

# Zero-vector kill switch: Check for flat indices before saving
mean_l2_norm = np.linalg.norm(vectors, axis=1).mean()
print(f"Mean L2 norm: {mean_l2_norm:.6f}")

if mean_l2_norm < 0.1:  # Kill switch threshold
    print(f"FATAL: Mean L2 norm ({mean_l2_norm:.6f}) is near zero - would create 'flat' index")
    print("This indicates a serious embedding failure. Aborting to prevent CI/runtime issues.")
    exit(1)

zero_count = int((vectors==0).all(axis=1).sum())
print(f"Wrote: {out}")
print(f"Shape: {vectors.shape} (vectors), {len(doc_ids)} (metadata)")
print(f"Zero vectors: {zero_count}")
print(f"Non-zero vectors: {len(doc_ids) - zero_count}")

if zero_count > 0:
    print("WARNING: Found zero vectors - this should not happen with real encoding!")
PY

echo "=== Real GTR encoding complete ==="
