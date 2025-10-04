#!/usr/bin/env python3
"""Quick GraphRAG test - 10 queries only."""
import sys
from pathlib import Path
import numpy as np
import time

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

print("Loading components...")
from src.db_faiss import FaissDB
from RAG.graphrag_backend import GraphRAGBackend
import json

# Load data
npz = np.load("artifacts/fw9k_vectors_tmd_fixed.npz", allow_pickle=True)
vectors = np.asarray(npz['vectors'], dtype=np.float32)
concept_texts = [str(x) for x in npz['concept_texts']]

# Load FAISS (use 784D index to match NPZ vectors)
index_path = "artifacts/fw9k_ivf_flat_ip_tmd_fixed.index"
db = FaissDB(index_path=index_path, meta_npz_path="artifacts/fw9k_vectors_tmd_fixed.npz")
db.load(index_path)

# GraphRAG
gb = GraphRAGBackend()

print(f"Loaded: {len(vectors)} vectors, {len(gb.text_to_idx)} graph mappings\n")

# Test 10 queries
test_indices = [0, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4400]
vec_hits = 0
gr_hits = 0

print("=" * 80)
print(f"{'Query':<50} {'vecRAG':<10} {'GraphRAG':<10}")
print("=" * 80)

for idx in test_indices:
    query = concept_texts[idx]
    qv = vectors[idx].reshape(1, -1)

    # vecRAG: pure vector search
    D, I = db.search(qv, 10)
    vec_result = [int(x) for x in I[0]]
    vec_hit = "✓" if idx in vec_result[:1] else "✗"
    if idx in vec_result[:1]:
        vec_hits += 1

    # GraphRAG: get neighbors
    neighbors = gb._get_1hop_neighbors(query)

    # Simple fusion: boost neighbors in vector results
    graph_boost = {n[0]: n[1] for n in neighbors}
    boosted_scores = []
    for i, (result_idx, vec_score) in enumerate(zip(vec_result, D[0])):
        result_text = concept_texts[result_idx] if result_idx < len(concept_texts) else ""
        boost = graph_boost.get(result_text, 0.0) * 0.3
        boosted_scores.append((result_idx, vec_score + boost))

    boosted_scores.sort(key=lambda x: -x[1])
    gr_result = [x[0] for x in boosted_scores]

    gr_hit = "✓" if idx in gr_result[:1] else "✗"
    if idx in gr_result[:1]:
        gr_hits += 1

    query_short = query[:47] + "..." if len(query) > 50 else query
    neighbor_info = f"({len(neighbors)} nbrs)" if neighbors else "(0 nbrs)"
    print(f"{query_short:<50} {vec_hit:<10} {gr_hit} {neighbor_info}")

print("=" * 80)
print(f"P@1 Results:")
print(f"  vecRAG:    {vec_hits}/10 = {vec_hits/10:.1%}")
print(f"  GraphRAG:  {gr_hits}/10 = {gr_hits/10:.1%}")
print(f"  Change:    {gr_hits - vec_hits:+d} queries")
print("=" * 80)

gb.close()
