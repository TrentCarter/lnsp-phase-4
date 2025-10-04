#!/usr/bin/env python3
"""Minimal GraphRAG benchmark - bypasses complex harness."""
import sys
from pathlib import Path
import numpy as np
import time

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from src.db_faiss import FaissDB
from src.vectorizer import EmbeddingBackend
from RAG.graphrag_backend import GraphRAGBackend, run_graphrag

print("=" * 70)
print("GraphRAG Minimal Benchmark")
print("=" * 70)

# Load corpus
print("\n[1/5] Loading corpus...")
npz_path = "artifacts/fw9k_vectors_tmd_fixed.npz"
npz = np.load(npz_path, allow_pickle=True)
vectors = np.asarray(npz['vectors'], dtype=np.float32)
concept_texts = [str(x) for x in npz['concept_texts']]
doc_ids = [str(x) for x in npz.get('doc_ids', range(len(vectors)))]
print(f"✓ Loaded {len(vectors)} vectors (dim={vectors.shape[1]})")

# Load FAISS
print("\n[2/5] Loading FAISS index...")
import json
meta = Path("artifacts/faiss_meta.json")
index_path = json.loads(meta.read_text()).get("index_path")
db = FaissDB(index_path=index_path, meta_npz_path=npz_path)
db.load(index_path)
print(f"✓ FAISS loaded ({db.index.ntotal} vectors)")

# Initialize GraphRAG
print("\n[3/5] Initializing GraphRAG...")
gb = GraphRAGBackend()
print(f"✓ GraphRAG connected ({len(gb.text_to_idx)} concept mappings)")

# Create test queries (every 10th concept)
print("\n[4/5] Creating test queries...")
n_queries = 50
indices = np.linspace(0, len(concept_texts)-1, n_queries, dtype=int)
queries_text = [concept_texts[i] for i in indices]
gold_pos = list(indices)

# Prepare query vectors
emb = EmbeddingBackend()
query_vecs = []
for qt in queries_text:
    v = emb.encode([qt])[0].astype(np.float32)
    # Add TMD padding if needed
    if vectors.shape[1] == 784:
        tmd = np.zeros(16, dtype=np.float32)
        v = np.concatenate([tmd, v])
    # Normalize
    norm = np.linalg.norm(v)
    if norm > 0:
        v = v / norm
    query_vecs.append(v)

print(f"✓ Created {len(queries_text)} test queries")

# Run benchmarks
print("\n[5/5] Running benchmarks...")
print("-" * 70)

# vecRAG baseline
print("\nBackend: vec (baseline)")
vec_indices = []
vec_scores = []
vec_latencies = []

for qv in query_vecs:
    qv = qv.reshape(1, -1)
    t0 = time.perf_counter()
    D, I = db.search(qv, 10)
    vec_latencies.append((time.perf_counter() - t0) * 1000.0)
    vec_indices.append([int(x) for x in I[0]])
    vec_scores.append([float(x) for x in D[0]])

# Calculate metrics
def precision_at_1(gold_positions, results):
    hits = sum(1 for gold, result_list in zip(gold_positions, results) if gold in result_list[:1])
    return hits / len(gold_positions)

def precision_at_5(gold_positions, results):
    hits = sum(1 for gold, result_list in zip(gold_positions, results) if gold in result_list[:5])
    return hits / len(gold_positions)

vec_p1 = precision_at_1(gold_pos, vec_indices)
vec_p5 = precision_at_5(gold_pos, vec_indices)
vec_lat = np.mean(vec_latencies)

print(f"  P@1: {vec_p1:.3f}")
print(f"  P@5: {vec_p5:.3f}")
print(f"  Latency: {vec_lat:.2f}ms")

# GraphRAG local
print("\nBackend: graphrag_local (1-2 hop neighbors)")
gr_indices, gr_scores, gr_latencies = run_graphrag(
    gb, vec_indices, vec_scores, queries_text, concept_texts, topk=10, mode="local"
)

gr_p1 = precision_at_1(gold_pos, gr_indices)
gr_p5 = precision_at_5(gold_pos, gr_indices)
gr_lat = np.mean(gr_latencies)

print(f"  P@1: {gr_p1:.3f}")
print(f"  P@5: {gr_p5:.3f}")
print(f"  Latency: {gr_lat:.2f}ms")

# Summary table
print("\n" + "=" * 70)
print("RESULTS SUMMARY")
print("=" * 70)
print(f"{'Backend':<20} {'P@1':<10} {'P@5':<10} {'Latency (ms)':<15}")
print("-" * 70)
print(f"{'vec (baseline)':<20} {vec_p1:<10.3f} {vec_p5:<10.3f} {vec_lat:<15.2f}")
print(f"{'graphrag_local':<20} {gr_p1:<10.3f} {gr_p5:<10.3f} {gr_lat:<15.2f}")
print("-" * 70)

# Calculate improvements
p1_improvement = ((gr_p1 - vec_p1) / vec_p1 * 100) if vec_p1 > 0 else 0
p5_improvement = ((gr_p5 - vec_p5) / vec_p5 * 100) if vec_p5 > 0 else 0

print(f"\nImprovements:")
print(f"  P@1: {p1_improvement:+.1f}%")
print(f"  P@5: {p5_improvement:+.1f}%")

# Cleanup
gb.close()

print("\n" + "=" * 70)
print("✅ Benchmark complete!")
print("=" * 70)
