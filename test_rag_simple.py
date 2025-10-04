#!/usr/bin/env python3
"""Simple vecRAG test for ontology data."""
import sys, time
import numpy as np
import faiss

print("=== Simple vecRAG Smoke Test ===\n")

# Load NPZ
print("Loading vectors...")
npz = np.load('artifacts/ontology_4k_full.npz', allow_pickle=True)
vectors = np.asarray(npz['vectors'], dtype=np.float32)
doc_ids = [str(x) for x in npz['doc_ids']]
concept_texts = [str(x) for x in npz['concept_texts']]

print(f"✓ Loaded {len(vectors)} vectors ({vectors.shape[1]}D)")
print(f"  Sample: '{concept_texts[0][:60]}...'\n")

# Load FAISS index
print("Loading FAISS index...")
index = faiss.read_index('artifacts/ontology_4k_full.index')
print(f"✓ Index loaded: {index.ntotal} vectors\n")

# Test self-retrieval
print("Testing self-retrieval on 20 samples...")
n_test = 20
k = 5
hits = 0
total_latency = 0

for i in range(n_test):
    query_vec = vectors[i:i+1]  # Shape: (1, 784)
    gold_doc_id = doc_ids[i]

    # Search
    start = time.time()
    D, I = index.search(query_vec, k)
    latency = (time.time() - start) * 1000
    total_latency += latency

    # Check if gold doc in top-k
    retrieved_ids = [doc_ids[idx] for idx in I[0]]
    if gold_doc_id in retrieved_ids:
        hits += 1
        rank = retrieved_ids.index(gold_doc_id) + 1
        status = "✓" if rank == 1 else "○"
        print(f"  {status} Query {i+1:2d}: Rank {rank} ({latency:4.1f}ms) - {concept_texts[i][:50]}...")
    else:
        print(f"  ✗ Query {i+1:2d}: NOT FOUND - {concept_texts[i][:50]}...")

precision = hits / n_test
avg_latency = total_latency / n_test

print(f"\n📊 Results:")
print(f"  P@{k} = {precision:.3f} ({hits}/{n_test})")
print(f"  Avg latency = {avg_latency:.2f}ms")

if precision >= 0.90:
    print("\n✅ Self-retrieval working well!")
    sys.exit(0)
elif precision >= 0.70:
    print("\n⚠️  Moderate precision - acceptable but could be better")
    sys.exit(0)
else:
    print("\n❌ Low precision - check index/embedding sync")
    sys.exit(1)
