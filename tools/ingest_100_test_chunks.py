#!/usr/bin/env python3
"""Ingest 100 test chunks with REAL GTR-T5 embeddings"""

import requests
import json
from pathlib import Path

print("\n" + "="*80)
print("INGESTING 100 TEST CHUNKS WITH REAL GTR-T5 EMBEDDINGS")
print("="*80 + "\n")

# Read test chunks
chunks_file = Path("test_data/100_test_chunks.jsonl")
chunks = []

with open(chunks_file) as f:
    for line in f:
        chunks.append(json.loads(line))

print(f"Loaded {len(chunks)} chunks from {chunks_file}\n")

# Ingest through API
print("Sending to ingestion API (http://localhost:8004/ingest)...")

response = requests.post(
    'http://localhost:8004/ingest',
    json={"chunks": chunks, "dataset_source": "test_vec2text_validation"},
    timeout=300
)

if response.status_code == 200:
    result = response.json()
    print("\n✅ Ingestion complete!")
    print(f"   Processed: {result.get('processed', 0)} chunks")
    print(f"   Errors: {result.get('errors', 0)}")
    print()
else:
    print(f"\n❌ Ingestion failed: HTTP {response.status_code}")
    print(f"   Response: {response.text}\n")

print("="*80 + "\n")
