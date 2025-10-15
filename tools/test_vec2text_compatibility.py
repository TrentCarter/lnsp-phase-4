#!/usr/bin/env python3
"""Test vec2text compatibility with database embeddings"""

import psycopg2
import requests
import numpy as np
import json

print("\n" + "="*80)
print("VEC2TEXT COMPATIBILITY TEST")
print("="*80 + "\n")

# Connect to database
conn = psycopg2.connect("dbname=lnsp")
cur = conn.cursor()

# Get the first chunk's embedding
cur.execute("""
    SELECT e.concept_text, v.concept_vec
    FROM cpe_entry e
    JOIN cpe_vectors v ON e.cpe_id = v.cpe_id
    WHERE e.dataset_source = 'test_vec2text_validation'
    LIMIT 1
""")

row = cur.fetchone()
if not row:
    print("❌ No test chunks found in database!")
    exit(1)

concept_text, concept_vec_json = row
concept_vec = np.array(json.loads(concept_vec_json), dtype=np.float32)

print(f"Original text: {concept_text}")
print(f"Vector shape: {concept_vec.shape}")
print(f"Vector norm: {np.linalg.norm(concept_vec):.4f}\n")

# Test with vec2text decoder
print("Sending to vec2text decoder...")
response = requests.post(
    'http://localhost:8766/decode',
    json={
        'vectors': [concept_vec.tolist()],
        'subscribers': 'jxe',
        'steps': 1
    },
    timeout=30
)

if response.status_code == 200:
    result = response.json()

    # Access results - response format: results[0]['subscribers']['gtr → jxe']
    decoded = result['results'][0]['subscribers']['gtr → jxe']

    print(f"✅ Vec2text output: {decoded['output']}")
    print(f"   Cosine similarity: {decoded['cosine']:.4f}\n")

    if decoded['cosine'] > 0.63:
        print(f"🎉 SUCCESS! Cosine {decoded['cosine']:.4f} > 0.63 (compatible!)")
        print(f"   The vec2text-compatible wrapper is working correctly!")
    else:
        print(f"❌ FAILED! Cosine {decoded['cosine']:.4f} < 0.63 (still incompatible)")
else:
    print(f"❌ Vec2text API failed: HTTP {response.status_code}")
    print(f"   Response: {response.text}")

print("\n" + "="*80 + "\n")

cur.close()
conn.close()
