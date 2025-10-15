#!/usr/bin/env python3
"""
Test Newly Ingested Vectors with Vec2Text
==========================================

Tests vectors from the fresh ingestion (test_vec2text_validation dataset)
to verify they decode correctly with vec2text.

EXPECTED: Cosine similarity 0.65-0.85 (working vectors)
BROKEN: Cosine similarity 0.05-0.15 (corrupted vectors)
"""

import psycopg2
from psycopg2.extras import RealDictCursor
import numpy as np
import requests

print("\n" + "="*80)
print("TESTING REAL GTR-T5 VECTORS → VEC2TEXT")
print("="*80 + "\n")

# Connect to database
conn = psycopg2.connect("dbname=lnsp")
cur = conn.cursor(cursor_factory=RealDictCursor)

# Get first 10 vectors from the new ingestion
cur.execute("""
    SELECT e.concept_text, v.concept_vec
    FROM cpe_entry e
    JOIN cpe_vectors v ON e.cpe_id = v.cpe_id
    WHERE e.dataset_source = 'test_vec2text_validation'
    ORDER BY e.created_at
    LIMIT 10
""")

rows = cur.fetchall()
print(f"Testing {len(rows)} vectors from fresh ingestion\n")

results = []

for i, row in enumerate(rows, 1):
    text = row['concept_text']
    vec = row['concept_vec']

    # Convert pgvector to numpy
    if isinstance(vec, str):
        vec_str = vec.strip('[]')
        vec_floats = [float(x) for x in vec_str.split(',')]
        vec_array = np.array(vec_floats, dtype=np.float32)
    else:
        vec_array = np.array(list(vec), dtype=np.float32)

    vec_norm = np.linalg.norm(vec_array)

    print(f"Sample {i}/10")
    print("─"*80)
    print(f"Original Text:\n  {text}\n")
    print(f"Vector norm: {vec_norm:.4f} (should be 1.0)\n")

    # Decode via vec2text
    try:
        response = requests.post(
            'http://localhost:8766/decode',
            json={
                'vectors': [vec_array.tolist()],
                'subscribers': 'jxe',
                'steps': 5,
                'device': 'cpu'
            },
            timeout=120
        )

        data = response.json()
        result = data['results'][0]['subscribers']['gtr → jxe']

        reconstructed_text = result['output']
        cosine = result['cosine']

        print(f"Vec2Text Reconstructed:\n  {reconstructed_text}\n")
        print(f"Cosine: {cosine:.4f}")

        if cosine >= 0.65:
            print("✅ EXCELLENT! Real GTR-T5 vectors working correctly!\n")
            results.append(('success', cosine))
        elif cosine >= 0.30:
            print("⚠️  MODERATE - Better than broken but not great\n")
            results.append(('moderate', cosine))
        else:
            print("❌ BROKEN - Same issue as before (fake vectors)\n")
            results.append(('broken', cosine))

    except Exception as e:
        print(f"✗ Error: {e}\n")
        results.append(('error', 0.0))

# Summary
print("="*80)
print("SUMMARY")
print("="*80 + "\n")

success = [r for r in results if r[0] == 'success']
moderate = [r for r in results if r[0] == 'moderate']
broken = [r for r in results if r[0] == 'broken']
errors = [r for r in results if r[0] == 'error']

print(f"✅ Excellent (cosine ≥ 0.65): {len(success)}/10")
print(f"⚠️  Moderate (0.30-0.65):      {len(moderate)}/10")
print(f"❌ Broken (< 0.30):            {len(broken)}/10")
print(f"✗  Errors:                     {len(errors)}/10")
print()

if results:
    all_cosines = [r[1] for r in results if r[1] > 0]
    if all_cosines:
        avg_cosine = np.mean(all_cosines)
        print(f"Average cosine similarity: {avg_cosine:.4f}")
        print()

        if avg_cosine >= 0.65:
            print("🎉 SUCCESS! Real GTR-T5 vectors are working perfectly!")
            print("   The pipeline is generating valid embeddings that decode correctly.")
            print()
        elif avg_cosine >= 0.30:
            print("⚠️  PARTIAL SUCCESS - Better than before but needs improvement")
            print()
        else:
            print("❌ FAILURE - Vectors still broken, same issue as before")
            print()

print("="*80 + "\n")
