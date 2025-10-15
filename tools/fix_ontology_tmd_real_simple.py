#!/usr/bin/env python3
"""
Fix TMD Vectors with Vec2Text-Compatible Encoder
==================================================
Reads database chunks, re-encodes with port 8767 (vec2text-compatible),
updates database, and regenerates training sequences NPZ.
"""
import psycopg2
import numpy as np
import requests
from pathlib import Path
import sys
import json

DB_CONFIG = {
    "dbname": "lnsp",
    "user": "trentcarter",
    "password": "",
    "host": "localhost",
    "port": 5432
}

VEC2TEXT_ENCODER_URL = "http://127.0.0.1:8767"
VEC2TEXT_DECODER_URL = "http://127.0.0.1:8766"
BATCH_SIZE = 100

def parse_pg_vector(vec_str):
    """Parse PostgreSQL vector format to numpy array."""
    if isinstance(vec_str, str):
        # Remove brackets and parse
        vec_str = vec_str.strip('[]')
        return np.array([float(x) for x in vec_str.split(',')], dtype=np.float32)
    return np.array(vec_str, dtype=np.float32)

def main():
    print("="*80)
    print("Regenerating Vec2Text-Compatible Vectors")
    print("="*80)
    
    # Step 1: Get all texts
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    
    cur.execute("""
        SELECT e.cpe_id, e.concept_text
        FROM cpe_entry e
        WHERE e.dataset_source = 'user_input'
        ORDER BY e.cpe_id
    """)
    rows = cur.fetchall()
    print(f"✓ Found {len(rows)} texts")
    
    # Step 2: Re-encode in batches
    print(f"\nRe-encoding in batches of {BATCH_SIZE}...")
    all_vectors = []
    
    for i in range(0, len(rows), BATCH_SIZE):
        batch = rows[i:i+BATCH_SIZE]
        texts = [row[1] for row in batch]
        
        response = requests.post(
            f"{VEC2TEXT_ENCODER_URL}/embed",
            json={"texts": texts},
            timeout=30
        )
        vectors = np.array(response.json()["embeddings"], dtype=np.float32)
        all_vectors.append(vectors)
        
        print(f"  Batch {i//BATCH_SIZE + 1}: {vectors.shape}")
    
    all_vectors = np.vstack(all_vectors)
    print(f"✓ Total: {all_vectors.shape}")
    
    # Step 3: Update database
    print("\nUpdating database...")
    for (cpe_id, _), vec in zip(rows, all_vectors):
        cur.execute("""
            UPDATE cpe_vectors
            SET concept_vec = %s::vector
            WHERE cpe_id = %s
        """, (vec.tolist(), cpe_id))
    
    conn.commit()
    print("✓ Database updated")
    
    # Step 4: Regenerate NPZ
    print("\nRegenerating NPZ files...")
    
    cur.execute("""
        SELECT e.cpe_id, e.concept_text, v.concept_vec
        FROM cpe_entry e
        JOIN cpe_vectors v ON e.cpe_id = v.cpe_id
        WHERE e.dataset_source = 'user_input'
        ORDER BY e.cpe_id
    """)
    rows = cur.fetchall()
    
    cpe_ids = [str(r[0]) for r in rows]
    texts = [r[1] for r in rows]
    vectors = [parse_pg_vector(r[2]) for r in rows]
    
    print(f"✓ Parsed {len(vectors)} vectors from database")
    
    # Create sequences
    context_size = 5
    sequences = []
    targets = []
    
    for i in range(len(vectors) - context_size):
        ctx = np.stack(vectors[i:i+context_size])
        tgt = vectors[i+context_size]
        sequences.append(ctx)
        targets.append(tgt)
    
    sequences = np.stack(sequences)
    targets = np.stack(targets)
    
    print(f"✓ Created {len(sequences)} training sequences")
    
    # Save files
    output_dir = Path("artifacts/lvm")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    ordered_path = output_dir / "wikipedia_42113_ordered.npz"
    np.savez_compressed(ordered_path, cpe_ids=np.array(cpe_ids), texts=np.array(texts), vectors=np.stack(vectors))
    
    sequences_path = output_dir / "training_sequences_ctx5.npz"
    np.savez_compressed(sequences_path, context_sequences=sequences, target_vectors=targets)
    
    print(f"✓ Saved: {ordered_path}")
    print(f"✓ Saved: {sequences_path}")
    
    # Step 5: Verify
    print("\nVerifying...")
    test_vectors = np.stack(vectors[:3])
    test_texts = texts[:3]
    
    for i, (vec, orig_text) in enumerate(zip(test_vectors, test_texts)):
        response = requests.post(f"{VEC2TEXT_DECODER_URL}/invert", json={"embeddings": [vec.tolist()]}, timeout=30)
        decoded = response.json()["texts"][0]
        
        enc_response = requests.post(f"{VEC2TEXT_ENCODER_URL}/embed", json={"texts": [decoded]}, timeout=30)
        decoded_vec = np.array(enc_response.json()["embeddings"][0])
        
        cosine = np.dot(vec, decoded_vec) / (np.linalg.norm(vec) * np.linalg.norm(decoded_vec))
        
        print(f"\nTest {i+1}:")
        print(f"  Original: {orig_text[:60]}...")
        print(f"  Decoded:  {decoded[:60]}...")
        print(f"  Cosine:   {cosine:.4f} {'✅' if cosine > 0.65 else '❌'}")
    
    cur.close()
    conn.close()
    
    print("\n" + "="*80)
    print("✅ DONE! New training data ready.")
    print("="*80)

if __name__ == "__main__":
    main()
