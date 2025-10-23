#!/usr/bin/env python3
"""Check vector quality for train/val/test splits"""

import psycopg2
import numpy as np
import sys

def check_vectors(split_name):
    """Check vector quality for a specific split"""
    conn = psycopg2.connect(dbname="lnsp")
    cur = conn.cursor()

    # Fetch all vectors for this split
    query = f"""
    SELECT
        v.cpe_id,
        v.concept_vec::text
    FROM cpe_vectors v
    JOIN lnsp_{split_name} t ON v.cpe_id = t.cpe_id
    """

    cur.execute(query)
    rows = cur.fetchall()

    if not rows:
        print(f"No vectors found for {split_name} split")
        return

    # Parse vectors and compute statistics
    null_count = 0
    zero_norm_count = 0
    high_norm_count = 0
    norms = []

    for cpe_id, vec_str in rows:
        if vec_str is None:
            null_count += 1
            continue

        # Parse pgvector format: [0.1,0.2,0.3,...]
        vec_str = vec_str.strip('[]')
        vec = np.array([float(x) for x in vec_str.split(',')])

        # Check for NaNs or Infs
        if np.any(np.isnan(vec)) or np.any(np.isinf(vec)):
            print(f"WARNING: NaN/Inf detected in {cpe_id}")
            continue

        # Compute L2 norm
        norm = np.linalg.norm(vec)
        norms.append(norm)

        if norm == 0:
            zero_norm_count += 1
        elif norm > 2.5:
            high_norm_count += 1

    norms = np.array(norms)

    print(f"\n=== {split_name.upper()} SPLIT VECTOR QUALITY ===")
    print(f"Total vectors:    {len(rows)}")
    print(f"Null vectors:     {null_count}")
    print(f"Zero norms:       {zero_norm_count}")
    print(f"High norms (>2.5): {high_norm_count}")
    if len(norms) > 0:
        print(f"Norm range:       [{norms.min():.4f}, {norms.max():.4f}]")
        print(f"Norm mean:        {norms.mean():.4f}")
        print(f"Norm std:         {norms.std():.4f}")

    cur.close()
    conn.close()

    # Return True if all checks pass
    return null_count == 0 and zero_norm_count == 0 and high_norm_count == 0

if __name__ == "__main__":
    splits = ['train', 'val', 'test']
    all_pass = True

    for split in splits:
        passed = check_vectors(split)
        if not passed:
            all_pass = False

    print("\n=== OVERALL STATUS ===")
    if all_pass:
        print("✅ All vector quality checks PASSED")
        sys.exit(0)
    else:
        print("⚠️  Some vector quality issues detected (see above)")
        sys.exit(1)
