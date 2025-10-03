#!/usr/bin/env python3
"""
Real fix for TMD vectors - assign non-zero codes to produce valid embeddings.

Uses domain=1, task=1, modifier=1 instead of 0,0,0 to avoid zero-norm vectors.
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import psycopg2
import numpy as np

# Import the connection and TMD encoder
from src.db_postgres import connect
from src.tmd_encoder import tmd16_deterministic

def fix_zero_tmd_vectors():
    """Update all zero TMD codes to (1,1,1) and recompute 16D vectors."""

    conn = connect()
    cur = conn.cursor()

    # Find entries with zero TMD codes (produces zero-norm vectors)
    print("🔍 Finding entries with zero TMD codes...")
    cur.execute("""
        SELECT cpe_id, domain_code, task_code, modifier_code
        FROM cpe_entry
        WHERE domain_code = 0 AND task_code = 0 AND modifier_code = 0
    """)

    zero_entries = cur.fetchall()
    total = len(zero_entries)
    print(f"   Found {total} entries with zero TMD codes\n")

    if total == 0:
        print("✅ No entries need fixing!")
        return

    # Update each entry with non-zero codes
    print("🔧 Fixing TMD codes and regenerating 16D vectors...")
    fixed = 0

    for cpe_id, old_d, old_t, old_m in zero_entries:
        # Assign non-zero codes: science (1), fact_retrieval (1), computational (1)
        new_domain = 1
        new_task = 1
        new_modifier = 1

        # Generate the 16D unit vector
        tmd_vec = tmd16_deterministic(new_domain, new_task, new_modifier)

        # Convert to pgvector format string
        tmd_str = '[' + ','.join(str(x) for x in tmd_vec) + ']'

        # Update cpe_entry with new codes
        cur.execute("""
            UPDATE cpe_entry
            SET domain_code = %s, task_code = %s, modifier_code = %s
            WHERE cpe_id = %s
        """, (new_domain, new_task, new_modifier, cpe_id))

        # Update cpe_vectors with new 16D vector (cast to vector type)
        cur.execute("""
            UPDATE cpe_vectors
            SET tmd_dense = %s::vector
            WHERE cpe_id = %s
        """, (tmd_str, cpe_id))

        fixed += 1
        if fixed % 500 == 0:
            print(f"   Fixed {fixed}/{total}...")
            conn.commit()

    conn.commit()
    print(f"\n✅ Fixed {fixed} TMD vectors\n")

    # Verify the fix
    print("🔬 Verifying fix...")
    cur.execute("""
        SELECT COUNT(*)
        FROM cpe_entry
        WHERE domain_code = 0 AND task_code = 0 AND modifier_code = 0
    """)
    remaining_zeros = cur.fetchone()[0]
    print(f"   Remaining zero codes: {remaining_zeros}")

    # Check actual vector norms
    cur.execute("""
        SELECT tmd_dense FROM cpe_vectors LIMIT 100
    """)
    sample_vecs = [np.array(row[0]) for row in cur.fetchall()]
    norms = [np.linalg.norm(v) for v in sample_vecs]
    zero_norms = sum(1 for n in norms if n == 0)

    print(f"   Sample of 100 vectors:")
    print(f"   - Zero norm: {zero_norms}/100")
    print(f"   - Non-zero norm: {100-zero_norms}/100")
    print(f"   - Mean norm: {np.mean(norms):.4f}")
    print(f"   - Std norm: {np.std(norms):.4f}")

    # Show one example
    cur.execute("""
        SELECT cv.tmd_dense, ce.domain_code, ce.task_code, ce.modifier_code, ce.concept_text
        FROM cpe_vectors cv
        JOIN cpe_entry ce ON cv.cpe_id = ce.cpe_id
        WHERE ce.domain_code = 1 AND ce.task_code = 1 AND ce.modifier_code = 1
        LIMIT 1
    """)
    example = cur.fetchone()
    if example:
        tmd_vec, d, t, m, concept = example
        tmd_arr = np.array(tmd_vec)
        print(f"\n   Example:")
        print(f"   - Concept: {concept[:60]}...")
        print(f"   - TMD codes: domain={d}, task={t}, modifier={m}")
        print(f"   - Vector norm: {np.linalg.norm(tmd_arr):.4f}")
        print(f"   - Vector[:4]: {tmd_arr[:4]}")

    cur.close()
    conn.close()

    if remaining_zeros == 0 and zero_norms == 0:
        print("\n✅ SUCCESS: All TMD vectors now have non-zero norms!")
    else:
        print(f"\n⚠️  Warning: Still have issues (zeros={remaining_zeros}, zero_norms={zero_norms})")

if __name__ == "__main__":
    fix_zero_tmd_vectors()
