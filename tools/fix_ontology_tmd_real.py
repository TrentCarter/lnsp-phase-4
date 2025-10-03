#!/usr/bin/env python3
"""
Real fix for TMD vectors - assign non-zero codes.

For ontology concepts without explicit TMD values:
- domain: 1 (science/research domain)
- task: 1 (fact_retrieval)
- modifier: 1 (computational - since these are computational ontology terms)

This produces valid 16D unit vectors instead of zeros.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import psycopg2
from src.db_config import get_connection_params
from src.tmd_encoder import tmd16_deterministic
import numpy as np

def fix_zero_tmd_vectors():
    """Update all zero TMD codes to (1,1,1) and recompute vectors."""

    conn = psycopg2.connect(**get_connection_params())
    cur = conn.cursor()

    # Find all entries with zero TMD codes
    cur.execute("""
        SELECT cpe_id, tmd_domain, tmd_task, tmd_modifier
        FROM cpe_entry
        WHERE tmd_domain = 0 AND tmd_task = 0 AND tmd_modifier = 0
    """)

    zero_entries = cur.fetchall()
    print(f"Found {len(zero_entries)} entries with zero TMD codes")

    if len(zero_entries) == 0:
        print("No entries to fix!")
        return

    # Update each entry with non-zero codes
    fixed = 0
    for cpe_id, _, _, _ in zero_entries:
        # Assign: science (1), fact_retrieval (1), computational (1)
        new_domain = 1
        new_task = 1
        new_modifier = 1

        # Generate the 16D vector
        tmd_vec = tmd16_deterministic(new_domain, new_task, new_modifier)
        tmd_list = tmd_vec.tolist()

        # Update cpe_entry with new codes
        cur.execute("""
            UPDATE cpe_entry
            SET tmd_domain = %s, tmd_task = %s, tmd_modifier = %s
            WHERE cpe_id = %s
        """, (new_domain, new_task, new_modifier, cpe_id))

        # Update cpe_vectors with new 16D vector
        cur.execute("""
            UPDATE cpe_vectors
            SET tmd_dense = %s
            WHERE cpe_id = %s
        """, (tmd_list, cpe_id))

        fixed += 1
        if fixed % 500 == 0:
            print(f"  Fixed {fixed}/{len(zero_entries)}...")
            conn.commit()

    conn.commit()
    print(f"✅ Fixed {fixed} TMD vectors")

    # Verify the fix
    cur.execute("""
        SELECT COUNT(*)
        FROM cpe_entry
        WHERE tmd_domain = 0 AND tmd_task = 0 AND tmd_modifier = 0
    """)
    remaining_zeros = cur.fetchone()[0]

    print(f"\nVerification:")
    print(f"  Remaining zero codes: {remaining_zeros}")

    # Check a sample vector
    cur.execute("""
        SELECT cv.tmd_dense, ce.tmd_domain, ce.tmd_task, ce.tmd_modifier
        FROM cpe_vectors cv
        JOIN cpe_entry ce ON cv.cpe_id = ce.cpe_id
        LIMIT 1
    """)
    sample = cur.fetchone()
    if sample:
        tmd_vec, d, t, m = sample
        tmd_arr = np.array(tmd_vec)
        print(f"  Sample TMD: domain={d}, task={t}, modifier={m}")
        print(f"  Sample vector norm: {np.linalg.norm(tmd_arr):.4f}")
        print(f"  Sample vector: {tmd_arr[:8]}...")

    cur.close()
    conn.close()

if __name__ == "__main__":
    fix_zero_tmd_vectors()
