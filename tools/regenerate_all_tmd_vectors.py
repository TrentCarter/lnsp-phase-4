#!/usr/bin/env python3
"""
Regenerate ALL TMD vectors from their domain/task/modifier codes.

This fixes entries that have valid codes but zero vectors.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from src.db_postgres import connect
from src.tmd_encoder import tmd16_deterministic

def regenerate_all_tmd_vectors():
    """Regenerate TMD 16D vectors for all entries."""

    conn = connect()
    cur = conn.cursor()

    # Get all entries with their TMD codes
    print("🔍 Loading all TMD codes...")
    cur.execute("""
        SELECT ce.cpe_id, ce.domain_code, ce.task_code, ce.modifier_code
        FROM cpe_entry ce
        ORDER BY ce.cpe_id
    """)

    all_entries = cur.fetchall()
    total = len(all_entries)
    print(f"   Found {total} entries\n")

    print("🔧 Regenerating TMD 16D vectors...")
    regenerated = 0

    for cpe_id, domain, task, modifier in all_entries:
        # Generate the 16D unit vector from codes
        tmd_vec = tmd16_deterministic(domain, task, modifier)

        # Convert to pgvector format string
        tmd_str = '[' + ','.join(str(x) for x in tmd_vec) + ']'

        # Update cpe_vectors with regenerated 16D vector
        cur.execute("""
            UPDATE cpe_vectors
            SET tmd_dense = %s::vector
            WHERE cpe_id = %s
        """, (tmd_str, cpe_id))

        regenerated += 1
        if regenerated % 500 == 0:
            print(f"   Regenerated {regenerated}/{total}...")
            conn.commit()

    conn.commit()
    print(f"\n✅ Regenerated {regenerated} TMD vectors\n")

    # Verify the fix
    print("🔬 Verifying regeneration...")

    # Check for zero vectors
    cur.execute("""
        SELECT COUNT(*)
        FROM cpe_vectors
        WHERE tmd_dense = '[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]'::vector
    """)
    zero_count = cur.fetchone()[0]
    print(f"   Vectors with all zeros: {zero_count}/{total}")

    # Check vector norms for a sample
    cur.execute("""
        SELECT tmd_dense <#> tmd_dense as neg_sq_norm
        FROM cpe_vectors
        LIMIT 100
    """)
    norms = [-row[0] for row in cur.fetchall()]  # Negate because <#> is negative inner product
    zero_norm_count = sum(1 for n in norms if abs(n) < 0.01)

    print(f"   Sample of 100 vectors:")
    print(f"   - Zero norm: {zero_norm_count}/100")
    print(f"   - Non-zero norm: {100-zero_norm_count}/100")
    print(f"   - Mean norm² : {np.mean(norms):.4f}")
    print(f"   - Std norm²: {np.std(norms):.4f}")

    # Show some examples
    cur.execute("""
        SELECT ce.concept_text, ce.domain_code, ce.task_code, ce.modifier_code,
               cv.tmd_dense <#> cv.tmd_dense as neg_sq_norm
        FROM cpe_entry ce
        JOIN cpe_vectors cv ON ce.cpe_id = cv.cpe_id
        ORDER BY RANDOM()
        LIMIT 3
    """)

    print(f"\n   Random examples:")
    for concept, d, t, m, neg_sq_norm in cur.fetchall():
        norm = np.sqrt(-neg_sq_norm) if neg_sq_norm < 0 else 0
        print(f"   - {concept[:50]:50s} | ({d},{t},{m}) | norm={norm:.3f}")

    cur.close()
    conn.close()

    if zero_count == 0:
        print("\n✅ SUCCESS: All TMD vectors regenerated successfully!")
    else:
        print(f"\n⚠️  Warning: Still have {zero_count} zero vectors")

if __name__ == "__main__":
    regenerate_all_tmd_vectors()
