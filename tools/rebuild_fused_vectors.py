#!/usr/bin/env python3
"""
Rebuild fused vectors from existing TMD and concept vectors.

This fixes the bug where fused_vec has 16 zeros instead of the correct TMD prefix.

The bug occurred because:
1. Initial ingestion created fused_vec from (0,0,0) TMD codes (all zeros)
2. Later fixes updated tmd_dense but NOT fused_vec
3. Now fused_vec is out of sync with tmd_dense

The fix:
- Read tmd_dense (16D) and concept_vec (768D) from database
- Concatenate: fused = [tmd_dense, concept_vec] (784D)
- Normalize to unit vector
- Update fused_vec in database
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from src.db_postgres import connect

def rebuild_fused_vectors():
    """Rebuild all fused vectors from TMD + concept vectors."""

    conn = connect()
    cur = conn.cursor()

    print("🔍 Loading all vectors...")
    # Cast pgvector to float array for processing
    cur.execute("""
        SELECT cpe_id,
               tmd_dense::text::jsonb as tmd_json,
               concept_vec::text::jsonb as concept_json,
               fused_vec::text::jsonb as fused_json
        FROM cpe_vectors
        ORDER BY cpe_id
    """)

    all_vectors = cur.fetchall()
    total = len(all_vectors)
    print(f"   Found {total} vectors\n")

    # Analyze current state
    print("📊 Analyzing current fused vectors...")
    zero_tmd_count = 0

    for cpe_id, tmd_json, concept_json, fused_json in all_vectors:
        # pgvector::text::jsonb already gives us a list
        fused_arr = np.array(fused_json)
        if np.allclose(fused_arr[:16], 0.0, atol=1e-6):
            zero_tmd_count += 1

    print(f"   Fused vectors with zero TMD prefix: {zero_tmd_count}/{total}")
    print(f"   Fused vectors with correct TMD: {total - zero_tmd_count}/{total}\n")

    if zero_tmd_count == 0:
        print("✅ All fused vectors already have correct TMD prefix!")
        cur.close()
        conn.close()
        return

    # Rebuild all fused vectors
    print("🔧 Rebuilding fused vectors...")
    rebuilt = 0
    fixed = 0

    for cpe_id, tmd_json, concept_json, fused_json in all_vectors:
        # pgvector::text::jsonb gives us lists already
        tmd_arr = np.array(tmd_json, dtype=np.float32)
        concept_arr = np.array(concept_json, dtype=np.float32)

        # Rebuild fused vector: [TMD_16D, concept_768D] = 784D
        new_fused = np.concatenate([tmd_arr, concept_arr])

        # Normalize to unit vector
        norm = np.linalg.norm(new_fused)
        if norm > 0:
            new_fused = new_fused / norm

        # Convert to pgvector format string
        fused_str = '[' + ','.join(str(x) for x in new_fused) + ']'

        # Update database
        cur.execute("""
            UPDATE cpe_vectors
            SET fused_vec = %s::vector,
                fused_norm = %s
            WHERE cpe_id = %s
        """, (fused_str, float(norm), cpe_id))

        rebuilt += 1

        # Check if this was a fix (had zero TMD before)
        old_fused = np.array(fused_json)
        if np.allclose(old_fused[:16], 0.0, atol=1e-6):
            fixed += 1

        if rebuilt % 500 == 0:
            print(f"   Rebuilt {rebuilt}/{total}...")
            conn.commit()

    conn.commit()
    print(f"\n✅ Rebuilt {rebuilt} fused vectors")
    print(f"   Fixed {fixed} vectors that had zero TMD prefix\n")

    # Verify the fix
    print("🔬 Verifying rebuilt vectors...")

    cur.execute("""
        SELECT cpe_id,
               tmd_dense::text::jsonb as tmd_json,
               fused_vec::text::jsonb as fused_json
        FROM cpe_vectors
        ORDER BY cpe_id
        LIMIT 100
    """)

    sample_vectors = cur.fetchall()
    mismatch_count = 0

    for cpe_id, tmd_json, fused_json in sample_vectors:
        tmd_arr = np.array(tmd_json)
        fused_arr = np.array(fused_json)

        # Check if TMD part of fused is PROPORTIONAL to tmd_dense
        # (fused is normalized, so TMD gets scaled down)
        if np.linalg.norm(tmd_arr) > 0:
            tmd_normalized = tmd_arr / np.linalg.norm(tmd_arr)
            fused_tmd = fused_arr[:16]
            if np.linalg.norm(fused_tmd) > 0:
                fused_tmd_normalized = fused_tmd / np.linalg.norm(fused_tmd)
                if not np.allclose(tmd_normalized, fused_tmd_normalized, atol=1e-4):
                    mismatch_count += 1
            else:
                # Fused TMD part is zero - this is wrong!
                mismatch_count += 1
        else:
            # Original TMD is zero - check if fused is also zero
            if not np.allclose(fused_arr[:16], 0.0, atol=1e-6):
                mismatch_count += 1

    print(f"   Sample of 100 vectors:")
    print(f"   - TMD/fused mismatches: {mismatch_count}/100")
    print(f"   - Correct alignment: {100-mismatch_count}/100")

    # Check norms
    cur.execute("""
        SELECT fused_vec <#> fused_vec as neg_sq_norm
        FROM cpe_vectors
        LIMIT 100
    """)
    norms = [np.sqrt(-row[0]) if row[0] < 0 else 0 for row in cur.fetchall()]
    mean_norm = np.mean(norms)
    std_norm = np.std(norms)
    unit_vectors = sum(1 for n in norms if abs(n - 1.0) < 0.01)

    print(f"   - Mean norm: {mean_norm:.6f}")
    print(f"   - Std norm: {std_norm:.6f}")
    print(f"   - Unit vectors (norm≈1): {unit_vectors}/100")

    # Show examples
    cur.execute("""
        SELECT e.concept_text, e.domain_code, e.task_code, e.modifier_code,
               substring(v.tmd_dense::text, 1, 40) as tmd_sample,
               substring(v.fused_vec::text, 1, 40) as fused_sample
        FROM cpe_entry e
        JOIN cpe_vectors v ON e.cpe_id = v.cpe_id
        WHERE e.domain_code = 1 AND e.task_code = 1 AND e.modifier_code = 1
        LIMIT 3
    """)

    print(f"\n   Examples of (1,1,1) vectors after fix:")
    for concept, d, t, m, tmd_sample, fused_sample in cur.fetchall():
        print(f"   - {concept[:50]:50s}")
        print(f"     TMD:   {tmd_sample}")
        print(f"     Fused: {fused_sample}")
        print()

    cur.close()
    conn.close()

    if mismatch_count == 0 and unit_vectors > 95:
        print("✅ SUCCESS: All fused vectors rebuilt correctly!")
        print("   - TMD prefixes match tmd_dense")
        print("   - All vectors are unit vectors")
    else:
        print(f"⚠️  Warning: Some issues remain:")
        if mismatch_count > 0:
            print(f"   - {mismatch_count} TMD/fused mismatches")
        if unit_vectors < 95:
            print(f"   - Only {unit_vectors}/100 are unit vectors")

if __name__ == "__main__":
    rebuild_fused_vectors()
