#!/usr/bin/env python3
"""
Resume re-encoding Wikipedia vectors from batch 241.
(Batches 1-240 already completed successfully)
"""

import os
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.db_postgres import connect as connect_pg
from app.vect_text_vect.vec_text_vect_isolated import IsolatedVecTextVectOrchestrator

# Set environment for vec2text
os.environ['VEC2TEXT_FORCE_PROJECT_VENV'] = '1'
os.environ['VEC2TEXT_DEVICE'] = 'cpu'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# Configuration
BATCH_SIZE = 100
DATASET_SOURCE = 'wikipedia_500k'
RESUME_FROM_BATCH = 241  # Skip batches 1-240 (already done)

def format_time(seconds: float) -> str:
    """Format seconds as human-readable time."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        mins = seconds / 60
        return f"{mins:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.2f}h"

def main():
    print("=" * 80)
    print("RESUMING Re-Encoding Wikipedia Vectors (CORRECT Encoder)")
    print("=" * 80)
    print()
    print("Configuration:")
    print(f"  Dataset source:  {DATASET_SOURCE}")
    print(f"  Batch size:      {BATCH_SIZE}")
    print(f"  Resume from:     Batch {RESUME_FROM_BATCH} (skipping 1-{RESUME_FROM_BATCH-1})")
    print()

    # ============================================================================
    # Step 1: Initialize Orchestrator
    # ============================================================================

    print("Step 1: Initializing vec2text orchestrator (CORRECT encoder)...")
    print()

    orchestrator = IsolatedVecTextVectOrchestrator()
    print("✓ IsolatedVecTextVectOrchestrator loaded")
    print()

    # ============================================================================
    # Step 2: Connect to Database and Count Chunks
    # ============================================================================

    print("Step 2: Connecting to database and counting chunks...")
    print()

    conn = connect_pg()
    cur = conn.cursor()

    count_query = f"""
    SELECT COUNT(*)
    FROM cpe_entry
    WHERE dataset_source = %s
    """
    cur.execute(count_query, (DATASET_SOURCE,))
    total_chunks = cur.fetchone()[0]

    print(f"✓ Found {total_chunks:,} total chunks")

    chunks_already_done = (RESUME_FROM_BATCH - 1) * BATCH_SIZE
    chunks_remaining = total_chunks - chunks_already_done

    print(f"  Already completed: {chunks_already_done:,} chunks (batches 1-{RESUME_FROM_BATCH-1})")
    print(f"  Remaining:         {chunks_remaining:,} chunks")
    print()

    if total_chunks == 0:
        print("✗ No chunks found! Exiting.")
        sys.exit(1)

    # Estimate time (based on test: 0.0102s per chunk with batch encoding)
    estimated_seconds = chunks_remaining * 0.0102
    print(f"Estimated time: {format_time(estimated_seconds)} (at 0.0102s/chunk)")
    print()

    # ============================================================================
    # Step 3: Re-encode in Batches (starting from RESUME_FROM_BATCH)
    # ============================================================================

    print(f"Step 3: Re-encoding vectors starting from batch {RESUME_FROM_BATCH}...")
    print()

    # Fetch all chunks (cpe_id + text)
    fetch_query = f"""
    SELECT cpe_id, concept_text
    FROM cpe_entry
    WHERE dataset_source = %s
    ORDER BY created_at
    """

    print(f"Fetching all {total_chunks:,} chunks from database...")
    cur.execute(fetch_query, (DATASET_SOURCE,))
    all_rows = cur.fetchall()
    print(f"✓ Fetched {len(all_rows):,} rows")
    print()

    # Calculate starting offset
    start_offset = (RESUME_FROM_BATCH - 1) * BATCH_SIZE
    rows_to_process = all_rows[start_offset:]

    print(f"Skipping first {start_offset:,} chunks (batches 1-{RESUME_FROM_BATCH-1})")
    print(f"Processing {len(rows_to_process):,} remaining chunks...")
    print()

    # Process in batches
    num_batches = (len(all_rows) + BATCH_SIZE - 1) // BATCH_SIZE
    total_updated = chunks_already_done  # Start counter from already-done chunks
    start_time = time.time()

    for batch_idx in range(start_offset, len(all_rows), BATCH_SIZE):
        batch_start = time.time()
        batch_rows = all_rows[batch_idx:batch_idx + BATCH_SIZE]

        cpe_ids = [row[0] for row in batch_rows]
        texts = [row[1] for row in batch_rows]

        # Encode batch with CORRECT encoder
        try:
            vectors_tensor = orchestrator.encode_texts(texts)
            vectors_np = vectors_tensor.cpu().detach().numpy()
        except Exception as e:
            print(f"✗ Error encoding batch {batch_idx // BATCH_SIZE + 1}: {e}")
            continue

        # Update database (one by one for safety)
        for cpe_id, vector in zip(cpe_ids, vectors_np):
            try:
                # Convert numpy array to PostgreSQL vector format
                vector_str = '[' + ','.join(map(str, vector)) + ']'

                # Update cpe_vectors table
                cur.execute("""
                    UPDATE cpe_vectors
                    SET concept_vec = %s::vector(768)
                    WHERE cpe_id = %s
                """, (vector_str, cpe_id))

                total_updated += 1

            except Exception as e:
                print(f"✗ Error updating {cpe_id}: {e}")
                continue

        # Commit batch
        conn.commit()

        # Progress report
        batch_time = time.time() - batch_start
        elapsed_time = time.time() - start_time
        chunks_processed = total_updated - chunks_already_done
        chunks_per_sec = chunks_processed / elapsed_time if elapsed_time > 0 else 0
        eta_seconds = (total_chunks - total_updated) / chunks_per_sec if chunks_per_sec > 0 else 0

        batch_num = (batch_idx // BATCH_SIZE) + 1
        progress_pct = (total_updated / total_chunks) * 100

        print(f"Batch {batch_num}/{num_batches} ({progress_pct:.1f}%): "
              f"{len(batch_rows)} chunks in {batch_time:.2f}s "
              f"| Total: {total_updated:,}/{total_chunks:,} "
              f"| Rate: {chunks_per_sec:.1f} chunks/s "
              f"| ETA: {format_time(eta_seconds)}")

    # ============================================================================
    # Step 4: Summary
    # ============================================================================

    total_time = time.time() - start_time

    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print()

    chunks_encoded_now = total_updated - chunks_already_done

    print(f"Re-encoding completed!")
    print(f"  Previously done: {chunks_already_done:,}")
    print(f"  Encoded now:     {chunks_encoded_now:,}")
    print(f"  Total updated:   {total_updated:,}/{total_chunks:,}")
    print(f"  Session time:    {format_time(total_time)}")
    print(f"  Rate:            {chunks_encoded_now / total_time:.1f} chunks/s")
    print()

    if total_updated == total_chunks:
        print("✅ All chunks re-encoded successfully!")
        print()
        print("Next step: Rebuild FAISS index with corrected vectors")
    else:
        print(f"⚠️ {total_chunks - total_updated} chunks still need updating")

    print()
    print("=" * 80)
    print("Re-encoding Complete!")
    print("=" * 80)

    # Cleanup
    cur.close()
    conn.close()

if __name__ == '__main__':
    main()
