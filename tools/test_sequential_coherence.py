#!/usr/bin/env python3
"""
Test Sequential Coherence in Database Chunks

Validates that ingested chunks maintain narrative/expository/instructional flow
by checking cosine similarity between consecutive vectors.

Usage:
    ./tools/test_sequential_coherence.py --dataset ALL --test-count 1000 --walk-count 10 --order random
    ./tools/test_sequential_coherence.py --dataset "watercycle-mini|semantic-75" --test-count 100 --order sequential
    ./tools/test_sequential_coherence.py --start-id 100 --end-id 500 --test-count 50 --walk-count 5
"""

import argparse
import os
import sys
import psycopg2
import numpy as np
from typing import List, Tuple, Optional, Dict
import random
from dataclasses import dataclass
from collections import defaultdict

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


@dataclass
class ChunkSequence:
    """A sequence of chunks for coherence testing"""
    start_id: int
    chunk_ids: List[int]
    concept_texts: List[str]
    vectors: np.ndarray  # Shape: [walk_count, 768]
    dataset_source: str


@dataclass
class CoherenceScore:
    """Coherence metrics for a sequence"""
    sequence: ChunkSequence
    cosine_similarities: List[float]  # Between consecutive chunks
    mean_similarity: float
    min_similarity: float
    is_coherent: bool  # True if mean > threshold
    coherence_threshold: float


# ANSI colors
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
BLUE = "\033[94m"
BOLD = "\033[1m"
RESET = "\033[0m"


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Compute cosine similarity between two vectors"""
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 < 1e-8 or norm2 < 1e-8:
        return 0.0
    return float(np.dot(vec1, vec2) / (norm1 * norm2))


def connect_db() -> psycopg2.extensions.connection:
    """Connect to PostgreSQL database"""
    return psycopg2.connect(
        dbname=os.getenv("LNSP_DB_NAME", "lnsp"),
        user=os.getenv("LNSP_DB_USER", "postgres"),
        password=os.getenv("LNSP_DB_PASSWORD", "password"),
        host=os.getenv("LNSP_DB_HOST", "localhost"),
        port=int(os.getenv("LNSP_DB_PORT", "5432"))
    )


def get_dataset_stats(conn) -> Dict[str, int]:
    """Get chunk counts per dataset_source"""
    cursor = conn.cursor()
    cursor.execute("""
        SELECT dataset_source, COUNT(*) as count
        FROM cpe_entry
        GROUP BY dataset_source
        ORDER BY count DESC
    """)

    stats = {}
    for row in cursor.fetchall():
        dataset_source, count = row
        stats[dataset_source] = count

    cursor.close()
    return stats


def get_chunk_id_range(conn, dataset_source: Optional[str] = None) -> Tuple[int, int]:
    """Get min and max chunk IDs for a dataset"""
    cursor = conn.cursor()

    if dataset_source and dataset_source != "ALL":
        cursor.execute("""
            SELECT MIN(id), MAX(id)
            FROM cpe_entry
            WHERE dataset_source = %s
        """, (dataset_source,))
    else:
        cursor.execute("""
            SELECT MIN(id), MAX(id)
            FROM cpe_entry
        """)

    result = cursor.fetchone()
    cursor.close()

    if not result or result[0] is None:
        return 0, 0

    return int(result[0]), int(result[1])


def fetch_chunk_sequence(
    conn,
    start_id: int,
    walk_count: int,
    dataset_source: Optional[str] = None
) -> Optional[ChunkSequence]:
    """Fetch a sequence of consecutive chunks with vectors"""
    cursor = conn.cursor()

    # Query for consecutive chunks
    if dataset_source and dataset_source != "ALL":
        query = """
            SELECT ce.id, ce.concept_text, ce.dataset_source, cv.concept_vec
            FROM cpe_entry ce
            JOIN cpe_vectors cv ON ce.cpe_id = cv.cpe_id
            WHERE ce.id >= %s AND ce.dataset_source = %s
            ORDER BY ce.id
            LIMIT %s
        """
        cursor.execute(query, (start_id, dataset_source, walk_count))
    else:
        query = """
            SELECT ce.id, ce.concept_text, ce.dataset_source, cv.concept_vec
            FROM cpe_entry ce
            JOIN cpe_vectors cv ON ce.cpe_id = cv.cpe_id
            WHERE ce.id >= %s
            ORDER BY ce.id
            LIMIT %s
        """
        cursor.execute(query, (start_id, walk_count))

    rows = cursor.fetchall()
    cursor.close()

    if len(rows) < 2:  # Need at least 2 for coherence check
        return None

    chunk_ids = [row[0] for row in rows]
    concept_texts = [row[1] for row in rows]
    dataset_src = rows[0][2]  # Use first chunk's dataset

    # Extract vectors (stored as JSONB arrays in PostgreSQL)
    vectors = []
    for row in rows:
        vec_data = row[3]  # concept_vec JSONB
        if isinstance(vec_data, list):
            vec = np.array(vec_data, dtype=np.float32)
        else:
            # Fallback if vector is missing
            vec = np.zeros(768, dtype=np.float32)
        vectors.append(vec)

    vectors_array = np.vstack(vectors)

    return ChunkSequence(
        start_id=start_id,
        chunk_ids=chunk_ids,
        concept_texts=concept_texts,
        vectors=vectors_array,
        dataset_source=dataset_src
    )


def compute_coherence(sequence: ChunkSequence, threshold: float = 0.5) -> CoherenceScore:
    """Compute coherence metrics for a chunk sequence"""
    similarities = []

    for i in range(len(sequence.vectors) - 1):
        sim = cosine_similarity(sequence.vectors[i], sequence.vectors[i + 1])
        similarities.append(sim)

    mean_sim = float(np.mean(similarities)) if similarities else 0.0
    min_sim = float(np.min(similarities)) if similarities else 0.0

    return CoherenceScore(
        sequence=sequence,
        cosine_similarities=similarities,
        mean_similarity=mean_sim,
        min_similarity=min_sim,
        is_coherent=(mean_sim >= threshold),
        coherence_threshold=threshold
    )


def print_summary_stats(
    scores: List[CoherenceScore],
    dataset_stats: Dict[str, int]
):
    """Print summary statistics"""
    print(f"\n{BOLD}{BLUE}{'=' * 80}{RESET}")
    print(f"{BOLD}{BLUE}Sequential Coherence Test Summary{RESET}")
    print(f"{BOLD}{BLUE}{'=' * 80}{RESET}\n")

    # Dataset distribution
    print(f"{BOLD}Datasets Tested:{RESET}")
    dataset_counts = defaultdict(int)
    for score in scores:
        dataset_counts[score.sequence.dataset_source] += 1

    for dataset, count in sorted(dataset_counts.items(), key=lambda x: -x[1]):
        total = dataset_stats.get(dataset, 0)
        pct = (count / len(scores) * 100) if scores else 0
        print(f"  {dataset:30s}: {count:4d} sequences ({pct:5.1f}% of tests, {total} total chunks)")

    # Overall coherence
    print(f"\n{BOLD}Overall Coherence:{RESET}")
    coherent_count = sum(1 for s in scores if s.is_coherent)
    coherent_pct = (coherent_count / len(scores) * 100) if scores else 0

    all_sims = [sim for score in scores for sim in score.cosine_similarities]
    mean_sims = [score.mean_similarity for score in scores]

    print(f"  Tests: {len(scores)}")
    print(f"  Coherent sequences: {coherent_count}/{len(scores)} ({coherent_pct:.1f}%)")
    print(f"  Mean similarity (overall): {np.mean(all_sims):.4f}")
    print(f"  Mean similarity (per sequence): {np.mean(mean_sims):.4f}")
    print(f"  Min similarity (worst): {np.min(all_sims):.4f}")
    print(f"  Max similarity (best): {np.max(all_sims):.4f}")

    # Quality assessment
    print(f"\n{BOLD}Quality Assessment:{RESET}")
    excellent = sum(1 for s in mean_sims if s >= 0.8)
    good = sum(1 for s in mean_sims if 0.6 <= s < 0.8)
    fair = sum(1 for s in mean_sims if 0.4 <= s < 0.6)
    poor = sum(1 for s in mean_sims if s < 0.4)

    print(f"  Excellent (≥0.8): {excellent:4d} ({excellent/len(scores)*100:5.1f}%) {GREEN}✓{RESET}")
    print(f"  Good (0.6-0.8):   {good:4d} ({good/len(scores)*100:5.1f}%) {GREEN}✓{RESET}")
    print(f"  Fair (0.4-0.6):   {fair:4d} ({fair/len(scores)*100:5.1f}%) {YELLOW}⚠{RESET}")
    print(f"  Poor (<0.4):      {poor:4d} ({poor/len(scores)*100:5.1f}%) {RED}✗{RESET}")

    # Verdict
    print(f"\n{BOLD}Verdict:{RESET}")
    if excellent + good >= len(scores) * 0.8:
        print(f"  {GREEN}✓ PASS{RESET} - Dataset is suitable for autoregressive LVM training")
        print(f"    {excellent + good}/{len(scores)} sequences show strong sequential coherence")
    elif excellent + good >= len(scores) * 0.5:
        print(f"  {YELLOW}⚠ MARGINAL{RESET} - Dataset may work but needs review")
        print(f"    Only {excellent + good}/{len(scores)} sequences show strong coherence")
    else:
        print(f"  {RED}✗ FAIL{RESET} - Dataset NOT suitable for autoregressive training")
        print(f"    Only {excellent + good}/{len(scores)} sequences show strong coherence")
        print(f"    Consider using document-based datasets instead of ontological hierarchies")


def print_worst_sequences(scores: List[CoherenceScore], count: int = 5):
    """Print details of worst-scoring sequences"""
    print(f"\n{BOLD}Worst {count} Sequences (Lowest Mean Coherence):{RESET}\n")

    sorted_scores = sorted(scores, key=lambda s: s.mean_similarity)

    for i, score in enumerate(sorted_scores[:count], 1):
        seq = score.sequence
        color = RED if score.mean_similarity < 0.4 else YELLOW

        print(f"{i}. {color}Mean: {score.mean_similarity:.4f}{RESET} | "
              f"Min: {score.min_similarity:.4f} | "
              f"Dataset: {seq.dataset_source}")
        print(f"   Start ID: {seq.start_id} | Chunks: {len(seq.chunk_ids)}")
        print(f"   Preview:")
        for j, text in enumerate(seq.concept_texts[:3]):
            preview = text[:80] + "..." if len(text) > 80 else text
            print(f"     [{j}] {preview}")
        print()


def print_best_sequences(scores: List[CoherenceScore], count: int = 5):
    """Print details of best-scoring sequences"""
    print(f"\n{BOLD}Best {count} Sequences (Highest Mean Coherence):{RESET}\n")

    sorted_scores = sorted(scores, key=lambda s: s.mean_similarity, reverse=True)

    for i, score in enumerate(sorted_scores[:count], 1):
        seq = score.sequence

        print(f"{i}. {GREEN}Mean: {score.mean_similarity:.4f}{RESET} | "
              f"Min: {score.min_similarity:.4f} | "
              f"Dataset: {seq.dataset_source}")
        print(f"   Start ID: {seq.start_id} | Chunks: {len(seq.chunk_ids)}")
        print(f"   Preview:")
        for j, text in enumerate(seq.concept_texts[:3]):
            preview = text[:80] + "..." if len(text) > 80 else text
            print(f"     [{j}] {preview}")
        print()


def main():
    parser = argparse.ArgumentParser(
        description="Test sequential coherence in database chunks"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="ALL",
        help="Dataset source to test (e.g., 'watercycle-mini|semantic-75' or 'ALL')"
    )
    parser.add_argument(
        "--start-id",
        type=int,
        help="Start ID for manual range (overrides --dataset)"
    )
    parser.add_argument(
        "--end-id",
        type=int,
        help="End ID for manual range (overrides --dataset)"
    )
    parser.add_argument(
        "--test-count",
        type=int,
        default=100,
        help="Number of sequences to test (default: 100)"
    )
    parser.add_argument(
        "--walk-count",
        type=int,
        default=10,
        help="Number of consecutive chunks per sequence (default: 10)"
    )
    parser.add_argument(
        "--order",
        type=str,
        choices=["random", "sequential"],
        default="random",
        help="Selection order: random or sequential (default: random)"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Coherence threshold (default: 0.5)"
    )
    parser.add_argument(
        "--show-examples",
        type=int,
        default=5,
        help="Number of best/worst examples to show (default: 5)"
    )

    args = parser.parse_args()

    # Connect to database
    print(f"{BOLD}Connecting to database...{RESET}")
    try:
        conn = connect_db()
    except Exception as e:
        print(f"{RED}✗ Failed to connect to database: {e}{RESET}")
        return 1

    print(f"{GREEN}✓ Connected{RESET}\n")

    # Get dataset statistics
    dataset_stats = get_dataset_stats(conn)

    print(f"{BOLD}Available Datasets:{RESET}")
    for dataset, count in sorted(dataset_stats.items(), key=lambda x: -x[1]):
        print(f"  {dataset:30s}: {count:6d} chunks")
    print()

    # Determine ID range
    if args.start_id is not None and args.end_id is not None:
        min_id, max_id = args.start_id, args.end_id
        dataset_filter = None
        print(f"{BOLD}Testing ID range:{RESET} {min_id} - {max_id}")
    else:
        dataset_filter = args.dataset
        min_id, max_id = get_chunk_id_range(conn, dataset_filter)
        if min_id == 0 and max_id == 0:
            print(f"{RED}✗ No chunks found for dataset '{dataset_filter}'{RESET}")
            conn.close()
            return 1
        print(f"{BOLD}Testing dataset:{RESET} {dataset_filter}")
        print(f"{BOLD}ID range:{RESET} {min_id} - {max_id}")

    print(f"{BOLD}Test parameters:{RESET}")
    print(f"  Test count: {args.test_count}")
    print(f"  Walk count: {args.walk_count} chunks per sequence")
    print(f"  Order: {args.order}")
    print(f"  Threshold: {args.threshold}")
    print()

    # Generate test starting IDs
    if args.order == "random":
        # Random sampling
        possible_starts = range(min_id, max_id - args.walk_count + 1)
        if len(possible_starts) < args.test_count:
            print(f"{YELLOW}⚠ Warning: Only {len(possible_starts)} possible starts, using all{RESET}")
            start_ids = list(possible_starts)
        else:
            start_ids = random.sample(possible_starts, args.test_count)
    else:
        # Sequential sampling
        step = max(1, (max_id - min_id) // args.test_count)
        start_ids = list(range(min_id, max_id - args.walk_count + 1, step))[:args.test_count]

    # Run tests
    print(f"{BOLD}Running coherence tests...{RESET}\n")

    scores = []
    failed_fetches = 0

    for i, start_id in enumerate(start_ids, 1):
        if i % 10 == 0:
            print(f"  Progress: {i}/{len(start_ids)} ({i/len(start_ids)*100:.1f}%)")

        sequence = fetch_chunk_sequence(conn, start_id, args.walk_count, dataset_filter)

        if sequence is None:
            failed_fetches += 1
            continue

        score = compute_coherence(sequence, args.threshold)
        scores.append(score)

    conn.close()

    if not scores:
        print(f"{RED}✗ No valid sequences found{RESET}")
        return 1

    if failed_fetches > 0:
        print(f"\n{YELLOW}⚠ {failed_fetches} sequences could not be fetched{RESET}")

    # Print results
    print_summary_stats(scores, dataset_stats)
    print_best_sequences(scores, args.show_examples)
    print_worst_sequences(scores, args.show_examples)

    print(f"\n{BOLD}{BLUE}{'=' * 80}{RESET}")
    print(f"{GREEN}✓ Coherence test complete{RESET}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
