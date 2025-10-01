#!/usr/bin/env python3
"""
6-Degree Shortcuts Implementation for Neo4j Graph

Based on Watts-Strogatz small-world network research:
- Add semantic shortcuts between distant but similar concepts
- Target: 0.5-2% shortcut rate (default 1%)
- Reduces retrieval hops from ~5-7 to <3

Author: [Architect]
Date: 2025-10-01
"""

import sys
import random
import numpy as np
import psycopg2
from neo4j import GraphDatabase
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from tqdm import tqdm


@dataclass
class Concept:
    """Concept with vector for similarity computation."""
    cpe_id: str
    text: str
    vector: np.ndarray
    tmd_lane: Optional[int]
    source: Optional[str]


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot_product / (norm1 * norm2)


def load_concept_vectors_from_postgres() -> Dict[str, Concept]:
    """
    Load all concepts with vectors from Postgres.

    Returns:
        Dict mapping cpe_id -> Concept with vector
    """
    conn = psycopg2.connect("dbname=lnsp")
    cur = conn.cursor()

    query = """
        SELECT
            e.cpe_id,
            e.concept_text,
            e.tmd_lane,
            e.dataset_source,
            v.concept_vec::text
        FROM cpe_entry e
        JOIN cpe_vectors v USING (cpe_id)
        WHERE e.validation_status = 'passed'
          AND v.concept_vec IS NOT NULL;
    """

    cur.execute(query)

    concepts = {}
    for row in cur.fetchall():
        cpe_id = row[0]
        # Convert Postgres vector to numpy array
        vec_data = row[4]
        if isinstance(vec_data, str):
            # Parse pgvector format: "[1.0,2.0,3.0]" or "{1.0,2.0,3.0}"
            vec_data = vec_data.strip('[]{}').split(',')
            vector = np.array([float(v.strip()) for v in vec_data], dtype=np.float32)
        elif isinstance(vec_data, list):
            vector = np.array(vec_data, dtype=np.float32)
        else:
            # Already an array
            vector = np.array(vec_data, dtype=np.float32)

        concepts[cpe_id] = Concept(
            cpe_id=cpe_id,
            text=row[1],
            vector=vector,
            tmd_lane=row[2],
            source=row[3]
        )

    cur.close()
    conn.close()

    return concepts


def find_distant_candidates(
    neo4j_session,
    source_id: str,
    min_hops: int = 4,
    max_hops: int = 10,
    max_candidates: int = 100
) -> List[str]:
    """
    Find concepts that are 4-10 hops away from source in Neo4j.

    Args:
        neo4j_session: Neo4j session
        source_id: Source concept cpe_id
        min_hops: Minimum hop distance (default 4)
        max_hops: Maximum hop distance (default 10)
        max_candidates: Max candidates to return

    Returns:
        List of candidate cpe_ids at distance [min_hops, max_hops]
    """
    # Query for paths of length min_hops to max_hops
    query = f"""
        MATCH path = (start:Concept {{cpe_id: $source_id}})-[*{min_hops}..{max_hops}]-(candidate:Concept)
        WHERE start <> candidate
        RETURN DISTINCT candidate.cpe_id as cpe_id, length(path) as distance
        ORDER BY distance
        LIMIT {max_candidates}
    """

    result = neo4j_session.run(query, source_id=source_id)
    candidates = [record['cpe_id'] for record in result]

    return candidates


def add_6deg_shortcuts(
    neo4j_uri: str = "bolt://localhost:7687",
    neo4j_user: str = "neo4j",
    neo4j_password: str = "password",
    shortcut_rate: float = 0.01,
    tmd_lane_match: bool = True,
    min_hops: int = 4,
    max_hops: int = 10,
    min_similarity: float = 0.5,
    random_seed: int = 42
) -> Dict[str, any]:
    """
    Add semantic shortcuts to Neo4j graph based on Watts-Strogatz research.

    Algorithm:
    1. Load all concepts with vectors from Postgres
    2. Load all concepts from Neo4j
    3. For each concept with probability = shortcut_rate:
       a. Find candidates 4-10 hops away
       b. Filter by TMD lane if tmd_lane_match=True
       c. Compute cosine similarity for all candidates
       d. Select candidate with highest similarity (if > min_similarity)
       e. Create 6DEG_SHORTCUT edge with confidence score
    4. Create index on 6DEG_SHORTCUT relationships

    Args:
        neo4j_uri: Neo4j connection URI
        neo4j_user: Neo4j username
        neo4j_password: Neo4j password
        shortcut_rate: Probability of adding shortcut (0.005-0.02, default 0.01)
        tmd_lane_match: Only shortcut within same TMD lane
        min_hops: Minimum hop distance for shortcuts (default 4)
        max_hops: Maximum hop distance for shortcuts (default 10)
        min_similarity: Minimum cosine similarity threshold (default 0.5)
        random_seed: Random seed for reproducibility

    Returns:
        Statistics dict with counts and metrics
    """
    random.seed(random_seed)
    np.random.seed(random_seed)

    print("=" * 60)
    print("6-DEGREE SHORTCUTS IMPLEMENTATION")
    print("=" * 60)
    print(f"Configuration:")
    print(f"  Shortcut rate: {shortcut_rate*100:.1f}%")
    print(f"  TMD lane matching: {tmd_lane_match}")
    print(f"  Hop range: {min_hops}-{max_hops}")
    print(f"  Min similarity: {min_similarity}")
    print()

    # Step 1: Load concept vectors from Postgres
    print("Loading concept vectors from Postgres...")
    concepts = load_concept_vectors_from_postgres()
    print(f"  Loaded {len(concepts)} concepts with vectors")
    print()

    # Step 2: Connect to Neo4j
    driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))

    with driver.session() as session:
        # Get all concept IDs from Neo4j
        result = session.run("MATCH (n:Concept) RETURN n.cpe_id as cpe_id")
        neo4j_concept_ids = [record['cpe_id'] for record in result]
        print(f"Neo4j has {len(neo4j_concept_ids)} concepts")

        # Filter to concepts that exist in both Postgres and Neo4j
        valid_concept_ids = [cid for cid in neo4j_concept_ids if cid in concepts]
        print(f"  {len(valid_concept_ids)} concepts have vectors")
        print()

        # Step 3: Add shortcuts
        shortcuts_to_add = []
        shortcuts_attempted = 0
        shortcuts_no_candidates = 0
        shortcuts_low_similarity = 0

        # Randomly select concepts for shortcuts
        num_shortcuts_target = int(len(valid_concept_ids) * shortcut_rate)
        selected_concepts = random.sample(valid_concept_ids, num_shortcuts_target)

        print(f"Attempting to add ~{num_shortcuts_target} shortcuts...")
        print()

        for source_id in tqdm(selected_concepts, desc="Finding shortcuts"):
            shortcuts_attempted += 1
            source_concept = concepts[source_id]

            # Find distant candidates
            candidates = find_distant_candidates(
                session, source_id, min_hops, max_hops, max_candidates=100
            )

            if not candidates:
                shortcuts_no_candidates += 1
                continue

            # Filter candidates by TMD lane if required
            if tmd_lane_match and source_concept.tmd_lane is not None:
                candidates = [
                    cid for cid in candidates
                    if cid in concepts and concepts[cid].tmd_lane == source_concept.tmd_lane
                ]

            if not candidates:
                shortcuts_no_candidates += 1
                continue

            # Compute similarities
            similarities = []
            for candidate_id in candidates:
                if candidate_id not in concepts:
                    continue

                candidate_concept = concepts[candidate_id]
                sim = cosine_similarity(source_concept.vector, candidate_concept.vector)

                if sim >= min_similarity:
                    similarities.append((candidate_id, sim))

            if not similarities:
                shortcuts_low_similarity += 1
                continue

            # Select candidate with highest similarity
            best_candidate_id, best_similarity = max(similarities, key=lambda x: x[1])

            shortcuts_to_add.append({
                'source_id': source_id,
                'target_id': best_candidate_id,
                'confidence': float(best_similarity)
            })

        print()
        print(f"Shortcut candidate analysis:")
        print(f"  Attempted: {shortcuts_attempted}")
        print(f"  No distant candidates: {shortcuts_no_candidates}")
        print(f"  Low similarity: {shortcuts_low_similarity}")
        print(f"  Valid shortcuts to add: {len(shortcuts_to_add)}")
        print()

        # Step 4: Create shortcut relationships in Neo4j
        if shortcuts_to_add:
            print("Creating shortcut relationships in Neo4j...")

            create_query = """
                MATCH (source:Concept {cpe_id: $source_id})
                MATCH (target:Concept {cpe_id: $target_id})
                MERGE (source)-[r:SHORTCUT_6DEG]->(target)
                SET r.confidence = $confidence,
                    r.created_at = timestamp()
            """

            for shortcut in tqdm(shortcuts_to_add, desc="Creating shortcuts"):
                session.run(
                    create_query,
                    source_id=shortcut['source_id'],
                    target_id=shortcut['target_id'],
                    confidence=shortcut['confidence']
                )

            print(f"  Created {len(shortcuts_to_add)} shortcuts")
            print()

            # Step 5: Create index on shortcuts
            print("Creating index on SHORTCUT_6DEG relationships...")
            try:
                session.run("CREATE INDEX FOR ()-[r:SHORTCUT_6DEG]-() ON (r.confidence)")
                print("  Index created successfully")
            except Exception as e:
                print(f"  Index may already exist: {e}")
            print()

    driver.close()

    # Compute statistics
    if shortcuts_to_add:
        confidences = [s['confidence'] for s in shortcuts_to_add]
        stats = {
            'shortcuts_added': len(shortcuts_to_add),
            'shortcuts_attempted': shortcuts_attempted,
            'success_rate': len(shortcuts_to_add) / shortcuts_attempted,
            'avg_confidence': np.mean(confidences),
            'min_confidence': np.min(confidences),
            'max_confidence': np.max(confidences),
            'no_candidates': shortcuts_no_candidates,
            'low_similarity': shortcuts_low_similarity
        }
    else:
        stats = {
            'shortcuts_added': 0,
            'shortcuts_attempted': shortcuts_attempted,
            'success_rate': 0.0,
            'avg_confidence': 0.0,
            'min_confidence': 0.0,
            'max_confidence': 0.0,
            'no_candidates': shortcuts_no_candidates,
            'low_similarity': shortcuts_low_similarity
        }

    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Shortcuts added: {stats['shortcuts_added']}")
    print(f"Success rate: {stats['success_rate']*100:.1f}%")
    print(f"Avg confidence: {stats['avg_confidence']:.4f}")
    print(f"Confidence range: [{stats['min_confidence']:.4f}, {stats['max_confidence']:.4f}]")
    print("=" * 60)

    return stats


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Add 6-degree shortcuts to Neo4j graph")
    parser.add_argument("--shortcut-rate", type=float, default=0.01,
                        help="Shortcut rate (0.005-0.02, default 0.01)")
    parser.add_argument("--no-tmd-match", action="store_true",
                        help="Don't require TMD lane matching")
    parser.add_argument("--min-hops", type=int, default=4,
                        help="Minimum hop distance (default 4)")
    parser.add_argument("--max-hops", type=int, default=10,
                        help="Maximum hop distance (default 10)")
    parser.add_argument("--min-similarity", type=float, default=0.5,
                        help="Minimum cosine similarity (default 0.5)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default 42)")

    args = parser.parse_args()

    stats = add_6deg_shortcuts(
        shortcut_rate=args.shortcut_rate,
        tmd_lane_match=not args.no_tmd_match,
        min_hops=args.min_hops,
        max_hops=args.max_hops,
        min_similarity=args.min_similarity,
        random_seed=args.seed
    )

    # Exit with error if no shortcuts added
    if stats['shortcuts_added'] == 0:
        print("ERROR: No shortcuts were added!")
        sys.exit(1)

    sys.exit(0)
