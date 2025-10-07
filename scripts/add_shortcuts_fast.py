#!/usr/bin/env python3
"""
FAST 6-Degree Shortcuts - Just random connections!

The existing code is doing expensive graph traversal (4-10 hops).
This version just randomly connects 1% of nodes - instant!
"""

import random
from neo4j import GraphDatabase
from tqdm import tqdm


def add_random_shortcuts(
    shortcut_rate: float = 0.01,
    neo4j_uri: str = "bolt://localhost:7687",
    neo4j_user: str = "neo4j",
    neo4j_password: str = "password",
    random_seed: int = 42
):
    """
    Add random shortcuts to Neo4j graph - FAST version.

    Algorithm:
    1. Get all concept IDs
    2. For 1% of concepts, pick a random other concept
    3. Create SHORTCUT_6DEG edge

    No graph traversal, no similarity computation, just random links!
    """
    random.seed(random_seed)

    print("=" * 60)
    print("FAST RANDOM SHORTCUTS")
    print("=" * 60)
    print(f"Shortcut rate: {shortcut_rate*100:.1f}%")
    print()

    driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))

    with driver.session() as session:
        # Get all concept IDs
        print("Loading concept IDs from Neo4j...")
        result = session.run("MATCH (c:Concept) RETURN c.cpe_id as cpe_id")
        all_concepts = [record['cpe_id'] for record in result]
        print(f"  Found {len(all_concepts)} concepts")
        print()

        # Calculate how many shortcuts to add
        num_shortcuts = int(len(all_concepts) * shortcut_rate)
        print(f"Adding {num_shortcuts} random shortcuts...")
        print()

        # Randomly select source concepts
        source_concepts = random.sample(all_concepts, num_shortcuts)

        shortcuts_added = 0
        for source_id in tqdm(source_concepts, desc="Creating shortcuts"):
            # Pick a random target (different from source)
            target_id = random.choice([c for c in all_concepts if c != source_id])

            # Create shortcut edge
            query = """
                MATCH (source:Concept {cpe_id: $source_id})
                MATCH (target:Concept {cpe_id: $target_id})
                MERGE (source)-[r:SHORTCUT_6DEG]->(target)
                SET r.confidence = 0.5,
                    r.created_at = timestamp()
            """

            session.run(query, source_id=source_id, target_id=target_id)
            shortcuts_added += 1

        print()
        print("=" * 60)
        print(f"✅ Added {shortcuts_added} shortcuts in Neo4j")
        print("=" * 60)

        # Verify
        result = session.run("MATCH ()-[r:SHORTCUT_6DEG]->() RETURN count(r) as count")
        total_shortcuts = result.single()['count']
        print(f"Total SHORTCUT_6DEG edges: {total_shortcuts}")

    driver.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Add random shortcuts (FAST version)")
    parser.add_argument("--rate", type=float, default=0.01, help="Shortcut rate (default 0.01)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    add_random_shortcuts(
        shortcut_rate=args.rate,
        random_seed=args.seed
    )
