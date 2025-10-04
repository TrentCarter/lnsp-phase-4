#!/usr/bin/env python3
"""
Fix Neo4j Graph: Convert Concept→Entity edges to Concept→Concept edges.

Root Cause:
    - Current: (Concept)-[:RELATES_TO]->(Entity {name: cpe_id, text: NULL})
    - Desired: (Concept)-[:RELATES_TO]->(Concept {cpe_id: ..., text: "..."})

Strategy:
    1. Find all Entity nodes that have CPE ID-like names
    2. For each Entity, find matching Concept with same cpe_id
    3. Rewrite Concept→Entity edges as Concept→Concept edges
    4. Delete orphaned Entity nodes

Usage:
    python scripts/fix_neo4j_concept_edges.py [--dry-run] [--batch-size 1000]
"""

import argparse
import sys
from typing import List, Dict, Tuple

try:
    from neo4j import GraphDatabase
except ImportError:
    print("ERROR: neo4j driver not installed")
    print("Install: pip install neo4j")
    sys.exit(1)


class Neo4jFixer:
    def __init__(self, uri: str = "bolt://localhost:7687", user: str = "neo4j", password: str = "password"):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def get_stats(self) -> Dict[str, int]:
        """Get current graph statistics."""
        with self.driver.session() as session:
            result = session.run("""
                OPTIONAL MATCH (c:Concept)
                WITH count(c) as concepts
                OPTIONAL MATCH (e:Entity)
                WITH concepts, count(e) as entities
                OPTIONAL MATCH (c1:Concept)-[r1:RELATES_TO]->(c2:Concept)
                WITH concepts, entities, count(r1) as c2c_edges
                OPTIONAL MATCH (c:Concept)-[r2:RELATES_TO]->(e:Entity)
                RETURN concepts, entities, c2c_edges, count(r2) as c2e_edges
            """).single()

            if result:
                return {
                    "concepts": result["concepts"],
                    "entities": result["entities"],
                    "concept_to_concept": result["c2c_edges"],
                    "concept_to_entity": result["c2e_edges"]
                }
            return {"concepts": 0, "entities": 0, "concept_to_concept": 0, "concept_to_entity": 0}

    def find_fixable_entities(self, limit: int = 1000) -> List[Dict]:
        """Find Entity nodes that should be Concepts (have matching text)."""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (e:Entity)
                WHERE e.name IS NOT NULL
                MATCH (c:Concept {text: e.name})
                RETURN e.name as entity_name, c.cpe_id as concept_cpe_id, c.text as concept_text
                LIMIT $limit
            """, limit=limit)

            return [dict(record) for record in result]

    def fix_edges_batch(self, batch_size: int = 100, dry_run: bool = False) -> Tuple[int, int]:
        """Fix Concept→Entity edges by converting to Concept→Concept.

        Returns:
            (fixed_edges, deleted_entities)
        """
        with self.driver.session() as session:
            if dry_run:
                print(f"\n[DRY RUN] Would fix edges in batches of {batch_size}")

                # Count how many would be fixed
                result = session.run("""
                    MATCH (src:Concept)-[r:RELATES_TO]->(e:Entity)
                    WHERE e.name IS NOT NULL
                    MATCH (dst:Concept {text: e.name})
                    RETURN count(r) as fixable_edges
                """).single()

                return (result["fixable_edges"] if result else 0, 0)

            # Actual fix: Rewrite edges and delete orphaned Entities
            fixed_edges = 0
            deleted_entities = 0

            while True:
                # Process one batch
                result = session.run("""
                    // Find Concept→Entity edges where Entity.name matches a Concept.text
                    MATCH (src:Concept)-[old_r:RELATES_TO]->(e:Entity)
                    WHERE e.name IS NOT NULL
                    WITH src, old_r, e LIMIT $batch_size

                    // Find the target Concept by text
                    MATCH (dst:Concept {text: e.name})

                    // Create new Concept→Concept edge (copy properties)
                    MERGE (src)-[new_r:RELATES_TO]->(dst)
                    SET new_r.type = old_r.type,
                        new_r.confidence = old_r.confidence,
                        new_r.weight = old_r.weight

                    // Delete old edge
                    DELETE old_r

                    RETURN count(new_r) as edges_fixed
                """, batch_size=batch_size).single()

                batch_count = result["edges_fixed"] if result else 0
                fixed_edges += batch_count

                print(f"  Fixed {batch_count} edges (total: {fixed_edges})")

                if batch_count == 0:
                    break  # No more edges to fix

            # Delete orphaned Entity nodes (no incoming or outgoing edges)
            result = session.run("""
                MATCH (e:Entity)
                WHERE NOT (e)-[]-()
                WITH e LIMIT 10000
                DELETE e
                RETURN count(e) as deleted
            """).single()

            deleted_entities = result["deleted"] if result else 0

            return (fixed_edges, deleted_entities)

    def create_concept_to_concept_index(self):
        """Create index on Concept.text for faster lookups (if not exists)."""
        with self.driver.session() as session:
            try:
                session.run("CREATE INDEX concept_text_idx IF NOT EXISTS FOR (c:Concept) ON (c.text)")
                print("[INDEX] Created/verified index on Concept.text")
            except Exception as e:
                print(f"[INDEX] Warning: {e}")


def main():
    parser = argparse.ArgumentParser(description="Fix Neo4j Concept→Entity edges to Concept→Concept")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be fixed without making changes")
    parser.add_argument("--batch-size", type=int, default=1000, help="Batch size for edge fixes (default: 1000)")
    parser.add_argument("--uri", default="bolt://localhost:7687", help="Neo4j URI")
    parser.add_argument("--user", default="neo4j", help="Neo4j username")
    parser.add_argument("--password", default="password", help="Neo4j password")
    args = parser.parse_args()

    print("=" * 70)
    print("Neo4j Graph Fixer: Concept→Entity → Concept→Concept")
    print("=" * 70)

    fixer = Neo4jFixer(uri=args.uri, user=args.user, password=args.password)

    try:
        # Step 1: Get initial stats
        print("\n[STEP 1] Current Graph Statistics")
        print("-" * 70)
        stats_before = fixer.get_stats()
        print(f"  Concepts:               {stats_before['concepts']:,}")
        print(f"  Entities:               {stats_before['entities']:,}")
        print(f"  Concept→Concept edges:  {stats_before['concept_to_concept']:,}")
        print(f"  Concept→Entity edges:   {stats_before['concept_to_entity']:,}")

        if stats_before['concept_to_entity'] == 0:
            print("\n✅ No Concept→Entity edges found. Graph is already correct!")
            return 0

        # Step 2: Sample fixable entities
        print("\n[STEP 2] Finding Fixable Entities")
        print("-" * 70)
        fixable = fixer.find_fixable_entities(limit=10)
        print(f"  Found {len(fixable)} sample fixable entities:")
        for entity in fixable[:5]:
            print(f"    - Entity '{entity['entity_name'][:50]}...' → Concept '{entity['concept_text'][:50]}...'")

        if len(fixable) >= 10:
            print(f"    ... and {len(fixable) - 5} more")

        # Step 3: Create index for performance
        print("\n[STEP 3] Optimizing for Performance")
        print("-" * 70)
        fixer.create_concept_to_concept_index()

        # Step 4: Fix edges
        print("\n[STEP 4] Fixing Edges")
        print("-" * 70)

        if args.dry_run:
            print("  [DRY RUN MODE] - No changes will be made")

        fixed_edges, deleted_entities = fixer.fix_edges_batch(
            batch_size=args.batch_size,
            dry_run=args.dry_run
        )

        if args.dry_run:
            print(f"\n  Would fix: {fixed_edges:,} edges")
            print(f"  Would delete: {deleted_entities:,} orphaned entities")
            print("\n  Run without --dry-run to apply changes")
            return 0

        print(f"\n  ✅ Fixed {fixed_edges:,} edges")
        print(f"  ✅ Deleted {deleted_entities:,} orphaned Entity nodes")

        # Step 5: Verify results
        print("\n[STEP 5] Final Graph Statistics")
        print("-" * 70)
        stats_after = fixer.get_stats()
        print(f"  Concepts:               {stats_after['concepts']:,}")
        print(f"  Entities:               {stats_after['entities']:,}")
        print(f"  Concept→Concept edges:  {stats_after['concept_to_concept']:,}")
        print(f"  Concept→Entity edges:   {stats_after['concept_to_entity']:,}")

        # Changes summary
        print("\n[SUMMARY] Changes Made")
        print("-" * 70)
        print(f"  Concept→Concept edges:  {stats_before['concept_to_concept']:,} → {stats_after['concept_to_concept']:,} "
              f"(+{stats_after['concept_to_concept'] - stats_before['concept_to_concept']:,})")
        print(f"  Concept→Entity edges:   {stats_before['concept_to_entity']:,} → {stats_after['concept_to_entity']:,} "
              f"(-{stats_before['concept_to_entity'] - stats_after['concept_to_entity']:,})")
        print(f"  Entity nodes deleted:   {stats_before['entities'] - stats_after['entities']:,}")

        if stats_after['concept_to_entity'] == 0:
            print("\n🎉 SUCCESS! All Concept→Entity edges converted to Concept→Concept")
            print("   GraphRAG should now work correctly!")
        else:
            print(f"\n⚠️  Warning: Still {stats_after['concept_to_entity']:,} Concept→Entity edges remaining")
            print("   These may be legitimate (non-Concept entities) or require manual review")

        print("\n" + "=" * 70)
        print("Next Steps:")
        print("  1. Verify graph: cypher-shell -u neo4j -p password")
        print("     MATCH (c1:Concept)-[r]->(c2:Concept) RETURN count(r)")
        print("  2. Re-run GraphRAG benchmark to test improvement")
        print("  3. Expected: P@1 ≈ 0.60-0.65 (up from 0.075)")
        print("=" * 70)

        return 0

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1

    finally:
        fixer.close()


if __name__ == "__main__":
    sys.exit(main())
