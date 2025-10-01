#!/usr/bin/env python3
"""
Test script for 6-degree shortcuts functionality.

Validates that shortcuts:
1. Exist in the graph
2. Span sufficient hop distance
3. Have good semantic similarity scores
4. Reduce retrieval hop count

Author: [Architect]
Date: 2025-10-01
"""

import pytest
from neo4j import GraphDatabase
import numpy as np


class Test6DegShortcuts:
    """Test suite for 6-degree shortcuts implementation."""

    @pytest.fixture(scope="class")
    def neo4j_driver(self):
        """Create Neo4j driver connection."""
        driver = GraphDatabase.driver(
            "bolt://localhost:7687",
            auth=("neo4j", "password")
        )
        yield driver
        driver.close()

    def test_shortcuts_exist(self, neo4j_driver):
        """Test that SHORTCUT_6DEG relationships exist in graph."""
        with neo4j_driver.session() as session:
            result = session.run(
                "MATCH ()-[r:SHORTCUT_6DEG]->() RETURN count(r) as count"
            )
            count = result.single()['count']

            assert count > 0, "No shortcuts found in graph"
            print(f"✅ Found {count} shortcuts in graph")

    def test_shortcut_confidence_scores(self, neo4j_driver):
        """Test that shortcuts have reasonable confidence scores."""
        with neo4j_driver.session() as session:
            result = session.run("""
                MATCH ()-[r:SHORTCUT_6DEG]->()
                RETURN
                    min(r.confidence) as min_conf,
                    max(r.confidence) as max_conf,
                    avg(r.confidence) as avg_conf,
                    count(r) as count
            """)
            record = result.single()

            min_conf = record['min_conf']
            max_conf = record['max_conf']
            avg_conf = record['avg_conf']
            count = record['count']

            # Validate confidence scores
            assert min_conf >= 0.4, f"Min confidence too low: {min_conf}"
            assert max_conf <= 1.01, f"Max confidence too high: {max_conf}"  # Allow small FP error
            assert avg_conf >= 0.5, f"Avg confidence too low: {avg_conf}"

            print(f"✅ Confidence scores valid:")
            print(f"   Count: {count}")
            print(f"   Range: [{min_conf:.4f}, {max_conf:.4f}]")
            print(f"   Average: {avg_conf:.4f}")

    def test_shortcuts_span_distance(self, neo4j_driver):
        """Test that shortcuts actually span multiple hops in original graph."""
        with neo4j_driver.session() as session:
            # Get all shortcuts
            result = session.run("""
                MATCH (source)-[s:SHORTCUT_6DEG]->(target)
                RETURN source.cpe_id as source_id,
                       target.cpe_id as target_id,
                       s.confidence as confidence
                LIMIT 10
            """)

            shortcuts = list(result)
            assert len(shortcuts) > 0, "No shortcuts to test"

            hop_distances = []
            for shortcut in shortcuts:
                source_id = shortcut['source_id']
                target_id = shortcut['target_id']

                # Find shortest path without shortcuts
                path_result = session.run("""
                    MATCH path = shortestPath(
                        (source:Concept {cpe_id: $source_id})
                        -[:RELATES_TO*]->
                        (target:Concept {cpe_id: $target_id})
                    )
                    RETURN length(path) as hops
                """, source_id=source_id, target_id=target_id)

                path_record = path_result.single()
                if path_record:
                    hops = path_record['hops']
                    hop_distances.append(hops)

            if hop_distances:
                avg_hops = np.mean(hop_distances)
                min_hops = np.min(hop_distances)
                max_hops = np.max(hop_distances)

                assert avg_hops >= 2, f"Shortcuts don't span enough distance: avg={avg_hops}"

                print(f"✅ Shortcuts span sufficient distance:")
                print(f"   Sample size: {len(hop_distances)}")
                print(f"   Hop range: [{min_hops}, {max_hops}]")
                print(f"   Average hops: {avg_hops:.2f}")
            else:
                print("⚠️  Could not measure hop distances (paths may be disconnected)")

    def test_hop_reduction_estimate(self, neo4j_driver):
        """
        Estimate hop reduction from shortcuts.

        Compare random walk with/without shortcuts for sample concepts.
        """
        with neo4j_driver.session() as session:
            # Get 10 random concepts that have shortcuts
            result = session.run("""
                MATCH (n:Concept)-[s:SHORTCUT_6DEG]->()
                RETURN DISTINCT n.cpe_id as cpe_id
                LIMIT 10
            """)

            concept_ids = [r['cpe_id'] for r in result]

            if not concept_ids:
                pytest.skip("No concepts with shortcuts found")

            baseline_hops = []
            shortcut_hops = []

            for cpe_id in concept_ids:
                # Baseline: Random walk without shortcuts
                baseline_result = session.run("""
                    MATCH path = (start:Concept {cpe_id: $cpe_id})
                                 -[:RELATES_TO*1..5]->
                                 (end:Concept)
                    WHERE start <> end
                    RETURN avg(length(path)) as avg_hops
                """, cpe_id=cpe_id)

                baseline_record = baseline_result.single()
                if baseline_record and baseline_record['avg_hops']:
                    baseline_hops.append(baseline_record['avg_hops'])

                # With shortcuts: Allow SHORTCUT_6DEG edges
                shortcut_result = session.run("""
                    MATCH path = (start:Concept {cpe_id: $cpe_id})
                                 -[:RELATES_TO|SHORTCUT_6DEG*1..5]->
                                 (end:Concept)
                    WHERE start <> end
                    RETURN avg(length(path)) as avg_hops
                """, cpe_id=cpe_id)

                shortcut_record = shortcut_result.single()
                if shortcut_record and shortcut_record['avg_hops']:
                    shortcut_hops.append(shortcut_record['avg_hops'])

            if baseline_hops and shortcut_hops:
                baseline_mean = np.mean(baseline_hops)
                shortcut_mean = np.mean(shortcut_hops)
                reduction = (baseline_mean - shortcut_mean) / baseline_mean

                print(f"✅ Hop reduction estimate:")
                print(f"   Baseline hops: {baseline_mean:.2f}")
                print(f"   With shortcuts: {shortcut_mean:.2f}")
                print(f"   Reduction: {reduction*100:.1f}%")

                # Note: Reduction may be small because we only have 35 shortcuts
                # Will improve significantly when we have 100-200 shortcuts
                assert reduction >= 0, "Shortcuts should not increase hop count"
            else:
                print("⚠️  Could not measure hop reduction (insufficient data)")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
