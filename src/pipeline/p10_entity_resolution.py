"""P10 — Entity resolution and cross-document linking for two-phase graph building.

Phase 1: Extract entities from individual documents (existing pipeline)
Phase 2: Resolve entities across documents and create cross-document relationships
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Any, List, Set, Tuple, Optional
from pathlib import Path

import numpy as np
from ..integrations import Triple
from ..integrations.lightrag import LightRAGGraphBuilderAdapter
from ..vectorizer import EmbeddingBackend


@dataclass
class Entity:
    """Represents an extracted entity with metadata."""
    text: str
    type: str  # person, location, organization, concept, etc.
    cpe_id: str
    confidence: float = 0.8
    aliases: Set[str] = None

    def __post_init__(self):
        if self.aliases is None:
            self.aliases = set()


@dataclass
class EntityMatch:
    """Represents a potential match between two entities."""
    entity1: Entity
    entity2: Entity
    similarity: float
    match_type: str  # exact, fuzzy, semantic, alias


class EntityResolver:
    """Resolves entities across documents and creates cross-document relationships."""

    def __init__(self, embedding_backend: Optional[EmbeddingBackend] = None):
        self.embedding_backend = embedding_backend or EmbeddingBackend()
        self.entities: Dict[str, Entity] = {}
        self.entity_embeddings: Dict[str, np.ndarray] = {}
        self.entity_clusters: Dict[str, Set[str]] = {}  # canonical_id -> {entity_ids}

    def extract_entities_from_relations(self, cpe_records: List[Dict[str, Any]]) -> None:
        """Phase 1: Extract all entities from existing relations in CPE records."""

        for record in cpe_records:
            cpe_id = record["cpe_id"]
            relations = record.get("relations_text", [])

            if not relations:
                continue

            for relation in relations:
                if not isinstance(relation, dict):
                    continue

                # Extract subject and object as entities
                subj = relation.get("subj", "").strip()
                obj = relation.get("obj", "").strip()

                if subj:
                    entity_id = f"{cpe_id}:subj:{subj}"
                    if entity_id not in self.entities:
                        self.entities[entity_id] = Entity(
                            text=subj,
                            type=self._classify_entity_type(subj),
                            cpe_id=cpe_id,
                            confidence=relation.get("confidence", 0.8)
                        )

                if obj:
                    entity_id = f"{cpe_id}:obj:{obj}"
                    if entity_id not in self.entities:
                        self.entities[entity_id] = Entity(
                            text=obj,
                            type=self._classify_entity_type(obj),
                            cpe_id=cpe_id,
                            confidence=relation.get("confidence", 0.8)
                        )

        print(f"[EntityResolver] Extracted {len(self.entities)} entities from {len(cpe_records)} records")

    def _classify_entity_type(self, entity_text: str) -> str:
        """Classify entity type based on text patterns and heuristics."""
        text = entity_text.lower()

        # Person indicators
        if any(pattern in text for pattern in ["lovelace", "singer", "artist", "programmer", "writer"]):
            return "person"

        # Location indicators
        if any(pattern in text for pattern in ["tower", "mars", "paris", "france", "city", "mountain"]):
            return "location"

        # Organization indicators
        if any(pattern in text for pattern in ["records", "label", "company", "band", "group"]):
            return "organization"

        # Temporal indicators
        if re.match(r"^\d{4}$", text) or "year" in text or "century" in text:
            return "temporal"

        # Process/concept indicators
        if any(pattern in text for pattern in ["photosynthesis", "energy", "programming", "analysis"]):
            return "process"

        # Creative work indicators
        if any(pattern in text for pattern in ["album", "song", "book", "engine", "theorem"]):
            return "creative_work"

        return "concept"  # Default fallback

    def compute_entity_embeddings(self) -> None:
        """Compute embeddings for all entities for semantic similarity."""
        texts = [entity.text for entity in self.entities.values()]
        entity_ids = list(self.entities.keys())

        if not texts:
            return

        try:
            embeddings = self.embedding_backend.encode(texts)
            for i, entity_id in enumerate(entity_ids):
                self.entity_embeddings[entity_id] = embeddings[i]
            print(f"[EntityResolver] Computed embeddings for {len(embeddings)} entities")
        except Exception as e:
            print(f"[EntityResolver] Embedding computation failed: {e}")

    def find_entity_matches(self, similarity_threshold: float = 0.85) -> List[EntityMatch]:
        """Phase 2: Find potential matches between entities across documents."""
        matches = []
        entity_ids = list(self.entities.keys())

        for i, id1 in enumerate(entity_ids):
            entity1 = self.entities[id1]

            for id2 in entity_ids[i+1:]:
                entity2 = self.entities[id2]

                # Skip entities from the same document
                if entity1.cpe_id == entity2.cpe_id:
                    continue

                # Only match entities of the same type
                if entity1.type != entity2.type:
                    continue

                match = self._compute_entity_match(entity1, entity2, id1, id2)
                if match and match.similarity >= similarity_threshold:
                    matches.append(match)

        print(f"[EntityResolver] Found {len(matches)} potential entity matches")
        return matches

    def _compute_entity_match(self, entity1: Entity, entity2: Entity, id1: str, id2: str) -> Optional[EntityMatch]:
        """Compute similarity between two entities using multiple methods."""

        # Exact match
        if entity1.text.lower() == entity2.text.lower():
            return EntityMatch(entity1, entity2, 1.0, "exact")

        # Fuzzy string match
        fuzzy_sim = self._fuzzy_similarity(entity1.text, entity2.text)
        if fuzzy_sim >= 0.9:
            return EntityMatch(entity1, entity2, fuzzy_sim, "fuzzy")

        # Semantic similarity via embeddings
        if id1 in self.entity_embeddings and id2 in self.entity_embeddings:
            vec1 = self.entity_embeddings[id1]
            vec2 = self.entity_embeddings[id2]
            cos_sim = float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))

            if cos_sim >= 0.8:
                return EntityMatch(entity1, entity2, cos_sim, "semantic")

        return None

    def _fuzzy_similarity(self, text1: str, text2: str) -> float:
        """Compute fuzzy string similarity using simple character overlap."""
        s1, s2 = set(text1.lower()), set(text2.lower())
        intersection = len(s1 & s2)
        union = len(s1 | s2)
        return intersection / union if union > 0 else 0.0

    def create_entity_clusters(self, matches: List[EntityMatch]) -> None:
        """Create clusters of equivalent entities based on matches."""
        # Build graph of entity connections
        connections = defaultdict(set)
        for match in matches:
            id1 = f"{match.entity1.cpe_id}:subj:{match.entity1.text}"
            id2 = f"{match.entity2.cpe_id}:subj:{match.entity2.text}"
            connections[id1].add(id2)
            connections[id2].add(id1)

        # Find connected components (clusters)
        visited = set()
        cluster_id = 0

        for entity_id in self.entities.keys():
            if entity_id in visited:
                continue

            # BFS to find all connected entities
            cluster = set()
            queue = [entity_id]

            while queue:
                current = queue.pop(0)
                if current in visited:
                    continue

                visited.add(current)
                cluster.add(current)

                for neighbor in connections[current]:
                    if neighbor not in visited:
                        queue.append(neighbor)

            if cluster:
                canonical_id = f"cluster_{cluster_id}"
                self.entity_clusters[canonical_id] = cluster
                cluster_id += 1

        print(f"[EntityResolver] Created {len(self.entity_clusters)} entity clusters")

    def generate_cross_document_relationships(self,
                                            graph_adapter: LightRAGGraphBuilderAdapter) -> List[Triple]:
        """Generate relationships between entities across documents."""
        triples = []

        for canonical_id, entity_ids in self.entity_clusters.items():
            if len(entity_ids) < 2:
                continue  # Need at least 2 entities to create relationships

            entity_list = [self.entities[eid] for eid in entity_ids if eid in self.entities]

            # Create relationships between all pairs in the cluster
            for i, entity1 in enumerate(entity_list):
                for entity2 in entity_list[i+1:]:
                    # Determine relationship type based on entity types
                    rel_type = self._determine_relationship_type(entity1, entity2)
                    confidence = min(entity1.confidence, entity2.confidence) * 0.9  # Slight penalty for cross-doc

                    triples.append(Triple(
                        src_cpe_id=entity1.cpe_id,
                        dst_cpe_id=entity2.cpe_id,
                        type=rel_type,
                        confidence=confidence,
                        properties={
                            "cross_document": True,
                            "entity1_text": entity1.text,
                            "entity2_text": entity2.text,
                            "entity_type": entity1.type,
                            "cluster_id": canonical_id
                        }
                    ))

        print(f"[EntityResolver] Generated {len(triples)} cross-document relationships")
        return triples

    def _determine_relationship_type(self, entity1: Entity, entity2: Entity) -> str:
        """Determine the type of relationship between two entities."""
        entity_type = entity1.type

        if entity_type == "person":
            return "same_person"
        elif entity_type == "location":
            return "same_location"
        elif entity_type == "organization":
            return "same_organization"
        elif entity_type == "creative_work":
            return "related_work"
        elif entity_type == "temporal":
            return "same_timeframe"
        elif entity_type == "process":
            return "related_process"
        else:
            return "same_concept"

    def save_entity_analysis(self, output_path: str) -> None:
        """Save entity analysis results for debugging and verification."""
        analysis = {
            "total_entities": len(self.entities),
            "entity_types": {},
            "clusters": {},
            "entities_by_type": defaultdict(list)
        }

        # Count entities by type
        for entity in self.entities.values():
            entity_type = entity.type
            if entity_type not in analysis["entity_types"]:
                analysis["entity_types"][entity_type] = 0
            analysis["entity_types"][entity_type] += 1
            analysis["entities_by_type"][entity_type].append({
                "text": entity.text,
                "cpe_id": entity.cpe_id,
                "confidence": entity.confidence
            })

        # Add cluster information
        for cluster_id, entity_ids in self.entity_clusters.items():
            cluster_entities = [self.entities[eid].text for eid in entity_ids if eid in self.entities]
            analysis["clusters"][cluster_id] = cluster_entities

        with open(output_path, 'w') as f:
            json.dump(analysis, f, indent=2)

        print(f"[EntityResolver] Saved entity analysis to {output_path}")


def run_two_phase_graph_extraction(
    cpe_records: List[Dict[str, Any]],
    graph_adapter: LightRAGGraphBuilderAdapter,
    neo_db,
    output_dir: str = "artifacts"
) -> Dict[str, Any]:
    """Run complete two-phase graph extraction process."""

    # Initialize entity resolver
    resolver = EntityResolver()

    # Phase 1: Extract entities from existing relations
    print("=== Phase 1: Entity Extraction ===")
    resolver.extract_entities_from_relations(cpe_records)
    resolver.compute_entity_embeddings()

    # Phase 2: Entity resolution and cross-document linking
    print("=== Phase 2: Entity Resolution ===")
    matches = resolver.find_entity_matches(similarity_threshold=0.85)
    resolver.create_entity_clusters(matches)

    # Generate cross-document relationships
    print("=== Phase 2: Cross-Document Linking ===")
    cross_doc_triples = resolver.generate_cross_document_relationships(graph_adapter)

    # Write cross-document relationships to Neo4j
    if neo_db and hasattr(neo_db, 'insert_relation_triple'):
        for triple in cross_doc_triples:
            try:
                neo_db.insert_relation_triple(triple.src_cpe_id, triple.dst_cpe_id, triple.type)
            except Exception as e:
                print(f"[TwoPhaseGraph] Neo4j insertion failed: {e}")

    # Save analysis results
    Path(output_dir).mkdir(exist_ok=True)
    resolver.save_entity_analysis(f"{output_dir}/entity_analysis.json")

    return {
        "total_entities": len(resolver.entities),
        "entity_clusters": len(resolver.entity_clusters),
        "cross_doc_relationships": len(cross_doc_triples),
        "entity_matches": len(matches),
        "phase1_entities_per_doc": len(resolver.entities) / len(cpe_records) if cpe_records else 0,
        "phase2_cross_links": len(cross_doc_triples)
    }