#!/usr/bin/env python3
"""
Parser for ConceptNet 5.7 dataset.

Format: CSV (gzipped)
Extracts: /r/IsA, /r/PartOf, /r/HasA relationships
Expected chains: ~20K
Quality: 82%

CSV Format:
/c/en/dog	/r/IsA	/c/en/animal	/d/conceptnet/4/en	weight
"""

import logging
import gzip
import csv
from pathlib import Path
from typing import Dict, List, Set
from collections import defaultdict

from base_parser import OntologyParser, OntologyChain

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ConceptNetParser(OntologyParser):
    """Parser for ConceptNet assertions (CSV format)."""

    # Hierarchical relations we care about
    HIERARCHICAL_RELATIONS = {
        '/r/IsA',      # dog IsA animal
        '/r/PartOf',   # wheel PartOf car
        '/r/HasA'      # car HasA wheel (inverse of PartOf)
    }

    def __init__(
        self,
        input_path: Path = None,
        output_path: Path = None,
        min_chain_length: int = 3,
        max_chain_length: int = 20,
        language: str = 'en'  # Focus on English concepts
    ):
        if input_path is None:
            input_path = Path("data/datasets/ontology_datasets/conceptnet/conceptnet-assertions-5.7.0.csv.gz")
        if output_path is None:
            output_path = Path("artifacts/ontology_chains/conceptnet_chains.jsonl")

        super().__init__(
            source_name="conceptnet",
            input_path=input_path,
            output_path=output_path,
            min_chain_length=min_chain_length,
            max_chain_length=max_chain_length
        )

        self.language = language
        self.concepts: Set[str] = set()
        self.parent_child_map: Dict[str, List[str]] = defaultdict(list)

    def parse(self) -> List[OntologyChain]:
        """Parse ConceptNet CSV and extract chains."""
        logger.info("=" * 60)
        logger.info("CONCEPTNET 5.7 PARSER")
        logger.info("=" * 60)
        logger.info(f"Input: {self.input_path}")
        logger.info(f"Output: {self.output_path}")
        logger.info(f"Language filter: {self.language}")

        # Parse CSV
        logger.info("Parsing CSV (this may take 2-3 minutes)...")
        self._parse_csv()

        # Build chains
        logger.info("Building concept chains...")
        self.chains = self.build_chains_from_graph(self.parent_child_map)

        # Filter valid chains
        logger.info("Filtering valid chains...")
        valid_chains = self.filter_valid_chains()

        # Compute stats
        self.compute_stats()
        self.print_stats()

        # Write chains
        self.write_chains(valid_chains)

        logger.info("=" * 60)
        logger.info("✅ CONCEPTNET PARSING COMPLETE")
        logger.info("=" * 60)

        return valid_chains

    def _parse_csv(self):
        """Parse ConceptNet CSV file."""
        line_count = 0
        relation_count = 0

        with gzip.open(self.input_path, 'rt', encoding='utf-8') as f:
            # ConceptNet CSV is tab-separated
            reader = csv.reader(f, delimiter='\t')

            for row in reader:
                line_count += 1

                # Progress update every 1M lines
                if line_count % 1000000 == 0:
                    logger.info(f"  Processed {line_count/1000000:.1f}M lines, found {relation_count} hierarchical relations")

                # CSV format: assertion_uri, relation, start, end, context, weight
                if len(row) < 4:
                    continue

                relation = row[1]
                start_concept = row[2]  # child
                end_concept = row[3]    # parent

                # Filter by relation type
                if relation not in self.HIERARCHICAL_RELATIONS:
                    continue

                # Filter by language
                if not self._is_target_language(start_concept) or not self._is_target_language(end_concept):
                    continue

                # Extract concept labels
                start_label = self._extract_label(start_concept)
                end_label = self._extract_label(end_concept)

                if not start_label or not end_label:
                    continue

                # Add concepts
                self.concepts.add(start_label)
                self.concepts.add(end_label)

                # Build parent→child relationship
                # For IsA and PartOf: end is parent of start
                # For HasA: start is parent of end
                if relation in ['/r/IsA', '/r/PartOf']:
                    self.parent_child_map[end_label].append(start_label)
                elif relation == '/r/HasA':
                    self.parent_child_map[start_label].append(end_label)

                relation_count += 1
                self.stats["total_relations"] += 1

        self.stats["total_concepts"] = len(self.concepts)
        logger.info(f"  Processed {line_count} total lines")
        logger.info(f"  Found {self.stats['total_concepts']} unique concepts")
        logger.info(f"  Found {self.stats['total_relations']} hierarchical relations")

    def _is_target_language(self, concept_uri: str) -> bool:
        """Check if concept is in target language."""
        # Format: /c/en/dog or /c/en/dog/n/animal
        parts = concept_uri.split('/')
        if len(parts) < 4:
            return False
        return parts[2] == self.language

    def _extract_label(self, concept_uri: str) -> str:
        """Extract readable label from ConceptNet URI."""
        # Format: /c/en/dog or /c/en/hot_dog/n
        parts = concept_uri.split('/')
        if len(parts) < 4:
            return ""

        # Get concept text (replace underscores with spaces)
        label = parts[3].replace('_', ' ')

        # Clean up
        label = label.strip()

        return label


if __name__ == "__main__":
    parser = ConceptNetParser()
    chains = parser.parse()

    logger.info(f"\n✅ Successfully extracted {len(chains)} valid chains from ConceptNet")
    logger.info(f"📁 Output: {parser.output_path}")
