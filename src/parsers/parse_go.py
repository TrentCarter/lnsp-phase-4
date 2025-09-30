#!/usr/bin/env python3
"""
Parser for Gene Ontology (GO) dataset.

Format: OBO (Open Biomedical Ontologies)
Extracts: is_a relationships
Expected chains: ~40K
Quality: 94%
"""

import logging
import re
from pathlib import Path
from typing import Dict, List, Optional
from collections import defaultdict

from base_parser import OntologyParser, OntologyChain

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class GOParser(OntologyParser):
    """Parser for Gene Ontology (OBO format)."""

    def __init__(
        self,
        input_path: Path = None,
        output_path: Path = None,
        min_chain_length: int = 3,
        max_chain_length: int = 20
    ):
        if input_path is None:
            input_path = Path("data/datasets/ontology_datasets/go/go.obo")
        if output_path is None:
            output_path = Path("artifacts/ontology_chains/go_chains.jsonl")

        super().__init__(
            source_name="go",
            input_path=input_path,
            output_path=output_path,
            min_chain_length=min_chain_length,
            max_chain_length=max_chain_length
        )

        self.concepts: Dict[str, str] = {}  # ID → name
        self.parent_child_map: Dict[str, List[str]] = defaultdict(list)

    def parse(self) -> List[OntologyChain]:
        """Parse GO OBO file and extract chains."""
        logger.info("=" * 60)
        logger.info("GENE ONTOLOGY (GO) PARSER")
        logger.info("=" * 60)
        logger.info(f"Input: {self.input_path}")
        logger.info(f"Output: {self.output_path}")

        # Parse OBO file
        logger.info("Parsing OBO format...")
        self._parse_obo_file()

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
        logger.info("✅ GO PARSING COMPLETE")
        logger.info("=" * 60)

        return valid_chains

    def _parse_obo_file(self):
        """Parse OBO format file."""
        with open(self.input_path, 'r', encoding='utf-8') as f:
            current_term = {}
            in_term = False

            for line in f:
                line = line.strip()

                # Start of term
                if line == '[Term]':
                    # Process previous term if exists
                    if current_term:
                        self._process_term(current_term)
                    current_term = {}
                    in_term = True
                    continue

                # End of term section
                if line.startswith('[') and line != '[Term]':
                    if current_term:
                        self._process_term(current_term)
                    current_term = {}
                    in_term = False
                    continue

                # Skip non-term sections
                if not in_term or not line or line.startswith('!'):
                    continue

                # Parse term fields
                if ':' in line:
                    key, value = line.split(':', 1)
                    key = key.strip()
                    value = value.strip()

                    if key == 'id':
                        current_term['id'] = value
                    elif key == 'name':
                        current_term['name'] = value
                    elif key == 'is_a':
                        # is_a: GO:0000001 ! parent term name
                        parent_id = value.split('!')[0].strip()
                        if 'is_a' not in current_term:
                            current_term['is_a'] = []
                        current_term['is_a'].append(parent_id)
                    elif key == 'is_obsolete':
                        current_term['is_obsolete'] = True

            # Process last term
            if current_term:
                self._process_term(current_term)

        logger.info(f"  Found {len(self.concepts)} terms")
        logger.info(f"  Found {self.stats['total_relations']} is_a relations")

    def _process_term(self, term: Dict):
        """Process a single OBO term."""
        # Skip if no ID or name
        if 'id' not in term or 'name' not in term:
            return

        # Skip obsolete terms
        if term.get('is_obsolete', False):
            return

        term_id = term['id']
        term_name = term['name']

        # Store concept
        self.concepts[term_id] = term_name
        self.stats["total_concepts"] += 1

        # Process is_a relationships
        if 'is_a' in term:
            for parent_id in term['is_a']:
                # We'll resolve parent names later
                # For now, store ID relationships
                if 'parent_ids' not in term:
                    term['parent_ids'] = []
                term['parent_ids'].append(parent_id)
                self.stats["total_relations"] += 1

    def _build_parent_child_map(self):
        """Build parent→child map with resolved names."""
        # Re-parse to build map with names
        with open(self.input_path, 'r', encoding='utf-8') as f:
            current_term = {}
            in_term = False

            for line in f:
                line = line.strip()

                if line == '[Term]':
                    if current_term and 'id' in current_term and 'is_a' in current_term:
                        self._add_relationships(current_term)
                    current_term = {}
                    in_term = True
                    continue

                if line.startswith('[') and line != '[Term]':
                    if current_term and 'id' in current_term and 'is_a' in current_term:
                        self._add_relationships(current_term)
                    current_term = {}
                    in_term = False
                    continue

                if not in_term or not line or line.startswith('!'):
                    continue

                if ':' in line:
                    key, value = line.split(':', 1)
                    key = key.strip()
                    value = value.strip()

                    if key == 'id':
                        current_term['id'] = value
                    elif key == 'name':
                        current_term['name'] = value
                    elif key == 'is_a':
                        parent_id = value.split('!')[0].strip()
                        if 'is_a' not in current_term:
                            current_term['is_a'] = []
                        current_term['is_a'].append(parent_id)

            if current_term and 'id' in current_term and 'is_a' in current_term:
                self._add_relationships(current_term)

    def _add_relationships(self, term: Dict):
        """Add parent→child relationships to map."""
        child_id = term['id']
        child_name = self.concepts.get(child_id)

        if not child_name:
            return

        for parent_id in term.get('is_a', []):
            parent_name = self.concepts.get(parent_id)
            if parent_name:
                self.parent_child_map[parent_name].append(child_name)

    def parse(self) -> List[OntologyChain]:
        """Parse GO OBO file and extract chains."""
        logger.info("=" * 60)
        logger.info("GENE ONTOLOGY (GO) PARSER")
        logger.info("=" * 60)
        logger.info(f"Input: {self.input_path}")
        logger.info(f"Output: {self.output_path}")

        # Parse OBO file (first pass: collect terms)
        logger.info("Parsing OBO format (pass 1: collecting terms)...")
        self._parse_obo_file()

        # Second pass: build relationships
        logger.info("Building relationships (pass 2)...")
        self._build_parent_child_map()

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
        logger.info("✅ GO PARSING COMPLETE")
        logger.info("=" * 60)

        return valid_chains


if __name__ == "__main__":
    parser = GOParser()
    chains = parser.parse()

    logger.info(f"\n✅ Successfully extracted {len(chains)} valid chains from GO")
    logger.info(f"📁 Output: {parser.output_path}")
