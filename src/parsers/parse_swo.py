#!/usr/bin/env python3
"""
Parser for Software Ontology (SWO) dataset.

Format: OWL/XML
Extracts: rdfs:subClassOf relationships
Expected chains: ~15K
Quality: 95%
"""

import logging
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Set
from collections import defaultdict

from base_parser import OntologyParser, OntologyChain

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SWOParser(OntologyParser):
    """Parser for Software Ontology (OWL format)."""

    # OWL/RDF namespace definitions
    NAMESPACES = {
        'owl': 'http://www.w3.org/2002/07/owl#',
        'rdf': 'http://www.w3.org/1999/02/22-rdf-syntax-ns#',
        'rdfs': 'http://www.w3.org/2000/01/rdf-schema#',
        'xml': 'http://www.w3.org/XML/1998/namespace'
    }

    def __init__(
        self,
        input_path: Path = None,
        output_path: Path = None,
        min_chain_length: int = 3,
        max_chain_length: int = 20
    ):
        if input_path is None:
            input_path = Path("data/datasets/ontology_datasets/swo/swo.owl")
        if output_path is None:
            output_path = Path("artifacts/ontology_chains/swo_chains.jsonl")

        super().__init__(
            source_name="swo",
            input_path=input_path,
            output_path=output_path,
            min_chain_length=min_chain_length,
            max_chain_length=max_chain_length
        )

        self.concept_labels: Dict[str, str] = {}  # URI → label
        self.parent_child_map: Dict[str, List[str]] = defaultdict(list)

    def parse(self) -> List[OntologyChain]:
        """Parse SWO OWL file and extract chains."""
        logger.info("=" * 60)
        logger.info("SOFTWARE ONTOLOGY (SWO) PARSER")
        logger.info("=" * 60)
        logger.info(f"Input: {self.input_path}")
        logger.info(f"Output: {self.output_path}")

        # Parse OWL file
        logger.info("Parsing OWL/XML...")
        tree = ET.parse(self.input_path)
        root = tree.getroot()

        # Extract classes and labels
        logger.info("Extracting classes and labels...")
        self._extract_classes(root)

        # Extract subClassOf relationships
        logger.info("Extracting subClassOf relationships...")
        self._extract_subclass_relations(root)

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
        logger.info("✅ SWO PARSING COMPLETE")
        logger.info("=" * 60)

        return valid_chains

    def _extract_classes(self, root: ET.Element):
        """Extract all OWL classes and their labels."""
        for cls in root.findall('.//owl:Class', self.NAMESPACES):
            # Get class URI
            uri = cls.get('{http://www.w3.org/1999/02/22-rdf-syntax-ns#}about')
            if not uri:
                continue

            # Get label (use rdfs:label or extract from URI)
            label_elem = cls.find('rdfs:label', self.NAMESPACES)
            if label_elem is not None and label_elem.text:
                label = label_elem.text.strip()
            else:
                # Extract label from URI (last part after # or /)
                label = uri.split('#')[-1].split('/')[-1]
                # Clean up: replace underscores, add spaces before capitals
                label = label.replace('_', ' ')

            self.concept_labels[uri] = label
            self.stats["total_concepts"] += 1

        logger.info(f"  Found {len(self.concept_labels)} classes")

    def _extract_subclass_relations(self, root: ET.Element):
        """Extract rdfs:subClassOf relationships."""
        for cls in root.findall('.//owl:Class', self.NAMESPACES):
            child_uri = cls.get('{http://www.w3.org/1999/02/22-rdf-syntax-ns#}about')
            if not child_uri or child_uri not in self.concept_labels:
                continue

            # Find subClassOf elements
            for subclass_elem in cls.findall('rdfs:subClassOf', self.NAMESPACES):
                parent_uri = subclass_elem.get('{http://www.w3.org/1999/02/22-rdf-syntax-ns#}resource')

                if parent_uri and parent_uri in self.concept_labels:
                    # Add parent→child relationship
                    parent_label = self.concept_labels[parent_uri]
                    child_label = self.concept_labels[child_uri]

                    self.parent_child_map[parent_label].append(child_label)
                    self.stats["total_relations"] += 1

        logger.info(f"  Found {self.stats['total_relations']} subClassOf relations")


if __name__ == "__main__":
    parser = SWOParser()
    chains = parser.parse()

    logger.info(f"\n✅ Successfully extracted {len(chains)} valid chains from SWO")
    logger.info(f"📁 Output: {parser.output_path}")
