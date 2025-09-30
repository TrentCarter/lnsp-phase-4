#!/usr/bin/env python3
"""
Base parser class for ontology datasets.

All dataset-specific parsers inherit from this class.
Provides common functionality for extracting parent→child chains.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class OntologyChain:
    """
    Represents a sequential parent→child concept chain.

    Example:
        chain_id: "swo_chain_001"
        concepts: ["Software", "Programming Language", "Python", "Django"]
        metadata: {"source": "swo", "quality": 0.95}
    """
    chain_id: str
    concepts: List[str]
    source: str
    metadata: Dict[str, Any]

    def to_jsonl(self) -> str:
        """Convert to JSONL format for ingestion."""
        return json.dumps({
            "chain_id": self.chain_id,
            "concepts": self.concepts,
            "source": self.source,
            "chain_length": len(self.concepts),
            "metadata": self.metadata
        })

    def is_valid(self, min_length: int = 3, max_length: int = 20) -> bool:
        """
        Validate chain quality.

        Args:
            min_length: Minimum chain length (default: 3)
            max_length: Maximum chain length (default: 20)

        Returns:
            True if valid, False otherwise
        """
        # Check length
        if not (min_length <= len(self.concepts) <= max_length):
            return False

        # Check for duplicates
        if len(self.concepts) != len(set(self.concepts)):
            return False

        # Check for empty strings
        if any(not c.strip() for c in self.concepts):
            return False

        return True


class OntologyParser:
    """Base class for parsing ontology datasets into chains."""

    def __init__(
        self,
        source_name: str,
        input_path: Path,
        output_path: Path,
        min_chain_length: int = 3,
        max_chain_length: int = 20
    ):
        """
        Initialize parser.

        Args:
            source_name: Dataset source name (e.g., "swo", "go")
            input_path: Path to input ontology file(s)
            output_path: Path to output JSONL file
            min_chain_length: Minimum valid chain length
            max_chain_length: Maximum valid chain length
        """
        self.source_name = source_name
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)
        self.min_chain_length = min_chain_length
        self.max_chain_length = max_chain_length

        self.chains: List[OntologyChain] = []
        self.stats = {
            "total_concepts": 0,
            "total_relations": 0,
            "generated_chains": 0,
            "valid_chains": 0,
            "invalid_chains": 0,
            "avg_chain_length": 0.0,
            "min_length": float('inf'),
            "max_length": 0
        }

    def parse(self) -> List[OntologyChain]:
        """
        Parse ontology and extract chains.

        Returns:
            List of extracted OntologyChain objects

        Override in subclasses to implement dataset-specific parsing.
        """
        raise NotImplementedError("Subclasses must implement parse()")

    def build_chains_from_graph(
        self,
        parent_child_map: Dict[str, List[str]],
        root_concepts: Optional[Set[str]] = None,
        max_chains_per_root: int = 1000
    ) -> List[OntologyChain]:
        """
        Build chains from parent→child graph using DFS traversal.

        Args:
            parent_child_map: Dict mapping parent concept to list of children
            root_concepts: Set of root concepts to start traversal (if None, auto-detect)
            max_chains_per_root: Maximum chains to extract per root (prevents explosion)

        Returns:
            List of OntologyChain objects
        """
        # Auto-detect roots if not provided
        if root_concepts is None:
            all_children = set()
            for children in parent_child_map.values():
                all_children.update(children)
            root_concepts = set(parent_child_map.keys()) - all_children

        logger.info(f"Found {len(root_concepts)} root concepts")

        chains = []
        chain_count = 0
        roots_processed = 0

        # DFS from each root
        for root in root_concepts:
            paths = self._dfs_paths(root, parent_child_map)

            # Limit chains per root to prevent explosion
            if len(paths) > max_chains_per_root:
                logger.warning(f"  Root '{root}' has {len(paths)} paths, limiting to {max_chains_per_root}")
                paths = paths[:max_chains_per_root]

            for path in paths:
                chain_count += 1
                chain = OntologyChain(
                    chain_id=f"{self.source_name}_chain_{chain_count:06d}",
                    concepts=path,
                    source=self.source_name,
                    metadata={
                        "root": root,
                        "depth": len(path) - 1
                    }
                )
                chains.append(chain)

            roots_processed += 1
            if roots_processed % 1000 == 0:
                logger.info(f"  Processed {roots_processed}/{len(root_concepts)} roots, {len(chains)} chains so far")

        return chains

    def _dfs_paths(
        self,
        node: str,
        graph: Dict[str, List[str]],
        current_path: Optional[List[str]] = None,
        max_depth: int = 50
    ) -> List[List[str]]:
        """
        Iterative DFS traversal to extract all paths from node to leaf nodes.
        Uses iterative approach to avoid stack overflow on deep graphs.

        Args:
            node: Current node
            graph: Parent→child adjacency map
            current_path: Current path being built
            max_depth: Maximum path depth to prevent infinite loops

        Returns:
            List of all paths from node to leaves
        """
        paths = []
        # Stack: (current_node, path_so_far)
        stack = [(node, [])]

        while stack:
            current_node, path = stack.pop()

            # Avoid cycles and depth limit
            if current_node in path or len(path) >= max_depth:
                if path:  # Save path if it's not empty
                    paths.append(path + [current_node])
                continue

            new_path = path + [current_node]

            # If leaf node, save path
            if current_node not in graph or not graph[current_node]:
                paths.append(new_path)
                continue

            # Add children to stack
            has_valid_children = False
            for child in graph[current_node]:
                if child not in new_path:  # Avoid cycles
                    stack.append((child, new_path))
                    has_valid_children = True

            # If all children are cycles, save current path
            if not has_valid_children:
                paths.append(new_path)

        return paths if paths else [[node]]

    def filter_valid_chains(self) -> List[OntologyChain]:
        """Filter chains to keep only valid ones."""
        valid_chains = []
        for chain in self.chains:
            if chain.is_valid(self.min_chain_length, self.max_chain_length):
                valid_chains.append(chain)
                self.stats["valid_chains"] += 1
            else:
                self.stats["invalid_chains"] += 1

        return valid_chains

    def compute_stats(self):
        """Compute statistics about extracted chains."""
        if not self.chains:
            return

        lengths = [len(chain.concepts) for chain in self.chains]
        self.stats["generated_chains"] = len(self.chains)
        self.stats["avg_chain_length"] = sum(lengths) / len(lengths)
        self.stats["min_length"] = min(lengths)
        self.stats["max_length"] = max(lengths)

    def write_chains(self, chains: Optional[List[OntologyChain]] = None):
        """
        Write chains to output JSONL file.

        Args:
            chains: List of chains to write (default: self.chains)
        """
        if chains is None:
            chains = self.chains

        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(self.output_path, 'w') as f:
            for chain in chains:
                f.write(chain.to_jsonl() + '\n')

        logger.info(f"✅ Wrote {len(chains)} chains to {self.output_path}")

    def print_stats(self):
        """Print parsing statistics."""
        logger.info("=" * 60)
        logger.info(f"PARSING STATISTICS - {self.source_name.upper()}")
        logger.info("=" * 60)
        logger.info(f"  Total concepts: {self.stats['total_concepts']}")
        logger.info(f"  Total relations: {self.stats['total_relations']}")
        logger.info(f"  Generated chains: {self.stats['generated_chains']}")
        logger.info(f"  Valid chains: {self.stats['valid_chains']}")
        logger.info(f"  Invalid chains: {self.stats['invalid_chains']}")
        logger.info(f"  Avg chain length: {self.stats['avg_chain_length']:.1f}")
        logger.info(f"  Min length: {self.stats['min_length']}")
        logger.info(f"  Max length: {self.stats['max_length']}")
        logger.info("=" * 60)
