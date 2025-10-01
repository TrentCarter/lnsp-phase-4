#!/usr/bin/env python3
"""
OPTIMIZED Multi-threaded ConceptNet Parser

Uses ALL available CPU cores to parse ConceptNet in parallel.
Target: 2-3 minutes total (vs 40+ minutes single-threaded)

Hardware optimizations:
- Multi-process CSV parsing (16 cores)
- Parallel chain building (16 cores)
- Shared memory for graph data
- Batch processing for efficiency
"""

import logging
import gzip
import csv
import json
from pathlib import Path
from typing import Dict, List, Set, Tuple
from collections import defaultdict
from multiprocessing import Pool, Manager, cpu_count
from functools import partial

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ConceptNetParallelParser:
    """Parallel parser for ConceptNet using all CPU cores."""

    HIERARCHICAL_RELATIONS = {
        '/r/IsA',      # dog IsA animal
        '/r/PartOf',   # wheel PartOf car
        '/r/HasA'      # car HasA wheel
    }

    def __init__(
        self,
        input_path: Path = None,
        output_path: Path = None,
        language: str = 'en',
        num_workers: int = None,
        max_chains_per_root: int = 1000
    ):
        if input_path is None:
            input_path = Path("data/datasets/ontology_datasets/conceptnet/conceptnet-assertions-5.7.0.csv.gz")
        if output_path is None:
            output_path = Path("artifacts/ontology_chains/conceptnet_chains.jsonl")

        self.input_path = input_path
        self.output_path = output_path
        self.language = language
        self.num_workers = num_workers or cpu_count()
        self.max_chains_per_root = max_chains_per_root

        logger.info(f"Initialized parallel parser with {self.num_workers} workers")

    def parse(self) -> int:
        """Parse ConceptNet in parallel and extract chains."""
        logger.info("=" * 60)
        logger.info("CONCEPTNET PARALLEL PARSER")
        logger.info("=" * 60)
        logger.info(f"Input: {self.input_path}")
        logger.info(f"Output: {self.output_path}")
        logger.info(f"Workers: {self.num_workers}")
        logger.info(f"Max chains/root: {self.max_chains_per_root}")

        # Step 1: Parse CSV in chunks (parallel)
        logger.info("\n[1/3] Parsing CSV in parallel...")
        parent_child_map, concepts = self._parse_csv_parallel()

        logger.info(f"  Found {len(concepts)} unique concepts")
        logger.info(f"  Found {sum(len(v) for v in parent_child_map.values())} relations")

        # Step 2: Find root concepts
        logger.info("\n[2/3] Finding root concepts...")
        all_children = set()
        for children in parent_child_map.values():
            all_children.update(children)
        root_concepts = list(set(parent_child_map.keys()) - all_children)
        logger.info(f"  Found {len(root_concepts)} root concepts")

        # Step 3: Build chains in parallel
        logger.info("\n[3/3] Building chains in parallel...")
        chains = self._build_chains_parallel(parent_child_map, root_concepts)

        # Step 4: Write chains
        logger.info(f"\nWriting {len(chains)} chains to {self.output_path}...")
        self._write_chains(chains)

        logger.info("=" * 60)
        logger.info(f"✅ COMPLETE: {len(chains)} chains extracted")
        logger.info("=" * 60)

        return len(chains)

    def _parse_csv_parallel(self) -> Tuple[Dict[str, Set[str]], Set[str]]:
        """Parse CSV in parallel chunks."""
        # Read entire file and split into chunks
        logger.info("  Reading CSV file...")
        with gzip.open(self.input_path, 'rt', encoding='utf-8') as f:
            lines = f.readlines()

        total_lines = len(lines)
        chunk_size = total_lines // self.num_workers
        logger.info(f"  Splitting {total_lines} lines into {self.num_workers} chunks")

        # Create chunks
        chunks = []
        for i in range(self.num_workers):
            start = i * chunk_size
            end = start + chunk_size if i < self.num_workers - 1 else total_lines
            chunks.append(lines[start:end])

        # Process chunks in parallel
        logger.info(f"  Processing chunks with {self.num_workers} workers...")
        with Pool(self.num_workers) as pool:
            results = pool.map(self._process_chunk, chunks)

        # Merge results
        logger.info("  Merging results...")
        parent_child_map = defaultdict(set)
        all_concepts = set()

        for chunk_relations, chunk_concepts in results:
            for parent, children in chunk_relations.items():
                parent_child_map[parent].update(children)
            all_concepts.update(chunk_concepts)

        # Convert sets to lists for JSON serialization
        parent_child_map = {k: list(v) for k, v in parent_child_map.items()}

        return parent_child_map, all_concepts

    def _process_chunk(self, lines: List[str]) -> Tuple[Dict[str, Set[str]], Set[str]]:
        """Process a chunk of CSV lines."""
        parent_child_map = defaultdict(set)
        concepts = set()

        for line in lines:
            parts = line.strip().split('\t')
            if len(parts) < 4:
                continue

            relation = parts[1]
            start_concept = parts[2]
            end_concept = parts[3]

            # Filter by relation and language
            if relation not in self.HIERARCHICAL_RELATIONS:
                continue

            if not self._is_target_language(start_concept) or not self._is_target_language(end_concept):
                continue

            # Extract labels
            start_label = self._extract_label(start_concept)
            end_label = self._extract_label(end_concept)

            if not start_label or not end_label:
                continue

            concepts.add(start_label)
            concepts.add(end_label)

            # Build parent→child relationship
            if relation in ['/r/IsA', '/r/PartOf']:
                parent_child_map[end_label].add(start_label)
            elif relation == '/r/HasA':
                parent_child_map[start_label].add(end_label)

        return parent_child_map, concepts

    def _build_chains_parallel(
        self,
        parent_child_map: Dict[str, List[str]],
        root_concepts: List[str]
    ) -> List[Dict]:
        """Build chains from roots in parallel."""
        # Split roots into batches
        batch_size = len(root_concepts) // self.num_workers
        batches = []
        for i in range(self.num_workers):
            start = i * batch_size
            end = start + batch_size if i < self.num_workers - 1 else len(root_concepts)
            batches.append(root_concepts[start:end])

        logger.info(f"  Processing {len(root_concepts)} roots in {self.num_workers} batches")

        # Process batches in parallel
        process_func = partial(
            self._process_root_batch,
            parent_child_map=parent_child_map,
            max_chains=self.max_chains_per_root
        )

        with Pool(self.num_workers) as pool:
            batch_results = pool.map(process_func, batches)

        # Merge chains
        all_chains = []
        for chains in batch_results:
            all_chains.extend(chains)

        return all_chains

    @staticmethod
    def _process_root_batch(
        roots: List[str],
        parent_child_map: Dict[str, List[str]],
        max_chains: int
    ) -> List[Dict]:
        """Process a batch of roots and extract chains."""
        chains = []
        chain_id = 0

        for root in roots:
            # DFS to find all paths from root
            paths = ConceptNetParallelParser._dfs_iterative(root, parent_child_map, max_depth=50)

            # Limit chains per root
            if len(paths) > max_chains:
                paths = paths[:max_chains]

            for path in paths:
                if 3 <= len(path) <= 20:  # Valid chain length
                    chain_id += 1
                    chains.append({
                        "chain_id": f"conceptnet_chain_{chain_id:06d}",
                        "concepts": path,
                        "source": "conceptnet",
                        "chain_length": len(path),
                        "metadata": {
                            "root": root,
                            "depth": len(path) - 1
                        }
                    })

        return chains

    @staticmethod
    def _dfs_iterative(
        node: str,
        graph: Dict[str, List[str]],
        max_depth: int = 50
    ) -> List[List[str]]:
        """Iterative DFS to extract paths."""
        paths = []
        stack = [(node, [])]

        while stack:
            current, path = stack.pop()

            if current in path or len(path) >= max_depth:
                if path:
                    paths.append(path + [current])
                continue

            new_path = path + [current]

            if current not in graph or not graph[current]:
                paths.append(new_path)
                continue

            has_children = False
            for child in graph[current]:
                if child not in new_path:
                    stack.append((child, new_path))
                    has_children = True

            if not has_children:
                paths.append(new_path)

        return paths if paths else [[node]]

    def _write_chains(self, chains: List[Dict]):
        """Write chains to JSONL file."""
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(self.output_path, 'w') as f:
            for chain in chains:
                f.write(json.dumps(chain) + '\n')

        logger.info(f"✅ Wrote {len(chains)} chains to {self.output_path}")

    def _is_target_language(self, concept_uri: str) -> bool:
        """Check if concept is in target language."""
        parts = concept_uri.split('/')
        if len(parts) < 4:
            return False
        return parts[2] == self.language

    def _extract_label(self, concept_uri: str) -> str:
        """Extract readable label from ConceptNet URI."""
        parts = concept_uri.split('/')
        if len(parts) < 4:
            return ""
        label = parts[3].replace('_', ' ').strip()
        return label


if __name__ == "__main__":
    import time
    start = time.time()

    parser = ConceptNetParallelParser(num_workers=16)  # Use all 16 cores
    num_chains = parser.parse()

    elapsed = time.time() - start
    logger.info(f"\n⏱️  Total time: {elapsed:.1f} seconds")
    logger.info(f"📊 Throughput: {num_chains/elapsed:.0f} chains/second")
    logger.info(f"✅ Successfully extracted {num_chains} chains from ConceptNet")
