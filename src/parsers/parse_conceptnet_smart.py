#!/usr/bin/env python3
"""
SMART ConceptNet Parser - Top Roots Only

Strategy: Extract chains from top 1,000 most-connected roots only.
This gives us high-quality, diverse chains in 2-3 minutes instead of 40+.

Target: 10K-15K chains in 2-3 minutes (vs 20K+ in 40+ minutes)
"""

import logging
import gzip
import csv
import json
from pathlib import Path
from typing import Dict, List, Set, Tuple
from collections import defaultdict, Counter

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ConceptNetSmartParser:
    """Smart ConceptNet parser - top roots only."""

    HIERARCHICAL_RELATIONS = {'/r/IsA', '/r/PartOf', '/r/HasA'}

    def __init__(
        self,
        input_path: Path = None,
        output_path: Path = None,
        language: str = 'en',
        top_roots: int = 1000,  # Only process top 1K roots
        max_chains_per_root: int = 50  # More aggressive limit
    ):
        if input_path is None:
            input_path = Path("data/datasets/ontology_datasets/conceptnet/conceptnet-assertions-5.7.0.csv.gz")
        if output_path is None:
            output_path = Path("artifacts/ontology_chains/conceptnet_chains.jsonl")

        self.input_path = input_path
        self.output_path = output_path
        self.language = language
        self.top_roots = top_roots
        self.max_chains_per_root = max_chains_per_root

    def parse(self) -> int:
        """Parse ConceptNet smartly."""
        logger.info("=" * 60)
        logger.info("CONCEPTNET SMART PARSER (Top Roots Only)")
        logger.info("=" * 60)
        logger.info(f"Top roots to process: {self.top_roots}")
        logger.info(f"Max chains/root: {self.max_chains_per_root}")

        # Step 1: Parse CSV
        logger.info("\n[1/4] Parsing CSV...")
        parent_child_map, concepts = self._parse_csv()

        # Step 2: Find roots and rank by connectivity
        logger.info("\n[2/4] Ranking roots by connectivity...")
        roots_ranked = self._rank_roots(parent_child_map)
        logger.info(f"  Total roots: {len(roots_ranked)}")
        logger.info(f"  Processing top {min(self.top_roots, len(roots_ranked))} roots")

        # Step 3: Build chains from top roots only
        logger.info("\n[3/4] Building chains from top roots...")
        chains = self._build_chains(parent_child_map, roots_ranked[:self.top_roots])

        # Step 4: Write
        logger.info(f"\n[4/4] Writing {len(chains)} chains...")
        self._write_chains(chains)

        logger.info("=" * 60)
        logger.info(f"✅ COMPLETE: {len(chains)} chains")
        logger.info("=" * 60)

        return len(chains)

    def _parse_csv(self) -> Tuple[Dict[str, List[str]], Set[str]]:
        """Parse CSV efficiently."""
        parent_child_map = defaultdict(list)
        concepts = set()
        line_count = 0

        with gzip.open(self.input_path, 'rt', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter='\t')

            for row in reader:
                line_count += 1
                if line_count % 5000000 == 0:
                    logger.info(f"  Processed {line_count/1000000:.1f}M lines, {len(concepts)} concepts, {len(parent_child_map)} parents")

                if len(row) < 4:
                    continue

                relation = row[1]
                if relation not in self.HIERARCHICAL_RELATIONS:
                    continue

                start = row[2]
                end = row[3]

                if not self._is_lang(start) or not self._is_lang(end):
                    continue

                start_label = self._label(start)
                end_label = self._label(end)

                if not start_label or not end_label:
                    continue

                concepts.add(start_label)
                concepts.add(end_label)

                if relation in ['/r/IsA', '/r/PartOf']:
                    parent_child_map[end_label].append(start_label)
                elif relation == '/r/HasA':
                    parent_child_map[start_label].append(end_label)

        logger.info(f"  Found {len(concepts)} concepts, {len(parent_child_map)} parents")
        return parent_child_map, concepts

    def _rank_roots(self, parent_child_map: Dict[str, List[str]]) -> List[Tuple[str, int]]:
        """Rank roots by total descendant count (connectivity)."""
        # Find roots
        all_children = set()
        for children in parent_child_map.values():
            all_children.update(children)
        roots = set(parent_child_map.keys()) - all_children

        # Count descendants for each root
        root_scores = []
        for root in roots:
            descendant_count = self._count_descendants(root, parent_child_map)
            root_scores.append((root, descendant_count))

        # Sort by descendant count (most connected first)
        root_scores.sort(key=lambda x: x[1], reverse=True)

        logger.info(f"  Top 10 roots:")
        for root, count in root_scores[:10]:
            logger.info(f"    {root}: {count} descendants")

        return [root for root, _ in root_scores]

    def _count_descendants(self, node: str, graph: Dict[str, List[str]]) -> int:
        """Count total descendants of a node (iterative BFS)."""
        if node not in graph:
            return 0

        visited = set()
        queue = [node]
        count = 0

        while queue:
            current = queue.pop(0)
            if current in visited:
                continue

            visited.add(current)

            if current in graph:
                for child in graph[current]:
                    if child not in visited:
                        queue.append(child)
                        count += 1

        return count

    def _build_chains(self, graph: Dict[str, List[str]], roots: List[str]) -> List[Dict]:
        """Build chains from top roots."""
        chains = []
        chain_id = 0

        for idx, root in enumerate(roots):
            if (idx + 1) % 100 == 0:
                logger.info(f"  Processed {idx+1}/{len(roots)} roots, {len(chains)} chains so far")

            paths = self._dfs(root, graph, max_depth=20)

            # Limit chains per root
            if len(paths) > self.max_chains_per_root:
                paths = paths[:self.max_chains_per_root]

            for path in paths:
                if 3 <= len(path) <= 20:
                    chain_id += 1
                    chains.append({
                        "chain_id": f"conceptnet_chain_{chain_id:06d}",
                        "concepts": path,
                        "source": "conceptnet",
                        "chain_length": len(path),
                        "metadata": {"root": root, "depth": len(path) - 1}
                    })

        return chains

    def _dfs(self, node: str, graph: Dict[str, List[str]], max_depth: int = 20) -> List[List[str]]:
        """Iterative DFS."""
        paths = []
        stack = [(node, [])]

        while stack:
            curr, path = stack.pop()

            if curr in path or len(path) >= max_depth:
                if path:
                    paths.append(path + [curr])
                continue

            new_path = path + [curr]

            if curr not in graph or not graph[curr]:
                paths.append(new_path)
                continue

            for child in graph[curr]:
                if child not in new_path:
                    stack.append((child, new_path))

        return paths if paths else [[node]]

    def _write_chains(self, chains: List[Dict]):
        """Write chains."""
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_path, 'w') as f:
            for chain in chains:
                f.write(json.dumps(chain) + '\n')
        logger.info(f"✅ Wrote {len(chains)} chains to {self.output_path}")

    def _is_lang(self, uri: str) -> bool:
        """Check language."""
        parts = uri.split('/')
        return len(parts) >= 4 and parts[2] == self.language

    def _label(self, uri: str) -> str:
        """Extract label."""
        parts = uri.split('/')
        if len(parts) < 4:
            return ""
        return parts[3].replace('_', ' ').strip()


if __name__ == "__main__":
    import time
    start = time.time()

    parser = ConceptNetSmartParser(top_roots=1000, max_chains_per_root=50)
    num_chains = parser.parse()

    elapsed = time.time() - start
    logger.info(f"\n⏱️  Total time: {elapsed/60:.1f} minutes ({elapsed:.0f} seconds)")
    logger.info(f"✅ Extracted {num_chains} chains")
