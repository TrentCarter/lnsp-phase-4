#!/usr/bin/env python3
"""
Simple Ontology → LNSP Ingestion

Converts ontology chains to factoid-like format and uses existing pipeline.
The LLM extraction will preserve the chain structure and create all metadata.

Usage:
    # Test with 10 chains
    python -m src.ingest_ontology_simple \
        --input artifacts/ontology_chains/swo_chains.jsonl \
        --limit 10 \
        --write-pg

    # Full ingestion
    python -m src.ingest_ontology_simple --ingest-all
"""

from __future__ import annotations
import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional

sys.path.insert(0, str(Path(__file__).parent))

from .ingest_factoid import load_factoid_samples, process_sample, SAMPLE_ITEMS
from .db_postgres import PostgresDB
from .db_neo4j import Neo4jDB
from .db_faiss import FaissDB
from .integrations.lightrag import LightRAGConfig, LightRAGGraphBuilderAdapter

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def chain_to_factoid_text(chain: Dict[str, Any]) -> str:
    """
    Convert ontology chain to natural language text.
    The LLM will extract the concept/probe/expected from this.
    
    Strategy: Create a hierarchical description that preserves the chain structure.
    Example: "BLAST is a type of analysis software, which is a type of software."
    """
    concepts = chain["concepts"]
    
    if len(concepts) == 2:
        return f"{concepts[1]} is a type of {concepts[0]}."
    elif len(concepts) == 3:
        return f"{concepts[2]} is a type of {concepts[1]}, which is a type of {concepts[0]}."
    else:
        # For longer chains: "D is C, C is B, B is A"
        parts = []
        for i in range(len(concepts) - 1, 0, -1):
            parts.append(f"{concepts[i]} is a type of {concepts[i-1]}")
        return ", and ".join(parts) + "."


def load_ontology_chains(file_path: Path, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """Load ontology chains and convert to factoid format."""
    samples = []
    
    with open(file_path, 'r') as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            
            chain = json.loads(line.strip())
            
            # Convert to factoid format (compatible with existing pipeline)
            samples.append({
                "id": chain["chain_id"],
                "contents": chain_to_factoid_text(chain),
                "source": chain["source"]  # Preserve source for tracking
            })
    
    return samples


def ingest_ontology_file(
    input_path: Path,
    write_pg: bool = False,
    write_neo4j: bool = False,
    write_faiss: bool = False,
    limit: Optional[int] = None
) -> Dict[str, int]:
    """Ingest ontology chains using the factoid pipeline."""
    
    logger.info("=" * 60)
    logger.info(f"ONTOLOGY INGESTION: {input_path.stem}")
    logger.info("=" * 60)
    logger.info(f"Input: {input_path}")
    logger.info(f"Limit: {limit if limit else 'None (all chains)'}")
    
    # Load and convert chains
    logger.info("Loading and converting chains to factoid format...")
    samples = load_ontology_chains(input_path, limit)
    logger.info(f"Loaded {len(samples)} chains")
    
    # Initialize databases
    # PostgreSQL and Neo4j have enabled flags, Faiss doesn't
    pg_db = PostgresDB(enabled=write_pg)
    neo_db = Neo4jDB(enabled=write_neo4j)

    # Faiss: initialize if needed, otherwise use stub
    if write_faiss:
        faiss_db = FaissDB()
    else:
        # Create stub Faiss DB
        class StubFaissDB:
            def add_vector(self, cpe_record): pass
        faiss_db = StubFaissDB()

    # Initialize graph adapter (required even if not writing to Neo4j)
    config = LightRAGConfig()
    graph_adapter = LightRAGGraphBuilderAdapter(config)
    
    # Generate batch ID (must be valid UUID for Postgres)
    import uuid
    batch_id = str(uuid.uuid4())
    
    # Process samples using existing factoid pipeline
    stats = {"total": len(samples), "processed": 0, "failed": 0}

    import time
    start_time = time.time()

    logger.info("Processing chains (LLM + 768D embeddings + TMD + Graph)...")
    for i, sample in enumerate(samples, 1):
        try:
            result = process_sample(
                sample=sample,
                pg_db=pg_db,
                neo_db=neo_db,
                faiss_db=faiss_db,
                batch_id=batch_id,
                graph_adapter=graph_adapter
            )

            if result:
                stats["processed"] += 1
                if i % 10 == 0:
                    elapsed = time.time() - start_time
                    rate = i / elapsed  # chains per second
                    remaining = stats['total'] - i
                    eta_seconds = remaining / rate if rate > 0 else 0
                    eta_minutes = eta_seconds / 60
                    logger.info(f"  ✓ {i}/{stats['total']} chains | {rate:.2f} chains/sec | ETA: {eta_minutes:.1f}min")
            else:
                stats["failed"] += 1

        except Exception as e:
            logger.error(f"✗ Error processing chain {sample['id']}: {e}")
            stats["failed"] += 1
    
    # Print summary
    logger.info("=" * 60)
    logger.info("INGESTION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Total chains:     {stats['total']:,}")
    logger.info(f"Processed:        {stats['processed']:,}")
    logger.info(f"Failed:           {stats['failed']:,}")
    logger.info(f"Success rate:     {stats['processed']/stats['total']*100:.1f}%")
    logger.info("=" * 60)
    
    return stats


def ingest_all_datasets(
    write_pg: bool = False,
    write_neo4j: bool = False,
    write_faiss: bool = False,
    limit_per_dataset: Optional[int] = None
) -> Dict[str, Dict[str, int]]:
    """Ingest all ontology datasets."""
    
    datasets = [
        "artifacts/ontology_chains/swo_chains.jsonl",
        "artifacts/ontology_chains/go_chains.jsonl",
        "artifacts/ontology_chains/dbpedia_chains.jsonl"
    ]
    
    all_stats = {}
    
    for dataset_path in datasets:
        path = Path(dataset_path)
        
        if not path.exists():
            logger.warning(f"Skipping {path.stem}: file not found at {dataset_path}")
            continue
        
        stats = ingest_ontology_file(
            input_path=path,
            write_pg=write_pg,
            write_neo4j=write_neo4j,
            write_faiss=write_faiss,
            limit=limit_per_dataset
        )
        
        all_stats[path.stem] = stats
    
    # Print combined summary
    logger.info("\n" + "=" * 60)
    logger.info("COMBINED INGESTION SUMMARY")
    logger.info("=" * 60)
    
    total_all = sum(s["total"] for s in all_stats.values())
    processed_all = sum(s["processed"] for s in all_stats.values())
    failed_all = sum(s["failed"] for s in all_stats.values())
    
    for name, stats in all_stats.items():
        logger.info(f"{name:20s}: {stats['processed']:,} / {stats['total']:,} chains")
    
    logger.info("-" * 60)
    logger.info(f"TOTAL:                {processed_all:,} / {total_all:,} chains")
    logger.info(f"Success:              {processed_all/total_all*100:.1f}%")
    logger.info("=" * 60)
    
    return all_stats


def main():
    parser = argparse.ArgumentParser(
        description="Ingest ontology chains into LNSP (simple factoid pipeline)"
    )
    
    # Input options
    parser.add_argument("--input", type=Path, help="Input JSONL file with ontology chains")
    parser.add_argument("--ingest-all", action="store_true", help="Ingest all available datasets")
    
    # Database options
    parser.add_argument("--write-pg", action="store_true", help="Write to PostgreSQL")
    parser.add_argument("--write-neo4j", action="store_true", help="Write to Neo4j")
    parser.add_argument("--write-faiss", action="store_true", help="Write to Faiss")
    
    # Processing options
    parser.add_argument("--limit", type=int, help="Limit number of chains to process")
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.ingest_all:
        ingest_all_datasets(
            write_pg=args.write_pg,
            write_neo4j=args.write_neo4j,
            write_faiss=args.write_faiss,
            limit_per_dataset=args.limit
        )
    elif args.input:
        if not args.input.exists():
            logger.error(f"Input file not found: {args.input}")
            sys.exit(1)
        
        ingest_ontology_file(
            input_path=args.input,
            write_pg=args.write_pg,
            write_neo4j=args.write_neo4j,
            write_faiss=args.write_faiss,
            limit=args.limit
        )
    else:
        parser.error("Must specify either --ingest-all or --input")


if __name__ == "__main__":
    main()
