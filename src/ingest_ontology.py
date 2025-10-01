#!/usr/bin/env python3
"""
Ontology Chain → LNSP Ingestion Pipeline

Reads ontology chain JSONL files (SWO, GO, DBpedia, ConceptNet),
converts to CPESH format, embeds concepts, and writes to Postgres, Neo4j, and Faiss.

Key differences from FactoidWiki ingestion:
- No LLM extraction needed (chains are already structured)
- Direct conversion to CPESH format
- Sequential relationships already validated
- Supports multiple ontology sources

Usage:
    # Ingest single dataset
    python -m src.ingest_ontology \
        --input artifacts/ontology_chains/swo_chains.jsonl \
        --source swo \
        --write-pg --write-neo4j --write-faiss

    # Ingest all datasets
    python -m src.ingest_ontology --ingest-all
"""

from __future__ import annotations
import argparse
import json
import logging
import sys
import uuid
from pathlib import Path
from typing import Dict, Any, List, Optional
from collections import defaultdict

# Add src to path for module imports
sys.path.insert(0, str(Path(__file__).parent))

from .db_postgres import PostgresDB
from .db_neo4j import Neo4jDB
from .db_faiss import FaissDB
from .tmd_encoder import pack_tmd, tmd16_deterministic
from .integrations.lightrag import (
    LightRAGConfig,
    LightRAGGraphBuilderAdapter,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def chain_to_cpesh(chain: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert an ontology chain to CPESH format.
    
    Strategy:
    - Concept: first concept in chain (root/most general)
    - Probe: last concept in chain (leaf/most specific)
    - Expected: all concepts in chain (sequential path)
    - Soft negatives: sibling concepts at same level (future enhancement)
    - Hard negatives: unrelated concepts from other chains (future enhancement)
    
    Args:
        chain: Ontology chain with format:
            {
                "chain_id": "swo_chain_001",
                "concepts": ["software", "analysis software", "blast"],
                "source": "swo",
                "chain_length": 3
            }
    
    Returns:
        CPESH extraction dict compatible with existing pipeline
    """
    concepts = chain["concepts"]
    
    # Validate chain structure
    if len(concepts) < 3:
        raise ValueError(f"Chain {chain['chain_id']} too short: {len(concepts)} < 3")
    
    return {
        "concept": concepts[0],  # Root/most general
        "probe": concepts[-1],  # Leaf/most specific
        "expected": concepts,  # Full sequential path
        "soft_negatives": [],  # TODO: Extract sibling concepts
        "hard_negatives": []  # TODO: Extract unrelated concepts
    }


def process_ontology_chain(
    chain: Dict[str, Any],
    pg_db: PostgresDB,
    neo_db: Neo4jDB,
    faiss_db: FaissDB,
    batch_id: str,
    graph_adapter: LightRAGGraphBuilderAdapter,
) -> Optional[str]:
    """Process a single ontology chain through the pipeline."""
    
    try:
        # Convert chain to CPESH format
        cpesh = chain_to_cpesh(chain)
        
        # Generate CPE ID
        chain_id = chain.get("chain_id", str(uuid.uuid4()))
        cpe_id = f"ont_{chain['source']}_{chain_id}"
        
        # Compute TMD (16-bit metadata)
        # Use default ontology domain/task/modifier codes
        # domain: 1 (ontology), task: 1 (classification), modifier: 1 (hierarchical)
        domain_code, task_code, modifier_code = 1, 1, 1
        tmd_packed = pack_tmd(domain_code, task_code, modifier_code)
        lane_idx = (domain_code << 11) | (task_code << 6) | modifier_code
        
        # Store in PostgreSQL (CPE metadata + TMD)
        pg_db.insert_cpe_entry(
            cpe_id=cpe_id,
            concept=cpesh["concept"],
            probe=cpesh["probe"],
            expected=cpesh["expected"],
            soft_negatives=cpesh.get("soft_negatives", []),
            hard_negatives=cpesh.get("hard_negatives", []),
            tmd_packed=tmd_packed,
            lane_index=lane_idx,
            batch_id=batch_id,
            source_id=chain_id,
            source_type="ontology",
            source_name=chain["source"]
        )
        
        # Build graph in Neo4j
        # Ontology chains have explicit sequential relationships
        for i in range(len(chain["concepts"]) - 1):
            parent = chain["concepts"][i]
            child = chain["concepts"][i + 1]
            
            # Create nodes and edge
            neo_db.merge_node(parent, "Concept", {"source": chain["source"]})
            neo_db.merge_node(child, "Concept", {"source": chain["source"]})
            neo_db.create_relationship(
                parent, child, "PARENT_OF",
                {"chain_id": chain_id, "source": chain["source"]}
            )
        
        # Embed and store in Faiss
        # Use concept as primary embedding target
        concept_vec = faiss_db.embed_text(cpesh["concept"])
        faiss_db.add_vector(cpe_id, concept_vec, metadata={
            "concept": cpesh["concept"],
            "probe": cpesh["probe"],
            "source": chain["source"]
        })
        
        logger.info(f"✓ Processed chain: {chain_id} ({len(chain['concepts'])} concepts)")
        return cpe_id
        
    except Exception as e:
        logger.error(f"Failed to process chain {chain.get('chain_id', 'unknown')}: {e}")
        return None


def ingest_ontology_file(
    input_path: Path,
    source: str,
    write_pg: bool = False,
    write_neo4j: bool = False,
    write_faiss: bool = False,
    limit: Optional[int] = None
) -> Dict[str, int]:
    """Ingest ontology chains from a JSONL file."""
    
    logger.info("=" * 60)
    logger.info(f"ONTOLOGY INGESTION: {source.upper()}")
    logger.info("=" * 60)
    logger.info(f"Input: {input_path}")
    logger.info(f"Limit: {limit if limit else 'None (all chains)'}")
    
    # Initialize databases
    pg_db = PostgresDB() if write_pg else None
    neo_db = Neo4jDB() if write_neo4j else None
    faiss_db = FaissDB() if write_faiss else None
    
    # Initialize graph adapter (if Neo4j enabled)
    graph_adapter = None
    if write_neo4j:
        config = LightRAGConfig()
        graph_adapter = LightRAGGraphBuilderAdapter(config)
    
    # Generate batch ID
    batch_id = f"ont_{source}_{uuid.uuid4().hex[:8]}"
    
    # Process chains
    stats = {
        "total": 0,
        "processed": 0,
        "failed": 0
    }
    
    with open(input_path, 'r') as f:
        for line in f:
            if limit and stats["total"] >= limit:
                break
            
            stats["total"] += 1
            
            try:
                chain = json.loads(line.strip())
                
                result = process_ontology_chain(
                    chain=chain,
                    pg_db=pg_db,
                    neo_db=neo_db,
                    faiss_db=faiss_db,
                    batch_id=batch_id,
                    graph_adapter=graph_adapter
                )
                
                if result:
                    stats["processed"] += 1
                else:
                    stats["failed"] += 1
                    
            except Exception as e:
                logger.error(f"Error processing line {stats['total']}: {e}")
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
    
    datasets = {
        "swo": "artifacts/ontology_chains/swo_chains.jsonl",
        "go": "artifacts/ontology_chains/go_chains.jsonl",
        "dbpedia": "artifacts/ontology_chains/dbpedia_chains.jsonl"
    }
    
    all_stats = {}
    
    for source, input_path in datasets.items():
        path = Path(input_path)
        
        if not path.exists():
            logger.warning(f"Skipping {source}: file not found at {input_path}")
            continue
        
        stats = ingest_ontology_file(
            input_path=path,
            source=source,
            write_pg=write_pg,
            write_neo4j=write_neo4j,
            write_faiss=write_faiss,
            limit=limit_per_dataset
        )
        
        all_stats[source] = stats
    
    # Print combined summary
    logger.info("\n" + "=" * 60)
    logger.info("COMBINED INGESTION SUMMARY")
    logger.info("=" * 60)
    
    total_all = sum(s["total"] for s in all_stats.values())
    processed_all = sum(s["processed"] for s in all_stats.values())
    failed_all = sum(s["failed"] for s in all_stats.values())
    
    for source, stats in all_stats.items():
        logger.info(f"{source.upper():10s}: {stats['processed']:,} / {stats['total']:,} chains")
    
    logger.info("-" * 60)
    logger.info(f"TOTAL:      {processed_all:,} / {total_all:,} chains")
    logger.info(f"Success:    {processed_all/total_all*100:.1f}%")
    logger.info("=" * 60)
    
    return all_stats


def main():
    parser = argparse.ArgumentParser(
        description="Ingest ontology chains into LNSP"
    )
    
    # Input options
    parser.add_argument(
        "--input",
        type=Path,
        help="Input JSONL file with ontology chains"
    )
    parser.add_argument(
        "--source",
        type=str,
        choices=["swo", "go", "dbpedia", "conceptnet"],
        help="Source ontology name"
    )
    parser.add_argument(
        "--ingest-all",
        action="store_true",
        help="Ingest all available ontology datasets"
    )
    
    # Database options
    parser.add_argument("--write-pg", action="store_true", help="Write to PostgreSQL")
    parser.add_argument("--write-neo4j", action="store_true", help="Write to Neo4j")
    parser.add_argument("--write-faiss", action="store_true", help="Write to Faiss")
    
    # Processing options
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit number of chains to process (per dataset if --ingest-all)"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.ingest_all:
        ingest_all_datasets(
            write_pg=args.write_pg,
            write_neo4j=args.write_neo4j,
            write_faiss=args.write_faiss,
            limit_per_dataset=args.limit
        )
    elif args.input and args.source:
        if not args.input.exists():
            logger.error(f"Input file not found: {args.input}")
            sys.exit(1)
        
        ingest_ontology_file(
            input_path=args.input,
            source=args.source,
            write_pg=args.write_pg,
            write_neo4j=args.write_neo4j,
            write_faiss=args.write_faiss,
            limit=args.limit
        )
    else:
        parser.error("Must specify either --ingest-all or both --input and --source")


if __name__ == "__main__":
    main()
