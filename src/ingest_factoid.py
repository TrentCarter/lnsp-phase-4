#!/usr/bin/env python3
"""
FactoidWiki → LNSP Ingestion Pipeline

Reads FactoidWiki JSONL items, extracts CPE + TMD, embeds concepts,
and writes to Postgres, Neo4j, and Faiss.

Usage:
    python -m src.ingest_factoid --help
"""

from __future__ import annotations
import argparse
import json
import os
import sys
import uuid
from pathlib import Path
from typing import Dict, Any, List, Optional

# Add src to path for module imports
sys.path.insert(0, str(Path(__file__).parent))

from .prompt_extractor import extract_cpe_from_text
from .tmd_encoder import pack_tmd, lane_index_from_bits, tmd16_deterministic
from .db_postgres import PostgresDB
from .db_neo4j import Neo4jDB
from .db_faiss import FaissDB
from .integrations.lightrag import (
    LightRAGConfig,
    LightRAGGraphBuilderAdapter,
    LightRAGHybridRetriever,
)
from .pipeline.p9_graph_extraction import run_graph_extraction


SAMPLE_ITEMS = [
    {
        "id": "sample-0",
        "contents": "Photosynthesis converts light energy into chemical energy in plants."
    },
    {
        "id": "sample-1",
        "contents": "Ada Lovelace is regarded as the first computer programmer for her work on the Analytical Engine."
    },
    {
        "id": "sample-2",
        "contents": "The Eiffel Tower was completed in 1889 for the Paris World's Fair."
    },
    {
        "id": "sample-3",
        "contents": "Olympus Mons on Mars is the tallest volcano in the solar system."
    },
]


def load_factoid_samples(
    file_path: str, 
    file_type: str = 'jsonl', 
    num_samples: Optional[int] = None
) -> List[Dict[str, Any]]:
    """Load samples from FactoidWiki file (JSONL or TSV)."""
    samples = []
    if file_type == 'tsv':
        import csv
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter='\t')
            for i, row in enumerate(reader):
                if num_samples and i >= num_samples:
                    break
                if len(row) >= 3:
                    # TSV format: id, type, question, answer
                    samples.append({'id': row[0], 'contents': row[2]})
    else:  # Default to jsonl
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if num_samples and i >= num_samples:
                    break
                samples.append(json.loads(line.strip()))
    return samples


def process_sample(
    sample: Dict[str, Any],
    pg_db: PostgresDB,
    neo_db: Neo4jDB,
    faiss_db: FaissDB,
    batch_id: str,
    graph_adapter: LightRAGGraphBuilderAdapter,
) -> Optional[str]:
    """Process a single FactoidWiki sample through the pipeline."""

    try:
        # Extract CPE using prompt template
        extraction = extract_cpe_from_text(sample["contents"])

        # Generate a deterministic CPE ID so eval sets can reference stable identifiers
        source_id = sample.get("id")
        if source_id:
            cpe_id = str(uuid.uuid5(uuid.NAMESPACE_URL, source_id))
        else:
            cpe_id = str(uuid.uuid4())

        # Build TMD encoding
        domain_code = extraction["domain_code"]
        task_code = extraction["task_code"]
        modifier_code = extraction["modifier_code"]

        tmd_bits = pack_tmd(domain_code, task_code, modifier_code)
        lane_index = lane_index_from_bits(tmd_bits)
        tmd_lane = f"{extraction['domain']}-{extraction['task']}-{extraction['modifier']}"
        tmd_dense = tmd16_deterministic(domain_code, task_code, modifier_code)

        # Create complete CPE record
        cpe_record = {
            "cpe_id": cpe_id,
            "mission_text": extraction["mission"],
            "source_chunk": sample["contents"],
            "concept_text": extraction["concept"],
            "probe_question": extraction["probe"],
            "expected_answer": extraction["expected"],
            "soft_negatives": extraction.get("soft_negatives", []),
            "hard_negatives": extraction.get("hard_negatives", []),
            "domain_code": domain_code,
            "task_code": task_code,
            "modifier_code": modifier_code,
            "content_type": "factual",
            "dataset_source": "factoid-wiki-large",
            "chunk_position": {
                "doc_id": sample.get("id", cpe_id),
                "start": 0,
                "end": len(sample["contents"])
            },
            "relations_text": extraction.get("relations", []),
            "tmd_bits": tmd_bits,
            "tmd_lane": tmd_lane,
            "lane_index": lane_index,
            "echo_score": extraction.get("echo_score", 0.95),
            "validation_status": extraction.get("validation_status", "passed"),
            "batch_id": batch_id,
            "tmd_dense": tmd_dense.tolist(),
            "concept_vec": extraction["concept_vec"],
            "question_vec": extraction["question_vec"],
            "fused_vec": extraction["fused_vec"],
            "fused_norm": extraction["fused_norm"]
        }

        # Write to databases
        pg_db.insert_cpe(cpe_record)
        neo_db.insert_concept(cpe_record)
        faiss_db.add_vector(cpe_record)

        # Run LightRAG graph extraction after concept insert
        run_graph_extraction(cpe_record, graph_adapter, neo_db)

        print(f" Processed {sample['id']} → CPE {cpe_id}")
        return cpe_id

    except Exception as e:
        print(f" Failed to process {sample['id']}: {e}")
        return None


def ingest(
    samples: List[Dict[str, Any]],
    *,
    write_pg: bool = False,
    write_neo4j: bool = False,
    faiss_out: str = "/tmp/factoid_vecs.npz",
    batch_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Programmatic ingestion helper used by tests and CLI wrappers."""

    lightrag_config = LightRAGConfig.from_env()
    graph_adapter = LightRAGGraphBuilderAdapter.from_config(lightrag_config)
    retriever_adapter = LightRAGHybridRetriever.from_config(config=lightrag_config, dim=784)

    pg_db = PostgresDB(enabled=write_pg)
    neo_db = Neo4jDB(enabled=write_neo4j)
    faiss_db = FaissDB(output_path=faiss_out, retriever_adapter=retriever_adapter)

    batch = batch_id or str(uuid.uuid4())
    processed = []

    for sample in samples:
        cpe_id = process_sample(sample, pg_db, neo_db, faiss_db, batch, graph_adapter)
        if cpe_id:
            processed.append(cpe_id)

    faiss_db.save()

    return {
        "count": len(processed),
        "batch_id": batch,
        "faiss_path": faiss_out,
    }


def main():
    parser = argparse.ArgumentParser(description="Ingest FactoidWiki into LNSP pipeline")
    parser.add_argument("--file-path", type=str,
                       default=str(Path(__file__).parent.parent / "data" / "datasets" / "factoid-wiki-large" / "factoid_wiki.jsonl"),
                       help="Path to data file (JSONL or TSV)")
    parser.add_argument("--file-type", type=str, default='jsonl', choices=['jsonl', 'tsv'],
                       help="Type of file to process (jsonl or tsv)")
    parser.add_argument("--num-samples", type=int, default=4,
                       help="Number of samples to process (None for all)")
    parser.add_argument("--write-pg", action="store_true",
                       help="Write to PostgreSQL")
    parser.add_argument("--write-neo4j", action="store_true",
                       help="Write to Neo4j")
    parser.add_argument("--faiss-out", type=str, default="/tmp/factoid_vecs.npz",
                       help="Path to save Faiss vectors")
    parser.add_argument("--batch-id", type=str, default=None,
                       help="Batch ID (auto-generated if not provided)")

    args = parser.parse_args()

    # Initialize LightRAG adapters
    lightrag_config = LightRAGConfig.from_env()
    graph_adapter = LightRAGGraphBuilderAdapter.from_config(lightrag_config)
    retriever_adapter = LightRAGHybridRetriever.from_config(
        config=lightrag_config, dim=784
    )

    # Initialize databases based on flags
    pg_db = PostgresDB(enabled=args.write_pg)
    neo_db = Neo4jDB(enabled=args.write_neo4j)
    faiss_db = FaissDB(output_path=args.faiss_out, retriever_adapter=retriever_adapter)

    batch_id = args.batch_id or str(uuid.uuid4())

    # Load samples
    file_path = Path(args.file_path)
    if not file_path.exists():
        print(f"Error: File not found at {file_path}")
        return 1

    print(f"Loading samples from {file_path} (type: {args.file_type})")
    if args.num_samples:
        print(f"Processing first {args.num_samples} samples")
    else:
        print("Processing all samples")

    samples = load_factoid_samples(str(file_path), args.file_type, args.num_samples)

    if not samples:
        print("No samples loaded!")
        return 1

    print(f"Starting ingestion with batch_id={batch_id}")
    print("=" * 60)

    # Process samples
    processed_ids = []
    for sample in samples:
        cpe_id = process_sample(sample, pg_db, neo_db, faiss_db, batch_id, graph_adapter)
        if cpe_id:
            processed_ids.append(cpe_id)

    # Finalize databases
    faiss_db.save()

    print("=" * 60)
    print(f" Completed ingestion of {len(processed_ids)}/{len(samples)} samples")
    print(f"  Batch ID: {batch_id}")
    if processed_ids:
        print(f"  Sample CPE IDs: {processed_ids[:3]}{'...' if len(processed_ids) > 3 else ''}")
    print(f"  Faiss saved to: {args.faiss_out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
