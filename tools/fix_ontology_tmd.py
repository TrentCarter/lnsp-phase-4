#!/usr/bin/env python3
"""Fix TMD classification for ontology concepts using LLM.

This script:
1. Loads ontology concepts from database
2. Uses Ollama Llama 3.1 to classify domain/task/modifier
3. Updates TMD vectors in database
4. Validates results

Usage:
    python tools/fix_ontology_tmd.py --mode classify
    python tools/fix_ontology_tmd.py --mode update-db
    python tools/fix_ontology_tmd.py --mode verify
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Tuple
import psycopg2
import numpy as np

# Add src to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.enums import DOMAIN_LABELS, TASK_LABELS, MODIFIER_LABELS
from src.tmd_encoder import pack_tmd
from src.llm.local_llama_client import call_local_llama

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def classify_concept_with_llm(concept_text: str, source: str) -> Tuple[int, int, int]:
    """Classify a concept using Ollama Llama 3.1.

    Args:
        concept_text: The concept to classify
        source: Origin (swo, go, dbpedia)

    Returns:
        (domain_code, task_code, modifier_code)
    """
    # Build prompt
    prompt = f"""You are classifying a concept from an ontology into three categories.

Concept: "{concept_text}"
Source: {source.upper()} ontology

Available domains:
- science (0): Scientific/research concepts
- engineering (1): Software/tools/engineering
- arts (2): Humanities/arts/philosophy
- biology (3): Biological/medical concepts
- geography (4): Geographic/spatial concepts
- law (5): Legal/regulatory concepts

Available tasks:
- fact_retrieval (0): Factual knowledge
- code_generation (1): Software/code related
- entailment (2): Logical reasoning
- qa (3): Question answering

Available modifiers:
- neutral (0): Standard concept
- robust (1): Core/fundamental concept
- multilingual (2): International/cross-language
- ethical (3): Has ethical considerations

Based on the concept and source, output ONLY a JSON object:
{{"domain": "domain_name", "task": "task_name", "modifier": "modifier_name"}}

Example for "BioConductor package metaArray":
{{"domain": "engineering", "task": "code_generation", "modifier": "neutral"}}

Example for "Gene Ontology: protein binding":
{{"domain": "biology", "task": "fact_retrieval", "modifier": "neutral"}}

Now classify: "{concept_text}"
Output:"""

    # Call LLM
    try:
        response = call_local_llama(prompt)
        result_text = response.text.strip()

        # Extract JSON (handle markdown code blocks)
        if "```json" in result_text:
            result_text = result_text.split("```json")[1].split("```")[0].strip()
        elif "```" in result_text:
            result_text = result_text.split("```")[1].split("```")[0].strip()

        # Parse JSON
        result = json.loads(result_text)

        # Map to codes
        domain_map = {label: idx for idx, label in enumerate(DOMAIN_LABELS)}
        task_map = {label: idx for idx, label in enumerate(TASK_LABELS)}
        modifier_map = {label: idx for idx, label in enumerate(MODIFIER_LABELS)}

        domain_code = domain_map.get(result["domain"].lower(), 0)  # default: science
        task_code = task_map.get(result["task"].lower(), 0)  # default: fact_retrieval
        modifier_code = modifier_map.get(result["modifier"].lower(), 0)  # default: neutral

        return (domain_code, task_code, modifier_code)

    except Exception as e:
        logger.warning(f"LLM classification failed for '{concept_text}': {e}")
        # Fallback based on source
        if source == "swo":
            return (2, 1, 0)  # engineering, code_generation, neutral
        elif source == "go":
            return (4, 0, 0)  # medicine (biology), fact_retrieval, neutral
        elif source == "dbpedia":
            return (9, 0, 0)  # art, fact_retrieval, neutral
        else:
            return (0, 0, 0)  # science, fact_retrieval, neutral


def classify_all_concepts(conn, limit: int = None):
    """Classify all concepts with zero or incorrect TMD vectors."""
    cur = conn.cursor()

    # Get concepts needing classification
    query = """
        SELECT v.cpe_id, e.concept_text, e.source
        FROM cpe_vectors v
        JOIN cpe_entry e ON v.cpe_id = e.cpe_id
        WHERE v.tmd_dense IS NOT NULL
        ORDER BY v.cpe_id
    """
    if limit:
        query += f" LIMIT {limit}"

    cur.execute(query)
    rows = cur.fetchall()
    logger.info(f"Classifying {len(rows)} concepts...")

    results = []
    for i, (cpe_id, concept_text, source) in enumerate(rows):
        if i % 100 == 0:
            logger.info(f"Progress: {i}/{len(rows)} ({100*i/len(rows):.1f}%)")

        domain, task, modifier = classify_concept_with_llm(concept_text, source or "unknown")

        # Pack into TMD vector
        tmd_dense = pack_tmd(domain, task, modifier)

        results.append({
            "cpe_id": cpe_id,
            "concept_text": concept_text,
            "source": source,
            "domain": domain,
            "task": task,
            "modifier": modifier,
            "tmd_dense": tmd_dense.tolist()
        })

    cur.close()
    logger.info(f"Classified {len(results)} concepts")
    return results


def update_database(conn, results: list):
    """Update TMD vectors in database."""
    cur = conn.cursor()

    logger.info(f"Updating {len(results)} TMD vectors in database...")

    for i, result in enumerate(results):
        if i % 100 == 0:
            logger.info(f"Update progress: {i}/{len(results)} ({100*i/len(results):.1f}%)")

        cur.execute("""
            UPDATE cpe_vectors
            SET tmd_dense = %s
            WHERE cpe_id = %s
        """, (json.dumps(result["tmd_dense"]), result["cpe_id"]))

    conn.commit()
    cur.close()
    logger.info("Database updated successfully")


def verify_results(conn):
    """Verify TMD classification results."""
    cur = conn.cursor()

    # Count zeros
    cur.execute("""
        SELECT COUNT(*)
        FROM cpe_vectors
        WHERE tmd_dense::jsonb = '[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]'::jsonb
    """)
    zero_count = cur.fetchone()[0]

    # Get sample classifications
    cur.execute("""
        SELECT e.concept_text, e.source, v.tmd_dense
        FROM cpe_vectors v
        JOIN cpe_entry e ON v.cpe_id = e.cpe_id
        WHERE v.tmd_dense IS NOT NULL
        ORDER BY random()
        LIMIT 20
    """)
    samples = cur.fetchall()

    cur.close()

    logger.info(f"\n{'='*60}")
    logger.info("VERIFICATION RESULTS")
    logger.info(f"{'='*60}")
    logger.info(f"Zero TMD vectors: {zero_count}")
    logger.info(f"\nSample classifications:")

    for concept_text, source, tmd_dense_json in samples:
        tmd = json.loads(tmd_dense_json)
        # Simple decode (first 3 values are domain, task, modifier approximation)
        domain = int(tmd[0]) if tmd[0] != 0 else 0
        task = int(tmd[4]) if len(tmd) > 4 and tmd[4] != 0 else 0
        modifier = int(tmd[8]) if len(tmd) > 8 and tmd[8] != 0 else 0

        logger.info(f"  [{source:8s}] {concept_text[:50]:50s} -> d={domain}, t={task}, m={modifier}")

    logger.info(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description="Fix ontology TMD classification")
    parser.add_argument("--mode", choices=["classify", "update-db", "verify", "all"],
                       default="all", help="Operation mode")
    parser.add_argument("--limit", type=int, help="Limit number of concepts (for testing)")
    parser.add_argument("--output", default="tmd_classifications.json",
                       help="Output file for classifications")
    args = parser.parse_args()

    # Connect to database
    conn = psycopg2.connect("dbname=lnsp")

    try:
        if args.mode in ["classify", "all"]:
            results = classify_all_concepts(conn, args.limit)

            # Save results
            output_path = ROOT / "artifacts" / args.output
            output_path.parent.mkdir(exist_ok=True)
            with open(output_path, "w") as f:
                json.dump(results, f, indent=2)
            logger.info(f"Saved classifications to {output_path}")

            if args.mode == "all":
                update_database(conn, results)

        if args.mode == "update-db":
            # Load from file
            input_path = ROOT / "artifacts" / args.output
            with open(input_path) as f:
                results = json.load(f)
            update_database(conn, results)

        if args.mode in ["verify", "all"]:
            verify_results(conn)

        logger.info("✅ All operations completed successfully")

    finally:
        conn.close()


if __name__ == "__main__":
    main()
