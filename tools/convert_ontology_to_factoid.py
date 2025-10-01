#!/usr/bin/env python3
"""
Convert ontology chains to FactoidWiki-compatible format.

This allows us to use the existing ingest_factoid.py pipeline
without modification.

Strategy:
- Convert chain concepts to a natural language "contents" field
- Create synthetic factoid-like items
- Use existing CPE extraction (which will preserve the chain structure)

Usage:
    python tools/convert_ontology_to_factoid.py \
        --input artifacts/ontology_chains/swo_chains.jsonl \
        --output data/ontology_factoids/swo_factoids.jsonl
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def chain_to_factoid(chain: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert an ontology chain to FactoidWiki-compatible format.
    
    Args:
        chain: {
            "chain_id": "swo_chain_001",
            "concepts": ["software", "analysis software", "blast"],
            "source": "swo",
            "chain_length": 3
        }
    
    Returns:
        FactoidWiki-compatible item:
        {
            "id": "swo_chain_001",
            "contents": "BLAST is a type of analysis software, which is a type of software."
        }
    """
    concepts = chain["concepts"]
    chain_id = chain["chain_id"]
    
    # Create natural language description of the chain
    # Strategy: "X is a type of Y, which is a type of Z"
    if len(concepts) == 2:
        contents = f"{concepts[1]} is a type of {concepts[0]}."
    elif len(concepts) == 3:
        contents = f"{concepts[2]} is a type of {concepts[1]}, which is a type of {concepts[0]}."
    else:
        # For longer chains: "D is C, C is B, B is A"
        parts = []
        for i in range(len(concepts) - 1, 0, -1):
            parts.append(f"{concepts[i]} is a type of {concepts[i-1]}")
        contents = ", and ".join(parts) + "."
    
    return {
        "id": chain_id,
        "contents": contents
    }


def convert_file(input_path: Path, output_path: Path, limit: int = None):
    """Convert an ontology chain file to FactoidWiki format."""
    
    logger.info(f"Converting {input_path} → {output_path}")
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    stats = {"total": 0, "converted": 0}
    
    with open(input_path, 'r') as fin, open(output_path, 'w') as fout:
        for line in fin:
            if limit and stats["total"] >= limit:
                break
            
            stats["total"] += 1
            
            try:
                chain = json.loads(line.strip())
                factoid = chain_to_factoid(chain)
                fout.write(json.dumps(factoid) + "\n")
                stats["converted"] += 1
                
            except Exception as e:
                logger.error(f"Error converting line {stats['total']}: {e}")
    
    logger.info(f"✓ Converted {stats['converted']}/{stats['total']} chains")
    logger.info(f"  Output: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert ontology chains to FactoidWiki format"
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, help="Limit number of chains")
    
    args = parser.parse_args()
    
    if not args.input.exists():
        logger.error(f"Input file not found: {args.input}")
        return 1
    
    convert_file(args.input, args.output, args.limit)
    return 0


if __name__ == "__main__":
    exit(main())
