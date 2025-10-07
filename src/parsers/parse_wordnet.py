#!/usr/bin/env python3
"""
Extract WordNet hypernym chains for LNSP ontology ingestion.

Generates JSONL file compatible with ingest_ontology_simple.py:
- Hypernym paths (word → hypernym → ... → entity)
- Multiple chains per synset (if multiple hypernym paths exist)
- Balanced sampling across semantic categories
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Set
import nltk
from nltk.corpus import wordnet as wn

def get_hypernym_chain(synset) -> List[str]:
    """Get hypernym path from synset to root (entity)."""
    paths = synset.hypernym_paths()
    if not paths:
        return [synset.name().split('.')[0]]

    # Use longest path (most specific)
    longest_path = max(paths, key=len)

    # Extract just the word names (remove POS tags)
    chain = [s.name().split('.')[0] for s in longest_path[::-1]]  # Reverse: root → leaf
    return chain

def extract_wordnet_chains(
    max_chains: int = 2000,
    min_depth: int = 3,
    max_depth: int = 15,
    pos_filters: Set[str] = None
) -> List[Dict]:
    """
    Extract diverse WordNet hypernym chains.

    Args:
        max_chains: Maximum number of chains to extract
        min_depth: Minimum chain length
        max_depth: Maximum chain length
        pos_filters: Part-of-speech tags to include (n, v, a, r)

    Returns:
        List of chain dictionaries compatible with LNSP format
    """
    if pos_filters is None:
        pos_filters = {'n', 'v', 'a'}  # nouns, verbs, adjectives (skip adverbs)

    print(f"Extracting WordNet chains (max={max_chains}, depth={min_depth}-{max_depth})...")

    chains = []
    chain_id = 0

    # Sample synsets by category for diversity
    categories = defaultdict(list)

    for synset in wn.all_synsets():
        pos = synset.pos()
        if pos not in pos_filters:
            continue

        chain = get_hypernym_chain(synset)

        # Filter by depth
        if not (min_depth <= len(chain) <= max_depth):
            continue

        # Skip single-concept chains
        if len(chain) < 2:
            continue

        # Categorize by top-level hypernym
        top_level = chain[1] if len(chain) > 1 else chain[0]
        categories[top_level].append({
            'synset': synset,
            'chain': chain,
            'depth': len(chain)
        })

    print(f"Found {sum(len(v) for v in categories.values())} valid chains across {len(categories)} categories")

    # Sample evenly across categories
    chains_per_category = max(1, max_chains // len(categories))

    for category, synsets in sorted(categories.items()):
        # Sort by depth for diversity (prefer deeper chains)
        synsets.sort(key=lambda x: x['depth'], reverse=True)

        for item in synsets[:chains_per_category]:
            if len(chains) >= max_chains:
                break

            chain_id += 1
            chains.append({
                'chain_id': f'wordnet_chain_{chain_id:06d}',
                'concepts': item['chain'],
                'source': 'wordnet',
                'chain_length': len(item['chain']),
                'metadata': {
                    'root': item['chain'][0],
                    'depth': item['depth'] - 1,
                    'synset': item['synset'].name(),
                    'pos': item['synset'].pos(),
                    'category': category
                }
            })

        if len(chains) >= max_chains:
            break

    print(f"✓ Extracted {len(chains)} chains")
    return chains

def main():
    parser = argparse.ArgumentParser(description='Extract WordNet hypernym chains')
    parser.add_argument('--output', default='artifacts/ontology_chains/wordnet_chains.jsonl',
                        help='Output JSONL file')
    parser.add_argument('--max-chains', type=int, default=2000,
                        help='Maximum number of chains to extract')
    parser.add_argument('--min-depth', type=int, default=3,
                        help='Minimum chain length')
    parser.add_argument('--max-depth', type=int, default=15,
                        help='Maximum chain length')
    parser.add_argument('--pos', nargs='+', default=['n', 'v', 'a'],
                        help='Part-of-speech tags to include (n=noun, v=verb, a=adj)')

    args = parser.parse_args()

    # Ensure WordNet is downloaded
    try:
        wn.all_synsets()
    except LookupError:
        print("Downloading WordNet corpus...")
        nltk.download('wordnet')
        nltk.download('omw-1.4')

    # Extract chains
    chains = extract_wordnet_chains(
        max_chains=args.max_chains,
        min_depth=args.min_depth,
        max_depth=args.max_depth,
        pos_filters=set(args.pos)
    )

    # Write to JSONL
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        for chain in chains:
            f.write(json.dumps(chain) + '\n')

    print(f"\n✓ Wrote {len(chains)} chains to {output_path}")
    print(f"  File size: {output_path.stat().st_size / 1024:.1f} KB")

    # Show sample
    print(f"\n=== Sample Chains ===")
    for i, chain in enumerate(chains[:5], 1):
        concepts = ' → '.join(chain['concepts'])
        print(f"{i}. [{chain['metadata']['pos']}] {concepts}")

if __name__ == '__main__':
    main()
