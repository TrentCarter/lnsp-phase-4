#!/usr/bin/env python3
"""
Script to read and print the first 10 items from FactoidWiki dataset.
"""

import json
from pathlib import Path


def read_factoid_wiki(file_path, num_items=10):
    """
    Read and print the first n items from FactoidWiki JSONL file.

    Args:
        file_path: Path to the factoid_wiki.jsonl file
        num_items: Number of items to read and print (default: 10)
    """
    file_path = Path(file_path)

    if not file_path.exists():
        print(f"Error: File not found at {file_path}")
        return

    print(f"Reading first {num_items} items from FactoidWiki dataset\n")
    print("=" * 80)

    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= num_items:
                break

            item = json.loads(line.strip())

            print(f"\nItem #{i + 1}")
            print("-" * 40)
            print(f"ID: {item['id']}")
            print(f"Contents: {item['contents']}")

            if 'metadata' in item:
                print(f"Metadata:")
                print(f"  - Title span: {item['metadata'].get('title_span', 'N/A')}")
                print(f"  - Section span: {item['metadata'].get('section_span', 'N/A')}")
                print(f"  - Content span: {item['metadata'].get('content_span', 'N/A')}")

    print("\n" + "=" * 80)
    print(f"Successfully read {min(i + 1, num_items)} items from the dataset")


def main():
    # Path to the FactoidWiki dataset
    dataset_path = "/Users/trentcarter/Artificial_Intelligence/AI_Projects/lnsp-phase-4/data/datasets/factoid-wiki-large/factoid_wiki.jsonl"

    # Read and print first 10 items
    read_factoid_wiki(dataset_path, num_items=10)


if __name__ == "__main__":
    main()