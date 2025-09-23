#!/usr/bin/env python3
"""
Creates a 10,000-item sample from the main FactoidWiki dataset.
"""

import json
from pathlib import Path

def create_sample(input_path, output_path, num_items=10000):
    """
    Reads the first `num_items` from the input JSONL file and writes them to the output file.
    """
    input_file = Path(input_path)
    output_file = Path(output_path)

    if not input_file.exists():
        print(f"Error: Input file not found at {input_file}")
        return

    output_file.parent.mkdir(parents=True, exist_ok=True)

    records = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= num_items:
                break
            record = json.loads(line)
            # The audit script expects 'doc_id', but the raw data has 'id'.
            # Let's create the 'doc_id' field.
            if 'id' in record and 'doc_id' not in record:
                record['doc_id'] = record['id']
            records.append(record)

    with open(output_file, 'w', encoding='utf-8') as f:
        for record in records:
            f.write(json.dumps(record) + '\n')

    print(f"Successfully created sample with {len(records)} items at {output_file}")


def main():
    root_dir = Path(__file__).parent.parent.parent
    input_dataset_path = root_dir / "data/datasets/factoid-wiki-large/factoid_wiki.jsonl"
    output_artifact_path = root_dir / "artifacts/fw10k_chunks.jsonl"

    create_sample(input_dataset_path, output_artifact_path, num_items=10000)

if __name__ == "__main__":
    main()
