#!/usr/bin/env python3
"""
Download Wikipedia dataset locally for sequential training.

Downloads Simple English Wikipedia articles to data/datasets/wikipedia/
for LVM training (sequential document data, not ontologies).

Usage:
    # Download 100 articles (pilot)
    ./.venv/bin/python tools/download_wikipedia.py --limit 100

    # Download 3000 articles (full)
    ./.venv/bin/python tools/download_wikipedia.py --limit 3000
"""

import argparse
import json
import os
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm


def download_wikipedia(
    output_dir: str = "data/datasets/wikipedia",
    limit: int = 100,
    language: str = "simple",  # Simple English
    min_length: int = 500,  # Skip stubs
):
    """Download Wikipedia articles locally."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"📥 Downloading {language} Wikipedia (limit={limit}, min_length={min_length})")
    print(f"   Output: {output_path}")

    # Load dataset from HuggingFace
    # Using wikimedia/wikipedia with Simple English
    print(f"\n1. Loading dataset from HuggingFace...")

    # For Simple English, use "20231101.simple" (date.language format)
    dataset = load_dataset(
        "wikimedia/wikipedia",
        f"20231101.{language}",
        split="train",
        streaming=True  # Stream to avoid loading all at once
    )

    print(f"2. Filtering and saving articles...")

    articles_saved = 0
    articles_skipped = 0
    output_file = output_path / f"wikipedia_{language}_articles.jsonl"

    with open(output_file, 'w') as f:
        for article in tqdm(dataset, total=limit, desc="Downloading"):
            # Skip short articles (stubs)
            if len(article['text']) < min_length:
                articles_skipped += 1
                continue

            # Save article
            record = {
                'title': article['title'],
                'text': article['text'],
                'url': article.get('url', ''),
                'id': article.get('id', ''),
            }
            f.write(json.dumps(record) + '\n')
            articles_saved += 1

            if articles_saved >= limit:
                break

    print(f"\n✅ Download complete!")
    print(f"   Articles saved: {articles_saved}")
    print(f"   Articles skipped: {articles_skipped} (too short)")
    print(f"   Output file: {output_file}")

    # Create metadata
    metadata = {
        'source': 'wikimedia/wikipedia',
        'language': language,
        'articles_count': articles_saved,
        'min_length': min_length,
        'output_file': str(output_file)
    }

    metadata_file = output_path / "metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"   Metadata: {metadata_file}")

    return str(output_file)


def main():
    parser = argparse.ArgumentParser(description="Download Wikipedia for LVM training")
    parser.add_argument("--limit", type=int, default=100, help="Number of articles to download")
    parser.add_argument("--language", type=str, default="simple", help="Wikipedia language (simple/en)")
    parser.add_argument("--min-length", type=int, default=500, help="Minimum article length (chars)")
    parser.add_argument("--output-dir", type=str, default="data/datasets/wikipedia", help="Output directory")

    args = parser.parse_args()

    download_wikipedia(
        output_dir=args.output_dir,
        limit=args.limit,
        language=args.language,
        min_length=args.min_length
    )


if __name__ == "__main__":
    main()
