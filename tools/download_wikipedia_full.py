#!/usr/bin/env python3
"""
Download Full Wikipedia Dataset from HuggingFace

Downloads the complete English Wikipedia dump (6.5M articles, ~25GB compressed).
Streams data to JSONL format for efficient processing.

Usage:
    # Download full Wikipedia
    python tools/download_wikipedia_full.py --language 20231101.en --limit 6500000

    # Download specific subset
    python tools/download_wikipedia_full.py --language 20231101.en --limit 100000
"""

import argparse
import json
from pathlib import Path
from tqdm import tqdm
from datasets import load_dataset


def download_full_wikipedia(
    dataset_name: str = "wikimedia/wikipedia",
    language: str = "20231101.en",
    output_path: str = "data/datasets/wikipedia/full_wikipedia.jsonl",
    limit: int = None,
    min_length: int = 500
):
    """
    Download full Wikipedia from HuggingFace and save to JSONL.

    Args:
        dataset_name: HuggingFace dataset identifier
        language: Wikipedia dump date + language (e.g., "20231101.en")
        output_path: Where to save the JSONL file
        limit: Maximum articles to download (None = all)
        min_length: Minimum article length in characters
    """

    print(f"📥 Downloading Wikipedia from HuggingFace")
    print(f"   Dataset: {dataset_name}")
    print(f"   Language: {language}")
    print(f"   Output: {output_path}")
    print(f"   Limit: {limit if limit else 'All articles'}")
    print(f"   Min length: {min_length} chars")
    print()

    # Create output directory
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Load dataset in streaming mode (doesn't download everything at once)
    print("1. Loading dataset from HuggingFace (streaming)...")
    try:
        dataset = load_dataset(
            dataset_name,
            language,
            split="train",
            streaming=True  # Stream instead of download all
        )
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        print(f"\nAvailable languages: Check https://huggingface.co/datasets/{dataset_name}")
        return

    # Process and save articles
    print("2. Processing and saving articles...")
    saved_count = 0
    skipped_count = 0

    with open(output_file, 'w', encoding='utf-8') as f:
        # Use tqdm for progress (unknown total in streaming mode)
        pbar = tqdm(desc="Articles", unit="article")

        for article in dataset:
            # Check if we've reached the limit
            if limit and saved_count >= limit:
                break

            # Extract fields
            title = article.get("title", "")
            text = article.get("text", "")
            url = article.get("url", "")
            article_id = article.get("id", "")

            # Filter short articles
            if len(text) < min_length:
                skipped_count += 1
                continue

            # Save to JSONL
            article_data = {
                "id": article_id,
                "title": title,
                "text": text,
                "url": url,
                "length": len(text)
            }
            f.write(json.dumps(article_data, ensure_ascii=False) + '\n')

            saved_count += 1
            pbar.update(1)
            pbar.set_postfix({
                "saved": saved_count,
                "skipped": skipped_count
            })

        pbar.close()

    print()
    print("✅ Download complete!")
    print(f"   Articles saved: {saved_count:,}")
    print(f"   Articles skipped: {skipped_count:,} (too short)")
    print(f"   Output file: {output_path}")
    print(f"   File size: {output_file.stat().st_size / 1024 / 1024 / 1024:.2f} GB")

    # Save metadata
    metadata = {
        "dataset": dataset_name,
        "language": language,
        "total_articles": saved_count,
        "skipped_articles": skipped_count,
        "min_length": min_length,
        "output_file": str(output_path)
    }
    metadata_path = output_file.parent / "full_wikipedia_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"   Metadata: {metadata_path}")


def main():
    parser = argparse.ArgumentParser(description="Download full Wikipedia from HuggingFace")
    parser.add_argument(
        "--dataset",
        default="wikimedia/wikipedia",
        help="HuggingFace dataset name"
    )
    parser.add_argument(
        "--language",
        default="20231101.en",
        help="Wikipedia dump date + language (e.g., 20231101.en)"
    )
    parser.add_argument(
        "--output",
        default="data/datasets/wikipedia/full_wikipedia.jsonl",
        help="Output JSONL file path"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum articles to download (default: all)"
    )
    parser.add_argument(
        "--min-length",
        type=int,
        default=500,
        help="Minimum article length in characters (default: 500)"
    )

    args = parser.parse_args()

    download_full_wikipedia(
        dataset_name=args.dataset,
        language=args.language,
        output_path=args.output,
        limit=args.limit,
        min_length=args.min_length
    )


if __name__ == "__main__":
    main()
