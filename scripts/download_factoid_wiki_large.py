#!/usr/bin/env python3
"""
Download the large FactoidWiki dataset from HuggingFace.

This script downloads the 10M+ row FactoidWiki-passage dataset
from chentong00/factoid-wiki-passage on HuggingFace.
"""

import os
import sys
from pathlib import Path
from typing import Optional
import argparse
import json


def download_large_factoid_dataset(output_dir: Optional[str] = None, sample_size: Optional[int] = None):
    """
    Download the large FactoidWiki dataset from HuggingFace.

    Args:
        output_dir: Directory to save the dataset
        sample_size: If specified, only download this many samples (for testing)
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("❌ Error: 'datasets' library not installed.")
        print("Please install it with: pip install datasets")
        return False

    # Set default output directory
    if output_dir is None:
        output_dir = Path(__file__).parent.parent / "data" / "datasets" / "factoid-wiki-large"
    else:
        output_dir = Path(output_dir)

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"📁 Output directory: {output_dir}")

    print("\n📥 Downloading FactoidWiki dataset from HuggingFace...")
    print("   Dataset: chentong00/factoid-wiki-passage")
    print("   Expected size: ~2.4 GB (10.3M rows)")

    try:
        # Load dataset
        if sample_size:
            print(f"   Loading first {sample_size} samples only...")
            dataset = load_dataset(
                "chentong00/factoid-wiki-passage",
                split=f"train[:{sample_size}]",
                cache_dir=output_dir / "cache"
            )
        else:
            print("   Loading full dataset (this may take a while)...")
            dataset = load_dataset(
                "chentong00/factoid-wiki-passage",
                split="train",
                cache_dir=output_dir / "cache"
            )

        print(f"\n✅ Dataset loaded successfully!")
        print(f"   Total rows: {len(dataset):,}")

        # Save to different formats
        print("\n💾 Saving dataset...")

        # Save as JSON Lines
        jsonl_path = output_dir / "factoid_wiki.jsonl"
        print(f"   Saving as JSONL to {jsonl_path}...")
        dataset.to_json(jsonl_path)

        # Save as Parquet (more efficient)
        parquet_path = output_dir / "factoid_wiki.parquet"
        print(f"   Saving as Parquet to {parquet_path}...")
        dataset.to_parquet(parquet_path)

        # Show statistics
        print("\n📊 Dataset Statistics:")
        print(f"   Total entries: {len(dataset):,}")

        # Check file sizes
        if jsonl_path.exists():
            jsonl_size = jsonl_path.stat().st_size
            print(f"   JSONL file size: {jsonl_size:,} bytes ({jsonl_size/1024/1024:.1f} MB)")

        if parquet_path.exists():
            parquet_size = parquet_path.stat().st_size
            print(f"   Parquet file size: {parquet_size:,} bytes ({parquet_size/1024/1024:.1f} MB)")

        # Show sample entries
        print("\n📝 Sample entries:")
        for i in range(min(3, len(dataset))):
            entry = dataset[i]
            print(f"\n   Entry {i+1}:")
            print(f"   ID: {entry['id']}")
            content_preview = entry['contents'][:200] + "..." if len(entry['contents']) > 200 else entry['contents']
            print(f"   Content: {content_preview}")
            if 'metadata' in entry:
                print(f"   Metadata: {entry['metadata']}")

        print("\n✅ Download complete!")
        print(f"   Dataset saved to: {output_dir}")

        return True

    except Exception as e:
        print(f"\n❌ Error downloading dataset: {e}")
        print("\nTroubleshooting tips:")
        print("1. Make sure you have enough disk space (~3-5 GB needed)")
        print("2. Check your internet connection")
        print("3. Try with a smaller sample first: --sample-size 1000")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Download the large FactoidWiki dataset from HuggingFace"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: ../data/datasets/factoid-wiki-large)"
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Only download this many samples (for testing)"
    )

    args = parser.parse_args()

    # Use specified path
    if args.output_dir is None:
        default_path = "/Users/trentcarter/Artificial_Intelligence/AI_Projects/lnsp-phase-4/data/datasets/factoid-wiki-large"
        print(f"Using default output directory: {default_path}")
        success = download_large_factoid_dataset(default_path, args.sample_size)
    else:
        success = download_large_factoid_dataset(args.output_dir, args.sample_size)

    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()