#!/usr/bin/env python3
"""
Download FactoidWiki (dataset-factoid-curated) dataset.

This script downloads the curated factoid question answering dataset
from the brmson/dataset-factoid-curated GitHub repository.
"""

import os
import sys
import json
import requests
from pathlib import Path
from typing import Optional
import argparse


def download_file(url: str, dest_path: Path) -> bool:
    """Download a file from URL to destination path."""
    try:
        print(f"Downloading: {url}")
        response = requests.get(url, stream=True)
        response.raise_for_status()

        # Get file size if available
        total_size = int(response.headers.get('content-length', 0))

        # Write file with progress
        with open(dest_path, 'wb') as f:
            downloaded = 0
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        print(f"Progress: {percent:.1f}%", end='\r')

        print(f"\n✅ Downloaded: {dest_path.name}")
        return True

    except requests.exceptions.RequestException as e:
        print(f"❌ Error downloading {url}: {e}")
        return False


def download_factoid_dataset(output_dir: Optional[str] = None) -> None:
    """
    Download the factoid-curated dataset from GitHub.

    Args:
        output_dir: Directory to save the dataset.
                   Defaults to ../data/datasets/factoid-wiki
    """
    # Set default output directory
    if output_dir is None:
        output_dir = Path(__file__).parent.parent / "data" / "datasets" / "factoid-wiki"
    else:
        output_dir = Path(output_dir)

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"📁 Output directory: {output_dir}")

    # GitHub raw content base URL
    base_url = "https://raw.githubusercontent.com/brmson/dataset-factoid-curated/master"

    # Files to download
    files = {
        "curated-full.tsv": "Complete dataset (all questions)",
        "curated-train.tsv": "Training split",
        "curated-test.tsv": "Test split (blind evaluation)",
        "large2470/large2470-train.tsv": "Larger noisy training set (~2470 questions)",
        "large2470/large2470-test.tsv": "Larger noisy test set",
        "README.md": "Dataset documentation"
    }

    downloaded_files = []
    total_size = 0

    print("\n📥 Downloading dataset files...")
    print("-" * 50)

    for file_path, description in files.items():
        print(f"\n📄 {file_path}: {description}")

        # Create subdirectory if needed
        if "/" in file_path:
            subdir = output_dir / Path(file_path).parent
            subdir.mkdir(parents=True, exist_ok=True)
            dest_path = output_dir / file_path
        else:
            dest_path = output_dir / file_path

        # Download file
        url = f"{base_url}/{file_path}"
        if download_file(url, dest_path):
            downloaded_files.append(dest_path)
            if dest_path.exists():
                size = dest_path.stat().st_size
                total_size += size
                print(f"   Size: {size:,} bytes")

    print("\n" + "=" * 50)
    print("📊 Download Summary:")
    print(f"   Files downloaded: {len(downloaded_files)}/{len(files)}")
    print(f"   Total size: {total_size:,} bytes ({total_size/1024/1024:.2f} MB)")
    print(f"   Location: {output_dir}")

    # Load and display basic statistics
    if (output_dir / "curated-full.tsv").exists():
        print("\n📈 Dataset Statistics:")
        try:
            # Read TSV file without pandas
            with open(output_dir / "curated-full.tsv", 'r', encoding='utf-8') as f:
                lines_full = f.readlines()
            print(f"   Full dataset rows: {len(lines_full)}")

            if (output_dir / "curated-train.tsv").exists():
                with open(output_dir / "curated-train.tsv", 'r', encoding='utf-8') as f:
                    lines_train = f.readlines()
                print(f"   Training set rows: {len(lines_train)}")

            if (output_dir / "curated-test.tsv").exists():
                with open(output_dir / "curated-test.tsv", 'r', encoding='utf-8') as f:
                    lines_test = f.readlines()
                print(f"   Test set rows: {len(lines_test)}")

            # Show sample questions
            print("\n📝 Sample questions from full dataset:")
            for i in range(min(3, len(lines_full))):
                parts = lines_full[i].strip().split('\t')
                question = parts[0] if parts else "N/A"
                print(f"   {i+1}. {question}")

        except Exception as e:
            print(f"   ⚠️  Could not load dataset statistics: {e}")

    print("\n✅ Dataset download complete!")
    print("\n📚 Note: This is the dataset-factoid-curated dataset from brmson/YodaQA,")
    print("   used for benchmarking factoid question answering systems.")
    print("   The questions are based on Wikipedia knowledge.")


def main():
    parser = argparse.ArgumentParser(
        description="Download FactoidWiki (dataset-factoid-curated) dataset"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: ../data/datasets/factoid-wiki)"
    )

    args = parser.parse_args()

    # Use the specified path from the user if not provided via argument
    if args.output_dir is None:
        default_path = "/Users/trentcarter/Artificial_Intelligence/AI_Projects/lnsp-phase-4/data/datasets/factoid-wiki"
        print(f"Using default output directory: {default_path}")
        download_factoid_dataset(default_path)
    else:
        download_factoid_dataset(args.output_dir)


if __name__ == "__main__":
    main()