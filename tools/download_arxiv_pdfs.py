#!/usr/bin/env python3
"""
Download missing arXiv PDFs and extract text.

Usage:
  python tools/download_arxiv_pdfs.py \
    --input data/datasets/arxiv/arxiv_cs_lg_ml.jsonl.gz \
    --output data/datasets/arxiv/arxiv_cs_lg_ml_complete.jsonl.gz \
    --max-downloads 100 \
    --pdf-dir data/datasets/arxiv/pdfs
"""

import argparse
import gzip
import json
import time
from pathlib import Path
from typing import Dict, Any, List
import requests
import pdfplumber
from tqdm import tqdm


def load_papers(jsonl_path: str) -> List[Dict[str, Any]]:
    """Load papers from JSONL.gz file."""
    papers = []
    open_fn = gzip.open if jsonl_path.endswith('.gz') else open

    with open_fn(jsonl_path, 'rt', encoding='utf-8') as f:
        for line in f:
            papers.append(json.loads(line))

    return papers


def download_pdf(pdf_url: str, output_path: str, timeout: int = 30) -> bool:
    """Download PDF from URL."""
    try:
        response = requests.get(pdf_url, timeout=timeout, stream=True)
        response.raise_for_status()

        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        return True
    except Exception as e:
        print(f"  [ERROR] Failed to download {pdf_url}: {e}")
        return False


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from PDF using pdfplumber."""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            text_parts = []
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    text_parts.append(text)

            return "\n".join(text_parts)
    except Exception as e:
        print(f"  [ERROR] Failed to extract text from {pdf_path}: {e}")
        return ""


def process_papers(
    papers: List[Dict[str, Any]],
    pdf_dir: str,
    max_downloads: int = None,
    delay: float = 1.0
) -> List[Dict[str, Any]]:
    """
    Download missing PDFs and extract text.

    Args:
        papers: List of paper metadata
        pdf_dir: Directory to save PDFs and text files
        max_downloads: Maximum number of papers to download (None = all)
        delay: Delay between downloads in seconds (respect arXiv rate limits)

    Returns:
        Updated list of papers with fulltext_path added
    """
    pdf_dir_path = Path(pdf_dir)
    pdf_dir_path.mkdir(parents=True, exist_ok=True)

    # Filter papers that need downloading
    papers_to_download = []
    for paper in papers:
        fulltext_path = paper.get('fulltext_path')
        if not fulltext_path or not Path(fulltext_path).exists():
            papers_to_download.append(paper)

    if max_downloads:
        papers_to_download = papers_to_download[:max_downloads]

    print(f"\nFound {len(papers_to_download)} papers to download (max: {max_downloads or 'all'})")

    downloaded = 0
    skipped = 0
    failed = 0

    for paper in tqdm(papers_to_download, desc="Downloading arXiv papers"):
        arxiv_id = paper.get('arxiv_id', 'unknown')
        pdf_url = paper.get('links', {}).get('pdf')

        if not pdf_url:
            skipped += 1
            continue

        # Download PDF
        pdf_path = pdf_dir_path / f"{arxiv_id}.pdf"
        txt_path = pdf_dir_path / f"{arxiv_id}.txt"

        # Skip if text already exists
        if txt_path.exists():
            paper['fulltext_path'] = str(txt_path)
            downloaded += 1
            continue

        # Download PDF
        if not download_pdf(pdf_url, str(pdf_path)):
            failed += 1
            time.sleep(delay)  # Still delay on failure
            continue

        # Extract text
        text = extract_text_from_pdf(str(pdf_path))
        if not text or len(text) < 500:
            print(f"  [WARN] {arxiv_id}: Extracted text too short ({len(text)} chars)")
            failed += 1
            # Clean up PDF
            pdf_path.unlink(missing_ok=True)
            time.sleep(delay)
            continue

        # Save text
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(text)

        # Update paper metadata
        paper['fulltext_path'] = str(txt_path)

        # Clean up PDF (we only need the text)
        pdf_path.unlink(missing_ok=True)

        downloaded += 1

        # Rate limiting (arXiv requests 3 seconds between requests)
        time.sleep(delay)

    print(f"\nDownload complete:")
    print(f"  Downloaded: {downloaded}")
    print(f"  Failed: {failed}")
    print(f"  Skipped: {skipped}")

    return papers


def save_papers(papers: List[Dict[str, Any]], output_path: str):
    """Save updated papers to JSONL.gz file."""
    open_fn = gzip.open if output_path.endswith('.gz') else open

    with open_fn(output_path, 'wt', encoding='utf-8') as f:
        for paper in papers:
            f.write(json.dumps(paper) + '\n')

    print(f"\nSaved updated metadata to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Download missing arXiv PDFs")
    parser.add_argument("--input", required=True, help="Input JSONL.gz file")
    parser.add_argument("--output", required=True, help="Output JSONL.gz file")
    parser.add_argument("--pdf-dir", default="data/datasets/arxiv/pdfs",
                        help="Directory to save PDFs and text")
    parser.add_argument("--max-downloads", type=int, default=None,
                        help="Maximum papers to download (None = all)")
    parser.add_argument("--delay", type=float, default=3.0,
                        help="Delay between downloads (seconds, arXiv recommends 3s)")

    args = parser.parse_args()

    print("=" * 80)
    print("ARXIV PDF DOWNLOADER")
    print("=" * 80)

    # Load papers
    print(f"\nLoading papers from {args.input}...")
    papers = load_papers(args.input)
    print(f"  Loaded {len(papers)} papers")

    # Count missing
    missing = sum(1 for p in papers if not p.get('fulltext_path') or not Path(p.get('fulltext_path', '')).exists())
    print(f"  Papers missing fulltext: {missing}")

    # Process papers
    print(f"\nStarting downloads (delay: {args.delay}s between requests)...")
    papers = process_papers(papers, args.pdf_dir, args.max_downloads, args.delay)

    # Save updated metadata
    save_papers(papers, args.output)

    # Final stats
    print("\n" + "=" * 80)
    print("DOWNLOAD COMPLETE")
    print("=" * 80)
    print(f"\nNext steps:")
    print(f"  1. Run ingestion with updated dataset:")
    print(f"     python tools/ingest_arxiv_to_npz_simple.py \\")
    print(f"       --input {args.output} \\")
    print(f"       --output artifacts/lvm/arxiv_papers_complete.npz \\")
    print(f"       --max-papers 1000")


if __name__ == "__main__":
    main()
