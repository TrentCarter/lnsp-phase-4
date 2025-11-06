#!/usr/bin/env python3
"""
Simplified arXiv ingestion using direct text splitting (no chunker service).

V2: Now includes robust pre-cleaning to filter non-prose content:
  - Headers/footers (arXiv IDs, page numbers)
  - Figure/Table/Algorithm captions
  - Pseudo-code blocks
  - ASCII tables and flowcharts
  - Section headers (References, Acknowledgments)

This prevents "franken-chunks" that would poison LVM training with
non-conceptual vectors. See 11/04 data quality review for details.

Usage:
  python tools/ingest_arxiv_to_npz_simple.py \
    --input data/datasets/arxiv/arxiv_cs_lg_ml.jsonl.gz \
    --output artifacts/lvm/arxiv_papers_768d.npz \
    --max-papers 210 \
    --encoder-url http://localhost:7001/encode
"""

import argparse
import gzip
import json
import re
import sys
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
import requests
from tqdm import tqdm


def load_arxiv_jsonl(jsonl_path: str, max_papers: int = None) -> List[Dict[str, Any]]:
    """Load arXiv papers from JSONL.gz file."""
    papers = []
    open_fn = gzip.open if jsonl_path.endswith('.gz') else open

    with open_fn(jsonl_path, 'rt', encoding='utf-8') as f:
        for line in f:
            if max_papers and len(papers) >= max_papers:
                break
            paper = json.loads(line)
            papers.append(paper)

    print(f"Loaded {len(papers)} papers from {jsonl_path}")
    return papers


def load_fulltext(fulltext_path: str) -> str:
    """Load full text from .txt file."""
    try:
        with open(fulltext_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        return ""


def _clean_and_reformat_text(raw_text: str) -> str:
    """
    Applies robust regex filters to remove non-prose content.
    This addresses "franken-chunks" identified in the 11/04 review
    by filtering headers, footers, tables, pseudo-code, and captions
    before the chunking logic is applied.

    FIX (2025-11-04): Split on sentence boundaries FIRST to handle
    single-line PDF extractions (no newlines). This prevents entire
    papers from being rejected as "ASCII art".
    """
    # CRITICAL FIX: Split on sentence boundaries FIRST
    # This handles PDF extractions that are single-line files
    # Split on: periods, newlines, and multiple spaces
    lines = re.split(r'(?<=[.!?])\s+|\n+|\s{2,}', raw_text)

    good_lines = []
    for line in lines:
        line_stripped = line.strip()

        # 1. Filter out empty lines
        if not line_stripped:
            continue

        # 2. Filter out headers/footers/metadata (from review)
        # e.g., "arXiv:2510.27688v1 [cs.CL] 31 Oct 2025", "13 Preprint"
        if re.match(r'^arXiv:\d{4}\.\d{4,5}v\d+', line_stripped, re.I):
            continue
        if re.match(r'^\d+\s+Preprint', line_stripped, re.I):
            continue
        if line_stripped == "Preprint":
            continue

        # 3. Filter out figure/table/algorithm captions (from review)
        # e.g., "Figure 1: Comparison between...", "Table 1: Performance..."
        if re.match(r'^(Figure|Table|Algorithm) \d+[:.]', line_stripped, re.I):
            continue

        # 4. Filter out pseudo-code (from review)
        # e.g., "1: procedure SAMPLE...", "Input:", "Output:"
        if re.match(r'^\d+:\s+(Input|Output|procedure|return|if|else|foreach|loop|end)', line_stripped, re.I):
            continue
        if re.match(r'^(Input|Output):', line_stripped, re.I):
            continue

        # 5. Filter out table-like structures (from review)
        # Lines with many | or --- or ===
        if line_stripped.count('|') > 3 or line_stripped.count('---') > 2 or line_stripped.count('===') > 2:
            continue
        # Lines that are *only* separators
        if re.match(r'^[|\s\-+=\.]+$', line_stripped) and len(line_stripped) > 5:
            continue

        # 6. Filter out code-like lines or ASCII art (from review)
        # Heuristic: if less than 60% of chars are alphanumeric, it's likely code/art
        alphanumeric_chars = sum(c.isalnum() for c in line_stripped)
        total_chars = len(line_stripped)
        if total_chars > 20 and (alphanumeric_chars / total_chars) < 0.6:
            continue

        # 7. Filter out common section headers that are not conceptual prose
        lower_line = line_stripped.lower()
        if lower_line in ["references", "acknowledgments", "appendix", "a proof",
                          "proof of theorem 1", "proof of theorem 2", "proof of theorem 3"]:
            continue
        if re.match(r'^a\.\d proof of', lower_line):
            continue

        # If all checks pass, keep the line
        good_lines.append(line_stripped)

    # Re-join good lines into paragraphs, separated by our marker
    # This becomes the input for the next splitting phase.
    return "||PARA||".join(good_lines)


def simple_chunk_text(text: str, target_size: int = 400) -> List[str]:
    """
    Simple text chunking using paragraphs and sentences.
    V2: Now includes robust pre-cleaning to filter non-prose.

    Strategy:
    1. Pre-clean text to remove tables, code, headers, captions.
    2. Split on "||PARA||" markers (paragraphs)
    3. If paragraph > target_size, split on sentences
    4. Combine small chunks to reach target_size
    """

    # 1. Pre-clean and reformat the text
    # This removes headers, footers, tables, captions, and pseudo-code.
    # It returns a single string with "||PARA||" as the paragraph separator.
    clean_text = _clean_and_reformat_text(text)

    # 2. Collapse all remaining whitespace
    clean_text = re.sub(r'\s+', ' ', clean_text)

    # 3. Split into paragraphs
    paragraphs = [p.strip() for p in clean_text.split('||PARA||') if p.strip()]

    chunks = []
    current_chunk = ""

    for para in paragraphs:
        # If paragraph is too long, split on sentences
        if len(para) > target_size * 2:
            sentences = re.split(r'(?<=[.!?])\s+', para)
            for sent in sentences:
                if len(current_chunk) + len(sent) < target_size:
                    current_chunk += " " + sent
                else:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = sent
        else:
            # Add paragraph to current chunk
            if len(current_chunk) + len(para) < target_size:
                current_chunk += " " + para
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = para

    # Add final chunk
    if current_chunk:
        chunks.append(current_chunk.strip())

    # Filter out very short chunks
    chunks = [c for c in chunks if len(c) > 50]

    return chunks


def embed_chunks_batched(chunks: List[str], encoder_url: str, batch_size: int = 32) -> np.ndarray:
    """Embed chunks using GTR-T5 encoder with batching."""
    if not chunks:
        return np.empty((0, 768), dtype=np.float32)

    all_embeddings = []

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i+batch_size]
        try:
            response = requests.post(
                encoder_url,
                json={"texts": batch},
                timeout=120
            )
            response.raise_for_status()
            embeddings = response.json()["embeddings"]
            all_embeddings.extend(embeddings)
        except Exception as e:
            print(f"  [ERROR] Batch {i//batch_size} failed: {e}", file=sys.stderr)
            # Add zero vectors for failed batch
            all_embeddings.extend([[0.0] * 768] * len(batch))

    return np.array(all_embeddings, dtype=np.float32)


def process_papers(
    papers: List[Dict[str, Any]],
    encoder_url: str,
    min_chunks_per_paper: int = 5,
    target_chunk_size: int = 400
) -> Dict[str, Any]:
    """Process all papers into vectors with metadata."""
    all_vectors = []
    all_article_ids = []
    all_concept_texts = []
    arxiv_metadata = {}

    papers_processed = 0
    papers_skipped = 0

    for paper in tqdm(papers, desc="Processing arXiv papers"):
        arxiv_id = paper.get("arxiv_id", "unknown")
        fulltext_path = paper.get("fulltext_path")

        if not fulltext_path:
            papers_skipped += 1
            continue

        # Load full text
        fulltext = load_fulltext(fulltext_path)
        if not fulltext or len(fulltext) < 500:
            papers_skipped += 1
            continue

        # Simple chunking
        chunks = simple_chunk_text(fulltext, target_size=target_chunk_size)
        if len(chunks) < min_chunks_per_paper:
            papers_skipped += 1
            continue

        # Embed chunks
        vectors = embed_chunks_batched(chunks, encoder_url, batch_size=32)
        if len(vectors) == 0 or np.all(vectors == 0):
            papers_skipped += 1
            continue

        # Store results
        all_vectors.append(vectors)
        all_article_ids.extend([arxiv_id] * len(vectors))
        all_concept_texts.extend(chunks)

        # Store metadata
        arxiv_metadata[arxiv_id] = {
            "title": paper.get("title", ""),
            "authors": paper.get("authors", []),
            "categories": paper.get("categories", []),
            "published": paper.get("published", ""),
            "num_chunks": len(chunks)
        }

        papers_processed += 1

    # Concatenate all vectors
    if all_vectors:
        all_vectors = np.vstack(all_vectors)
    else:
        all_vectors = np.empty((0, 768), dtype=np.float32)

    print(f"\nProcessing complete:")
    print(f"  Papers processed: {papers_processed}")
    print(f"  Papers skipped: {papers_skipped}")
    print(f"  Total vectors: {len(all_vectors)}")
    print(f"  Avg vectors/paper: {len(all_vectors)/papers_processed if papers_processed > 0 else 0:.1f}")

    return {
        "vectors": all_vectors,
        "article_ids": np.array(all_article_ids, dtype=object),
        "concept_texts": np.array(all_concept_texts, dtype=object),
        "arxiv_metadata": arxiv_metadata
    }


def save_npz(data: Dict[str, Any], output_path: str):
    """Save processed data to NPZ file."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        output_path,
        vectors=data["vectors"],
        article_ids=data["article_ids"],
        concept_texts=data["concept_texts"],
        arxiv_metadata=np.array([data["arxiv_metadata"]], dtype=object)[0]
    )

    print(f"\nSaved to: {output_path}")
    print(f"  Vectors: {data['vectors'].shape}")
    print(f"  Articles: {len(data['arxiv_metadata'])}")


def main():
    parser = argparse.ArgumentParser(description="Ingest arXiv papers to NPZ (simplified)")
    parser.add_argument("--input", required=True, help="Input JSONL.gz file")
    parser.add_argument("--output", required=True, help="Output NPZ file path")
    parser.add_argument("--max-papers", type=int, default=None, help="Maximum papers to process")
    parser.add_argument("--encoder-url", default="http://localhost:7001/encode",
                        help="GTR-T5 encoder endpoint")
    parser.add_argument("--min-chunks-per-paper", type=int, default=5,
                        help="Minimum chunks required per paper")
    parser.add_argument("--target-chunk-size", type=int, default=400,
                        help="Target characters per chunk")

    args = parser.parse_args()

    # Load papers
    papers = load_arxiv_jsonl(args.input, args.max_papers)
    if not papers:
        print("ERROR: No papers loaded!", file=sys.stderr)
        sys.exit(1)

    # Process papers
    print(f"\nProcessing with:")
    print(f"  Encoder: {args.encoder_url}")
    print(f"  Chunking: V2 paragraph/sentence splitting (with pre-cleaning)")
    print(f"  Target chunk size: {args.target_chunk_size} chars")
    print(f"  Min chunks/paper: {args.min_chunks_per_paper}")
    print(f"  Pre-cleaning: Filters tables, pseudo-code, headers, captions")

    data = process_papers(
        papers,
        args.encoder_url,
        args.min_chunks_per_paper,
        args.target_chunk_size
    )

    if len(data["vectors"]) == 0:
        print("ERROR: No vectors generated!", file=sys.stderr)
        sys.exit(1)

    # Save NPZ
    save_npz(data, args.output)

    print("\n✓ arXiv ingestion complete!")
    print(f"\nNext steps:")
    print(f"  1. Create training sequences:")
    print(f"     python tools/create_sequences_from_npz.py \\")
    print(f"       --input {args.output} \\")
    print(f"       --output artifacts/lvm/arxiv_sequences_ctx5.npz \\")
    print(f"       --context-size 5")
    print(f"")
    print(f"  2. Measure Δ (data quality):")
    print(f"     python tools/tests/diagnose_data_direction.py \\")
    print(f"       artifacts/lvm/arxiv_sequences_ctx5.npz --n-samples 5000")


if __name__ == "__main__":
    main()
