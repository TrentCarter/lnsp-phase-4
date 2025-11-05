#!/usr/bin/env python3
"""
Ingest arXiv papers from JSONL into NPZ training sequences for LVM.

Pipeline:
1. Read arXiv JSONL.gz files (metadata + fulltext paths)
2. Load full text from .txt files
3. Chunk using Episode Chunker (semantic boundaries)
4. Embed with GTR-T5 encoder (768D via port 7001)
5. Create NPZ with article-based splits for train/val/OOD

Output NPZ fields:
- vectors: (N, 768) float32 - GTR-T5 embeddings
- article_ids: (N,) string - arXiv IDs for article-based splitting
- concept_texts: (N,) string - chunk text content
- arxiv_metadata: dict - titles, authors, categories per article

Usage:
  python tools/ingest_arxiv_to_npz.py \
    --input data/datasets/arxiv/arxiv_cs_lg_ml.jsonl.gz \
    --output artifacts/lvm/arxiv_papers_768d.npz \
    --max-papers 210 \
    --encoder-url http://localhost:7001/encode

Requirements:
- Episode chunker service running on port 8900
- GTR-T5 encoder service running on port 7001
"""

import argparse
import gzip
import json
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
        print(f"  [WARN] Failed to load {fulltext_path}: {e}", file=sys.stderr)
        return ""


def chunk_text(text: str, chunker_url: str = "http://localhost:8900/chunk") -> List[str]:
    """Chunk text using Episode Chunker service."""
    try:
        response = requests.post(
            chunker_url,
            json={"text": text, "max_chunk_size": 512},
            timeout=60
        )
        response.raise_for_status()
        result = response.json()
        return result.get("chunks", [])
    except Exception as e:
        print(f"  [WARN] Chunking failed: {e}", file=sys.stderr)
        # Fallback: simple splitting
        sentences = text.split('. ')
        chunk_size = 3
        return ['. '.join(sentences[i:i+chunk_size]) + '.'
                for i in range(0, len(sentences), chunk_size)]


def embed_chunks(chunks: List[str], encoder_url: str) -> np.ndarray:
    """Embed chunks using GTR-T5 encoder (768D)."""
    if not chunks:
        return np.empty((0, 768), dtype=np.float32)

    try:
        response = requests.post(
            encoder_url,
            json={"texts": chunks},
            timeout=120
        )
        response.raise_for_status()
        embeddings = response.json()["embeddings"]
        return np.array(embeddings, dtype=np.float32)
    except Exception as e:
        print(f"  [ERROR] Embedding failed: {e}", file=sys.stderr)
        return np.empty((0, 768), dtype=np.float32)


def process_papers(
    papers: List[Dict[str, Any]],
    encoder_url: str,
    chunker_url: str,
    min_chunks_per_paper: int = 5
) -> Dict[str, Any]:
    """
    Process all papers into vectors with metadata.

    Returns:
        dict with keys: vectors, article_ids, concept_texts, arxiv_metadata
    """
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

        # Load and chunk full text
        fulltext = load_fulltext(fulltext_path)
        if not fulltext or len(fulltext) < 500:
            print(f"  [SKIP] {arxiv_id}: text too short ({len(fulltext)} chars)")
            papers_skipped += 1
            continue

        chunks = chunk_text(fulltext, chunker_url)
        if len(chunks) < min_chunks_per_paper:
            print(f"  [SKIP] {arxiv_id}: too few chunks ({len(chunks)} < {min_chunks_per_paper})")
            papers_skipped += 1
            continue

        # Embed chunks
        vectors = embed_chunks(chunks, encoder_url)
        if len(vectors) == 0:
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
    print(f"  Vector shape: {all_vectors.shape}")

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
        arxiv_metadata=np.array([data["arxiv_metadata"]], dtype=object)[0]  # Store dict
    )

    print(f"\nSaved to: {output_path}")
    print(f"  Vectors: {data['vectors'].shape}")
    print(f"  Articles: {len(data['arxiv_metadata'])}")


def main():
    parser = argparse.ArgumentParser(description="Ingest arXiv papers to NPZ for LVM training")
    parser.add_argument("--input", required=True, help="Input JSONL.gz file")
    parser.add_argument("--output", required=True, help="Output NPZ file path")
    parser.add_argument("--max-papers", type=int, default=None, help="Maximum papers to process")
    parser.add_argument("--encoder-url", default="http://localhost:7001/encode",
                        help="GTR-T5 encoder endpoint")
    parser.add_argument("--chunker-url", default="http://localhost:8900/chunk",
                        help="Episode chunker endpoint")
    parser.add_argument("--min-chunks-per-paper", type=int, default=5,
                        help="Minimum chunks required per paper")

    args = parser.parse_args()

    # Load papers
    papers = load_arxiv_jsonl(args.input, args.max_papers)
    if not papers:
        print("ERROR: No papers loaded!", file=sys.stderr)
        sys.exit(1)

    # Process papers
    print(f"\nProcessing with:")
    print(f"  Encoder: {args.encoder_url}")
    print(f"  Chunker: {args.chunker_url}")
    print(f"  Min chunks/paper: {args.min_chunks_per_paper}")

    data = process_papers(
        papers,
        args.encoder_url,
        args.chunker_url,
        args.min_chunks_per_paper
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
