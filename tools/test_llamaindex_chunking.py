#!/usr/bin/env python3
"""
Test LlamaIndex SemanticSplitter (Simple mode) on arXiv papers.

Usage:
    python tools/test_llamaindex_chunking.py
"""

import os
import sys
from pathlib import Path
from typing import List, Dict
import json

# LlamaIndex imports
from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter


def load_arxiv_paper(txt_path: str) -> str:
    """Load arXiv paper text from file."""
    with open(txt_path, 'r', encoding='utf-8') as f:
        return f.read()


def chunk_with_simple_mode(text: str, paper_id: str, chunk_size: int = 512, chunk_overlap: int = 50) -> List[Dict]:
    """
    Chunk text using LlamaIndex's SentenceSplitter (Simple mode).

    Args:
        text: Full paper text
        paper_id: arXiv paper ID
        chunk_size: Target chunk size in tokens (default 512)
        chunk_overlap: Overlap between chunks in tokens (default 50)

    Returns:
        List of chunk dictionaries with metadata
    """
    # Create LlamaIndex Document
    doc = Document(
        text=text,
        metadata={"paper_id": paper_id}
    )

    # Initialize SentenceSplitter (Simple mode with sentence-level splitting)
    splitter = SentenceSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separator=" ",
        paragraph_separator="\n\n",
    )

    # Split into nodes (chunks)
    nodes = splitter.get_nodes_from_documents([doc])

    # Convert to simple format
    chunks = []
    for i, node in enumerate(nodes):
        chunks.append({
            "chunk_id": i,
            "paper_id": paper_id,
            "text": node.text,
            "char_count": len(node.text),
            "metadata": node.metadata
        })

    return chunks


def display_chunks(chunks: List[Dict], paper_id: str, max_preview: int = 200):
    """Display chunks with statistics."""
    print(f"\n{'='*80}")
    print(f"Paper: {paper_id}")
    print(f"{'='*80}")
    print(f"Total chunks: {len(chunks)}")

    if chunks:
        char_counts = [c['char_count'] for c in chunks]
        print(f"Avg chunk size: {sum(char_counts) / len(char_counts):.0f} chars")
        print(f"Min/Max: {min(char_counts)} / {max(char_counts)} chars")

    print(f"\n{'-'*80}")
    print("Chunk Preview:")
    print(f"{'-'*80}")

    for i, chunk in enumerate(chunks[:5]):  # Show first 5 chunks
        preview = chunk['text'][:max_preview]
        if len(chunk['text']) > max_preview:
            preview += "..."

        print(f"\n[Chunk {i}] ({chunk['char_count']} chars)")
        print(preview)

    if len(chunks) > 5:
        print(f"\n... ({len(chunks) - 5} more chunks)")


def main():
    # Get arXiv text files
    pdfs_dir = Path("data/datasets/arxiv/pdfs")

    if not pdfs_dir.exists():
        print(f"ERROR: Directory not found: {pdfs_dir}")
        sys.exit(1)

    txt_files = sorted(list(pdfs_dir.glob("*.txt")))

    if len(txt_files) == 0:
        print(f"ERROR: No .txt files found in {pdfs_dir}")
        sys.exit(1)

    print(f"Found {len(txt_files)} arXiv text files")
    print(f"Testing LlamaIndex SentenceSplitter (Simple mode) on 4 papers...")

    # Test on first 4 papers
    test_papers = txt_files[:4]

    # Chunk each paper
    all_results = []

    for txt_path in test_papers:
        paper_id = txt_path.stem  # e.g., "2510.25701v1"

        try:
            # Load paper text
            text = load_arxiv_paper(str(txt_path))

            if len(text) < 100:
                print(f"\nWARNING: {paper_id} is too short ({len(text)} chars), skipping")
                continue

            print(f"\nLoaded {paper_id}: {len(text)} chars")

            # Chunk with Simple mode
            chunks = chunk_with_simple_mode(text, paper_id)

            # Display results
            display_chunks(chunks, paper_id)

            # Store results
            all_results.append({
                "paper_id": paper_id,
                "original_length": len(text),
                "num_chunks": len(chunks),
                "chunks": chunks
            })

        except Exception as e:
            print(f"ERROR processing {paper_id}: {e}")
            continue

    # Summary statistics
    print(f"\n\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")

    if all_results:
        total_papers = len(all_results)
        total_chunks = sum(r['num_chunks'] for r in all_results)
        avg_chunks_per_paper = total_chunks / total_papers

        print(f"Papers processed: {total_papers}")
        print(f"Total chunks: {total_chunks}")
        print(f"Avg chunks per paper: {avg_chunks_per_paper:.1f}")

        # Save results to JSON for inspection
        output_file = "artifacts/lvm/llamaindex_chunking_test_results.json"
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        with open(output_file, 'w') as f:
            json.dump(all_results, f, indent=2)

        print(f"\nFull results saved to: {output_file}")
    else:
        print("No papers were successfully processed")


if __name__ == "__main__":
    main()
