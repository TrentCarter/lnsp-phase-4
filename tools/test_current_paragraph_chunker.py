#!/usr/bin/env python3
"""
Test the CURRENT paragraph chunker (custom implementation) against LlamaIndex modes.

This is the chunker that produced Δ = +0.18 in the arXiv validation.
"""

import re
import time
from pathlib import Path
from typing import List, Dict
import json


def simple_chunk_text(text: str, target_size: int = 400) -> List[str]:
    """
    Current custom paragraph chunker (from ingest_arxiv_to_npz_simple.py).

    Strategy:
    1. Split on double newlines (paragraphs)
    2. If paragraph > target_size*2, split on sentences
    3. Combine small chunks to reach target_size
    4. Filter out chunks < 50 chars
    """
    # Clean text
    text = re.sub(r'\s+', ' ', text)  # Collapse whitespace
    text = text.replace('\n\n', '||PARA||')  # Mark paragraphs
    text = re.sub(r'\s+', ' ', text)  # Collapse again

    # Split into paragraphs
    paragraphs = [p.strip() for p in text.split('||PARA||') if p.strip()]

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


def load_arxiv_paper(txt_path: str) -> str:
    """Load arXiv paper text from file."""
    with open(txt_path, 'r', encoding='utf-8') as f:
        return f.read()


def analyze_chunks(chunks: List[str], mode_name: str) -> Dict:
    """Calculate statistics for a set of chunks."""
    if not chunks:
        return {
            "mode": mode_name,
            "num_chunks": 0,
            "avg_chars": 0,
            "min_chars": 0,
            "max_chars": 0,
            "avg_words": 0,
            "status": "no_chunks"
        }

    char_counts = [len(c) for c in chunks]
    word_counts = [len(c.split()) for c in chunks]

    return {
        "mode": mode_name,
        "num_chunks": len(chunks),
        "avg_chars": sum(char_counts) / len(char_counts) if char_counts else 0,
        "min_chars": min(char_counts) if char_counts else 0,
        "max_chars": max(char_counts) if char_counts else 0,
        "avg_words": sum(word_counts) / len(word_counts) if word_counts else 0,
        "status": "success"
    }


def main():
    print("Testing CURRENT Paragraph Chunker (Custom Implementation)")
    print("="*100)

    # Get arXiv text files (same 4 papers as LlamaIndex test)
    pdfs_dir = Path("data/datasets/arxiv/pdfs")
    txt_files = sorted(list(pdfs_dir.glob("*.txt")))
    test_papers = txt_files[:4]

    all_results = []
    total_time = 0

    for txt_path in test_papers:
        paper_id = txt_path.stem

        try:
            text = load_arxiv_paper(str(txt_path))

            if len(text) < 100:
                print(f"\nWARNING: {paper_id} too short, skipping")
                continue

            print(f"\n{'='*100}")
            print(f"Paper: {paper_id} ({len(text):,} chars)")
            print(f"{'='*100}")

            # Test current paragraph chunker
            print("\nTesting Current Paragraph Chunker (target_size=400)...")
            start = time.time()
            chunks = simple_chunk_text(text, target_size=400)
            elapsed = time.time() - start
            total_time += elapsed

            stats = analyze_chunks(chunks, "Current Paragraph")

            print(f"  ✓ {len(chunks)} chunks in {elapsed:.4f}s")
            print(f"  Avg: {stats['avg_chars']:.0f} chars, Min: {stats['min_chars']}, Max: {stats['max_chars']}")
            print(f"  Sample chunk 0: \"{chunks[0][:100]}...\"")
            if len(chunks) > 1:
                print(f"  Sample chunk 1: \"{chunks[1][:100]}...\"")

            all_results.append({
                "paper_id": paper_id,
                "original_length": len(text),
                "num_chunks": len(chunks),
                "stats": stats,
                "time_sec": elapsed,
                "sample_chunks": chunks[:3]  # First 3 for inspection
            })

        except Exception as e:
            print(f"ERROR processing {paper_id}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Summary
    print(f"\n\n{'='*100}")
    print("SUMMARY - Current Paragraph Chunker")
    print(f"{'='*100}")

    if all_results:
        total_papers = len(all_results)
        total_chunks = sum(r['num_chunks'] for r in all_results)
        avg_chunks_per_paper = total_chunks / total_papers

        all_char_counts = []
        for r in all_results:
            # Reconstruct char counts from stats
            all_char_counts.append(r['stats']['avg_chars'])

        avg_chars_overall = sum(all_char_counts) / len(all_char_counts)

        print(f"Papers processed: {total_papers}")
        print(f"Total chunks: {total_chunks}")
        print(f"Avg chunks per paper: {avg_chunks_per_paper:.1f}")
        print(f"Avg chars per chunk: {avg_chars_overall:.0f}")
        print(f"Total time: {total_time:.4f}s")
        print(f"Speed: {total_time / total_papers:.4f}s per paper")

        # Save results
        output_file = "artifacts/lvm/current_paragraph_chunker_test.json"
        with open(output_file, 'w') as f:
            json.dump(all_results, f, indent=2)

        print(f"\nResults saved to: {output_file}")

        # Print per-paper breakdown
        print(f"\n{'='*100}")
        print("PER-PAPER BREAKDOWN")
        print(f"{'='*100}")
        print(f"{'Paper':<20} {'Chunks':>10} {'Avg Chars':>12} {'Min':>8} {'Max':>8} {'Time (s)':>12}")
        print("-"*100)
        for r in all_results:
            print(f"{r['paper_id']:<20} {r['num_chunks']:>10} {r['stats']['avg_chars']:>12.0f} "
                  f"{r['stats']['min_chars']:>8} {r['stats']['max_chars']:>8} {r['time_sec']:>12.4f}")


if __name__ == "__main__":
    main()
