#!/usr/bin/env python3
"""
Test ALL LlamaIndex chunking modes on arXiv papers:
- Simple: Sentence-based splitting (TRUE sentence level)
- Semantic: Embedding-based semantic boundaries
- Proposition: LLM-based atomic proposition extraction
- Hybrid: Semantic + Proposition combined

Usage:
    python tools/test_llamaindex_all_modes.py
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Optional
import json
import time

# LlamaIndex imports
from llama_index.core import Document
from llama_index.core.node_parser import (
    SentenceSplitter,
    SemanticSplitterNodeParser,
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


def load_arxiv_paper(txt_path: str) -> str:
    """Load arXiv paper text from file."""
    with open(txt_path, 'r', encoding='utf-8') as f:
        return f.read()


def chunk_simple_mode(text: str, paper_id: str) -> List[Dict]:
    """
    TRUE Simple mode: Individual sentence-level splitting.
    No grouping, just split at sentence boundaries (. ! ?)
    """
    doc = Document(text=text, metadata={"paper_id": paper_id})

    # TRUE sentence-level splitting: very small chunk_size to avoid grouping
    splitter = SentenceSplitter(
        chunk_size=128,  # Small enough to avoid grouping sentences
        chunk_overlap=0,  # No overlap for true sentence boundaries
        separator=" ",
        paragraph_separator="\n\n",
    )

    nodes = splitter.get_nodes_from_documents([doc])

    chunks = []
    for i, node in enumerate(nodes):
        chunks.append({
            "chunk_id": i,
            "text": node.text,
            "char_count": len(node.text),
            "word_count": len(node.text.split()),
        })

    return chunks


def chunk_semantic_mode(text: str, paper_id: str, embed_model) -> List[Dict]:
    """
    Semantic mode: Uses embeddings to find semantic boundaries.
    Splits where meaning changes significantly.
    """
    doc = Document(text=text, metadata={"paper_id": paper_id})

    # Semantic splitter with embedding model
    splitter = SemanticSplitterNodeParser(
        buffer_size=1,  # Number of sentences to group
        breakpoint_percentile_threshold=95,  # Sensitivity to semantic breaks
        embed_model=embed_model,
    )

    nodes = splitter.get_nodes_from_documents([doc])

    chunks = []
    for i, node in enumerate(nodes):
        chunks.append({
            "chunk_id": i,
            "text": node.text,
            "char_count": len(node.text),
            "word_count": len(node.text.split()),
        })

    return chunks


def chunk_proposition_mode(text: str, paper_id: str) -> List[Dict]:
    """
    Proposition mode: Uses LLM to extract atomic propositions.
    NOTE: This requires LLM access (OpenAI or local Ollama).
    For now, we'll return a placeholder or skip if not available.
    """
    # TODO: Implement with LLM (requires ollama or OpenAI API)
    # For now, return note that this needs LLM setup
    return [{
        "chunk_id": 0,
        "text": "[Proposition mode requires LLM setup - not implemented yet]",
        "char_count": 0,
        "word_count": 0,
    }]


def chunk_hybrid_mode(text: str, paper_id: str, embed_model) -> List[Dict]:
    """
    Hybrid mode: Combination of Semantic and Proposition.
    NOTE: This requires both embeddings and LLM.
    For now, we'll use semantic mode as approximation.
    """
    # TODO: Implement true hybrid (semantic + proposition)
    # For now, use semantic mode as approximation
    return chunk_semantic_mode(text, paper_id, embed_model)


def analyze_chunks(chunks: List[Dict], mode_name: str) -> Dict:
    """Calculate statistics for a set of chunks."""
    if not chunks or chunks[0].get("char_count", 0) == 0:
        return {
            "mode": mode_name,
            "num_chunks": 0,
            "avg_chars": 0,
            "min_chars": 0,
            "max_chars": 0,
            "avg_words": 0,
            "status": "not_available"
        }

    char_counts = [c['char_count'] for c in chunks]
    word_counts = [c['word_count'] for c in chunks]

    return {
        "mode": mode_name,
        "num_chunks": len(chunks),
        "avg_chars": sum(char_counts) / len(char_counts) if char_counts else 0,
        "min_chars": min(char_counts) if char_counts else 0,
        "max_chars": max(char_counts) if char_counts else 0,
        "avg_words": sum(word_counts) / len(word_counts) if word_counts else 0,
        "status": "success"
    }


def display_comparison_table(results: List[Dict]):
    """Display comparison table across all modes."""
    print(f"\n{'='*100}")
    print("CHUNKING MODE COMPARISON TABLE")
    print(f"{'='*100}")
    print(f"{'Mode':<15} {'Chunks':>10} {'Avg Chars':>12} {'Min Chars':>12} {'Max Chars':>12} {'Avg Words':>12} {'Status':<15}")
    print(f"{'-'*100}")

    for r in results:
        if r['status'] == 'not_available':
            print(f"{r['mode']:<15} {'N/A':>10} {'N/A':>12} {'N/A':>12} {'N/A':>12} {'N/A':>12} {'Not Available':<15}")
        else:
            print(f"{r['mode']:<15} {r['num_chunks']:>10} {r['avg_chars']:>12.0f} {r['min_chars']:>12} {r['max_chars']:>12} {r['avg_words']:>12.1f} {r['status']:<15}")


def main():
    print("Testing LlamaIndex Chunking Modes (Simple, Semantic, Proposition, Hybrid)")
    print("="*100)

    # Get arXiv text files
    pdfs_dir = Path("data/datasets/arxiv/pdfs")
    txt_files = sorted(list(pdfs_dir.glob("*.txt")))

    if len(txt_files) == 0:
        print(f"ERROR: No .txt files found in {pdfs_dir}")
        sys.exit(1)

    # Test on first 4 papers
    test_papers = txt_files[:4]

    # Initialize embedding model for Semantic/Hybrid modes
    print("\nLoading embedding model for Semantic mode...")
    embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
    print("✓ Embedding model loaded")

    # Process each paper with all modes
    all_paper_results = []

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

            paper_results = {
                "paper_id": paper_id,
                "original_length": len(text),
                "modes": {}
            }

            # Test Simple mode (TRUE sentence-level)
            print("\n[1/4] Testing Simple mode (sentence-level)...")
            start = time.time()
            simple_chunks = chunk_simple_mode(text, paper_id)
            simple_time = time.time() - start
            paper_results["modes"]["simple"] = {
                "chunks": simple_chunks[:5],  # Store first 5 for inspection
                "stats": analyze_chunks(simple_chunks, "Simple"),
                "time_sec": simple_time
            }
            print(f"  ✓ {len(simple_chunks)} chunks in {simple_time:.2f}s")
            print(f"  Sample: \"{simple_chunks[0]['text'][:100]}...\"")

            # Test Semantic mode
            print("\n[2/4] Testing Semantic mode (embedding-based)...")
            start = time.time()
            semantic_chunks = chunk_semantic_mode(text, paper_id, embed_model)
            semantic_time = time.time() - start
            paper_results["modes"]["semantic"] = {
                "chunks": semantic_chunks[:5],
                "stats": analyze_chunks(semantic_chunks, "Semantic"),
                "time_sec": semantic_time
            }
            print(f"  ✓ {len(semantic_chunks)} chunks in {semantic_time:.2f}s")
            print(f"  Sample: \"{semantic_chunks[0]['text'][:100]}...\"")

            # Test Proposition mode (placeholder)
            print("\n[3/4] Testing Proposition mode (LLM-based)...")
            prop_chunks = chunk_proposition_mode(text, paper_id)
            paper_results["modes"]["proposition"] = {
                "chunks": [],
                "stats": analyze_chunks(prop_chunks, "Proposition"),
                "time_sec": 0
            }
            print(f"  ⚠ Requires LLM setup (not implemented)")

            # Test Hybrid mode (semantic approximation)
            print("\n[4/4] Testing Hybrid mode (Semantic+Proposition)...")
            start = time.time()
            hybrid_chunks = chunk_hybrid_mode(text, paper_id, embed_model)
            hybrid_time = time.time() - start
            paper_results["modes"]["hybrid"] = {
                "chunks": hybrid_chunks[:5],
                "stats": analyze_chunks(hybrid_chunks, "Hybrid"),
                "time_sec": hybrid_time
            }
            print(f"  ✓ {len(hybrid_chunks)} chunks in {hybrid_time:.2f}s (using semantic)")

            all_paper_results.append(paper_results)

            # Display comparison table for this paper
            mode_stats = [
                paper_results["modes"]["simple"]["stats"],
                paper_results["modes"]["semantic"]["stats"],
                paper_results["modes"]["proposition"]["stats"],
                paper_results["modes"]["hybrid"]["stats"],
            ]
            display_comparison_table(mode_stats)

        except Exception as e:
            print(f"ERROR processing {paper_id}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Save results
    output_file = "artifacts/lvm/llamaindex_all_modes_comparison.json"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, 'w') as f:
        json.dump(all_paper_results, f, indent=2)

    print(f"\n\n{'='*100}")
    print(f"Results saved to: {output_file}")
    print(f"{'='*100}")

    # Final summary across all papers
    print("\n\nAGGREGATE SUMMARY ACROSS ALL PAPERS:")
    print("="*100)

    if all_paper_results:
        # Aggregate stats
        agg_stats = {
            "simple": {"total_chunks": 0, "total_chars": [], "total_time": 0},
            "semantic": {"total_chunks": 0, "total_chars": [], "total_time": 0},
            "hybrid": {"total_chunks": 0, "total_chars": [], "total_time": 0},
        }

        for paper in all_paper_results:
            for mode in ["simple", "semantic", "hybrid"]:
                if mode in paper["modes"]:
                    stats = paper["modes"][mode]["stats"]
                    if stats["status"] == "success":
                        agg_stats[mode]["total_chunks"] += stats["num_chunks"]
                        agg_stats[mode]["total_time"] += paper["modes"][mode]["time_sec"]
                        # Collect all chunk sizes
                        chunks = paper["modes"][mode]["chunks"]
                        agg_stats[mode]["total_chars"].extend([c["char_count"] for c in chunks])

        # Print aggregate table
        print(f"{'Mode':<15} {'Total Chunks':>15} {'Avg Chars':>15} {'Total Time':>15}")
        print("-"*60)
        for mode_name, data in agg_stats.items():
            if data["total_chunks"] > 0:
                avg_chars = sum(data["total_chars"]) / len(data["total_chars"]) if data["total_chars"] else 0
                print(f"{mode_name.capitalize():<15} {data['total_chunks']:>15} {avg_chars:>15.0f} {data['total_time']:>15.2f}s")


if __name__ == "__main__":
    main()
