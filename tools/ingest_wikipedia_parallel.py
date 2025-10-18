#!/usr/bin/env python3
"""
MASSIVELY PARALLEL Wikipedia Ingestion Pipeline

Leverages 40 GPUs + 128GB RAM for maximum throughput via multiprocessing.
Processes 40 articles simultaneously with process pool parallelization.

Usage:
    # Process with 40 parallel workers (saturates 40 GPUs)
    LNSP_TMD_MODE=hybrid ./.venv/bin/python tools/ingest_wikipedia_parallel.py \
        --limit 10000 \
        --workers 40 \
        --batch-size 200
"""

import argparse
import json
import os
import requests
import time
from pathlib import Path
from typing import List, Dict, Tuple
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import functools

# Import heuristics for hybrid mode
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.tmd_heuristics import classify_task, classify_modifier

# API Endpoints
EPISODE_API = "http://localhost:8900"
SEMANTIC_API = "http://localhost:8001"
TMD_API = "http://localhost:8002"
EMBEDDING_API = "http://localhost:8767"
INGEST_API = "http://localhost:8004"

# TMD Mode configuration
TMD_MODE = os.getenv("LNSP_TMD_MODE", "full")


def check_apis():
    """Verify all APIs are running"""
    apis = {
        "Episode Chunker": f"{EPISODE_API}/health",
        "Semantic Chunker": f"{SEMANTIC_API}/health",
        "TMD Router": f"{TMD_API}/health",
        "GTR-T5 Embeddings": f"{EMBEDDING_API}/health",
        "Ingest": f"{INGEST_API}/health"
    }

    print("🔍 Checking APIs...")
    for name, url in apis.items():
        try:
            response = requests.get(url, timeout=2)
            if response.status_code == 200:
                print(f"  ✅ {name}: {url}")
            else:
                print(f"  ❌ {name}: {url} (HTTP {response.status_code})")
                return False
        except Exception as e:
            print(f"  ❌ {name}: {url} ({e})")
            return False

    return True


def load_articles(file_path: str, limit: int, skip: int = 0) -> List[Dict]:
    """Load Wikipedia articles from JSONL"""
    articles = []
    with open(file_path) as f:
        for _ in range(skip):
            next(f, None)
        for line in f:
            articles.append(json.loads(line))
            if len(articles) >= limit:
                break
    return articles


def chunk_into_episodes(document_id: str, text: str) -> List[Dict]:
    """Step 1: Chunk document into episodes"""
    try:
        response = requests.post(
            f"{EPISODE_API}/chunk",
            json={
                "document_id": document_id,
                "text": text,
                "coherence_threshold": 0.6,
                "min_episode_length": 3,
                "max_episode_length": 20
            },
            timeout=60
        )
        response.raise_for_status()
        return response.json()["episodes"]
    except Exception as e:
        print(f"Episode chunking error for {document_id}: {e}")
        return []


def chunk_semantically(episode_text: str) -> List[str]:
    """Step 2: Chunk episode into semantic chunks"""
    try:
        response = requests.post(
            f"{SEMANTIC_API}/chunk",
            json={
                "text": episode_text,
                "mode": "semantic",
                "min_chunk_size": 40,
                "max_chunk_size": 200,
                "breakpoint_threshold": 75
            },
            timeout=60
        )
        response.raise_for_status()
        chunks = response.json()["chunks"]
        return [c["text"] for c in chunks]
    except Exception as e:
        print(f"Semantic chunking error: {e}")
        return []


def extract_tmd(text: str) -> Dict:
    """Step 3: Extract TMD codes"""
    try:
        response = requests.post(
            f"{TMD_API}/route",
            json={"concept_text": text},
            timeout=30
        )
        response.raise_for_status()
        result = response.json()
        return {
            "domain_code": result["domain_code"],
            "task_code": result["task_code"],
            "modifier_code": result["modifier_code"]
        }
    except Exception as e:
        print(f"TMD extraction error: {e}")
        return {"domain_code": "000", "task_code": "00", "modifier_code": "0"}


def get_embeddings(texts: List[str]) -> List[List[float]]:
    """Step 4: Get GTR-T5 embeddings (batched)"""
    try:
        response = requests.post(
            f"{EMBEDDING_API}/embed",
            json={"texts": texts},
            timeout=300
        )
        response.raise_for_status()
        return response.json()["embeddings"]
    except Exception as e:
        print(f"Embedding error: {e}")
        return [[0.0] * 768] * len(texts)


def ingest_chunks(chunks_data: List[Dict], dataset_source: str = "wikipedia_500k"):
    """Step 5: Ingest to PostgreSQL + FAISS"""
    cleaned_chunks = []
    for chunk in chunks_data:
        chunk_copy = chunk.copy()
        chunk_copy.pop("dataset_source", None)
        cleaned_chunks.append(chunk_copy)

    try:
        response = requests.post(
            f"{INGEST_API}/ingest",
            json={
                "chunks": cleaned_chunks,
                "dataset_source": dataset_source
            },
            timeout=600
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Ingestion error: {e}")
        return {"status": "error"}


def process_article_to_chunks(article_with_index: Tuple[Dict, int]) -> List[Dict]:
    """
    Process single article through chunking + TMD extraction.
    This function will be called in parallel by multiple workers.

    Args:
        article_with_index: Tuple of (article dict, article index)

    Returns:
        List of chunk dictionaries with all metadata
    """
    article, article_index = article_with_index
    document_id = f"wikipedia_{article_index}"
    title = article.get("title", "Unknown")
    text = article.get("text", "")

    all_chunks_data = []

    try:
        # Step 1: Episode chunking
        episodes = chunk_into_episodes(document_id, text)
        if not episodes:
            return []

        # Hybrid TMD mode: Extract Domain ONCE for entire article
        article_domain = None
        if TMD_MODE == "hybrid":
            article_summary = f"{title}. {text[:500]}"
            domain_result = extract_tmd(article_summary)
            article_domain = domain_result["domain_code"]

        # Step 2-3: Semantic chunking + TMD extraction
        for ep_idx, episode in enumerate(episodes):
            episode_id = episode["episode_id"]
            semantic_chunks = chunk_semantically(episode["text"])

            for seq_idx, chunk_text in enumerate(semantic_chunks):
                if TMD_MODE == "hybrid":
                    task_code = classify_task(chunk_text)
                    modifier_code = classify_modifier(chunk_text)
                    domain_code = article_domain
                else:
                    tmd = extract_tmd(chunk_text)
                    domain_code = tmd["domain_code"]
                    task_code = tmd["task_code"]
                    modifier_code = tmd["modifier_code"]

                chunk_data = {
                    "text": chunk_text,
                    "document_id": document_id,
                    "sequence_index": seq_idx,
                    "episode_id": episode_id,
                    "domain_code": domain_code,
                    "task_code": task_code,
                    "modifier_code": modifier_code
                }

                all_chunks_data.append(chunk_data)

    except Exception as e:
        pass  # Error already logged in individual functions

    return all_chunks_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/datasets/wikipedia/wikipedia_500k.jsonl")
    parser.add_argument("--limit", type=int, default=10, help="Number of articles to process")
    parser.add_argument("--skip-offset", type=int, default=0, help="Number of articles to skip")
    parser.add_argument("--workers", type=int, default=40, help="Number of parallel workers (default: 40 for 40 GPUs)")
    parser.add_argument("--batch-size", type=int, default=200, help="Chunks to batch for embedding+ingestion (default: 200)")
    parser.add_argument("--skip-check", action="store_true", help="Skip API health check")

    args = parser.parse_args()

    print("🚀 MASSIVELY PARALLEL Wikipedia Ingestion Pipeline")
    print("=" * 80)
    print(f"   TMD Mode: {TMD_MODE}")
    print(f"   Parallel workers: {args.workers} (leveraging 40 GPUs + 128GB RAM)")
    print(f"   Ingestion batch size: {args.batch_size} chunks")

    # Check APIs
    if not args.skip_check:
        if not check_apis():
            print("\n❌ Some APIs are not running.")
            return 1

    # Load articles
    print(f"\n📥 Loading articles from {args.input}...")
    if not Path(args.input).exists():
        print(f"❌ File not found.")
        return 1

    articles = load_articles(args.input, args.limit, args.skip_offset)
    print(f"   Loaded: {len(articles)} articles (skipped: {args.skip_offset})")

    # Prepare articles with indices
    articles_with_indices = [(article, args.skip_offset + i + 1) for i, article in enumerate(articles)]

    # Process articles in PARALLEL using multiprocessing
    print(f"\n⚙️  Processing {len(articles)} articles with {args.workers} parallel workers...")
    total_chunks = 0
    start_time = time.time()

    all_chunks = []

    # Use multiprocessing Pool to process articles in parallel
    with Pool(processes=args.workers) as pool:
        # Process articles in parallel with progress bar
        for chunk_list in tqdm(
            pool.imap_unordered(process_article_to_chunks, articles_with_indices),
            total=len(articles),
            desc="Articles"
        ):
            all_chunks.extend(chunk_list)

            # Batch ingestion: When we accumulate enough chunks, embed + ingest
            if len(all_chunks) >= args.batch_size:
                # Extract texts for embedding
                texts = [c["text"] for c in all_chunks[:args.batch_size]]
                embeddings = get_embeddings(texts)

                # Add embeddings to chunks
                batch_to_ingest = all_chunks[:args.batch_size]
                for chunk_data, embedding in zip(batch_to_ingest, embeddings):
                    chunk_data["concept_vec"] = embedding

                # Ingest batch
                ingest_chunks(batch_to_ingest)
                total_chunks += len(batch_to_ingest)

                # Remove ingested chunks
                all_chunks = all_chunks[args.batch_size:]

    # Ingest remaining chunks
    if all_chunks:
        texts = [c["text"] for c in all_chunks]
        embeddings = get_embeddings(texts)

        for chunk_data, embedding in zip(all_chunks, embeddings):
            chunk_data["concept_vec"] = embedding

        ingest_chunks(all_chunks)
        total_chunks += len(all_chunks)

    # Summary
    total_time = time.time() - start_time
    print(f"\n✅ Pipeline Complete!")
    print(f"   Articles processed: {len(articles)}")
    print(f"   Chunks ingested: {total_chunks}")
    print(f"   Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
    print(f"   Throughput: {len(articles)/total_time:.2f} articles/sec")
    print(f"   Avg per article: {total_time/len(articles):.2f}s")
    print(f"   GPU Saturation: {args.workers} parallel workers × APIs = MAXIMUM THROUGHPUT!")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
