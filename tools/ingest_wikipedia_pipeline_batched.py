#!/usr/bin/env python3
"""
Batched Wikipedia Ingestion Pipeline (10x faster via parallel processing)

Collects chunks from multiple articles and sends in large batches to saturate
the ingest API's 10-thread pool for GPU parallelization.

Usage:
    # Batch 20 articles at a time (recommended)
    LNSP_TMD_MODE=hybrid ./.venv/bin/python tools/ingest_wikipedia_pipeline_batched.py --limit 10000 --article-batch-size 20
"""

import argparse
import json
import os
import requests
import time
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm
from dataclasses import dataclass, field

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


def chunk_semantically(episode_text: str) -> List[str]:
    """Step 2: Chunk episode into semantic chunks"""
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


def extract_tmd(text: str) -> Dict:
    """Step 3: Extract TMD codes"""
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


def get_embeddings(texts: List[str]) -> List[List[float]]:
    """Step 4: Get GTR-T5 embeddings"""
    response = requests.post(
        f"{EMBEDDING_API}/embed",
        json={"texts": texts},
        timeout=120
    )
    response.raise_for_status()
    return response.json()["embeddings"]


def ingest_chunks(chunks_data: List[Dict], dataset_source: str = "wikipedia_500k"):
    """Step 5: Ingest to PostgreSQL + FAISS"""
    cleaned_chunks = []
    for chunk in chunks_data:
        chunk_copy = chunk.copy()
        chunk_copy.pop("dataset_source", None)
        cleaned_chunks.append(chunk_copy)

    response = requests.post(
        f"{INGEST_API}/ingest",
        json={
            "chunks": cleaned_chunks,
            "dataset_source": dataset_source
        },
        timeout=300
    )
    response.raise_for_status()
    return response.json()


def process_article_to_chunks(article: Dict, article_index: int) -> List[Dict]:
    """
    Process article through chunking + TMD extraction, return raw chunk data.
    Does NOT call embeddings or ingestion - those are batched later.
    """
    document_id = f"wikipedia_{article_index}"
    title = article.get("title", "Unknown")
    text = article.get("text", "")

    all_chunks_data = []

    try:
        # Step 1: Episode chunking
        episodes = chunk_into_episodes(document_id, text)

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
                    "source_document": document_id,  # Fixed: was "document_id"
                    "chunk_index": seq_idx,          # Fixed: was "sequence_index"
                    "episode_id": episode_id,
                    "domain_code": domain_code,
                    "task_code": task_code,
                    "modifier_code": modifier_code
                }

                all_chunks_data.append(chunk_data)

    except Exception as e:
        print(f"\n  ⚠️  Error processing article {article_index} ({title}): {e}")

    return all_chunks_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/datasets/wikipedia/wikipedia_500k.jsonl")
    parser.add_argument("--limit", type=int, default=10, help="Number of articles to process")
    parser.add_argument("--skip-offset", type=int, default=0, help="Number of articles to skip")
    parser.add_argument("--article-batch-size", type=int, default=20, help="Articles to batch before ingesting (default: 20)")
    parser.add_argument("--skip-check", action="store_true", help="Skip API health check")

    args = parser.parse_args()

    print("🚀 Wikipedia Batched Ingestion Pipeline")
    print("=" * 80)
    print(f"   TMD Mode: {TMD_MODE}")
    print(f"   Article batch size: {args.article_batch_size} (parallel GPU processing)")

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

    # Process articles in batches
    print(f"\n⚙️  Processing articles in batches of {args.article_batch_size}...")
    total_chunks = 0
    start_time = time.time()

    article_batches = [articles[i:i + args.article_batch_size] for i in range(0, len(articles), args.article_batch_size)]

    for batch_idx, article_batch in enumerate(tqdm(article_batches, desc="Article Batches"), 1):
        batch_start = time.time()

        # Step 1-3: Process all articles in this batch (chunking + TMD)
        all_batch_chunks = []
        for i, article in enumerate(article_batch):
            article_index = args.skip_offset + (batch_idx - 1) * args.article_batch_size + i + 1
            chunks = process_article_to_chunks(article, article_index)
            all_batch_chunks.extend(chunks)

        if not all_batch_chunks:
            continue

        # Step 4: Batch embeddings for ALL chunks in this article batch
        texts = [c["text"] for c in all_batch_chunks]
        embeddings = get_embeddings(texts)

        for chunk_data, embedding in zip(all_batch_chunks, embeddings):
            chunk_data["concept_vec"] = embedding

        # Step 5: Ingest all chunks from this article batch at once
        # The ingest API will parallelize across its 10 workers
        ingest_chunks(all_batch_chunks)

        total_chunks += len(all_batch_chunks)
        batch_time = time.time() - batch_start

        # Progress update
        articles_processed = batch_idx * args.article_batch_size
        if articles_processed > len(articles):
            articles_processed = len(articles)

        elapsed = time.time() - start_time
        rate = articles_processed / elapsed if elapsed > 0 else 0

        tqdm.write(f"  Batch {batch_idx}: {len(all_batch_chunks)} chunks in {batch_time:.1f}s | "
                  f"Total: {articles_processed}/{len(articles)} articles ({rate:.2f} art/s)")

    # Summary
    total_time = time.time() - start_time
    print(f"\n✅ Pipeline Complete!")
    print(f"   Articles processed: {len(articles)}")
    print(f"   Chunks ingested: {total_chunks}")
    print(f"   Total time: {total_time:.1f}s")
    print(f"   Throughput: {len(articles)/total_time:.2f} articles/sec")
    print(f"   Avg per article: {total_time/len(articles):.2f}s")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
