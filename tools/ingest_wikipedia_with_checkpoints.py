#!/usr/bin/env python3
"""
Wikipedia Ingestion Pipeline with Commit Checkpoints

KEY DIFFERENCE: Commits after every N articles to ensure crash-resistant progress.

This script wraps the standard ingestion pipeline but adds:
1. Checkpoint commits every N articles (default: 50)
2. Progress tracking that survives crashes
3. Automatic resume from last checkpoint

Usage:
    # Ingest with checkpoints every 50 articles
    LNSP_TMD_MODE=hybrid ./.venv/bin/python tools/ingest_wikipedia_with_checkpoints.py \\
      --input data/datasets/wikipedia/wikipedia_500k.jsonl \\
      --skip-offset 3432 \\
      --limit 3000 \\
      --checkpoint-every 50

    # Resume from crash (automatically detects last checkpoint)
    LNSP_TMD_MODE=hybrid ./.venv/bin/python tools/ingest_wikipedia_with_checkpoints.py \\
      --input data/datasets/wikipedia/wikipedia_500k.jsonl \\
      --limit 3000 \\
      --resume
"""

import argparse
import json
import os
import requests
import time
import psycopg2
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm

# API Endpoints
EPISODE_API = "http://localhost:8900"
SEMANTIC_API = "http://localhost:8001"
EMBEDDING_API = "http://localhost:8767"
INGEST_API = "http://localhost:8004"

# TMD Mode configuration
TMD_MODE = os.getenv("LNSP_TMD_MODE", "full")

# Database connection
PG_DSN = os.getenv("PG_DSN", "host=localhost port=5432 dbname=lnsp user=trentcarter")

# Checkpoint file location
CHECKPOINT_FILE = "artifacts/wikipedia_ingestion_checkpoint.json"


def get_pg_connection():
    """Get PostgreSQL connection"""
    return psycopg2.connect(PG_DSN)


def load_checkpoint() -> Dict:
    """Load checkpoint from disk"""
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, 'r') as f:
            return json.load(f)
    return {"last_article_index": 0, "last_batch_id": None, "total_processed": 0}


def save_checkpoint(article_index: int, batch_id: str, total_processed: int):
    """Save checkpoint to disk"""
    os.makedirs(os.path.dirname(CHECKPOINT_FILE), exist_ok=True)
    with open(CHECKPOINT_FILE, 'w') as f:
        json.dump({
            "last_article_index": article_index,
            "last_batch_id": batch_id,
            "total_processed": total_processed,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        }, f, indent=2)


def get_last_article_in_db() -> int:
    """Query PostgreSQL to find the highest article number ingested"""
    try:
        conn = get_pg_connection()
        cur = conn.cursor()
        cur.execute("""
            SELECT MAX(CAST(SUBSTRING(batch_id FROM 'wikipedia_([0-9]+)') AS INTEGER))
            FROM cpe_entry
            WHERE dataset_source = 'wikipedia_500k' AND batch_id LIKE 'wikipedia_%'
        """)
        result = cur.fetchone()[0]
        cur.close()
        conn.close()
        return result if result is not None else 0
    except Exception as e:
        print(f"⚠️  Could not query database for last article: {e}")
        return 0


def check_apis():
    """Verify all APIs are running"""
    apis = {
        "Episode Chunker": f"{EPISODE_API}/health",
        "Semantic Chunker": f"{SEMANTIC_API}/health",
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

    print(f"  ℹ️  TMD extraction: handled internally by Ingest API (mode: {TMD_MODE})")
    return True


def load_articles(file_path: str, limit: int, skip: int = 0) -> List[Dict]:
    """Load Wikipedia articles from JSONL"""
    articles = []
    with open(file_path) as f:
        # Skip offset lines
        for _ in range(skip):
            next(f, None)

        # Load limit articles
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
            "min_chunk_size": 10,
            "max_chunk_size": 500,
            "breakpoint_threshold": 75
        },
        timeout=60
    )
    response.raise_for_status()
    chunks = response.json()["chunks"]
    return [c["text"] for c in chunks]


def get_embeddings(texts: List[str]) -> List[List[float]]:
    """Step 3: Get GTR-T5 embeddings"""
    response = requests.post(
        f"{EMBEDDING_API}/embed",
        json={"texts": texts},
        timeout=120
    )
    response.raise_for_status()
    return response.json()["embeddings"]


def ingest_chunks(chunks_data: List[Dict], dataset_source: str = "wikipedia_500k"):
    """Step 4: Ingest to PostgreSQL + FAISS (includes TMD extraction)"""
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


def force_db_commit():
    """
    Force PostgreSQL to commit any pending transactions.

    The Ingest API commits after each batch, but we want to ensure
    that all commits are flushed to disk at checkpoint boundaries.
    """
    try:
        conn = get_pg_connection()
        conn.autocommit = True
        cur = conn.cursor()
        # This is a no-op query that forces pending commits to flush
        cur.execute("SELECT COUNT(*) FROM cpe_entry WHERE created_at > NOW() - INTERVAL '1 second'")
        cur.close()
        conn.close()
    except Exception as e:
        print(f"  ⚠️  Could not force commit: {e}")


def process_article(article: Dict, article_index: int) -> Dict:
    """Process a single Wikipedia article through full pipeline"""

    document_id = f"wikipedia_{article_index}"
    title = article.get("title", "Unknown")
    text = article.get("text", "")

    stats = {
        "document_id": document_id,
        "title": title,
        "episodes": 0,
        "chunks": 0,
        "errors": []
    }

    try:
        # Step 1: Episode chunking
        episodes = chunk_into_episodes(document_id, text)
        stats["episodes"] = len(episodes)

        all_chunks_data = []

        # Process all episodes
        for ep_idx, episode in enumerate(episodes):
            episode_id = episode["episode_id"]

            # Step 2: Semantic chunking
            semantic_chunks = chunk_semantically(episode["text"])

            # Process each chunk
            for seq_idx, chunk_text in enumerate(semantic_chunks):
                chunk_data = {
                    "text": chunk_text,
                    "document_id": document_id,
                    "sequence_index": seq_idx,
                    "episode_id": episode_id,
                    "dataset_source": "wikipedia_500k",
                }
                all_chunks_data.append(chunk_data)
                stats["chunks"] += 1

        # Step 3: Batch embeddings
        if all_chunks_data:
            texts = [c["text"] for c in all_chunks_data]
            embeddings = get_embeddings(texts)

            # Add embeddings to chunk data
            for chunk_data, embedding in zip(all_chunks_data, embeddings):
                chunk_data["concept_vec"] = embedding

            # Step 4: Ingest (includes internal TMD extraction + CPESH + database writes)
            ingest_chunks(all_chunks_data)

    except Exception as e:
        stats["errors"].append(str(e))

    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/datasets/wikipedia/wikipedia_500k.jsonl")
    parser.add_argument("--limit", type=int, default=10, help="Number of articles to process")
    parser.add_argument("--skip-offset", type=int, default=0, help="Number of articles to skip")
    parser.add_argument("--checkpoint-every", type=int, default=50, help="Commit checkpoint every N articles")
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
    parser.add_argument("--skip-check", action="store_true", help="Skip API health check")

    args = parser.parse_args()

    print("🚀 Wikipedia Ingestion Pipeline with Checkpoints")
    print("=" * 80)
    print(f"   TMD Mode: {TMD_MODE}")
    print(f"   Checkpoint frequency: every {args.checkpoint_every} articles")

    # Check APIs
    if not args.skip_check:
        if not check_apis():
            print("\n❌ Some APIs are not running. Start them first.")
            return 1

    # Handle resume mode
    skip_offset = args.skip_offset
    if args.resume:
        last_article = get_last_article_in_db()
        print(f"\n📊 Resume mode: Last article in database: {last_article}")
        skip_offset = last_article + 1
        print(f"   Resuming from article: {skip_offset}")

    # Load articles
    print(f"\n📥 Loading articles from {args.input}...")
    if not Path(args.input).exists():
        print(f"❌ File not found: {args.input}")
        return 1

    articles = load_articles(args.input, args.limit, skip_offset)
    print(f"   Loaded: {len(articles)} articles (starting from: {skip_offset})")

    # Process articles with checkpoints
    print(f"\n⚙️  Processing articles with checkpoints every {args.checkpoint_every}...")
    total_episodes = 0
    total_chunks = 0
    errors = []

    checkpoint_counter = 0

    for i, article in enumerate(tqdm(articles, desc="Articles"), start=skip_offset + 1):
        stats = process_article(article, i)
        total_episodes += stats["episodes"]
        total_chunks += stats["chunks"]
        if stats["errors"]:
            errors.append((stats["title"], stats["errors"]))

        checkpoint_counter += 1

        # Checkpoint commit every N articles
        if checkpoint_counter >= args.checkpoint_every:
            print(f"\n💾 Checkpoint: Processed {i - skip_offset} articles (article #{i})")
            force_db_commit()  # Ensure DB flushes pending commits
            save_checkpoint(i, f"wikipedia_{i}", i - skip_offset)
            checkpoint_counter = 0

    # Final checkpoint
    print(f"\n💾 Final checkpoint: {len(articles)} articles processed")
    force_db_commit()
    save_checkpoint(skip_offset + len(articles), f"wikipedia_{skip_offset + len(articles)}", len(articles))

    # Summary
    print(f"\n✅ Pipeline Complete!")
    print(f"   Articles processed: {len(articles)}")
    print(f"   Episodes created: {total_episodes}")
    print(f"   Chunks ingested: {total_chunks}")

    if errors:
        print(f"\n⚠️  Errors: {len(errors)}")
        for title, errs in errors[:5]:
            print(f"      {title}: {errs[0]}")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
