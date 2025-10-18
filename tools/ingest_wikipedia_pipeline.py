#!/usr/bin/env python3
"""
Complete Wikipedia Ingestion Pipeline

Downloads Wikipedia articles → Episodes → Semantic Chunks → Embeddings → PostgreSQL + FAISS

Pipeline:
1. Download Wikipedia articles (local)
2. Episode Chunker API :8900 (coherence-based episodes)
3. Semantic Chunker API :8001 (fine-grain chunks)
4. Vec2Text-Compatible GTR-T5 API :8767 (768D vectors)
5. Ingest API :8004 (CPESH + TMD extraction + PostgreSQL + FAISS)

TMD Modes (configured via LNSP_TMD_MODE, handled internally by Ingest API):
- full: LLM extraction per chunk (slow, accurate)
- hybrid: LLM for Domain + heuristics for Task/Modifier (fast, good) [default for Wikipedia]

Chunk Sizing:
- Min: 10 chars (small chunks allowed)
- Max: 500 chars (17 tokens × 2.5 chars typical)

Usage:
    # Pilot (10 articles, hybrid TMD mode)
    LNSP_TMD_MODE=hybrid ./.venv/bin/python tools/ingest_wikipedia_pipeline.py --limit 10

    # Full mode (slower, more accurate TMD)
    LNSP_TMD_MODE=full ./.venv/bin/python tools/ingest_wikipedia_pipeline.py --limit 10

    # Production (500k articles, hybrid mode recommended)
    LNSP_TMD_MODE=hybrid ./.venv/bin/python tools/ingest_wikipedia_pipeline.py --limit 500000
"""

import argparse
import json
import os
import pickle
import requests
import signal
import sys
import time
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm
from dataclasses import dataclass, field

# API Endpoints
EPISODE_API = "http://localhost:8900"
SEMANTIC_API = "http://localhost:8001"
EMBEDDING_API = "http://localhost:8767"
INGEST_API = "http://localhost:8004"

def save_checkpoint(article_index, stats, timings):
    """Save current progress to checkpoint file"""
    checkpoint = {
        'article_index': article_index,
        'stats': stats,
        'timings': timings
    }
    with open('artifacts/ingestion_checkpoint.pkl', 'wb') as f:
        pickle.dump(checkpoint, f)
    print(f"💾 Checkpoint saved at article {article_index}")

def load_checkpoint():
    """Load progress from checkpoint file if it exists"""
    checkpoint_file = 'artifacts/ingestion_checkpoint.pkl'
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'rb') as f:
            checkpoint = pickle.load(f)
        print(f"📖 Resuming from checkpoint at article {checkpoint['article_index']}")
        return checkpoint
    return None

def signal_handler(sig, frame):
    """Handle SIGTERM/SIGINT to save checkpoint before exit"""
    print(f"\n⚠️  Received signal {sig}. Saving checkpoint before exit...")
    save_checkpoint(current_article_index, stats_so_far, timings)
    sys.exit(0)


@dataclass
class PipelineTimings:
    """Track detailed pipeline timing metrics"""
    episode_chunking_ms: List[float] = field(default_factory=list)
    semantic_chunking_ms: List[float] = field(default_factory=list)
    tmd_extraction_ms: List[float] = field(default_factory=list)
    embedding_ms: List[float] = field(default_factory=list)
    ingestion_ms: List[float] = field(default_factory=list)
    total_ms: List[float] = field(default_factory=list)

    def add_timings(self, episode: float, semantic: float, tmd: float, embedding: float, ingest: float, total: float):
        self.episode_chunking_ms.append(episode)
        self.semantic_chunking_ms.append(semantic)
        self.tmd_extraction_ms.append(tmd)
        self.embedding_ms.append(embedding)
        self.ingestion_ms.append(ingest)
        self.total_ms.append(total)

    def summary(self) -> Dict:
        import statistics
        def safe_mean(lst):
            return statistics.mean(lst) if lst else 0

        return {
            "episode_chunking": {
                "avg_ms": safe_mean(self.episode_chunking_ms),
                "total_ms": sum(self.episode_chunking_ms)
            },
            "semantic_chunking": {
                "avg_ms": safe_mean(self.semantic_chunking_ms),
                "total_ms": sum(self.semantic_chunking_ms)
            },
            "tmd_extraction": {
                "avg_ms": safe_mean(self.tmd_extraction_ms),
                "total_ms": sum(self.tmd_extraction_ms)
            },
            "embedding": {
                "avg_ms": safe_mean(self.embedding_ms),
                "total_ms": sum(self.embedding_ms)
            },
            "ingestion": {
                "avg_ms": safe_mean(self.ingestion_ms),
                "total_ms": sum(self.ingestion_ms)
            },
            "total_pipeline": {
                "avg_ms": safe_mean(self.total_ms),
                "total_ms": sum(self.total_ms)
            }
        }


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
            "min_chunk_size": 10,  # Updated: allow small chunks
            "max_chunk_size": 500,  # Updated: 17 tokens × 2.5 chars, max 500
            "breakpoint_threshold": 75  # Lower = more chunks, Higher = fewer chunks
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
    # Remove dataset_source from individual chunks (it's request-level metadata)
    cleaned_chunks = []
    for chunk in chunks_data:
        chunk_copy = chunk.copy()
        chunk_copy.pop("dataset_source", None)  # Remove if present
        cleaned_chunks.append(chunk_copy)

    response = requests.post(
        f"{INGEST_API}/ingest",
        json={
            "chunks": cleaned_chunks,
            "dataset_source": dataset_source  # Send at request level
        },
        timeout=300
    )
    response.raise_for_status()
    return response.json()


def process_article(article: Dict, article_index: int, timings: PipelineTimings) -> Dict:
    """Process a single Wikipedia article through full pipeline"""

    document_id = f"wikipedia_{article_index}"
    title = article.get("title", "Unknown")
    text = article.get("text", "")

    stats = {
        "document_id": document_id,
        "title": title,
        "episodes": 0,
        "chunks": 0,
        "errors": [],
        "timings": {}
    }

    article_start = time.time()
    episode_time = 0
    semantic_time = 0
    embedding_time = 0
    ingest_time = 0

    try:
        # Step 1: Episode chunking
        t0 = time.time()
        episodes = chunk_into_episodes(document_id, text)
        episode_time = (time.time() - t0) * 1000
        stats["episodes"] = len(episodes)

        all_chunks_data = []

        # Process all episodes
        for ep_idx, episode in enumerate(episodes):
            episode_id = episode["episode_id"]

            # Step 2: Semantic chunking
            t1 = time.time()
            semantic_chunks = chunk_semantically(episode["text"])
            semantic_time += (time.time() - t1) * 1000

            # Process each chunk
            for seq_idx, chunk_text in enumerate(semantic_chunks):
                # Prepare chunk data (TMD extraction will be handled by Ingest API)
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

            t3 = time.time()
            embeddings = get_embeddings(texts)
            embedding_time = (time.time() - t3) * 1000

            # Add embeddings to chunk data
            for chunk_data, embedding in zip(all_chunks_data, embeddings):
                chunk_data["concept_vec"] = embedding

            # Step 4: Ingest (includes internal TMD extraction + CPESH + database writes)
            t4 = time.time()
            ingest_chunks(all_chunks_data)
            ingest_time = (time.time() - t4) * 1000

        # Calculate total time
        total_time = (time.time() - article_start) * 1000

        # Record timings (TMD time now included in ingest_time)
        timings.add_timings(episode_time, semantic_time, 0, embedding_time, ingest_time, total_time)

        stats["timings"] = {
            "episode_chunking_ms": episode_time,
            "semantic_chunking_ms": semantic_time,
            "tmd_extraction_ms": 0,  # Now handled internally by Ingest API
            "embedding_ms": embedding_time,
            "ingestion_ms": ingest_time,  # Includes TMD + CPESH + DB writes
            "total_ms": total_time
        }

    except Exception as e:
        stats["errors"].append(str(e))

    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/datasets/wikipedia/wikipedia_simple_articles.jsonl")
    parser.add_argument("--limit", type=int, default=10, help="Number of articles to process")
    parser.add_argument("--skip-offset", type=int, default=0, help="Number of articles to skip (for batching)")
    parser.add_argument("--skip-check", action="store_true", help="Skip API health check")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")

    args = parser.parse_args()

    print("🚀 Wikipedia Ingestion Pipeline")
    print("=" * 80)
    print(f"   TMD Mode: {TMD_MODE}")

    # Set up signal handler for graceful shutdown
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    # Check APIs
    if not args.skip_check:
        if not check_apis():
            print("\n❌ Some APIs are not running. Start them first:")
            print("   Episode: ./.venv/bin/uvicorn app.api.episode_chunker:app --port 8900 &")
            print("   Semantic: ./.venv/bin/uvicorn app.api.chunking:app --port 8001 &")
            print("   Embeddings: ./.venv/bin/uvicorn app.api.vec2text_embedding_server:app --port 8767 &")
            print("   Ingest: ./.venv/bin/uvicorn app.api.ingest_chunks:app --port 8004 &")
            print("\n   Note: TMD extraction handled internally by Ingest API")
            return 1

    # Load articles
    print(f"\n📥 Loading articles from {args.input}...")
    if not Path(args.input).exists():
        print(f"❌ File not found. Download first:")
        print(f"   ./.venv/bin/python tools/download_wikipedia.py --limit {args.limit}")
        return 1

    articles = load_articles(args.input, args.limit, args.skip_offset)
    print(f"   Loaded: {len(articles)} articles (skipped: {args.skip_offset})")

    # Check for checkpoint
    checkpoint = load_checkpoint() if args.resume else None
    start_index = 0
    total_episodes = 0
    total_chunks = 0
    errors = []
    timings = PipelineTimings()

    if checkpoint:
        start_index = checkpoint['article_index']
        total_episodes = sum(s['episodes'] for s in checkpoint['stats'])
        total_chunks = sum(s['chunks'] for s in checkpoint['stats'])
        errors = checkpoint['stats']
        timings = checkpoint['timings']
        print(f"   Resuming from article {start_index}")

    # Process articles
    print(f"\n⚙️  Processing articles...")
    for i, article in enumerate(tqdm(articles, desc="Articles"), 1):
        if i <= start_index:
            continue

        stats = process_article(article, i, timings)
        total_episodes += stats["episodes"]
        total_chunks += stats["chunks"]
        if stats["errors"]:
            errors.append((stats["title"], stats["errors"]))

        # Save checkpoint every 100 articles
        if i % 100 == 0:
            save_checkpoint(i, errors, timings)

    # Performance summary
    timing_summary = timings.summary()

    # Summary
    print(f"\n✅ Pipeline Complete!")
    print(f"   Articles processed: {len(articles) - start_index}")
    print(f"   Episodes created: {total_episodes}")
    print(f"   Chunks ingested: {total_chunks}")

    print(f"\n⏱️  Performance Metrics (per article average):")
    print(f"   Episode Chunking: {timing_summary['episode_chunking']['avg_ms']:.1f}ms")
    print(f"   Semantic Chunking: {timing_summary['semantic_chunking']['avg_ms']:.1f}ms")
    print(f"   TMD Extraction: {timing_summary['tmd_extraction']['avg_ms']:.1f}ms")
    print(f"   Embeddings: {timing_summary['embedding']['avg_ms']:.1f}ms")
    print(f"   Ingestion: {timing_summary['ingestion']['avg_ms']:.1f}ms")
    print(f"   Total Pipeline: {timing_summary['total_pipeline']['avg_ms']:.1f}ms")

    print(f"\n📊 Total Time:")
    print(f"   Pipeline: {timing_summary['total_pipeline']['total_ms']/1000:.1f}s")
    if timing_summary['total_pipeline']['total_ms'] > 0:
        print(f"   Throughput: {(len(articles) - start_index)/(timing_summary['total_pipeline']['total_ms']/1000):.2f} articles/sec")
    else:
        print(f"   Throughput: N/A (timing data missing)")

    if errors:
        print(f"\n⚠️  Errors: {len(errors)}")
        for title, errs in errors[:5]:
            print(f"      {title}: {errs[0]}")

    # Save timing metrics
    metrics_file = "artifacts/pipeline_metrics.json"
    with open(metrics_file, 'w') as f:
        json.dump({
            "summary": timing_summary,
            "articles_processed": len(articles) - start_index,
            "total_episodes": total_episodes,
            "total_chunks": total_chunks,
            "errors": len(errors)
        }, f, indent=2)
    print(f"\n💾 Metrics saved to: {metrics_file}")

    # Clean up checkpoint on successful completion
    checkpoint_file = 'artifacts/ingestion_checkpoint.pkl'
    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)
        print(f"🧹 Checkpoint cleaned up")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
