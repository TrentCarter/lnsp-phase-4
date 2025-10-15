#!/usr/bin/env python3
"""
Complete Wikipedia Ingestion Pipeline

Downloads Wikipedia articles → Episodes → Semantic Chunks → TMD → Embeddings → PostgreSQL + FAISS

Pipeline:
1. Download Wikipedia articles (local)
2. Episode Chunker API :8900 (coherence-based episodes)
3. Semantic Chunker API :8001 (fine-grain chunks)
4. TMD Router API :8002 (Domain/Task/Modifier extraction)
5. Vec2Text-Compatible GTR-T5 API :8767 (768D vectors)
6. Ingest API :8004 (PostgreSQL + FAISS)

TMD Modes:
- full: LLM extraction per chunk (slow, accurate)
- hybrid: LLM for Domain per article + heuristics for Task/Modifier (fast, good)

Usage:
    # Pilot (10 articles, full TMD)
    ./.venv/bin/python tools/ingest_wikipedia_pipeline.py --limit 10

    # Hybrid mode (fast)
    LNSP_TMD_MODE=hybrid ./.venv/bin/python tools/ingest_wikipedia_pipeline.py --limit 10

    # Full (3000 articles)
    ./.venv/bin/python tools/ingest_wikipedia_pipeline.py --limit 3000
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
TMD_MODE = os.getenv("LNSP_TMD_MODE", "full")  # "full" or "hybrid"


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


def load_articles(file_path: str, limit: int) -> List[Dict]:
    """Load Wikipedia articles from JSONL"""
    articles = []
    with open(file_path) as f:
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
            "max_chunk_size": 320,
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


def ingest_chunks(chunks_data: List[Dict]):
    """Step 5: Ingest to PostgreSQL + FAISS"""
    response = requests.post(
        f"{INGEST_API}/ingest",
        json={"chunks": chunks_data},
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
    tmd_time = 0
    embedding_time = 0
    ingest_time = 0

    try:
        # Step 1: Episode chunking
        t0 = time.time()
        episodes = chunk_into_episodes(document_id, text)
        episode_time = (time.time() - t0) * 1000
        stats["episodes"] = len(episodes)

        all_chunks_data = []

        # Hybrid TMD mode: Extract Domain ONCE for entire article
        article_domain = None
        if TMD_MODE == "hybrid":
            t_domain = time.time()
            # Use article title + first 500 chars for domain classification
            article_summary = f"{title}. {text[:500]}"
            domain_result = extract_tmd(article_summary)
            article_domain = domain_result["domain_code"]
            tmd_time += (time.time() - t_domain) * 1000

        for ep_idx, episode in enumerate(episodes):
            episode_id = episode["episode_id"]

            # Step 2: Semantic chunking
            t1 = time.time()
            semantic_chunks = chunk_semantically(episode["text"])
            semantic_time += (time.time() - t1) * 1000

            # Process each chunk
            for seq_idx, chunk_text in enumerate(semantic_chunks):
                # Step 3: TMD extraction (mode-dependent)
                t2 = time.time()

                if TMD_MODE == "hybrid":
                    # Hybrid: Use article-level Domain + heuristics for T/M
                    task_code = classify_task(chunk_text)
                    modifier_code = classify_modifier(chunk_text)
                    domain_code = article_domain
                else:
                    # Full: LLM extraction per chunk
                    tmd = extract_tmd(chunk_text)
                    domain_code = tmd["domain_code"]
                    task_code = tmd["task_code"]
                    modifier_code = tmd["modifier_code"]

                tmd_time += (time.time() - t2) * 1000

                # Prepare chunk data
                chunk_data = {
                    "text": chunk_text,
                    "document_id": document_id,
                    "sequence_index": seq_idx,
                    "episode_id": episode_id,
                    "dataset_source": f"wikipedia_{title}",
                    "domain_code": domain_code,
                    "task_code": task_code,
                    "modifier_code": modifier_code
                }

                all_chunks_data.append(chunk_data)
                stats["chunks"] += 1

        # Step 4: Batch embeddings
        if all_chunks_data:
            texts = [c["text"] for c in all_chunks_data]

            t3 = time.time()
            embeddings = get_embeddings(texts)
            embedding_time = (time.time() - t3) * 1000

            # Add embeddings to chunk data
            for chunk_data, embedding in zip(all_chunks_data, embeddings):
                chunk_data["concept_vec"] = embedding

            # Step 5: Ingest
            t4 = time.time()
            ingest_chunks(all_chunks_data)
            ingest_time = (time.time() - t4) * 1000

        # Calculate total time
        total_time = (time.time() - article_start) * 1000

        # Record timings
        timings.add_timings(episode_time, semantic_time, tmd_time, embedding_time, ingest_time, total_time)

        stats["timings"] = {
            "episode_chunking_ms": episode_time,
            "semantic_chunking_ms": semantic_time,
            "tmd_extraction_ms": tmd_time,
            "embedding_ms": embedding_time,
            "ingestion_ms": ingest_time,
            "total_ms": total_time
        }

    except Exception as e:
        stats["errors"].append(str(e))

    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/datasets/wikipedia/wikipedia_simple_articles.jsonl")
    parser.add_argument("--limit", type=int, default=10, help="Number of articles to process")
    parser.add_argument("--skip-check", action="store_true", help="Skip API health check")

    args = parser.parse_args()

    print("🚀 Wikipedia Ingestion Pipeline")
    print("=" * 80)
    print(f"   TMD Mode: {TMD_MODE}")

    # Check APIs
    if not args.skip_check:
        if not check_apis():
            print("\n❌ Some APIs are not running. Start them first:")
            print("   Episode: ./.venv/bin/uvicorn app.api.episode_chunker:app --port 8900 &")
            print("   Semantic: ./.venv/bin/uvicorn app.api.chunking:app --port 8001 &")
            print("   TMD: ./.venv/bin/uvicorn app.api.tmd_router:app --port 8002 &")
            print("   Embeddings: ./.venv/bin/uvicorn app.api.vec2text_embedding_server:app --port 8767 &")
            print("   Ingest: ./.venv/bin/uvicorn app.api.ingest_chunks:app --port 8004 &")
            return 1

    # Load articles
    print(f"\n📥 Loading articles from {args.input}...")
    if not Path(args.input).exists():
        print(f"❌ File not found. Download first:")
        print(f"   ./.venv/bin/python tools/download_wikipedia.py --limit {args.limit}")
        return 1

    articles = load_articles(args.input, args.limit)
    print(f"   Loaded: {len(articles)} articles")

    # Process articles
    print(f"\n⚙️  Processing articles...")
    total_episodes = 0
    total_chunks = 0
    errors = []
    timings = PipelineTimings()

    for i, article in enumerate(tqdm(articles, desc="Articles"), 1):
        stats = process_article(article, i, timings)
        total_episodes += stats["episodes"]
        total_chunks += stats["chunks"]
        if stats["errors"]:
            errors.append((stats["title"], stats["errors"]))

    # Performance summary
    timing_summary = timings.summary()

    # Summary
    print(f"\n✅ Pipeline Complete!")
    print(f"   Articles processed: {len(articles)}")
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
        print(f"   Throughput: {len(articles)/(timing_summary['total_pipeline']['total_ms']/1000):.2f} articles/sec")
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
            "articles_processed": len(articles),
            "total_episodes": total_episodes,
            "total_chunks": total_chunks,
            "errors": len(errors)
        }, f, indent=2)
    print(f"\n💾 Metrics saved to: {metrics_file}")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
