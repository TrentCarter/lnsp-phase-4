#!/usr/bin/env python3
"""
Optimized Wikipedia Ingestion Pipeline - Simple Chunking Only

Pipeline (streamlined):
1. Download Wikipedia articles (local)
2. Simple Chunker API :8001 (word-based chunks, NO episode/semantic overhead)
3. Vec2Text-Compatible GTR-T5 API :8767 (768D vectors with caching)
4. Ingest API :8004 (heuristic TMD + PostgreSQL)

Key Optimizations:
- NO episode chunking (removed bottleneck)
- NO semantic chunking overhead
- Simple word-based chunking only (target: 40 words, max: 500 chars)
- Heuristic TMD mode (no LLM calls during ingest)
- Per-article profiling to logs/ingest_profile.jsonl
- Embedding cache for repeated text (headers, etc.)

Usage:
    # Fresh start (optimized settings)
    LNSP_TMD_MODE=heuristic OMP_NUM_THREADS=8 \\
    ./.venv/bin/python tools/ingest_wikipedia_pipeline_optimized.py --limit 100

    # Check profiling results
    python -c "
import json, statistics as st
L=[json.loads(x) for x in open('logs/ingest_profile.jsonl')]
for k in ['t_chunk_ms','t_embed_ms','t_db_ms']:
    xs=[r[k] for r in L]
    print(k, 'median', int(st.median(xs)), 'p90', int(sorted(xs)[int(.9*len(xs))]))
print('chunks/article median', int(st.median([r['n_chunks'] for r in L])))
"
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm

# API Endpoints
SEMANTIC_API = "http://localhost:8001"  # Only using simple chunker
EMBEDDING_API = "http://localhost:8767"
INGEST_API = "http://localhost:8004"

# Profiling output
PROFILE_LOG = "logs/ingest_profile.jsonl"


def check_apis() -> bool:
    """Check API health"""
    import requests

    tmd_mode = os.environ.get("LNSP_TMD_MODE", "heuristic")
    print("🔍 Checking APIs...")

    endpoints = [
        ("Simple Chunker", f"{SEMANTIC_API}/health"),
        ("GTR-T5 Embeddings", f"{EMBEDDING_API}/health"),
        ("Ingest", f"{INGEST_API}/health"),
    ]

    for name, url in endpoints:
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                print(f"  ✅ {name}: {url}")
            else:
                print(f"  ❌ {name}: {url} (HTTP {response.status_code})")
                return False
        except Exception as e:
            print(f"  ❌ {name}: {url} ({e})")
            return False

    print(f"  ℹ️  TMD mode: {tmd_mode} (heuristic recommended for speed)")
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


def chunk_simple(text: str, target_words: int = 40, max_chars: int = 500) -> List[str]:
    """
    Simple word-based chunking with optimized parameters

    Args:
        text: Full article text
        target_words: Target words per chunk (default 40 = ~160-200 chars)
        max_chars: Hard max to avoid mid-sentence splits

    Returns:
        List of chunk texts
    """
    import requests

    response = requests.post(
        f"{SEMANTIC_API}/chunk",
        json={
            "text": text,
            "mode": "simple",
            "min_chunk_size": 10,  # Allow short concepts
            "max_chunk_size": max_chars,  # Hard cap
            "target_words": target_words,  # New: control chunk density
            "breakpoint_threshold": 85  # Ignored in simple mode
        },
        timeout=60
    )
    response.raise_for_status()
    chunks = response.json()["chunks"]
    return [c["text"] for c in chunks]


def get_embeddings(texts: List[str]) -> List[List[float]]:
    """Get GTR-T5 embeddings (with server-side caching if available)"""
    import requests

    response = requests.post(
        f"{EMBEDDING_API}/embed",
        json={"texts": texts},
        timeout=120
    )
    response.raise_for_status()
    return response.json()["embeddings"]


def ingest_chunks(chunks_data: List[Dict], dataset_source: str = "wikipedia_500k"):
    """Ingest to PostgreSQL (heuristic TMD mode)"""
    import requests

    # Remove dataset_source from individual chunks
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


def process_article(article: Dict, article_index: int) -> Dict:
    """
    Process single article with profiling

    Returns profiling dict for JSONL logging
    """
    document_id = f"wikipedia_{article_index}"
    title = article.get("title", "Unknown")
    text = article.get("text", "")

    # === PROFILED SECTION START ===
    t0 = time.time()

    # Step 1: Simple chunking ONLY (no episode overhead)
    chunks = chunk_simple(text, target_words=40, max_chars=500)
    t1 = time.time()

    # Prepare chunk data
    chunks_data = []
    for chunk_idx, chunk_text in enumerate(chunks):
        chunk_data = {
            "text": chunk_text,
            "source_document": document_id,
            "article_title": title,
            "article_index": article_index,
            "chunk_index": chunk_idx,  # Sequential 0..N-1 per article
            "dataset_source": "wikipedia_500k",
        }
        chunks_data.append(chunk_data)

    # Step 2: Batch embeddings
    texts = [c["text"] for c in chunks_data]
    embeddings = get_embeddings(texts)
    t2 = time.time()

    # Add embeddings to chunk data
    for chunk_data, embedding in zip(chunks_data, embeddings):
        chunk_data["concept_vec"] = embedding

    # Step 3: Database write (heuristic TMD included)
    ingest_chunks(chunks_data)
    t3 = time.time()

    # === PROFILED SECTION END ===

    # Build profiling record
    profile = {
        "article_index": article_index,
        "article_title": title,
        "n_chunks": len(chunks),
        "t_chunk_ms": int((t1 - t0) * 1000),
        "t_embed_ms": int((t2 - t1) * 1000),
        "t_db_ms": int((t3 - t2) * 1000),
        "t_total_ms": int((t3 - t0) * 1000),
    }

    # Append to profile log
    with open(PROFILE_LOG, "a") as f:
        f.write(json.dumps(profile) + "\n")

    return profile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/datasets/wikipedia/wikipedia_500k.jsonl")
    parser.add_argument("--limit", type=int, default=100, help="Number of articles to process")
    parser.add_argument("--skip-offset", type=int, default=0, help="Articles to skip (for batching)")
    parser.add_argument("--skip-check", action="store_true", help="Skip API health check")

    args = parser.parse_args()

    # Verify TMD mode
    TMD_MODE = os.environ.get("LNSP_TMD_MODE", "heuristic")
    if TMD_MODE != "heuristic":
        print(f"⚠️  Warning: TMD_MODE={TMD_MODE}. Recommended: heuristic for speed")

    print("🚀 Wikipedia Ingestion Pipeline (Optimized - Simple Chunking Only)")
    print("=" * 80)
    print(f"   TMD Mode: {TMD_MODE}")
    print(f"   Chunking: Simple word-based (target: 40 words, max: 500 chars)")
    print(f"   Profiling: {PROFILE_LOG}")

    # Ensure logs directory exists
    os.makedirs("logs", exist_ok=True)

    # Clear old profile log
    if os.path.exists(PROFILE_LOG):
        os.remove(PROFILE_LOG)
        print(f"   Cleared old profile log")

    # Check APIs
    if not args.skip_check:
        if not check_apis():
            print("\n❌ Some APIs are not running. Start them first:")
            print("   Semantic: ./.venv/bin/uvicorn app.api.chunking:app --port 8001 &")
            print("   Embeddings: ./.venv/bin/uvicorn app.api.vec2text_embedding_server:app --port 8767 &")
            print("   Ingest: ./.venv/bin/uvicorn app.api.ingest_chunks:app --port 8004 &")
            return 1

    # Load articles
    print(f"\n📥 Loading articles from {args.input}...")
    if not Path(args.input).exists():
        print(f"❌ File not found: {args.input}")
        return 1

    articles = load_articles(args.input, args.limit, args.skip_offset)
    print(f"   Loaded: {len(articles)} articles (skipped: {args.skip_offset})")

    # Process articles
    print(f"\n⚙️  Processing articles...")
    total_chunks = 0
    errors = []

    for i, article in enumerate(tqdm(articles, desc="Articles"), args.skip_offset + 1):
        try:
            profile = process_article(article, i)
            total_chunks += profile["n_chunks"]
        except Exception as e:
            print(f"\n❌ Error processing article {i}: {e}")
            errors.append({"article_index": i, "error": str(e)})

    # Print summary
    print(f"\n✅ Pipeline complete!")
    print(f"   Articles: {len(articles)}")
    print(f"   Chunks: {total_chunks}")
    print(f"   Errors: {len(errors)}")
    print(f"   Profile: {PROFILE_LOG}")

    # Quick profiling summary
    print(f"\n📊 Performance Summary:")
    if os.path.exists(PROFILE_LOG):
        import statistics as st
        profiles = [json.loads(x) for x in open(PROFILE_LOG)]

        for key, label in [("t_chunk_ms", "Chunking"), ("t_embed_ms", "Embedding"), ("t_db_ms", "Database")]:
            vals = [p[key] for p in profiles]
            print(f"   {label}: median={int(st.median(vals))}ms, p90={int(sorted(vals)[int(0.9*len(vals))])}ms")

        chunks_per_article = [p["n_chunks"] for p in profiles]
        print(f"   Chunks/article: median={int(st.median(chunks_per_article))}, p90={int(sorted(chunks_per_article)[int(0.9*len(chunks_per_article))])}")

        total_times = [p["t_total_ms"] for p in profiles]
        print(f"   Total/article: median={int(st.median(total_times))}ms, p90={int(sorted(total_times)[int(0.9*len(total_times))])}ms")

    return 0 if not errors else 1


if __name__ == "__main__":
    sys.exit(main())
