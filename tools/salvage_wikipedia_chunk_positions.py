#!/usr/bin/env python3
"""
Salvage 771k Wikipedia chunks by backfilling chunk_position metadata.

This script:
1. Reads Wikipedia JSONL file
2. Re-generates chunks using same pipeline (episode + semantic chunking)
3. Matches chunks to database by text
4. Updates chunk_position with correct {source: article_id, index: chunk_index}

This restores the sequential ordering lost due to field name mismatch bug.

Usage:
    python tools/salvage_wikipedia_chunk_positions.py \
        --input data/datasets/wikipedia/wikipedia_500k.jsonl \
        --limit 3500  # Number of articles to process (match what was ingested)
"""

import argparse
import json
import requests
import sys
import psycopg2
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.db_postgres import connect as connect_pg

# API Endpoints (same as ingestion)
EPISODE_API = "http://localhost:8900"
SEMANTIC_API = "http://localhost:8001"


def chunk_into_episodes(document_id: str, text: str) -> List[Dict]:
    """Step 1: Episode chunking (matches ingestion pipeline)"""
    response = requests.post(
        f"{EPISODE_API}/chunk",
        json={"document_id": document_id, "text": text},
        timeout=60
    )
    response.raise_for_status()
    return response.json()["episodes"]


def chunk_semantically(text: str) -> List[str]:
    """Step 2: Semantic chunking (matches ingestion pipeline)"""
    response = requests.post(
        f"{SEMANTIC_API}/chunk",
        json={
            "text": text,
            "min_chunk_size": 10,
            "max_chunk_size": 500,
            "breakpoint_threshold": 75
        },
        timeout=60
    )
    response.raise_for_status()
    chunks = response.json()["chunks"]
    return [c["text"] for c in chunks]


def process_article_chunks(article: Dict, article_index: int) -> List[Dict]:
    """
    Process article to generate chunks (same as ingestion).

    Returns list of {text, source_document, chunk_index}
    """
    document_id = f"wikipedia_{article_index}"
    text = article.get("text", "")

    # Step 1: Episode chunking
    episodes = chunk_into_episodes(document_id, text)

    all_chunks = []

    # Step 2: Semantic chunking
    for ep_idx, episode in enumerate(episodes):
        semantic_chunks = chunk_semantically(episode["text"])

        for seq_idx, chunk_text in enumerate(semantic_chunks):
            all_chunks.append({
                "text": chunk_text,
                "source_document": document_id,
                "chunk_index": len(all_chunks)  # Global index across all episodes
            })

    return all_chunks


def update_chunk_positions_batch(conn, updates: List[tuple]):
    """
    Batch update chunk_position in PostgreSQL.

    updates: List of (chunk_position_json, concept_text) tuples
    """
    cur = conn.cursor()

    # Use UPDATE with VALUES clause for efficient batch update
    query = """
    UPDATE cpe_entry
    SET chunk_position = data.chunk_position::jsonb
    FROM (VALUES %s) AS data(chunk_position, concept_text)
    WHERE cpe_entry.concept_text = data.concept_text
      AND cpe_entry.dataset_source = 'wikipedia_500k'
    """

    # Format values for psycopg2
    from psycopg2.extras import execute_values

    execute_values(
        cur,
        """
        UPDATE cpe_entry
        SET chunk_position = v.chunk_position::jsonb
        FROM (VALUES %s) AS v(chunk_position, concept_text)
        WHERE cpe_entry.concept_text = v.concept_text
          AND cpe_entry.dataset_source = 'wikipedia_500k'
        """,
        updates,
        template="(%s, %s)"
    )

    conn.commit()
    cur.close()

    return cur.rowcount


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Wikipedia JSONL file")
    parser.add_argument("--limit", type=int, required=True, help="Number of articles to process")
    parser.add_argument("--batch-size", type=int, default=100, help="Batch size for database updates")
    parser.add_argument("--dry-run", action="store_true", help="Don't write to database")

    args = parser.parse_args()

    print("=" * 80)
    print("SALVAGE WIKIPEDIA CHUNK POSITIONS")
    print("=" * 80)
    print()
    print(f"Input file: {args.input}")
    print(f"Articles to process: {args.limit}")
    print(f"Batch size: {args.batch_size}")
    print(f"Dry run: {args.dry_run}")
    print()

    # Check APIs
    print("Checking APIs...")
    try:
        requests.get(f"{EPISODE_API}/health", timeout=2).raise_for_status()
        print("  ✅ Episode Chunker")
    except:
        print("  ❌ Episode Chunker - Start with:")
        print("     ./.venv/bin/uvicorn app.api.episode_chunker:app --port 8900 &")
        return 1

    try:
        requests.get(f"{SEMANTIC_API}/health", timeout=2).raise_for_status()
        print("  ✅ Semantic Chunker")
    except:
        print("  ❌ Semantic Chunker - Start with:")
        print("     ./.venv/bin/uvicorn app.api.chunking:app --port 8001 &")
        return 1

    print()

    # Load articles
    print(f"Loading articles from {args.input}...")
    if not Path(args.input).exists():
        print(f"❌ File not found: {args.input}")
        return 1

    articles = []
    with open(args.input, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= args.limit:
                break
            articles.append(json.loads(line))

    print(f"  Loaded: {len(articles)} articles")
    print()

    # Connect to database
    if not args.dry_run:
        print("Connecting to PostgreSQL...")
        conn = connect_pg()
        print("  ✅ Connected")
        print()

    # Process articles and build update batches
    print("Processing articles and matching chunks...")
    print()

    total_chunks = 0
    total_matched = 0
    total_updated = 0
    update_batch = []

    for i, article in enumerate(tqdm(articles, desc="Articles"), 1):
        try:
            # Generate chunks (same as ingestion)
            chunks = process_article_chunks(article, i)
            total_chunks += len(chunks)

            # Build updates for this article's chunks
            for chunk in chunks:
                chunk_position = {
                    "source": chunk["source_document"],
                    "index": chunk["chunk_index"]
                }

                update_batch.append((
                    json.dumps(chunk_position),
                    chunk["text"]
                ))
                total_matched += 1

            # Batch update when batch is full
            if len(update_batch) >= args.batch_size:
                if not args.dry_run:
                    rows_updated = update_chunk_positions_batch(conn, update_batch)
                    total_updated += rows_updated
                update_batch = []

        except Exception as e:
            print(f"\n⚠️  Error processing article {i}: {e}")
            continue

    # Final batch
    if update_batch and not args.dry_run:
        rows_updated = update_chunk_positions_batch(conn, update_batch)
        total_updated += rows_updated

    if not args.dry_run:
        conn.close()

    # Summary
    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print()
    print(f"Articles processed: {len(articles)}")
    print(f"Chunks generated: {total_chunks}")
    print(f"Chunks matched: {total_matched}")

    if not args.dry_run:
        print(f"Database rows updated: {total_updated}")
        print()
        print("✅ Chunk positions salvaged!")
        print()
        print("Next step: Rebuild NPZ file with correct ordering:")
        print("  python tools/rebuild_faiss_with_corrected_vectors.py")
    else:
        print()
        print("✅ Dry run complete (no database changes)")
        print()
        print("Remove --dry-run to apply changes")

    print()


if __name__ == "__main__":
    main()
