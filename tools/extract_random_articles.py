#!/usr/bin/env python3
"""
Extract random articles from vecRAG database for compliance checking.

This script pulls 2 random articles from the database and saves them to a JSON file
so users can verify data quality and compliance.

Usage:
    python tools/extract_random_articles.py --output sample_articles.json
"""

import argparse
import json
import psycopg2
import psycopg2.extras
import sys
from typing import Dict, List
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.db_postgres import connect as connect_pg


def get_distinct_article_indices(conn) -> List[int]:
    """Get all distinct article indices from chunk_position."""
    cur = conn.cursor()

    query = """
    SELECT DISTINCT (chunk_position->>'source') as source
    FROM cpe_entry
    WHERE dataset_source = 'wikipedia_500k'
      AND chunk_position->>'source' LIKE 'wikipedia_%'
    ORDER BY source
    """

    cur.execute(query)
    results = cur.fetchall()
    cur.close()

    # Extract article indices from "wikipedia_123" format
    article_indices = []
    for row in results:
        source = row[0]
        if source.startswith('wikipedia_'):
            try:
                article_idx = int(source.split('_')[1])
                article_indices.append(article_idx)
            except (IndexError, ValueError):
                continue

    return sorted(article_indices)


def get_article_chunks(conn, article_index: int) -> List[Dict]:
    """Get all chunks for a specific article."""
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

    query = """
    SELECT
        cpe_id,
        mission_text,
        source_chunk,
        concept_text,
        probe_question,
        expected_answer,
        domain_code,
        task_code,
        modifier_code,
        content_type,
        chunk_position,
        relations_text,
        echo_score,
        validation_status,
        tmd_bits,
        tmd_lane,
        lane_index,
        created_at
    FROM cpe_entry
    WHERE dataset_source = 'wikipedia_500k'
      AND chunk_position->>'source' = %s
    ORDER BY (chunk_position->>'index')::int
    """

    source = f'wikipedia_{article_index}'
    cur.execute(query, (source,))
    results = cur.fetchall()
    cur.close()

    # Convert to list of dicts
    chunks = []
    for row in results:
        chunk_data = dict(row)
        # Convert datetime to string for JSON serialization
        if chunk_data['created_at']:
            chunk_data['created_at'] = chunk_data['created_at'].isoformat()
        chunks.append(chunk_data)

    return chunks


def main():
    parser = argparse.ArgumentParser(description="Extract random articles from database")
    parser.add_argument("--output", required=True, help="Output JSON file path")
    parser.add_argument("--article-1", type=int, help="Specific first article index")
    parser.add_argument("--article-2", type=int, help="Specific second article index")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible selection")

    args = parser.parse_args()

    print("🔍 Extracting random articles from vecRAG database")
    print("=" * 60)

    # Connect to database
    print("Connecting to PostgreSQL...")
    try:
        conn = connect_pg()
        print("✅ Connected successfully")
    except Exception as e:
        print(f"❌ Failed to connect to database: {e}")
        return 1

    try:
        # Get all available article indices
        print("Finding available articles...")
        article_indices = get_distinct_article_indices(conn)
        print(f"Found {len(article_indices)} articles (indices {article_indices[0]} to {article_indices[-1]})")

        if len(article_indices) < 2:
            print("❌ Need at least 2 articles in database")
            return 1

        # Select 2 random articles
        import random
        random.seed(args.seed)

        if args.article_1 is not None and args.article_2 is not None:
            selected_indices = [args.article_1, args.article_2]
            print(f"Using specified articles: {selected_indices[0]}, {selected_indices[1]}")
        else:
            # Ensure we don't select the same article twice
            selected_indices = random.sample(article_indices, 2)
            print(f"Selected random articles: {selected_indices[0]}, {selected_indices[1]}")

        # Extract articles
        articles_data = {}

        for i, article_idx in enumerate(selected_indices, 1):
            print(f"\n📄 Extracting Article {article_idx}...")

            chunks = get_article_chunks(conn, article_idx)

            if not chunks:
                print(f"⚠️  No chunks found for article {article_idx}")
                continue

            print(f"  Found {len(chunks)} chunks")

            # Group chunks by article
            articles_data[f"article_{article_idx}"] = {
                "article_index": article_idx,
                "total_chunks": len(chunks),
                "chunks": chunks,
                "sample_chunk_text": chunks[0]["concept_text"][:100] + "..." if chunks[0]["concept_text"] else "N/A"
            }

        # Write to JSON file
        print(f"\n💾 Writing to {args.output}...")
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(articles_data, f, indent=2, ensure_ascii=False)

        print("✅ Successfully extracted articles!")
        print(f"📊 Summary:")
        print(f"  Article 1 (index {selected_indices[0]}): {len(articles_data.get(f'article_{selected_indices[0]}', {}).get('chunks', []))} chunks")
        print(f"  Article 2 (index {selected_indices[1]}): {len(articles_data.get(f'article_{selected_indices[1]}', {}).get('chunks', []))} chunks")
        print(f"  Output file: {args.output}")

        print("\n📋 Next steps:")
        print(f"  1. Review the extracted articles in: {args.output}")
        print(f"  2. Check data quality and compliance")
        print(f"  3. Continue monitoring the ingestion process")

    except Exception as e:
        print(f"❌ Error during extraction: {e}")
        return 1
    finally:
        conn.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
