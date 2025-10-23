#!/usr/bin/env python3
"""
Test Ingestion with Sequential UUIDs and Article-Level Domain

Requirements:
1. Sequential UUIDs: 00000001-..., 00000002-..., etc.
2. Article IDs: Article 1, Article 2, etc.
3. Chunk IDs reset per article: Chunk 0, 1, 2... within each article
4. Timestamps on every chunk
5. Domain (D) extracted once per article, shared by all chunks
6. Task/Modifier (T/M) extracted per chunk

Format:
    UUID 1, Article 1, Chunk 0
    UUID 2, Article 1, Chunk 1
    UUID 3, Article 1, Chunk 2
    UUID 4, Article 2, Chunk 0
    UUID 5, Article 2, Chunk 1
    ...

Usage:
    python tools/test_sequential_ingestion.py --limit 10
"""

import argparse
import json
import os
import requests
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Dict

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.db_postgres import connect as connect_pg

# API Endpoints
EPISODE_API = "http://localhost:8900"
SEMANTIC_API = "http://localhost:8001"
EMBEDDING_API = "http://localhost:8767"

# LLM for TMD extraction
LLM_ENDPOINT = os.environ.get("LNSP_LLM_ENDPOINT", "http://localhost:11434")
LLM_MODEL = os.environ.get("LNSP_LLM_MODEL", "llama3.1:8b")


def generate_sequential_uuid(counter: int) -> str:
    """Generate sequential UUID: 00000001-0000-0000-0000-000000000000"""
    return f"{counter:08d}-0000-0000-0000-000000000000"


def chunk_into_episodes(document_id: str, text: str) -> List[Dict]:
    """Episode chunking"""
    response = requests.post(
        f"{EPISODE_API}/chunk",
        json={"document_id": document_id, "text": text},
        timeout=60
    )
    response.raise_for_status()
    return response.json()["episodes"]


def chunk_semantically(text: str) -> List[str]:
    """Semantic chunking"""
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


def get_embeddings(texts: List[str]) -> List[List[float]]:
    """Get GTR-T5 embeddings"""
    response = requests.post(
        f"{EMBEDDING_API}/embed",
        json={"texts": texts},
        timeout=120
    )
    response.raise_for_status()
    return response.json()["embeddings"]


def extract_article_domain(title: str, first_paragraph: str) -> int:
    """
    Extract domain code for entire article using LLM.

    Domain codes:
    0 = General/Unknown
    1 = Science/Technology
    2 = History/Culture
    3 = Geography/Places
    4 = Arts/Entertainment
    5 = Sports/Recreation
    """
    prompt = f"""Given this Wikipedia article title and opening:

Title: {title}
Opening: {first_paragraph[:500]}

What is the PRIMARY domain? Reply with ONLY the number:
0 = General/Unknown
1 = Science/Technology
2 = History/Culture
3 = Geography/Places
4 = Arts/Entertainment
5 = Sports/Recreation"""

    try:
        response = requests.post(
            f"{LLM_ENDPOINT}/api/chat",
            json={
                "model": LLM_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False
            },
            timeout=30
        )
        response.raise_for_status()

        answer = response.json()["message"]["content"].strip()
        # Extract first digit
        for char in answer:
            if char.isdigit():
                code = int(char)
                if 0 <= code <= 5:
                    return code

        return 0  # Default to General

    except Exception as e:
        print(f"⚠️  LLM domain extraction failed: {e}, defaulting to 0")
        return 0


def extract_chunk_task_modifier(chunk_text: str) -> tuple:
    """
    Extract task and modifier codes for a chunk using LLM.

    Task codes:
    0 = General/Unknown
    1 = Define/Explain
    2 = Describe/Detail
    3 = Compare/Contrast

    Modifier codes:
    0 = None
    1 = Historical
    2 = Technical
    3 = Conceptual
    """
    prompt = f"""Given this text chunk:

{chunk_text[:300]}

What is the primary TASK and MODIFIER? Reply with ONLY two numbers separated by comma.

Task:
0 = General/Unknown
1 = Define/Explain
2 = Describe/Detail
3 = Compare/Contrast

Modifier:
0 = None
1 = Historical
2 = Technical
3 = Conceptual

Format: task,modifier (e.g., "1,2")"""

    try:
        response = requests.post(
            f"{LLM_ENDPOINT}/api/chat",
            json={
                "model": LLM_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False
            },
            timeout=30
        )
        response.raise_for_status()

        answer = response.json()["message"]["content"].strip()
        # Parse "task,modifier"
        parts = answer.replace(" ", "").split(",")
        if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
            task = int(parts[0])
            modifier = int(parts[1])
            if 0 <= task <= 3 and 0 <= modifier <= 3:
                return (task, modifier)

        return (0, 0)  # Default

    except Exception as e:
        print(f"⚠️  LLM task/modifier extraction failed: {e}, defaulting to 0,0")
        return (0, 0)


def insert_chunk_to_db(conn, uuid: str, article_id: str, chunk_index: int,
                      chunk_text: str, embedding: List[float],
                      domain_code: int, task_code: int, modifier_code: int,
                      timestamp: datetime):
    """Insert chunk directly to PostgreSQL with sequential UUID"""

    cur = conn.cursor()

    # Calculate TMD bits
    tmd_bits = (domain_code << 6) | (task_code << 3) | modifier_code
    tmd_lane = f"lane_{tmd_bits % 16}"
    lane_index = tmd_bits % 32768

    # Insert into cpe_entry
    cur.execute("""
        INSERT INTO cpe_entry (
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
            dataset_source,
            chunk_position,
            tmd_bits,
            tmd_lane,
            lane_index,
            created_at
        ) VALUES (
            %s::uuid,
            %s,
            %s,
            %s,
            '',
            '',
            %s,
            %s,
            %s,
            'semantic_chunk',
            'test_sequential',
            %s::jsonb,
            %s,
            %s,
            %s,
            %s
        )
    """, (
        uuid,
        chunk_text,  # mission_text
        chunk_text,  # source_chunk
        chunk_text,  # concept_text
        domain_code,
        task_code,
        modifier_code,
        json.dumps({"source": article_id, "index": chunk_index}),
        tmd_bits,
        tmd_lane,
        lane_index,
        timestamp
    ))

    # Insert into cpe_vectors
    embedding_str = '[' + ','.join(str(x) for x in embedding) + ']'

    cur.execute("""
        INSERT INTO cpe_vectors (
            cpe_id,
            concept_vec
        ) VALUES (
            %s::uuid,
            %s
        )
    """, (uuid, embedding_str))

    cur.close()


def process_article(article: Dict, article_num: int, uuid_counter: int, conn) -> int:
    """
    Process one article with sequential UUIDs.

    Returns: updated uuid_counter
    """
    article_id = f"article_{article_num}"
    title = article.get("title", "Unknown")
    text = article.get("text", "")

    print(f"\n📄 Article {article_num}: {title}")

    # Step 1: Episode chunking
    episodes = chunk_into_episodes(article_id, text)
    print(f"   Episodes: {len(episodes)}")

    # Step 2: Semantic chunking (all episodes)
    all_chunk_texts = []
    for episode in episodes:
        semantic_chunks = chunk_semantically(episode["text"])
        all_chunk_texts.extend(semantic_chunks)

    print(f"   Chunks: {len(all_chunk_texts)}")

    if not all_chunk_texts:
        print(f"   ⚠️  No chunks generated, skipping")
        return uuid_counter

    # Step 3: Get embeddings (batch)
    print(f"   Getting embeddings...")
    embeddings = get_embeddings(all_chunk_texts)

    # Step 4: Extract DOMAIN once for entire article
    print(f"   Extracting article domain...")
    first_chunk = all_chunk_texts[0]
    domain_code = extract_article_domain(title, first_chunk)
    print(f"   Domain: {domain_code}")

    # Step 5: Extract task/modifier per chunk and insert
    timestamp = datetime.now()

    for chunk_index, (chunk_text, embedding) in enumerate(zip(all_chunk_texts, embeddings)):
        # Extract task/modifier for this chunk
        task_code, modifier_code = extract_chunk_task_modifier(chunk_text)

        # Generate sequential UUID
        uuid = generate_sequential_uuid(uuid_counter)

        # Insert to database
        insert_chunk_to_db(
            conn, uuid, article_id, chunk_index,
            chunk_text, embedding,
            domain_code, task_code, modifier_code,
            timestamp
        )

        uuid_counter += 1

        if chunk_index == 0:
            print(f"   Chunk 0: UUID={uuid}, D={domain_code}, T={task_code}, M={modifier_code}")
        elif chunk_index == len(all_chunk_texts) - 1:
            print(f"   Chunk {chunk_index}: UUID={uuid}, D={domain_code}, T={task_code}, M={modifier_code}")

    conn.commit()
    print(f"   ✅ Inserted {len(all_chunk_texts)} chunks")

    return uuid_counter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/datasets/wikipedia/wikipedia_500k.jsonl")
    parser.add_argument("--limit", type=int, default=10, help="Number of articles")

    args = parser.parse_args()

    print("=" * 80)
    print("TEST SEQUENTIAL INGESTION")
    print("=" * 80)
    print(f"Articles: {args.limit}")
    print(f"Dataset: test_sequential")
    print()

    # Check APIs
    print("Checking APIs...")
    try:
        requests.get(f"{EPISODE_API}/health", timeout=2).raise_for_status()
        print("  ✅ Episode Chunker")
    except:
        print("  ❌ Episode Chunker not running!")
        return 1

    try:
        requests.get(f"{SEMANTIC_API}/health", timeout=2).raise_for_status()
        print("  ✅ Semantic Chunker")
    except:
        print("  ❌ Semantic Chunker not running!")
        return 1

    try:
        requests.get(f"{EMBEDDING_API}/health", timeout=2).raise_for_status()
        print("  ✅ GTR-T5 Embeddings")
    except:
        print("  ❌ GTR-T5 Embeddings not running!")
        return 1

    try:
        requests.get(f"{LLM_ENDPOINT}/api/tags", timeout=2).raise_for_status()
        print(f"  ✅ LLM ({LLM_MODEL})")
    except:
        print(f"  ❌ LLM not running!")
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
    print("Connecting to PostgreSQL...")
    conn = connect_pg()

    # Clear existing test data
    print("Clearing existing test_sequential data...")
    cur = conn.cursor()
    cur.execute("DELETE FROM cpe_entry WHERE dataset_source = 'test_sequential'")
    deleted = cur.rowcount
    conn.commit()
    cur.close()
    print(f"  Deleted {deleted} old rows")
    print()

    # Process articles
    print("Processing articles...")
    uuid_counter = 1  # Start from 1

    for article_num, article in enumerate(articles, 1):
        try:
            uuid_counter = process_article(article, article_num, uuid_counter, conn)
        except Exception as e:
            print(f"❌ Error processing article {article_num}: {e}")
            import traceback
            traceback.print_exc()
            continue

    conn.close()

    # Summary
    print()
    print("=" * 80)
    print("VERIFICATION")
    print("=" * 80)
    print()

    # Query to verify
    conn = connect_pg()
    cur = conn.cursor()

    cur.execute("""
        SELECT
            cpe_id::text,
            chunk_position->>'source' as article_id,
            (chunk_position->>'index')::int as chunk_index,
            domain_code,
            task_code,
            modifier_code,
            LEFT(concept_text, 60) as preview
        FROM cpe_entry
        WHERE dataset_source = 'test_sequential'
        ORDER BY cpe_id
        LIMIT 20
    """)

    print("First 20 chunks (ordered by UUID):")
    print()
    print(f"{'UUID':<40} {'Article':<12} {'Chunk':<7} {'D':<3} {'T':<3} {'M':<3} {'Preview'}")
    print("-" * 140)

    for row in cur.fetchall():
        uuid, article_id, chunk_idx, d, t, m, preview = row
        print(f"{uuid:<40} {article_id:<12} {chunk_idx:<7} {d:<3} {t:<3} {m:<3} {preview}...")

    cur.execute("""
        SELECT
            chunk_position->>'source' as article_id,
            COUNT(*) as chunk_count,
            MIN(domain_code) as domain_min,
            MAX(domain_code) as domain_max
        FROM cpe_entry
        WHERE dataset_source = 'test_sequential'
        GROUP BY article_id
        ORDER BY article_id
    """)

    print()
    print("Summary by article:")
    print(f"{'Article':<15} {'Chunks':<10} {'Domain':<10}")
    print("-" * 40)

    for article_id, chunk_count, d_min, d_max in cur.fetchall():
        domain_str = str(d_min) if d_min == d_max else f"{d_min}-{d_max} ⚠️"
        print(f"{article_id:<15} {chunk_count:<10} {domain_str}")

    cur.close()
    conn.close()

    print()
    print("=" * 80)
    print("✅ TEST INGESTION COMPLETE!")
    print("=" * 80)
    print()


if __name__ == "__main__":
    main()
