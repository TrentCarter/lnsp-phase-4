#!/usr/bin/env python3
"""
Create Out-of-Distribution (OOD) Test Set

Ingests NEW Wikipedia articles (not in training data) and creates test sequences.
This tests model generalization to truly unseen data.

Usage:
    python tools/create_ood_test_set.py \
        --start-article 8471 \
        --num-articles 500 \
        --output artifacts/lvm/wikipedia_ood_test_ctx5.npz
"""

import argparse
import json
import numpy as np
import psycopg2
import uuid
from pathlib import Path
from tqdm import tqdm
import requests


def simple_chunk_text(text: str, target_words: int = 60, max_chars: int = 500):
    """Simple word-based chunking (same as training data)"""
    words = text.split()
    chunks = []

    for i in range(0, len(words), target_words):
        chunk_words = words[i:i+target_words]
        chunk = ' '.join(chunk_words)[:max_chars]
        if len(chunk) > 20:  # Skip very short chunks
            chunks.append(chunk)

    return chunks


def encode_batch(texts: list, batch_size: int = 32):
    """Encode texts using GTR-T5 API"""
    url = "http://localhost:8767/embed"
    all_vectors = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]

        try:
            response = requests.post(url, json={
                "texts": batch,
                "normalize": True
            }, timeout=30)
            response.raise_for_status()

            result = response.json()
            vectors = np.array(result['embeddings'], dtype=np.float32)
            all_vectors.append(vectors)

        except Exception as e:
            print(f"⚠️  Error encoding batch: {e}")
            all_vectors.append(np.zeros((len(batch), 768), dtype=np.float32))

    return np.vstack(all_vectors)


def ingest_ood_articles(
    jsonl_path: str,
    start_article: int,
    num_articles: int,
    output_path: str
):
    """
    Ingest NEW Wikipedia articles for OOD testing.

    Process:
    1. Read articles from JSONL (starting at start_article)
    2. Chunk each article
    3. Encode chunks with GTR-T5
    4. Create sequences (5 context → 1 target)
    5. Save as test NPZ
    """
    print(f"📖 Processing NEW Wikipedia articles for OOD test")
    print(f"   Source: {jsonl_path}")
    print(f"   Article range: {start_article} - {start_article + num_articles - 1}")
    print()

    # Check encoder availability
    try:
        response = requests.get("http://localhost:8767/health", timeout=5)
        response.raise_for_status()
        print("✅ GTR-T5 encoder (port 8767) is available")
    except:
        print("❌ ERROR: GTR-T5 encoder not available on port 8767")
        print("   Please start: ./.venv/bin/uvicorn app.api.vec2text_embedding_server:app --host 127.0.0.1 --port 8767")
        return

    print()

    # Read and process articles
    all_sequences = []
    all_targets = []
    article_ids = []

    articles_processed = 0
    sequences_created = 0

    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(tqdm(f, desc="Reading articles", unit="article")):
            # Skip until start_article
            if idx + 1 < start_article:
                continue

            # Stop after num_articles
            if articles_processed >= num_articles:
                break

            try:
                article = json.loads(line)
                title = article.get('title', f'Article_{idx+1}')
                text = article.get('text', '')

                # Sanitize text
                title = title.encode('utf-8', errors='replace').decode('utf-8')
                text = text.encode('utf-8', errors='replace').decode('utf-8')

                # Chunk article
                chunks = simple_chunk_text(text)

                if len(chunks) < 7:  # Need at least 7 chunks (5 context + 1 target + 1 extra)
                    continue

                # Encode chunks
                vectors = encode_batch(chunks)

                # Create sequences
                for i in range(len(vectors) - 5):
                    ctx = vectors[i:i+5]      # [5, 768]
                    tgt = vectors[i+5]         # [768]

                    all_sequences.append(ctx)
                    all_targets.append(tgt)
                    article_ids.append(idx + 1)
                    sequences_created += 1

                articles_processed += 1

                if articles_processed % 100 == 0:
                    print(f"   Processed {articles_processed} articles → {sequences_created} sequences")

            except Exception as e:
                print(f"⚠️  Error processing article {idx+1}: {e}")
                continue

    print()
    print(f"✅ Processing complete:")
    print(f"   Articles: {articles_processed}")
    print(f"   Test sequences: {sequences_created}")
    print()

    if sequences_created == 0:
        print("❌ No test sequences created!")
        return

    # Convert to numpy
    sequences = np.array(all_sequences, dtype=np.float32)  # [N, 5, 768]
    targets = np.array(all_targets, dtype=np.float32)      # [N, 768]
    article_ids = np.array(article_ids, dtype=np.int32)    # [N]

    # Save
    print(f"💾 Saving OOD test set to {output_path}")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        output_path,
        context_sequences=sequences,
        target_vectors=targets,
        article_ids=article_ids
    )

    file_size = Path(output_path).stat().st_size / (1024**2)  # MB
    print(f"✅ Saved {sequences_created:,} OOD test sequences ({file_size:.1f} MB)")
    print()


def main():
    parser = argparse.ArgumentParser(description="Create OOD test set from NEW Wikipedia articles")
    parser.add_argument(
        "--jsonl",
        default="data/datasets/wikipedia/wikipedia_500k.jsonl",
        help="Wikipedia JSONL file"
    )
    parser.add_argument(
        "--start-article",
        type=int,
        default=8471,
        help="Starting article index (default: 8471, after training data)"
    )
    parser.add_argument(
        "--num-articles",
        type=int,
        default=500,
        help="Number of articles to process (default: 500)"
    )
    parser.add_argument(
        "--output",
        default="artifacts/lvm/wikipedia_ood_test_ctx5.npz",
        help="Output NPZ file"
    )

    args = parser.parse_args()

    print("=" * 80)
    print("CREATE OUT-OF-DISTRIBUTION (OOD) TEST SET")
    print("=" * 80)
    print()

    ingest_ood_articles(
        args.jsonl,
        args.start_article,
        args.num_articles,
        args.output
    )

    print("=" * 80)
    print("✅ OOD TEST SET READY!")
    print("=" * 80)
    print()
    print("Next step: Run comprehensive benchmark")
    print(f"  ./.venv/bin/python tools/benchmark_all_lvms_comprehensive.py")
    print()


if __name__ == "__main__":
    main()
