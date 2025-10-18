#!/usr/bin/env python3
"""
Prepare LVM Training Data from Wikipedia 500K

Converts Wikipedia articles into training sequences:
- Split articles into sentences
- Create sequences: 5 context sentences → 1 target sentence
- Encode with GTR-T5 (768D vectors)
- Save as train/val/test NPZ

Usage:
    python tools/prepare_wikipedia_training_data.py \
        --input data/datasets/wikipedia/wikipedia_500k.jsonl \
        --output artifacts/lvm/data/wikipedia_500k_training.npz \
        --max-articles 500000 \
        --context-size 5
"""

import argparse
import json
import numpy as np
import requests
import re
from pathlib import Path
from tqdm import tqdm
from typing import List, Tuple


def split_into_sentences(text: str) -> List[str]:
    """
    Split text into sentences using simple regex.
    Better than nltk.sent_tokenize for speed on large datasets.
    """
    # Split on period, exclamation, question mark followed by space/newline
    sentences = re.split(r'(?<=[.!?])\s+', text)

    # Filter out very short sentences (likely errors)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 20]

    return sentences


def encode_sentences_batch(sentences: List[str], batch_size: int = 32) -> np.ndarray:
    """
    Encode sentences using GTR-T5 encoder API (port 8767).
    Processes in batches for efficiency.
    """
    url = "http://localhost:8767/embed"
    all_vectors = []

    for i in range(0, len(sentences), batch_size):
        batch = sentences[i:i+batch_size]

        try:
            response = requests.post(url, json={
                "texts": batch,
                "normalize": True
            }, timeout=30)
            response.raise_for_status()

            result = response.json()
            vectors = np.array(result['embeddings'])
            all_vectors.append(vectors)

        except Exception as e:
            print(f"⚠️  Error encoding batch {i//batch_size}: {e}")
            # Return zeros for failed batches (will be filtered later)
            all_vectors.append(np.zeros((len(batch), 768)))

    return np.vstack(all_vectors)


def process_wikipedia_articles(
    input_path: str,
    context_size: int = 5,
    max_articles: int = None,
    batch_size: int = 32
) -> Tuple[List[np.ndarray], np.ndarray]:
    """
    Process Wikipedia articles into training sequences.

    For each article:
    1. Split into sentences
    2. Create sliding windows: [s0...s4] → s5, [s1...s5] → s6, etc.
    3. Encode all sentences
    4. Store context sequences + targets

    Returns:
        contexts: List of context arrays (each is context_size x 768)
        targets: Array of target vectors (N x 768)
    """
    print(f"📖 Processing Wikipedia articles from {input_path}")
    print(f"   Context size: {context_size} sentences")
    print(f"   Max articles: {max_articles if max_articles else 'All'}")
    print()

    all_contexts = []
    all_targets = []

    articles_processed = 0
    sequences_created = 0
    skipped_articles = 0

    with open(input_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Processing articles", unit="article"):
            if max_articles and articles_processed >= max_articles:
                break

            try:
                article = json.loads(line)
                text = article.get('text', '')

                # Split into sentences
                sentences = split_into_sentences(text)

                # Need at least context_size + 1 sentences
                if len(sentences) < context_size + 1:
                    skipped_articles += 1
                    continue

                # Encode all sentences for this article
                vectors = encode_sentences_batch(sentences, batch_size=batch_size)

                # Create sliding window sequences
                for i in range(len(vectors) - context_size):
                    context = vectors[i:i+context_size]  # (5, 768)
                    target = vectors[i+context_size]      # (768,)

                    all_contexts.append(context)
                    all_targets.append(target)
                    sequences_created += 1

                articles_processed += 1

                # Progress update every 1000 articles
                if articles_processed % 1000 == 0:
                    print(f"   Processed {articles_processed} articles → {sequences_created} sequences")

            except Exception as e:
                print(f"⚠️  Error processing article: {e}")
                skipped_articles += 1
                continue

    print()
    print(f"✅ Processing complete:")
    print(f"   Articles processed: {articles_processed}")
    print(f"   Articles skipped: {skipped_articles}")
    print(f"   Training sequences: {sequences_created}")
    print()

    return all_contexts, np.array(all_targets)


def save_train_val_test(
    contexts: List[np.ndarray],
    targets: np.ndarray,
    output_path: str,
    train_split: float = 0.7,
    val_split: float = 0.15
):
    """
    Convert to fixed-size arrays and split into train/val/test.
    """
    print(f"💾 Saving training data...")

    # Convert contexts to numpy array (all same size now)
    contexts = np.array(contexts)  # Shape: (N, context_size, 768)

    print(f"   Context sequences shape: {contexts.shape}")
    print(f"   Target vectors shape: {targets.shape}")

    # Shuffle data
    N = len(contexts)
    indices = np.random.permutation(N)
    contexts = contexts[indices]
    targets = targets[indices]

    # Split
    train_end = int(N * train_split)
    val_end = int(N * (train_split + val_split))

    train_ctx = contexts[:train_end]
    train_tgt = targets[:train_end]

    val_ctx = contexts[train_end:val_end]
    val_tgt = targets[train_end:val_end]

    test_ctx = contexts[val_end:]
    test_tgt = targets[val_end:]

    # Save
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        context_sequences=train_ctx,
        target_vectors=train_tgt,
        val_context_sequences=val_ctx,
        val_target_vectors=val_tgt,
        test_context_sequences=test_ctx,
        test_target_vectors=test_tgt
    )

    print(f"✅ Saved to {output_path}")
    print(f"   Train: {len(train_ctx)} sequences ({len(train_ctx)/N*100:.1f}%)")
    print(f"   Val:   {len(val_ctx)} sequences ({len(val_ctx)/N*100:.1f}%)")
    print(f"   Test:  {len(test_ctx)} sequences ({len(test_ctx)/N*100:.1f}%)")
    print()

    # Calculate file size
    file_size = Path(output_path).stat().st_size / (1024**3)  # GB
    print(f"   File size: {file_size:.2f} GB")
    print()


def main():
    parser = argparse.ArgumentParser(description="Prepare Wikipedia training data for LVM")
    parser.add_argument(
        "--input",
        default="data/datasets/wikipedia/wikipedia_500k.jsonl",
        help="Input Wikipedia JSONL file"
    )
    parser.add_argument(
        "--output",
        default="artifacts/lvm/data/wikipedia_500k_training.npz",
        help="Output NPZ file"
    )
    parser.add_argument(
        "--context-size",
        type=int,
        default=5,
        help="Number of context sentences (default: 5)"
    )
    parser.add_argument(
        "--max-articles",
        type=int,
        default=None,
        help="Maximum articles to process (default: all)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Encoding batch size (default: 32)"
    )
    parser.add_argument(
        "--train-split",
        type=float,
        default=0.7,
        help="Train split ratio (default: 0.7)"
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.15,
        help="Validation split ratio (default: 0.15)"
    )

    args = parser.parse_args()

    print("=" * 80)
    print("WIKIPEDIA TRAINING DATA PREPARATION")
    print("=" * 80)
    print()

    # Check if encoder is available
    try:
        response = requests.get("http://localhost:8767/health", timeout=5)
        response.raise_for_status()
        print("✅ GTR-T5 encoder (port 8767) is available")
    except:
        print("❌ ERROR: GTR-T5 encoder not available on port 8767")
        print("   Please start: ./.venv/bin/uvicorn app.api.vec2text_embedding_server:app --host 127.0.0.1 --port 8767")
        return

    print()

    # Process articles
    contexts, targets = process_wikipedia_articles(
        args.input,
        context_size=args.context_size,
        max_articles=args.max_articles,
        batch_size=args.batch_size
    )

    if len(contexts) == 0:
        print("❌ No training sequences created!")
        return

    # Save train/val/test
    save_train_val_test(
        contexts,
        targets,
        args.output,
        train_split=args.train_split,
        val_split=args.val_split
    )

    print("=" * 80)
    print("✅ TRAINING DATA PREPARATION COMPLETE!")
    print("=" * 80)
    print()
    print("Next steps:")
    print("  1. Train Transformer: nohup ./.venv/bin/python app/lvm/train_transformer.py \\")
    print(f"                             --data {args.output} \\")
    print("                             --epochs 20 --device cpu > /tmp/training.log 2>&1 &")
    print()
    print("  2. Monitor progress: tail -f /tmp/training.log")
    print()


if __name__ == "__main__":
    main()
