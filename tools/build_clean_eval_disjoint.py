#!/usr/bin/env python3
"""
Build clean Phase-2 eval set with strict article disjointness.

Usage:
    python3 tools/build_clean_eval_disjoint.py \
        --train-npz artifacts/lvm/train_payload_aligned.npz \
        --out-train artifacts/lvm/train_clean.npz \
        --out-eval artifacts/lvm/eval_clean.npz \
        --eval-articles 50 \
        --min-chunks-per-article 10
"""
import argparse
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-npz', type=Path, required=True)
    parser.add_argument('--out-train', type=Path, required=True)
    parser.add_argument('--out-eval', type=Path, required=True)
    parser.add_argument('--eval-articles', type=int, default=50,
                        help='Number of articles for eval')
    parser.add_argument('--min-chunks-per-article', type=int, default=10,
                        help='Minimum chunks per article')
    args = parser.parse_args()

    print("=" * 80)
    print("BUILD CLEAN DISJOINT EVAL SET")
    print("=" * 80)
    print(f"Train NPZ: {args.train_npz}")
    print(f"Eval articles: {args.eval_articles}")
    print(f"Min chunks/article: {args.min_chunks_per_article}")
    print()

    # Load data
    print("Loading data...")
    data = np.load(args.train_npz, allow_pickle=True)
    contexts = data['context_sequences']
    targets = data['target_vectors']
    truth_keys = data['truth_keys']
    print(f"  Total samples: {len(contexts)}")
    print()

    # Extract article IDs
    article_ids = truth_keys[:, 0]
    unique_articles = np.unique(article_ids)
    print(f"  Unique articles: {len(unique_articles)}")
    print()

    # Count chunks per article
    article_counts = {}
    for art_id in unique_articles:
        count = np.sum(article_ids == art_id)
        article_counts[art_id] = count

    # Filter articles with enough chunks
    eligible_articles = [
        art_id for art_id, count in article_counts.items()
        if count >= args.min_chunks_per_article
    ]
    print(f"  Eligible articles (≥{args.min_chunks_per_article} chunks): {len(eligible_articles)}")

    if len(eligible_articles) < args.eval_articles:
        print(f"  ⚠️  Warning: Only {len(eligible_articles)} eligible, requested {args.eval_articles}")
        args.eval_articles = len(eligible_articles)

    # Sample eval articles (use last N to avoid train leakage)
    eval_articles = set(sorted(eligible_articles)[-args.eval_articles:])
    train_articles = set(unique_articles) - eval_articles

    print(f"  Train articles: {len(train_articles)}")
    print(f"  Eval articles: {len(eval_articles)}")
    print()

    # Split by article
    train_mask = np.array([art_id in train_articles for art_id in article_ids])
    eval_mask = np.array([art_id in eval_articles for art_id in article_ids])

    train_contexts = contexts[train_mask]
    train_targets = targets[train_mask]
    train_truth_keys = truth_keys[train_mask]

    eval_contexts = contexts[eval_mask]
    eval_targets = targets[eval_mask]
    eval_truth_keys = truth_keys[eval_mask]

    print("Split results:")
    print(f"  Train: {len(train_contexts)} samples")
    print(f"  Eval: {len(eval_contexts)} samples")
    print()

    # Verify disjointness
    train_articles_actual = set(train_truth_keys[:, 0])
    eval_articles_actual = set(eval_truth_keys[:, 0])
    overlap = train_articles_actual & eval_articles_actual

    if len(overlap) == 0:
        print("✅ STRICT DISJOINTNESS: No article overlap")
    else:
        print(f"❌ ERROR: {len(overlap)} articles overlap!")
        return

    print()

    # Save
    print("Saving...")
    args.out_train.parent.mkdir(parents=True, exist_ok=True)
    args.out_eval.parent.mkdir(parents=True, exist_ok=True)

    # Preserve original metadata
    np.savez(
        args.out_train,
        context_sequences=train_contexts,
        target_vectors=train_targets,
        truth_keys=train_truth_keys,
        context_length=data.get('context_length', 5),
        num_sequences=len(train_contexts),
        vector_dim=data.get('vector_dim', 768),
        provenance=data.get('provenance', 'disjoint_train'),
    )

    np.savez(
        args.out_eval,
        context_sequences=eval_contexts,
        target_vectors=eval_targets,
        truth_keys=eval_truth_keys,
        context_length=data.get('context_length', 5),
        num_sequences=len(eval_contexts),
        vector_dim=data.get('vector_dim', 768),
        provenance=data.get('provenance', 'disjoint_eval'),
    )

    print(f"✅ Train: {args.out_train}")
    print(f"✅ Eval: {args.out_eval}")
    print("=" * 80)


if __name__ == '__main__':
    main()
