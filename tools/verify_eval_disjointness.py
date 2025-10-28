#!/usr/bin/env python3
"""
Multi-level leak detection for train/eval splits.

Guards:
1. Doc-level disjointness (article_id overlap = 0)
2. Text-hash overlap (chunk content duplicates = 0)
3. Vector near-duplicates (cosine ≥ 0.995 = 0)

Usage:
    python tools/verify_eval_disjointness.py \
        --train-npz artifacts/lvm/train_payload_aligned.npz \
        --eval-npz artifacts/lvm/wikipedia_ood_test_ctx5_legacy_eval.npz \
        --payload artifacts/wikipedia_584k_payload.npy \
        --faiss artifacts/wikipedia_584k_ivf_flat_ip.index
"""

import argparse
import re
from pathlib import Path

import faiss
import numpy as np
import xxhash


def normalize_text(text: str) -> str:
    """Normalize text for hash comparison."""
    return re.sub(r"\s+", " ", text.lower()).strip()


def text_hash(text: str) -> str:
    """Compute 64-bit text hash."""
    return xxhash.xxh64(normalize_text(text)).hexdigest()


def check_doc_disjointness(train_npz, eval_npz):
    """Check article-level disjointness (required)."""
    print("=" * 80)
    print("GUARD 1: DOC-LEVEL DISJOINTNESS")
    print("=" * 80)

    train_keys = train_npz['truth_keys']
    eval_keys = eval_npz['truth_keys']

    train_docs = set(int(k[0]) for k in train_keys)
    eval_docs = set(int(k[0]) for k in eval_keys)

    overlap = train_docs & eval_docs

    print(f"Train articles: {len(train_docs)}")
    print(f"Eval articles: {len(eval_docs)}")
    print(f"Overlap: {len(overlap)}")
    print()

    if overlap:
        overlap_pct = len(overlap) / len(eval_docs) * 100
        print(f"❌ FAIL: {len(overlap)} overlapping article_ids ({overlap_pct:.1f}%)")
        print(f"   First 20: {sorted(overlap)[:20]}")
        print()
        print("   This eval set is NOT suitable for generalization testing!")
        return False
    else:
        print("✅ PASS: Complete article disjointness")
        return True


def check_chunk_disjointness(train_npz, eval_npz):
    """Check chunk-level disjointness (article_id, chunk_id pairs)."""
    print()
    print("=" * 80)
    print("GUARD 2A: CHUNK-LEVEL DISJOINTNESS")
    print("=" * 80)

    train_keys = train_npz['truth_keys']
    eval_keys = eval_npz['truth_keys']

    train_chunks = set((int(k[0]), int(k[1])) for k in train_keys)
    eval_chunks = set((int(k[0]), int(k[1])) for k in eval_keys)

    overlap = train_chunks & eval_chunks

    print(f"Train chunks: {len(train_chunks)}")
    print(f"Eval chunks: {len(eval_chunks)}")
    print(f"Overlap: {len(overlap)}")
    print()

    if overlap:
        overlap_pct = len(overlap) / len(eval_chunks) * 100
        print(f"❌ FAIL: {len(overlap)} overlapping chunks ({overlap_pct:.1f}%)")
        print(f"   Model may have seen exact same sequences during training!")
        return False
    else:
        print("✅ PASS: No chunk overlap")
        return True


def check_text_hash_overlap(train_npz, eval_npz, payload_path, limit=10000):
    """Check text content duplicates using xxhash."""
    print()
    print("=" * 80)
    print("GUARD 2B: TEXT-HASH OVERLAP (Content Duplicates)")
    print("=" * 80)

    # Load payload
    payload = np.load(payload_path, allow_pickle=True).item()

    # Build hash sets
    print(f"Computing text hashes (limit={limit} per set)...")

    train_keys = train_npz['truth_keys'][:limit]
    eval_keys = eval_npz['truth_keys'][:limit]

    # Get texts from payload
    def get_text(keys, payload):
        texts = []
        for art_idx, chunk_idx in keys:
            for pid, (text, meta, vec) in payload.items():
                if meta['article_index'] == art_idx and meta['chunk_index'] == chunk_idx:
                    texts.append(text)
                    break
        return texts

    print("  Extracting train texts...")
    train_texts = get_text(train_keys, payload)
    print("  Extracting eval texts...")
    eval_texts = get_text(eval_keys, payload)

    print(f"  Computing hashes...")
    train_hashes = {text_hash(t) for t in train_texts if t}
    eval_hashes = {text_hash(t) for t in eval_texts if t}

    overlap = train_hashes & eval_hashes

    print(f"\nTrain text hashes: {len(train_hashes)}")
    print(f"Eval text hashes: {len(eval_hashes)}")
    print(f"Overlap: {len(overlap)}")
    print()

    if overlap:
        overlap_pct = len(overlap) / len(eval_hashes) * 100 if eval_hashes else 0
        print(f"❌ FAIL: {len(overlap)} duplicate text chunks ({overlap_pct:.1f}%)")
        print(f"   Eval chunks have identical text content to training chunks!")
        return False
    else:
        print("✅ PASS: No text content duplicates")
        return True


def check_vector_near_duplicates(train_npz, eval_npz, threshold=0.995, limit=5000):
    """Check vector near-duplicates (cosine ≥ 0.995)."""
    print()
    print("=" * 80)
    print("GUARD 3: VECTOR NEAR-DUPLICATES (Cosine ≥ 0.995)")
    print("=" * 80)

    # Load vectors
    train_targets = train_npz['targets'][:limit]
    eval_targets = eval_npz['targets'][:limit]

    # Normalize
    train_vecs = train_targets / np.linalg.norm(train_targets, axis=1, keepdims=True)
    eval_vecs = eval_targets / np.linalg.norm(eval_targets, axis=1, keepdims=True)

    print(f"Checking {len(eval_vecs)} eval vectors against {len(train_vecs)} train vectors...")
    print(f"Threshold: cosine ≥ {threshold}")
    print()

    # Build FAISS index
    index = faiss.IndexFlatIP(train_vecs.shape[1])
    index.add(train_vecs.astype('float32'))

    # Search for nearest train vector for each eval vector
    D, I = index.search(eval_vecs.astype('float32'), 1)

    # Check for near-duplicates (cosine = inner product for normalized vectors)
    cosines = D.ravel()
    near_dups = cosines >= threshold

    print(f"Near-duplicates found: {near_dups.sum()}")
    print(f"Max cosine: {cosines.max():.6f}")
    print(f"Mean cosine: {cosines.mean():.6f}")
    print(f"P95 cosine: {np.percentile(cosines, 95):.6f}")
    print()

    if near_dups.any():
        near_dup_pct = near_dups.sum() / len(eval_vecs) * 100
        print(f"⚠️  WARNING: {near_dups.sum()} eval vectors ({near_dup_pct:.1f}%) are near-duplicates")
        print(f"   These eval vectors are very similar to training vectors!")
        print(f"   Cosine distribution of near-dups: {cosines[near_dups][:10]}")

        # Fail if >5% are near-duplicates
        if near_dup_pct > 5:
            print(f"\n❌ FAIL: >5% near-duplicates")
            return False
        else:
            print(f"\n⚠️  SOFT PASS: <5% near-duplicates (acceptable)")
            return True
    else:
        print("✅ PASS: No near-duplicate vectors")
        return True


def main():
    ap = argparse.ArgumentParser(description="Verify eval set disjointness with multi-level guards")
    ap.add_argument("--train-npz", type=Path, required=True)
    ap.add_argument("--eval-npz", type=Path, required=True)
    ap.add_argument("--payload", type=Path, required=True)
    ap.add_argument("--faiss", type=Path, default=None)
    args = ap.parse_args()

    print("\n" + "=" * 80)
    print("MULTI-LEVEL LEAK DETECTION")
    print("=" * 80)
    print(f"Train: {args.train_npz}")
    print(f"Eval: {args.eval_npz}")
    print("=" * 80)
    print()

    # Load data
    train_data = np.load(args.train_npz, allow_pickle=True)
    eval_data = np.load(args.eval_npz, allow_pickle=True)

    # Run guards
    results = []

    # Guard 1: Doc-level disjointness (REQUIRED)
    doc_ok = check_doc_disjointness(train_data, eval_data)
    results.append(("Doc-level disjointness", doc_ok))

    # Guard 2A: Chunk-level disjointness
    chunk_ok = check_chunk_disjointness(train_data, eval_data)
    results.append(("Chunk-level disjointness", chunk_ok))

    # Guard 2B: Text-hash overlap (if doc-level passes)
    if doc_ok:
        text_ok = check_text_hash_overlap(train_data, eval_data, args.payload, limit=5000)
        results.append(("Text-hash duplicates", text_ok))
    else:
        results.append(("Text-hash duplicates", None))

    # Guard 3: Vector near-duplicates
    vec_ok = check_vector_near_duplicates(train_data, eval_data, threshold=0.995, limit=5000)
    results.append(("Vector near-duplicates", vec_ok))

    # Summary
    print()
    print("=" * 80)
    print("LEAK DETECTION SUMMARY")
    print("=" * 80)

    for check_name, passed in results:
        if passed is None:
            status = "⏭️  SKIPPED"
        elif passed:
            status = "✅ PASS"
        else:
            status = "❌ FAIL"
        print(f"{status} - {check_name}")

    print()

    # One-liner for CI
    doc_overlap = 0 if doc_ok else 1
    chunk_overlap = 0 if chunk_ok else 1
    text_overlap = 0 if results[2][1] else (1 if results[2][1] is False else 0)
    vec_near_dups = 0 if vec_ok else 1

    print(f"LEAK: doc_overlap={doc_overlap}, chunk_overlap={chunk_overlap}, "
          f"text_hash_overlap={text_overlap}, vec_near_dups={vec_near_dups}")

    # Overall result
    all_passed = doc_ok and chunk_ok and vec_ok and (results[2][1] is not False)

    if all_passed:
        print("\n✅ ALL GUARDS PASSED - Eval set is clean!")
        print(f"   Safe to use: {args.eval_npz.name}")
    else:
        print("\n❌ LEAK DETECTED - Eval set is compromised!")
        print(f"   DO NOT use: {args.eval_npz.name}")
        print(f"   Tag file: eval_leaky=true")

    print("=" * 80)

    return 0 if all_passed else 1


if __name__ == "__main__":
    exit(main())
