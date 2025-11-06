#!/usr/bin/env python3
"""
Build sentence-level vector bank from paragraph data.

Splits paragraphs into sentences, embeds each sentence, and stores
with metadata linking back to paragraphs, sections, and articles.

Usage:
    python tools/build_sentence_bank.py \
        --in-jsonl data/paragraphs.jsonl \
        --out-npz artifacts/sentence_bank.npz
"""
import argparse
import os
import json
import gzip
import numpy as np
import re
import requests
from typing import List, Tuple

# Sentence splitting regex: split on .!? followed by space and capital letter
SPLIT = re.compile(r'(?<=[.!?])\s+(?=[A-Z"\'])')
ENCODER = os.environ.get("ENCODER_URL", "http://localhost:8767/embed")

def enc(batch: List[str], batch_size=128) -> np.ndarray:
    """Encode texts in batches using GTR-T5 encoder."""
    if not batch:
        return np.zeros((0, 768), np.float32)

    all_vecs = []
    for i in range(0, len(batch), batch_size):
        sub_batch = batch[i:i+batch_size]
        r = requests.post(ENCODER, json={"texts": sub_batch})
        r.raise_for_status()
        V = np.array(r.json()["embeddings"], dtype=np.float32)
        # Normalize
        V /= (np.linalg.norm(V, axis=1, keepdims=True) + 1e-9)
        all_vecs.append(V)

    return np.vstack(all_vecs)

def split_sentences(text: str) -> List[str]:
    """Split text into sentences, filtering short fragments."""
    # Replace common abbreviations to avoid false splits
    abbr_map = [
        ("Mr.", "Mr<DOT>"), ("Mrs.", "Mrs<DOT>"), ("Ms.", "Ms<DOT>"),
        ("Dr.", "Dr<DOT>"), ("Prof.", "Prof<DOT>"), ("Sr.", "Sr<DOT>"),
        ("Jr.", "Jr<DOT>"), ("St.", "St<DOT>"), ("vs.", "vs<DOT>"),
        ("etc.", "etc<DOT>"), ("e.g.", "e<DOT>g<DOT>"), ("i.e.", "i<DOT>e<DOT>")
    ]
    for abbr, placeholder in abbr_map:
        text = text.replace(abbr, placeholder)

    # Split on sentence boundaries
    sents = [s.strip() for s in SPLIT.split(text) if s.strip()]

    # Restore abbreviations
    sents = [s.replace("<DOT>", ".") for s in sents]

    # Filter: at least 20 alphabetic characters
    return [s for s in sents if sum(ch.isalpha() for ch in s) >= 20]

def main():
    ap = argparse.ArgumentParser(description="Build sentence-level vector bank")
    ap.add_argument("--in-jsonl", required=True,
                   help="Paragraph bank JSONL: {text, para_id, section_id, article_id}")
    ap.add_argument("--out-npz", required=True,
                   help="Output NPZ file for sentence bank")
    ap.add_argument("--batch-size", type=int, default=128,
                   help="Encoding batch size")
    args = ap.parse_args()

    print(f"Building sentence bank from: {args.in_jsonl}")

    S = []  # All sentences
    meta = []  # Metadata: (sent_id, para_id, section_id, article_id, sent_idx)

    opener = gzip.open if args.in_jsonl.endswith(".gz") else open

    with opener(args.in_jsonl, "rt", encoding="utf-8", errors="ignore") as f:
        sent_id = 0
        para_count = 0

        for line in f:
            j = json.loads(line)
            txt = j["text"]
            para_id = j.get("para_id", para_count)
            section_id = j.get("section_id", -1)
            article_id = j.get("article_id", -1)

            # Split into sentences
            sents = split_sentences(txt)

            # Store with metadata
            for i, s in enumerate(sents):
                S.append(s)
                meta.append((sent_id, para_id, section_id, article_id, i))
                sent_id += 1

            para_count += 1

            if para_count % 1000 == 0:
                print(f"  Processed {para_count} paragraphs, {sent_id} sentences...")

    print(f"\nTotal: {len(S)} sentences from {para_count} paragraphs")
    print(f"Encoding with batch_size={args.batch_size}...")

    # Encode all sentences
    V = enc(S, batch_size=args.batch_size)

    # Build output arrays
    out = {
        "sent_vecs": V,
        "sent_ids": np.array([m[0] for m in meta], np.int64),
        "para_ids": np.array([m[1] for m in meta], np.int64),
        "section_ids": np.array([m[2] for m in meta], np.int32),
        "article_ids": np.array([m[3] for m in meta], np.int64),
        "sent_idx": np.array([m[4] for m in meta], np.int32),
        "sent_texts": np.array(S, dtype=object)
    }

    # Save
    os.makedirs(os.path.dirname(args.out_npz) or ".", exist_ok=True)
    np.savez_compressed(args.out_npz, **out)

    print(f"\n✓ Wrote {args.out_npz} with {V.shape[0]} sentences")
    print(f"  Dimensions: {V.shape}")
    print(f"  Size: {os.path.getsize(args.out_npz) / 1024 / 1024:.1f} MB")

if __name__ == "__main__":
    main()
