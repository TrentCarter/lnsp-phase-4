#!/usr/bin/env python3
"""
Convert paragraph-level NPZ to JSONL format for sentence bank builder.

Usage:
    python tools/npz_to_paragraph_jsonl.py \
        --in-npz artifacts/lvm/arxiv_papers_210_768d.npz \
        --out-jsonl data/arxiv_paragraphs.jsonl
"""
import argparse
import json
import numpy as np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-npz", required=True)
    ap.add_argument("--out-jsonl", required=True)
    args = ap.parse_args()

    print(f"Loading: {args.in_npz}")
    npz = np.load(args.in_npz, allow_pickle=True)

    print(f"Keys: {list(npz.keys())}")

    # Extract data
    texts = npz["concept_texts"] if "concept_texts" in npz else npz.get("texts", [])
    article_ids = npz.get("article_ids", np.arange(len(texts)))

    # Metadata if available
    metadata = npz.get("arxiv_metadata", None)

    print(f"Found {len(texts)} paragraphs")

    # Write JSONL
    with open(args.out_jsonl, "w", encoding="utf-8") as f:
        for i, text in enumerate(texts):
            para_id = i
            # Handle string or int article IDs
            if i < len(article_ids):
                aid = article_ids[i]
                # If it's a string (like arXiv ID), use hash for numeric ID
                article_id = hash(str(aid)) % (2**31) if isinstance(aid, str) else int(aid)
            else:
                article_id = i

            record = {
                "text": str(text),
                "para_id": para_id,
                "section_id": -1,  # Unknown
                "article_id": article_id,
                "arxiv_id": str(article_ids[i]) if i < len(article_ids) else None
            }

            f.write(json.dumps(record) + "\n")

    print(f"✓ Wrote {args.out_jsonl} with {len(texts)} paragraphs")

if __name__ == "__main__":
    main()
