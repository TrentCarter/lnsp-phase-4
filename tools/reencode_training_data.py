#!/usr/bin/env python3
"""Re-encode ordered concept vectors with the vec2text-compatible encoder."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import requests

# Allow importing helper in tools.extract_ordered_training_data
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.extract_ordered_training_data import create_training_sequences  # noqa: E402


def batched(iterable: Iterable[str], size: int) -> Iterable[list[str]]:
    batch: list[str] = []
    for item in iterable:
        batch.append(item)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch


def encode_texts(texts: list[str], endpoint: str, timeout: float = 30.0) -> np.ndarray:
    payload = {"texts": texts}
    response = requests.post(endpoint, json=payload, timeout=timeout)
    response.raise_for_status()
    data = response.json()
    vectors = np.array(data["embeddings"], dtype=np.float32)
    if vectors.shape[0] != len(texts):
        raise RuntimeError(f"Encoder returned {vectors.shape[0]} rows for {len(texts)} inputs")
    return vectors


def reencode_ordered_dataset(
    ordered_path: Path,
    endpoint: str,
    batch_size: int,
) -> Path:
    print(f"Loading ordered dataset: {ordered_path}")
    raw = np.load(ordered_path, allow_pickle=True)
    data = {key: raw[key] for key in raw.files}
    concept_key = "concept_texts" if "concept_texts" in data else "texts"
    concept_texts = data[concept_key].astype(object)
    print(f"Total concepts: {len(concept_texts):,}")

    vectors = []
    start_time = time.time()
    for idx, batch in enumerate(batched(concept_texts, batch_size), start=1):
        vecs = encode_texts(batch, endpoint)
        vectors.append(vecs)
        if idx % 20 == 0:
            elapsed = time.time() - start_time
            processed = idx * batch_size
            print(f"  Encoded {processed:,} vectors in {elapsed:.1f}s ({processed/elapsed:.1f} vec/s)")

    new_vectors = np.vstack(vectors)
    print("Encoding complete")
    print(f"Vector shape: {new_vectors.shape}")
    print(f"Mean norm: {np.linalg.norm(new_vectors, axis=1).mean():.6f}")

    out_path = ordered_path.with_name(ordered_path.stem + "_vec2text.npz")
    print(f"Writing re-encoded dataset to {out_path}")

    data["vectors"] = new_vectors
    np.savez_compressed(out_path, **data)

    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Re-encode ordered concepts using vec2text encoder")
    parser.add_argument(
        "--ordered-npz",
        default="artifacts/lvm/wikipedia_42113_ordered.npz",
        help="Path to ordered NPZ file",
    )
    parser.add_argument(
        "--endpoint",
        default="http://127.0.0.1:8767/embed",
        help="Vec2text-compatible embedding endpoint",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Texts per embedding request",
    )
    parser.add_argument(
        "--context-size",
        type=int,
        default=5,
        help="Context window for sequence generation",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/lvm",
        help="Directory for regenerated training sequences",
    )
    args = parser.parse_args()

    ordered_path = Path(args.ordered_npz)
    new_ordered = reencode_ordered_dataset(ordered_path, args.endpoint, args.batch_size)

    print("Generating training sequences from re-encoded dataset...")
    seq_path = create_training_sequences(
        npz_file=new_ordered,
        context_size=args.context_size,
        output_dir=args.output_dir,
    )

    print("\nRe-encoding complete")
    print(f"Ordered dataset: {new_ordered}")
    print(f"Training sequences: {seq_path}")


if __name__ == "__main__":
    main()
