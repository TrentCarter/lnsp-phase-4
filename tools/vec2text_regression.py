#!/usr/bin/env python3
"""Light-weight regression check for the vec2text pipeline."""

from __future__ import annotations

import statistics
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
import sys
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.vect_text_vect.vec2text_processor import create_vec2text_processor

SAMPLES = [
    "The water cycle includes evaporation, condensation, and precipitation.",
    "Binary search trees maintain sorted order to enable logarithmic lookups.",
    "Interrupt handlers must acknowledge the PIC before returning.",
]


def main() -> None:
    processor = create_vec2text_processor(device="cpu", random_seed=7, debug=False)

    embeddings = torch.stack([
        processor.get_vector_from_source(text)
        for text in SAMPLES
    ])

    results = processor.decode_embeddings(
        embeddings,
        num_iterations=4,
        beam_width=2,
        prompts=SAMPLES,
    )

    final_cos = []
    for original, info in zip(SAMPLES, results):
        decoded = info.get("final_text", "")
        cos = info.get("final_cosine", float("nan"))
        print("―" * 72)
        print(f"Original : {original}")
        print(f"Decoded  : {decoded}")
        print(f"Cosine   : {cos:.3f}")
        final_cos.append(cos)

    avg_cos = statistics.fmean(final_cos)
    print("=" * 72)
    print(f"Average cosine similarity: {avg_cos:.3f}")
    if avg_cos < 0.45:
        raise SystemExit(f"Average cosine {avg_cos:.3f} below expected threshold 0.45")


if __name__ == "__main__":  # pragma: no cover
    main()
