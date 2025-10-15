#!/usr/bin/env python3
"""Historic JXE vec2text wrapper retained for compatibility."""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.vect_text_vect.vec2text_processor import create_vec2text_processor


def main() -> None:
    if len(sys.argv) != 3:
        print("Usage: jxe_wrapper.py <input_file> <output_file>", file=sys.stderr)
        sys.exit(1)

    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])

    try:
        with input_path.open("rb") as handle:
            payload = pickle.load(handle)

        vectors = np.asarray(payload.get("vectors"), dtype=np.float32)
        metadata = payload.get("metadata") or {}
        steps = int(payload.get("steps", 1))

        processor = create_vec2text_processor(
            teacher_model_name="data/teacher_models/gtr-t5-base",
            device="cpu",
            random_seed=42,
        )

        texts = metadata.get("original_texts") or [" "] * vectors.shape[0]
        prompts = [str(t) if t is not None else " " for t in texts]
        info = processor.decode_embeddings(
            vectors,
            num_iterations=max(1, steps),
            beam_width=1,
            prompts=prompts,
        )
        results = [entry.get("final_text", "") or "<decode_error>" for entry in info]
        output = {"status": "success", "result": results, "details": info}

    except Exception as exc:
        output = {"status": "error", "error": str(exc)}

    with output_path.open("wb") as handle:
        pickle.dump(output, handle)


if __name__ == "__main__":  # pragma: no cover
    main()
