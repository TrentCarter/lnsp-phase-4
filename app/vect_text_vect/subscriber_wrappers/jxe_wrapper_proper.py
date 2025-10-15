#!/usr/bin/env python3
"""JXE vec2text wrapper that uses the shared processor pipeline."""

from __future__ import annotations

import os
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from app.vect_text_vect.vec2text_processor import create_vec2text_processor


def _load_request(path: Path) -> Dict[str, Any]:
    with path.open("rb") as handle:
        data = pickle.load(handle)
    return data if isinstance(data, dict) else {}


def _ensure_float32_matrix(vectors: Any) -> np.ndarray:
    arr = np.asarray(vectors, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"expected 2D array of embeddings, received shape {arr.shape}")
    if arr.shape[1] != 768:
        raise ValueError(f"expected embeddings with dim 768, received {arr.shape[1]}")
    return arr


def _resolve_prompts(metadata: Dict[str, Any], count: int) -> List[str]:
    prompts = metadata.get("original_texts") if isinstance(metadata, dict) else None
    if isinstance(prompts, list) and len(prompts) == count:
        return [str(p) if p is not None else " " for p in prompts]
    return [" "] * count


def main() -> None:
    if len(sys.argv) != 3:
        print("Usage: jxe_wrapper_proper.py <input_file> <output_file>", file=sys.stderr)
        sys.exit(1)

    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])

    try:
        payload = _load_request(input_path)
        vectors = _ensure_float32_matrix(payload.get("vectors"))
        metadata = payload.get("metadata") or {}
        steps = int(payload.get("steps", 1))
        beam_width = max(1, int(payload.get("beam_width", 1)))  # JXE default: greedy
        device_override = payload.get("device_override")
        debug = bool(payload.get("debug", False))

        teacher_hint = (
            metadata.get("teacher_model_path")
            or metadata.get("teacher_model")
            or "data/teacher_models/gtr-t5-base"
        )

        processor = create_vec2text_processor(
            teacher_model_name=teacher_hint,
            device=device_override,
            random_seed=42,
            debug=debug,
        )

        prompts = _resolve_prompts(metadata, vectors.shape[0])
        info = processor.decode_embeddings(
            vectors,
            num_iterations=max(1, steps),
            beam_width=beam_width,
            prompts=prompts,
        )

        decoded_texts: List[str] = []
        for record in info:
            text = str(record.get("final_text", "")).strip()
            decoded_texts.append(text if text else "<decode_error>")

        output = {
            "status": "success",
            "result": decoded_texts,
            "details": info,
        }

    except Exception as exc:  # pragma: no cover - defensive
        output = {"status": "error", "error": str(exc)}

    with output_path.open("wb") as handle:
        pickle.dump(output, handle)


if __name__ == "__main__":  # pragma: no cover
    main()
