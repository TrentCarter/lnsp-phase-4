#!/usr/bin/env python3
"""IELab vec2text wrapper that routes through the shared processor."""

from __future__ import annotations

import os
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("ACCELERATE_DISABLE_INIT_EMPTY_WEIGHTS", "1")
os.environ.setdefault("TRANSFORMERS_NO_ACCELERATE", "1")

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


def _maybe_apply_mamba(vectors: torch.Tensor, checkpoint: str, debug: bool) -> torch.Tensor:
    checkpoint_path = Path(checkpoint)
    if not checkpoint_path.exists():
        if debug:
            print(f"[IELab] Mamba checkpoint not found: {checkpoint}", file=sys.stderr)
        return vectors

    try:
        from app.nemotron_vmmoe.minimal_mamba import MinimalMamba
        from app.nemotron_vmmoe.minimal_mamba_trainer import MambaVectorConfig
    except Exception as exc:  # pragma: no cover - optional dependency
        if debug:
            print(f"[IELab] Failed to import Mamba modules: {exc}", file=sys.stderr)
        return vectors

    if debug:
        print(f"[IELab] Loading Mamba checkpoint from {checkpoint_path}", file=sys.stderr)

    try:
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        model = MinimalMamba(MambaVectorConfig())
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        with torch.no_grad():
            transformed = model(vectors.unsqueeze(1))  # [B, 1, D]
        transformed = F.normalize(transformed.squeeze(1), dim=-1)
        if debug:
            cos = F.cosine_similarity(vectors, transformed, dim=-1).mean().item()
            print(f"[IELab] Mamba transform mean cosine: {cos:.4f}", file=sys.stderr)
        return transformed
    except Exception as exc:  # pragma: no cover - optional path
        if debug:
            print(f"[IELab] Mamba transformation failed: {exc}", file=sys.stderr)
        return vectors


def main() -> None:
    if len(sys.argv) != 3:
        print("Usage: ielab_wrapper.py <input_file> <output_file>", file=sys.stderr)
        sys.exit(1)

    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])

    try:
        payload = _load_request(input_path)
        metadata = payload.get("metadata") or {}
        vectors = _ensure_float32_matrix(payload.get("vectors"))
        steps = int(payload.get("steps", 20))
        beam_width = max(1, int(payload.get("beam_width", 2)))
        device_override = payload.get("device_override")
        debug = bool(payload.get("debug", False))
        normalize_input = bool(payload.get("normalize", False))
        mamba_checkpoint = payload.get("mamba_checkpoint")

        teacher_hint = (
            metadata.get("teacher_model_path")
            or metadata.get("teacher_model")
            or "data/teacher_models/gtr-t5-base"
        )

        processor = create_vec2text_processor(
            teacher_model_name=teacher_hint,
            device=device_override,
            random_seed=123,
            debug=debug,
        )

        tensor = torch.from_numpy(vectors)
        if normalize_input:
            tensor = F.normalize(tensor, dim=-1)
        if mamba_checkpoint and str(mamba_checkpoint).lower() not in {"", "none"}:
            tensor = _maybe_apply_mamba(tensor, str(mamba_checkpoint), debug)
        vectors_for_decode = tensor.cpu().numpy()

        prompts = _resolve_prompts(metadata, vectors_for_decode.shape[0])
        info = processor.decode_embeddings(
            vectors_for_decode,
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
