#!/usr/bin/env python3
"""IELab vec2text subscriber using the shared processor."""

from __future__ import annotations

import torch
from typing import List

from app.vect_text_vect.vec2text_processor import create_vec2text_processor


class IELabVec2TextSubscriber:
    """Vec2Text subscriber matching the IELab configuration."""

    def __init__(
        self,
        steps: int = 20,
        beam_width: int = 2,
        device: str | None = None,
        debug: bool = False,
    ):
        self.steps = steps
        self.beam_width = max(1, beam_width)
        self.debug = debug
        self.name = "ielab"
        self.output_type = "text"

        # IELab models have historically been CPU-only for stability.
        requested = device or "cpu"
        if requested != "cpu" and debug:
            print(f"[DEBUG] IELab subscriber forcing CPU despite request '{requested}'")
        self.device = torch.device("cpu")

        try:
            self.processor = create_vec2text_processor(
                teacher_model_name="data/teacher_models/gtr-t5-base",
                device="cpu",
                random_seed=123,
                debug=self.debug,
            )
        except Exception as exc:
            raise RuntimeError(f"Failed to initialise vec2text processor: {exc}")

    def process(self, vectors: torch.Tensor, metadata=None) -> List[str]:
        metadata = metadata or {}
        original_texts = metadata.get("original_texts", [])
        decoded_texts: List[str] = []

        for idx in range(vectors.shape[0]):
            embedding = vectors[idx].detach().to(self.processor.device)
            prompt = original_texts[idx] if idx < len(original_texts) else " "
            try:
                result = self.processor.iterative_vec2text_process(
                    input_text=prompt,
                    vector_source="teacher",
                    num_iterations=max(1, self.steps),
                    beam_width=self.beam_width,
                    target_embedding=embedding,
                )
                decoded = (result or {}).get("final_text")
                decoded_texts.append(decoded.strip() if decoded else "[IELab: No text returned]")
            except Exception as exc:
                decoded_texts.append(f"[IELab decode error: {exc}]")

        return decoded_texts
