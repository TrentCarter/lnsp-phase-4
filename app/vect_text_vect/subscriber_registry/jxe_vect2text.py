#!/usr/bin/env python3
# 20250825T091424_v1.1
"""
JXE Vec2Text subscriber using LNSP processor
"""

import torch
from typing import Any, Dict, List

from app.vect_text_vect.vec2text_processor import create_vec2text_processor


class JXEVec2TextSubscriber:
    """Vec2Text implementation using the shared processor."""

    def __init__(
        self,
        teacher_model_path: str = "data/teacher_models/gtr-t5-base",
        steps: int = 1,
        device: str | None = None,
        debug: bool = False,
    ):
        self.teacher_model_path = teacher_model_path
        self.steps = steps
        self.debug = debug
        self.name = "jxe"
        self.output_type = "text"

        if device:
            self.device = torch.device(device)
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        if self.debug:
            print(f"[DEBUG] JXE Vec2Text using device: {self.device}")
            print(f"[DEBUG] Teacher model path: {self.teacher_model_path}")
            print(f"[DEBUG] Steps: {self.steps}")

        try:
            self.processor = create_vec2text_processor(
                teacher_model_name=self.teacher_model_path,
                device=self.device.type,
                random_seed=42,
                debug=self.debug,
            )
        except Exception as exc:
            raise RuntimeError(f"Failed to initialise vec2text processor: {exc}")
    
    def process(self, vectors: torch.Tensor, metadata: Dict[str, Any] | None = None) -> List[str]:
        """
        Decode vectors to text
        
        Args:
            vectors: [N, 768] tensor
            metadata: Optional dict with 'original_texts' key
            
        Returns:
            List of decoded texts
        """
        original_texts = metadata.get('original_texts', []) if metadata else []
        batch_size = vectors.shape[0]
        decoded_texts = []
        
        for i in range(batch_size):
            vector = vectors[i].detach().to(self.processor.device)
            input_text = original_texts[i] if i < len(original_texts) else " "

            try:
                result = self.processor.iterative_vec2text_process(
                    input_text=input_text,
                    vector_source="teacher",
                    num_iterations=max(1, self.steps),
                    beam_width=1,
                    target_embedding=vector,
                )
                decoded = (result or {}).get("final_text")
                decoded_texts.append(decoded.strip() if decoded else "[JXE: No text returned]")
            except Exception as exc:
                decoded_texts.append(f"[JXE decode error: {exc}]")

        return decoded_texts
