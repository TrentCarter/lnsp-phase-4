#!/usr/bin/env python3
# 20250825T091424_v1.1
"""
JXE Vec2Text subscriber using LNSP processor
"""

import torch
from typing import List, Dict, Any


class JXEVec2TextSubscriber:
    """Original vec2text implementation using LNSP processor"""
    
    def __init__(self, 
                 teacher_model_path: str = "data/teacher_models/gtr-t5-base",
                 steps: int = 1,
                 device: str = None,
                 debug: bool = False):
        """Initialize JXE vec2text decoder"""
        self.teacher_model_path = teacher_model_path
        self.steps = steps
        self.debug = debug
        self.name = "jxe"
        self.output_type = "text"
        
        # Device setup
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
        
        # Load processor
        try:
            import sys
            from pathlib import Path
            # Add project root to path if not already there
            project_root = Path(__file__).resolve().parents[3]
            if str(project_root) not in sys.path:
                sys.path.insert(0, str(project_root))
            
            from app.agents.vec2text_agent import create_vec2text_processor
            self.processor = create_vec2text_processor(teacher_model_name=self.teacher_model_path)
            self._original_get_vector = self.processor.get_vector_from_source
        except Exception as e:
            raise RuntimeError(f"Failed to load LNSP processor: {e}")
    
    def process(self, vectors: torch.Tensor, metadata: Dict[str, Any] = None) -> List[str]:
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
            vector = vectors[i]
            input_text = original_texts[i] if i < len(original_texts) else " "
            
            # Prepare embedding: flatten to (768,) on correct device
            embedding_flat = vector.squeeze().detach()
            if embedding_flat.dim() > 1:
                embedding_flat = embedding_flat.view(-1)
            # Move to processor device and L2-normalize
            try:
                embedding_flat = embedding_flat.to(self.processor.device)
                norm = torch.linalg.norm(embedding_flat, ord=2)
                if norm > 0:
                    embedding_flat = embedding_flat / (norm + 1e-12)
                if self.debug:
                    print(f"[DEBUG] Inject vec norm={norm.item():.6f} device={embedding_flat.device} shape={tuple(embedding_flat.shape)}")
            except (RuntimeError, AttributeError) as e:
                if self.debug:
                    print(f"[DEBUG] Vector prep error: {e}")
            
            # Mock the get_vector function to return our embedding
            def _mock_get_vector(text, vector_source):
                if vector_source == 'teacher':
                    return embedding_flat
                return self._original_get_vector(text, vector_source)
            
            # Temporarily replace the method
            self.processor.get_vector_from_source = _mock_get_vector
            
            try:
                result = self.processor.iterative_vec2text_process(
                    input_text=input_text,
                    vector_source='teacher',
                    num_iterations=max(1, self.steps)
                )
                
                decoded = (result or {}).get('final_text', None)
                if decoded:
                    decoded_texts.append(decoded.strip())
                else:
                    decoded_texts.append("[JXE: No text returned]")
                    
            except Exception as e:
                decoded_texts.append(f"[JXE decode error: {e}]")
                
            finally:
                # Restore original method
                self.processor.get_vector_from_source = self._original_get_vector
        
        return decoded_texts