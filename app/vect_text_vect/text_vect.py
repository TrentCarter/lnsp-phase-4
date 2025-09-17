#!/usr/bin/env python3
"""
Text to vector encoder using GTR-T5
"""

import torch
from sentence_transformers import SentenceTransformer
from typing import List


class TextToVectorEncoder:
    """Efficient batch encoding of text to vectors using GTR-T5"""
    
    def __init__(self, 
                 model_path: str = "sentence-transformers/gtr-t5-base",
                 device: str = None,
                 batch_size: int = 32):
        """Initialize GTR-T5 encoder"""
        
        # Device setup
        if device:
            self.device = device
        elif torch.backends.mps.is_available():
            self.device = "mps"
        elif torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"
        
        self.batch_size = batch_size
        
        # Load model
        self.model = SentenceTransformer(model_path, device=self.device)
        
    def encode(self, texts: List[str], normalize: bool = True) -> torch.Tensor:
        """
        Encode texts to vectors
        
        Args:
            texts: List of texts to encode
            normalize: Whether to normalize embeddings
            
        Returns:
            Tensor of shape [N, 768]
        """
        # Use sentence transformers batch encoding
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=normalize,
            convert_to_tensor=True,
            device=self.device
        )
        
        return embeddings