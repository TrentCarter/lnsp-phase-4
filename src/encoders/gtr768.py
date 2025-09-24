"""
GTR-T5 768D Encoder Implementation
Uses sentence-transformers for consistent 768D embeddings.
"""
from __future__ import annotations
from typing import List, Optional
import numpy as np
from sentence_transformers import SentenceTransformer


class GTR768Encoder:
    """Encoder for GTR-T5-base model producing 768D normalized embeddings."""

    def __init__(self, model_name: str = "sentence-transformers/gtr-t5-base"):
        self.model_name = model_name
        self.model: Optional[SentenceTransformer] = None

    def load_model(self) -> None:
        """Load the sentence transformer model."""
        if self.model is None:
            self.model = SentenceTransformer(self.model_name)

    def encode_batch(self, texts: List[str], normalize: bool = True) -> np.ndarray:
        """
        Encode a batch of texts to 768D vectors.

        Args:
            texts: List of text strings to encode
            normalize: Whether to L2 normalize the embeddings

        Returns:
            Array of shape (len(texts), 768) with float32 embeddings
        """
        if self.model is None:
            self.load_model()

        embeddings = self.model.encode(texts, normalize_embeddings=normalize)
        return embeddings.astype(np.float32)

    def encode_single(self, text: str, normalize: bool = True) -> np.ndarray:
        """
        Encode a single text to 768D vector.

        Args:
            text: Text string to encode
            normalize: Whether to L2 normalize the embedding

        Returns:
            Array of shape (768,) with float32 embedding
        """
        return self.encode_batch([text], normalize=normalize)[0]

    def get_dimension(self) -> int:
        """Return the embedding dimension."""
        return 768


# Global instance for convenience
_encoder = GTR768Encoder()


def encode_texts(texts: List[str]) -> np.ndarray:
    """Convenience function to encode texts using global encoder."""
    return _encoder.encode_batch(texts)


def encode_text(text: str) -> np.ndarray:
    """Convenience function to encode a single text using global encoder."""
    return _encoder.encode_single(text)
