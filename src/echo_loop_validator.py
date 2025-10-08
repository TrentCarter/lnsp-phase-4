#!/usr/bin/env python3
"""
Echo Loop Validator

Validates generated CPESH structures by comparing embedding similarity
between input concept and generated output. Implements the Echo Loop
quality control mechanism from TMD-LS architecture.
"""
import numpy as np
from typing import Dict, Optional, Tuple
import sys
from pathlib import Path

# Add src to path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from src.vectorizer import EmbeddingBackend
    HAS_VECTORIZER = True
except ImportError:
    HAS_VECTORIZER = False

from scipy.spatial.distance import cosine as scipy_cosine


# ============================================================================
# Embedding Generation
# ============================================================================

class EchoLoopValidator:
    """
    Echo Loop validator for CPESH quality control.

    Uses embedding similarity to validate that generated CPESH structures
    preserve semantic meaning from original concept.
    """

    def __init__(self, threshold: float = 0.82):
        """
        Initialize Echo Loop validator.

        Args:
            threshold: Cosine similarity threshold (default 0.82 from PRD)
        """
        if not HAS_VECTORIZER:
            raise RuntimeError("EmbeddingBackend not available. Install dependencies.")

        self.threshold = threshold
        self.embedding_backend = EmbeddingBackend()

        # Statistics
        self._validations = 0
        self._accepts = 0
        self._requeues = 0
        self._escalates = 0

    def compute_concept_embedding(self, text: str) -> np.ndarray:
        """
        Generate 768D GTR-T5 embedding for concept.

        Args:
            text: Concept text

        Returns:
            768D embedding vector
        """
        embeddings = self.embedding_backend.encode([text])
        return embeddings[0]

    def compute_cpesh_embedding(self, cpesh: Dict) -> np.ndarray:
        """
        Generate embedding for CPESH structure.

        Combines concept + expected answer embeddings to create
        representative vector for the CPESH structure.

        Args:
            cpesh: CPESH dictionary with keys:
                - concept: str
                - probe: str (optional, not used)
                - expected: str
                - soft_negatives: list (optional, not used)
                - hard_negatives: list (optional, not used)

        Returns:
            768D embedding vector
        """
        # Strategy: Embed concept and expected answer, then average
        # This captures the core semantic content of the CPESH

        concept = cpesh.get('concept', '')
        expected = cpesh.get('expected', '')

        if not concept or not expected:
            raise ValueError("CPESH must contain 'concept' and 'expected' fields")

        # Embed both
        embeddings = self.embedding_backend.encode([concept, expected])
        concept_emb = embeddings[0]
        expected_emb = embeddings[1]

        # Average (simple fusion)
        cpesh_emb = (concept_emb + expected_emb) / 2.0

        return cpesh_emb

    def compute_cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Compute cosine similarity between two vectors.

        Args:
            vec1: First vector
            vec2: Second vector

        Returns:
            Cosine similarity (0-1, where 1 is identical)
        """
        # scipy.spatial.distance.cosine returns distance (1 - similarity)
        # So we compute: similarity = 1 - distance
        distance = scipy_cosine(vec1, vec2)
        similarity = 1.0 - distance
        return similarity

    def validate_cpesh(
        self,
        concept_text: str,
        cpesh: Dict,
        threshold: Optional[float] = None
    ) -> Dict:
        """
        Validate CPESH structure using Echo Loop.

        Args:
            concept_text: Original concept text
            cpesh: Generated CPESH structure
            threshold: Cosine similarity threshold (overrides instance default)

        Returns:
            {
                'valid': bool,
                'cosine_similarity': float,
                'threshold': float,
                'action': 'accept' | 're_queue' | 'escalate',
                'reason': str
            }
        """
        if threshold is None:
            threshold = self.threshold

        # Generate embeddings
        concept_emb = self.compute_concept_embedding(concept_text)
        cpesh_emb = self.compute_cpesh_embedding(cpesh)

        # Compute similarity
        similarity = self.compute_cosine_similarity(concept_emb, cpesh_emb)

        # Update statistics
        self._validations += 1

        # Determine validation result and action
        if similarity >= 0.85:
            # High confidence - accept immediately
            action = 'accept'
            valid = True
            reason = "High confidence (>= 0.85)"
            self._accepts += 1

        elif similarity >= threshold:
            # Acceptable - accept with monitoring
            action = 'accept'
            valid = True
            reason = f"Acceptable (>= {threshold})"
            self._accepts += 1

        elif similarity >= 0.70:
            # Borderline - re-queue for improvement
            action = 're_queue'
            valid = False
            reason = "Re-queue for improvement (0.70-threshold)"
            self._requeues += 1

        else:
            # Poor quality - escalate to fallback model
            action = 'escalate'
            valid = False
            reason = "Escalate to fallback (< 0.70)"
            self._escalates += 1

        return {
            'valid': valid,
            'cosine_similarity': similarity,
            'threshold': threshold,
            'action': action,
            'reason': reason
        }

    def validate_smoothing(
        self,
        original_text: str,
        smoothed_text: str,
        threshold: Optional[float] = None
    ) -> Dict:
        """
        Validate output smoothing preserved semantics.

        Used for LVM output smoothing validation (P17).

        Args:
            original_text: Original LVM output
            smoothed_text: Smoothed/refined output
            threshold: Cosine similarity threshold (overrides instance default)

        Returns:
            {
                'valid': bool,
                'cosine_similarity': float,
                'threshold': float,
                'action': 'accept' | 'reject',
                'reason': str
            }
        """
        if threshold is None:
            threshold = self.threshold

        # Generate embeddings
        original_emb = self.compute_concept_embedding(original_text)
        smoothed_emb = self.compute_concept_embedding(smoothed_text)

        # Compute similarity
        similarity = self.compute_cosine_similarity(original_emb, smoothed_emb)

        # Update statistics
        self._validations += 1

        # Smoothing validation is stricter: either accept or reject
        if similarity >= threshold:
            action = 'accept'
            valid = True
            reason = f"Semantics preserved (>= {threshold})"
            self._accepts += 1
        else:
            action = 'reject'
            valid = False
            reason = f"Semantic drift detected (< {threshold})"
            self._requeues += 1  # Count as re-queue

        return {
            'valid': valid,
            'cosine_similarity': similarity,
            'threshold': threshold,
            'action': action,
            'reason': reason
        }

    def stats(self) -> Dict:
        """
        Get validation statistics.

        Returns:
            {
                'validations': int,
                'accepts': int,
                'requeues': int,
                'escalates': int,
                'accept_rate': float,
                'requeue_rate': float,
                'escalate_rate': float
            }
        """
        total = self._validations
        if total == 0:
            return {
                'validations': 0,
                'accepts': 0,
                'requeues': 0,
                'escalates': 0,
                'accept_rate': 0.0,
                'requeue_rate': 0.0,
                'escalate_rate': 0.0
            }

        return {
            'validations': total,
            'accepts': self._accepts,
            'requeues': self._requeues,
            'escalates': self._escalates,
            'accept_rate': self._accepts / total,
            'requeue_rate': self._requeues / total,
            'escalate_rate': self._escalates / total
        }

    def reset_stats(self) -> None:
        """Reset validation statistics."""
        self._validations = 0
        self._accepts = 0
        self._requeues = 0
        self._escalates = 0


# ============================================================================
# Global Validator Instance
# ============================================================================

_global_validator: Optional[EchoLoopValidator] = None


def get_validator(threshold: float = 0.82) -> EchoLoopValidator:
    """
    Get global Echo Loop validator instance (singleton).

    Args:
        threshold: Cosine similarity threshold

    Returns:
        EchoLoopValidator instance
    """
    global _global_validator
    if _global_validator is None:
        _global_validator = EchoLoopValidator(threshold=threshold)
    return _global_validator


# ============================================================================
# Convenience Functions
# ============================================================================

def validate_cpesh(
    concept_text: str,
    cpesh: Dict,
    threshold: float = 0.82
) -> Dict:
    """
    Validate CPESH structure (convenience function).

    Args:
        concept_text: Original concept text
        cpesh: Generated CPESH structure
        threshold: Cosine similarity threshold

    Returns:
        Validation result dictionary
    """
    validator = get_validator(threshold)
    return validator.validate_cpesh(concept_text, cpesh, threshold)


def validate_smoothing(
    original_text: str,
    smoothed_text: str,
    threshold: float = 0.82
) -> Dict:
    """
    Validate output smoothing (convenience function).

    Args:
        original_text: Original text
        smoothed_text: Smoothed text
        threshold: Cosine similarity threshold

    Returns:
        Validation result dictionary
    """
    validator = get_validator(threshold)
    return validator.validate_smoothing(original_text, smoothed_text, threshold)


# ============================================================================
# CLI for Testing
# ============================================================================

if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Echo Loop Validator - Test CPESH quality")
    parser.add_argument('--concept', required=True, help='Original concept text')
    parser.add_argument('--cpesh-json', help='CPESH structure as JSON string')
    parser.add_argument('--cpesh-file', help='CPESH structure from JSON file')
    parser.add_argument('--threshold', type=float, default=0.82, help='Cosine similarity threshold (default: 0.82)')
    parser.add_argument('--stats', action='store_true', help='Show validation statistics')

    args = parser.parse_args()

    validator = get_validator(threshold=args.threshold)

    if args.stats:
        stats = validator.stats()
        print("📊 Echo Loop Validation Statistics")
        print(f"  Total Validations: {stats['validations']}")
        print(f"  Accepts: {stats['accepts']} ({stats['accept_rate']:.1%})")
        print(f"  Re-queues: {stats['requeues']} ({stats['requeue_rate']:.1%})")
        print(f"  Escalates: {stats['escalates']} ({stats['escalate_rate']:.1%})")
        sys.exit(0)

    # Load CPESH
    if args.cpesh_json:
        cpesh = json.loads(args.cpesh_json)
    elif args.cpesh_file:
        with open(args.cpesh_file, 'r') as f:
            cpesh = json.load(f)
    else:
        print("❌ Error: Must provide --cpesh-json or --cpesh-file")
        sys.exit(1)

    # Validate
    print(f"🔍 Validating CPESH for concept: '{args.concept}'")
    print()

    result = validator.validate_cpesh(args.concept, cpesh)

    # Display result
    status_emoji = "✅" if result['valid'] else "❌"
    print(f"{status_emoji} Validation Result")
    print(f"  Valid: {result['valid']}")
    print(f"  Cosine Similarity: {result['cosine_similarity']:.4f}")
    print(f"  Threshold: {result['threshold']:.4f}")
    print(f"  Action: {result['action']}")
    print(f"  Reason: {result['reason']}")
    print()

    # Show CPESH details
    print("📄 CPESH Structure")
    print(f"  Concept: {cpesh.get('concept', 'N/A')}")
    print(f"  Probe: {cpesh.get('probe', 'N/A')}")
    print(f"  Expected: {cpesh.get('expected', 'N/A')}")
    print(f"  Soft Negatives: {len(cpesh.get('soft_negatives', []))} items")
    print(f"  Hard Negatives: {len(cpesh.get('hard_negatives', []))} items")
