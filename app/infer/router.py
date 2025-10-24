"""
Smart LVM Router with Vector-Based Confidence
==============================================

Routes queries to AMN (primary), Transformer(opt) (fallback), or GRU (secondary)
based on confidence metrics and query characteristics.

CRITICAL FIX: LVM outputs are 768D vectors, NOT logit distributions!
Cannot use "logit margin" - use cosine similarity to training set instead.
"""

import numpy as np
import torch
from pathlib import Path
from typing import Tuple, Optional

# Will be initialized on first call
_models = None
_training_vectors = None


def _load_models():
    """Lazy load production models"""
    global _models, _training_vectors

    if _models is not None:
        return

    # Import here to avoid circular dependencies
    import sys
    sys.path.insert(0, 'app/lvm')
    from models import create_model

    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

    # Load AMN (primary)
    amn_path = Path('artifacts/lvm/production_model/best_model.pt')
    amn_ckpt = torch.load(amn_path, map_location=device, weights_only=False)
    amn_config = amn_ckpt.get('model_config', {})
    amn = create_model('amn', **amn_config)
    amn.load_state_dict(amn_ckpt['model_state_dict'])
    amn.to(device)
    amn.eval()

    # Load Transformer (fallback accuracy)
    txf_path = Path('artifacts/lvm/fallback_accuracy/best_model.pt')
    txf_ckpt = torch.load(txf_path, map_location=device, weights_only=False)
    txf_config = txf_ckpt.get('model_config', {})
    txf = create_model('transformer', **txf_config)
    txf.load_state_dict(txf_ckpt['model_state_dict'])
    txf.to(device)
    txf.eval()

    # Load GRU (fallback secondary)
    gru_path = Path('artifacts/lvm/fallback_secondary/best_model.pt')
    gru_ckpt = torch.load(gru_path, map_location=device, weights_only=False)
    gru_config = gru_ckpt.get('model_config', {})
    gru = create_model('gru', **gru_config)
    gru.load_state_dict(gru_ckpt['model_state_dict'])
    gru.to(device)
    gru.eval()

    _models = {
        'amn': amn,
        'transformer': txf,
        'gru': gru,
        'device': device
    }

    # Load training vectors for confidence calculation
    # Use a sample of target vectors from training set
    data = np.load('artifacts/lvm/wikipedia_fresh_sequences_ctx5.npz')
    # Sample 10k vectors for efficiency (full set is 489k)
    indices = np.random.choice(len(data['train_target_vectors']), 10000, replace=False)
    _training_vectors = data['train_target_vectors'][indices]
    _training_vectors = torch.from_numpy(_training_vectors).float().to(device)

    print(f"✅ Loaded models: AMN, Transformer(opt), GRU on {device}")
    print(f"✅ Loaded {len(_training_vectors):,} reference vectors for confidence")


# High-stakes lanes that escalate to Transformer
LANE_HIGH_STAKES = {
    'sci-fact',
    'math-derivation',
    'med-guideline',
    'policy/procedure',
    'legal-interpretation',
    'safety-critical'
}


def compute_confidence(pred_vec: torch.Tensor, k: int = 50) -> float:
    """
    Compute confidence as mean cosine similarity to top-k nearest training vectors.

    CORRECT metric for LVM (vector outputs), NOT logit margin!

    Args:
        pred_vec: Predicted vector (768D)
        k: Number of nearest neighbors to average

    Returns:
        float: Confidence score (0-1, higher = more confident)
    """
    _load_models()  # Ensure training vectors loaded

    # Normalize
    pred_norm = pred_vec / (pred_vec.norm() + 1e-8)
    train_norm = _training_vectors / (_training_vectors.norm(dim=1, keepdim=True) + 1e-8)

    # Compute cosine similarities
    cosines = (pred_norm @ train_norm.T).squeeze()

    # Mean of top-k
    topk_cosines, _ = torch.topk(cosines, k=min(k, len(cosines)))
    confidence = topk_cosines.mean().item()

    return confidence


def infer(
    query_vec_768: np.ndarray,
    lane: Optional[str] = None,
    ctx_len: Optional[int] = None,
    confidence_threshold: float = 0.70
) -> Tuple[np.ndarray, str, dict]:
    """
    Route query to appropriate LVM model based on confidence and context.

    Args:
        query_vec_768: Input context vectors [5, 768] or [batch, 5, 768]
        lane: Query domain/lane (e.g., 'sci-fact', 'general')
        ctx_len: Context length in tokens (optional, for long-context detection)
        confidence_threshold: Min confidence to stay with AMN (default 0.70)

    Returns:
        Tuple of (prediction_vec [768], model_name, metadata)
    """
    _load_models()

    device = _models['device']
    amn = _models['amn']
    txf = _models['transformer']
    gru = _models['gru']

    # Convert to tensor
    if isinstance(query_vec_768, np.ndarray):
        query_vec = torch.from_numpy(query_vec_768).float().to(device)
    else:
        query_vec = query_vec_768.to(device)

    # Ensure correct shape [batch=1, 5, 768] or [5, 768]
    if query_vec.ndim == 2:
        query_vec = query_vec.unsqueeze(0)  # Add batch dim

    # Step 1: Try AMN (fast primary)
    with torch.no_grad():
        y_amn = amn(query_vec).squeeze()  # [768]

    confidence = compute_confidence(y_amn)

    # Check escalation criteria
    needs_accuracy = (
        confidence < confidence_threshold or
        (lane in LANE_HIGH_STAKES) or
        (ctx_len and ctx_len > 384)
    )

    metadata = {
        'amn_confidence': confidence,
        'lane': lane,
        'ctx_len': ctx_len,
        'escalated': needs_accuracy
    }

    if not needs_accuracy:
        # AMN sufficient
        return y_amn.cpu().numpy(), 'AMN', metadata

    # Step 2: Escalate to Transformer (accuracy fallback)
    with torch.no_grad():
        y_txf = txf(query_vec).squeeze()  # [768]

    txf_confidence = compute_confidence(y_txf)
    metadata['txf_confidence'] = txf_confidence

    if txf_confidence >= confidence_threshold:
        # Transformer confident
        return y_txf.cpu().numpy(), 'Transformer(opt)', metadata

    # Step 3: Last resort - GRU (secondary fallback)
    with torch.no_grad():
        y_gru = gru(query_vec).squeeze()  # [768]

    gru_confidence = compute_confidence(y_gru)
    metadata['gru_confidence'] = gru_confidence

    # Return GRU prediction (best effort)
    return y_gru.cpu().numpy(), 'GRU', metadata


def infer_batch(
    query_vecs: np.ndarray,
    lanes: Optional[list] = None,
    confidence_threshold: float = 0.70
) -> list:
    """
    Batch inference with smart routing.

    Args:
        query_vecs: [batch, 5, 768] input vectors
        lanes: List of lane identifiers (one per batch item)
        confidence_threshold: Min confidence threshold

    Returns:
        List of (prediction_vec, model_name, metadata) tuples
    """
    batch_size = len(query_vecs)
    lanes = lanes or [None] * batch_size

    results = []
    for i in range(batch_size):
        pred, model, meta = infer(
            query_vecs[i],
            lane=lanes[i],
            confidence_threshold=confidence_threshold
        )
        results.append((pred, model, meta))

    return results


def get_router_stats(results: list) -> dict:
    """
    Compute routing statistics from batch results.

    Args:
        results: List of (pred, model_name, metadata) tuples

    Returns:
        Dict with routing statistics
    """
    model_counts = {}
    total_escalated = 0
    confidence_scores = []

    for pred, model, meta in results:
        model_counts[model] = model_counts.get(model, 0) + 1
        if meta.get('escalated', False):
            total_escalated += 1
        if 'amn_confidence' in meta:
            confidence_scores.append(meta['amn_confidence'])

    total = len(results)

    return {
        'total_queries': total,
        'amn_count': model_counts.get('AMN', 0),
        'transformer_count': model_counts.get('Transformer(opt)', 0),
        'gru_count': model_counts.get('GRU', 0),
        'amn_pct': 100 * model_counts.get('AMN', 0) / total if total > 0 else 0,
        'escalation_rate': 100 * total_escalated / total if total > 0 else 0,
        'mean_confidence': np.mean(confidence_scores) if confidence_scores else 0
    }
