"""
Ranking-based loss functions for P7 "Directional Ranker" LVM

Replaces regression objective (predict exact next vector) with ranking objective
(rank next vector highest among candidates). This prevents "orthogonal escape"
by making the model compete with concrete negatives instead of drifting to
arbitrary low-similarity space.

Key components:
1. InfoNCE ranking loss with in-batch negatives
2. Prev-repel margin: explicitly push away from previous chunk
3. Directional gate: down-weight sequences with weak forward signal
4. Cosine floor with teacher pull during warmup

Author: Claude Code
Date: 2025-11-04
Status: P7 Architecture
"""

import torch
import torch.nn.functional as F
from typing import Dict, Tuple, Optional


def info_nce_ranking_loss(
    query: torch.Tensor,
    positive: torch.Tensor,
    negatives: torch.Tensor,
    temperature: float = 0.07
) -> torch.Tensor:
    """
    InfoNCE contrastive ranking loss

    Args:
        query: (B, D) - predicted vectors (model output)
        positive: (B, D) - target next vectors
        negatives: (B, N, D) - negative samples (previous + in-batch + in-article distractors)
        temperature: softmax temperature for scaling logits

    Returns:
        loss: scalar - InfoNCE loss encouraging high similarity with positive
    """
    # Normalize all vectors to unit sphere
    query = F.normalize(query, dim=-1)
    positive = F.normalize(positive, dim=-1)
    negatives = F.normalize(negatives, dim=-1)

    # Compute similarity scores
    pos_score = torch.sum(query * positive, dim=-1) / temperature  # (B,)
    neg_scores = torch.bmm(negatives, query.unsqueeze(-1)).squeeze(-1) / temperature  # (B, N)

    # InfoNCE: -log(exp(pos) / (exp(pos) + sum(exp(neg))))
    # = -pos + log(exp(pos) + sum(exp(neg)))
    # = -pos + logsumexp([pos, neg...])
    all_scores = torch.cat([pos_score.unsqueeze(1), neg_scores], dim=1)  # (B, N+1)
    loss = -pos_score + torch.logsumexp(all_scores, dim=1)

    return loss.mean()


def prev_repel_margin_loss(
    query: torch.Tensor,
    positive: torch.Tensor,
    previous: torch.Tensor,
    margin: float = 0.07
) -> torch.Tensor:
    """
    Hard negative margin loss explicitly pushing away from previous chunk

    Args:
        query: (B, D) - predicted vectors
        positive: (B, D) - target next vectors
        previous: (B, D) - previous chunk vectors (hard negatives)
        margin: target margin between pos and prev scores

    Returns:
        loss: scalar - margin violation loss
    """
    # Normalize to unit sphere
    query = F.normalize(query, dim=-1)
    positive = F.normalize(positive, dim=-1)
    previous = F.normalize(previous, dim=-1)

    # Compute cosine similarities
    pos_score = torch.sum(query * positive, dim=-1)  # (B,)
    prev_score = torch.sum(query * previous, dim=-1)  # (B,)

    # Margin loss: max(0, margin - pos_score + prev_score)
    # If pos_score > prev_score + margin, loss = 0 (satisfied)
    # Otherwise, penalize the violation
    margin_violation = margin - pos_score + prev_score
    loss = torch.clamp(margin_violation, min=0.0)

    return loss.mean()


def semantic_anchor_blend(
    query_raw: torch.Tensor,
    context_vectors: torch.Tensor,
    lambda_blend: float = 0.8,
    learnable: bool = False
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Blend model output with context subspace to prevent orthogonal escape

    Args:
        query_raw: (B, D) - raw model prediction before anchoring
        context_vectors: (B, K, D) - context sequence vectors (e.g., 5 chunks)
        lambda_blend: blend weight (0.6-0.9), higher = more model, lower = more anchor
        learnable: if True, return learnable lambda parameter

    Returns:
        query_anchored: (B, D) - blended and normalized query
        lambda_param: if learnable, returns nn.Parameter for optimization
    """
    # Compute context centroid (simple mean, could use attention weights)
    context_centroid = context_vectors.mean(dim=1)  # (B, D)

    # Blend: q' = norm(λ·q_raw + (1-λ)·c_hat)
    query_blended = lambda_blend * query_raw + (1.0 - lambda_blend) * context_centroid
    query_anchored = F.normalize(query_blended, dim=-1)

    if learnable:
        # Return a lambda parameter that can be optimized
        # (Caller should create nn.Parameter and pass value)
        return query_anchored, None

    return query_anchored, None


def semantic_anchor_blend_attention(
    query_raw: torch.Tensor,
    context_vectors: torch.Tensor,
    attention_weights: torch.Tensor,
    lambda_blend: float = 0.8
) -> torch.Tensor:
    """
    Blend model output with attention-weighted context subspace

    Args:
        query_raw: (B, D) - raw model prediction
        context_vectors: (B, K, D) - context sequence vectors
        attention_weights: (B, K) - attention weights over context (softmax'd)
        lambda_blend: blend weight

    Returns:
        query_anchored: (B, D) - blended and normalized query
    """
    # Weighted context centroid
    context_weighted = torch.bmm(
        attention_weights.unsqueeze(1),  # (B, 1, K)
        context_vectors  # (B, K, D)
    ).squeeze(1)  # (B, D)

    # Blend and normalize
    query_blended = lambda_blend * query_raw + (1.0 - lambda_blend) * context_weighted
    query_anchored = F.normalize(query_blended, dim=-1)

    return query_anchored


def directional_gate_weights(
    context_vectors: torch.Tensor,
    target_next: torch.Tensor,
    target_prev: torch.Tensor,
    threshold: float = 0.03,
    weak_weight: float = 0.25
) -> torch.Tensor:
    """
    Compute per-sequence loss weights based on directional signal strength

    Down-weight sequences with weak forward bias (Δ < threshold) to avoid
    training on ambiguous or backward-biased samples.

    Args:
        context_vectors: (B, K, D) - context sequence vectors
        target_next: (B, D) - next chunk vectors
        target_prev: (B, D) - previous chunk vectors (context[-1])
        threshold: minimum Δ for full weight (default 0.03)
        weak_weight: weight for weak-signal sequences (default 0.25)

    Returns:
        weights: (B,) - per-sequence loss weights in [weak_weight, 1.0]
    """
    # Use last context vector as reference point
    ref_vector = context_vectors[:, -1, :]  # (B, D)

    # Compute forward and backward similarities
    cos_next = F.cosine_similarity(ref_vector, target_next, dim=-1)  # (B,)
    cos_prev = F.cosine_similarity(ref_vector, target_prev, dim=-1)  # (B,)

    # Directional margin Δ = cos_next - cos_prev
    delta = cos_next - cos_prev

    # Weight: 1.0 if Δ >= threshold, weak_weight if Δ < threshold
    # (Could use smooth interpolation instead of hard threshold)
    weights = torch.where(delta >= threshold,
                         torch.ones_like(delta),
                         torch.full_like(delta, weak_weight))

    return weights


def cosine_floor_teacher_loss(
    query: torch.Tensor,
    target: torch.Tensor,
    floor_threshold: float = 0.20,
    warmup_only: bool = True
) -> Tuple[torch.Tensor, int]:
    """
    Teacher pull loss when predictions are too far from target during warmup

    Prevents early training collapse by gently pulling predictions toward
    targets when cosine similarity drops below floor_threshold.

    Args:
        query: (B, D) - predicted vectors
        target: (B, D) - target next vectors
        floor_threshold: minimum cosine similarity (default 0.20)
        warmup_only: only apply during first 1-2 epochs

    Returns:
        loss: scalar - teacher pull loss (0 if all above threshold)
        n_violations: number of sequences below threshold
    """
    # Normalize
    query = F.normalize(query, dim=-1)
    target = F.normalize(target, dim=-1)

    # Compute cosine similarity
    cos_sim = torch.sum(query * target, dim=-1)  # (B,)

    # Find violations (cos < floor_threshold)
    violations_mask = cos_sim < floor_threshold
    n_violations = violations_mask.sum().item()

    if n_violations == 0:
        return torch.tensor(0.0, device=query.device), 0

    # Teacher loss: pull toward target (minimize 1 - cos)
    teacher_loss = (1.0 - cos_sim[violations_mask]).mean()

    return teacher_loss, n_violations


def p7_combined_loss(
    query: torch.Tensor,
    positive: torch.Tensor,
    previous: torch.Tensor,
    negatives: torch.Tensor,
    context_vectors: torch.Tensor,
    weights_dict: Dict[str, float],
    epoch: int = 0,
    warmup_epochs: int = 2
) -> Dict[str, torch.Tensor]:
    """
    Combined P7 ranking loss with all components

    Args:
        query: (B, D) - predicted vectors (already anchored)
        positive: (B, D) - target next vectors
        previous: (B, D) - previous chunk vectors
        negatives: (B, N, D) - negative samples pool
        context_vectors: (B, K, D) - context sequence for gating
        weights_dict: dictionary with loss component weights:
            - 'w_rank': InfoNCE weight (default 1.0)
            - 'w_margin': prev-repel weight (default 0.5)
            - 'w_teacher': teacher pull weight (default 0.2, warmup only)
            - 'margin': margin value (default 0.07)
            - 'temperature': InfoNCE temperature (default 0.07)
            - 'gate_threshold': Δ threshold (default 0.03)
            - 'gate_weak_weight': weak sequence weight (default 0.25)
            - 'floor_threshold': teacher floor (default 0.20)
        epoch: current training epoch
        warmup_epochs: number of epochs for teacher pull

    Returns:
        loss_dict: dictionary with total loss and components
    """
    # Extract weights with defaults
    w_rank = weights_dict.get('w_rank', 1.0)
    w_margin = weights_dict.get('w_margin', 0.5)
    w_teacher = weights_dict.get('w_teacher', 0.2)
    margin = weights_dict.get('margin', 0.07)
    temperature = weights_dict.get('temperature', 0.07)
    gate_threshold = weights_dict.get('gate_threshold', 0.03)
    gate_weak_weight = weights_dict.get('gate_weak_weight', 0.25)
    floor_threshold = weights_dict.get('floor_threshold', 0.20)

    # 1. Directional gate: compute per-sequence weights
    gate_weights = directional_gate_weights(
        context_vectors, positive, previous,
        threshold=gate_threshold,
        weak_weight=gate_weak_weight
    )

    # 2. InfoNCE ranking loss
    loss_rank = info_nce_ranking_loss(query, positive, negatives, temperature=temperature)

    # 3. Prev-repel margin loss
    loss_margin = prev_repel_margin_loss(query, positive, previous, margin=margin)

    # 4. Teacher pull (warmup only)
    if epoch < warmup_epochs:
        loss_teacher, n_violations = cosine_floor_teacher_loss(
            query, positive, floor_threshold=floor_threshold
        )
    else:
        loss_teacher = torch.tensor(0.0, device=query.device)
        n_violations = 0

    # 5. Apply directional gate weights
    # (For simplicity, apply globally; could apply per-component if needed)
    gate_weight_mean = gate_weights.mean()

    # 6. Combined loss
    loss_total = (
        w_rank * loss_rank * gate_weight_mean +
        w_margin * loss_margin * gate_weight_mean +
        w_teacher * loss_teacher  # Teacher not gated (always pull back)
    )

    # Return detailed breakdown
    return {
        'loss': loss_total,
        'loss_rank': loss_rank,
        'loss_margin': loss_margin,
        'loss_teacher': loss_teacher,
        'gate_weight_mean': gate_weight_mean,
        'n_teacher_violations': n_violations
    }


def compute_directional_metrics(
    query: torch.Tensor,
    positive: torch.Tensor,
    previous: torch.Tensor,
    context_vectors: torch.Tensor
) -> Dict[str, float]:
    """
    Compute directional alignment metrics for monitoring

    Args:
        query: (B, D) - predicted vectors
        positive: (B, D) - target next vectors
        previous: (B, D) - previous chunk vectors
        context_vectors: (B, K, D) - context sequence

    Returns:
        metrics: dictionary with alignment statistics
    """
    # Normalize
    query = F.normalize(query, dim=-1)
    positive = F.normalize(positive, dim=-1)
    previous = F.normalize(previous, dim=-1)

    # Cosine similarities
    cos_next = F.cosine_similarity(query, positive, dim=-1).mean().item()
    cos_prev = F.cosine_similarity(query, previous, dim=-1).mean().item()
    margin = cos_next - cos_prev

    # Anchor alignment: how close to context centroid?
    context_centroid = context_vectors.mean(dim=1)
    context_centroid = F.normalize(context_centroid, dim=-1)
    cos_anchor = F.cosine_similarity(query, context_centroid, dim=-1).mean().item()

    return {
        'cos_next': cos_next,
        'cos_prev': cos_prev,
        'margin': margin,
        'cos_anchor': cos_anchor
    }
