"""
P8 Listwise Ranking Loss

Key differences from P7 InfoNCE:
1. Task-specific candidates (no random in-batch negatives)
2. Listwise softmax (not global InfoNCE contrast)
3. Prev-repel margin is explicit
4. Order verifier auxiliary loss

Candidate set construction:
- Positive: true next chunk
- Hard negatives:
  * Previous chunk (index -1)
  * 2-4 same-article distractors (nearby chunks)
- Global negatives: in-batch samples (optional, limited)

This focuses the ranking signal on temporal ordering within articles,
avoiding spurious correlations from random batch composition.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple


def listwise_ranking_loss(
    query: torch.Tensor,
    candidates: torch.Tensor,
    temperature: float = 0.07
) -> Dict[str, torch.Tensor]:
    """
    Listwise ranking loss: rank true next (index 0) highest

    Args:
        query: (B, D) - predicted vectors (unit-normalized)
        candidates: (B, L, D) - candidate vectors (unit-normalized)
                    candidates[:, 0] = true next chunk
                    candidates[:, 1] = previous chunk
                    candidates[:, 2:] = other negatives
        temperature: temperature for softmax sharpening

    Returns:
        loss_dict: dictionary with:
            - loss: listwise ranking loss
            - scores_pos: scores for positive samples
            - scores_neg_prev: scores for previous chunk
            - rank_pos: rank of positive in candidate list (1-indexed)
    """
    # Compute cosine similarities (all vectors are unit-normalized)
    # scores: (B, L)
    scores = torch.bmm(
        query.unsqueeze(1),  # (B, 1, D)
        candidates.transpose(1, 2)  # (B, D, L)
    ).squeeze(1)  # (B, L)

    # Listwise softmax: rank true next (index 0) highest
    log_probs = F.log_softmax(scores / temperature, dim=-1)
    loss = -log_probs[:, 0].mean()  # Negative log-likelihood for index 0

    # Diagnostics
    scores_pos = scores[:, 0]  # Scores for true next
    scores_neg_prev = scores[:, 1]  # Scores for previous chunk

    # Rank of positive (how many candidates score higher?)
    # rank = 1 + number of negatives with score > score_pos
    rank_pos = 1 + (scores[:, 1:] > scores_pos.unsqueeze(-1)).sum(dim=-1).float()

    return {
        'loss': loss,
        'scores_pos': scores_pos.mean().item(),
        'scores_neg_prev': scores_neg_prev.mean().item(),
        'rank_pos': rank_pos.mean().item()
    }


def prev_repel_margin_loss(
    query: torch.Tensor,
    target_next: torch.Tensor,
    target_prev: torch.Tensor,
    margin: float = 0.07
) -> Dict[str, torch.Tensor]:
    """
    Explicit margin: cos(q, next) > cos(q, prev) + margin

    Args:
        query: (B, D) - predicted vectors
        target_next: (B, D) - true next chunk
        target_prev: (B, D) - previous chunk
        margin: required margin (default 0.07)

    Returns:
        loss_dict: dictionary with:
            - loss: margin violation loss
            - cos_next: cosine similarity to next
            - cos_prev: cosine similarity to prev
            - margin_actual: actual margin (cos_next - cos_prev)
    """
    # Cosine similarities
    cos_next = F.cosine_similarity(query, target_next, dim=-1)  # (B,)
    cos_prev = F.cosine_similarity(query, target_prev, dim=-1)  # (B,)

    # Margin loss: max(0, margin - (cos_next - cos_prev))
    margin_actual = cos_next - cos_prev
    loss = torch.clamp(margin - margin_actual, min=0.0).mean()

    return {
        'loss': loss,
        'cos_next': cos_next.mean().item(),
        'cos_prev': cos_prev.mean().item(),
        'margin': margin_actual.mean().item()
    }


def order_verifier_loss(
    order_verifier: nn.Module,
    c_i: torch.Tensor,
    c_j: torch.Tensor,
    labels: torch.Tensor
) -> Dict[str, torch.Tensor]:
    """
    Auxiliary loss: predict if j > i

    Args:
        order_verifier: OrderVerifier module
        c_i: (B, D) - earlier chunks
        c_j: (B, D) - later chunks
        labels: (B,) - binary labels (1 if j > i, 0 if j < i)

    Returns:
        loss_dict: dictionary with:
            - loss: binary cross-entropy loss
            - accuracy: classification accuracy
    """
    logits = order_verifier(c_i, c_j)  # (B,)
    loss = F.binary_cross_entropy_with_logits(logits, labels.float())

    # Accuracy
    preds = (torch.sigmoid(logits) > 0.5).float()
    accuracy = (preds == labels).float().mean()

    return {
        'loss': loss,
        'accuracy': accuracy.item()
    }


def combined_p8_loss(
    query: torch.Tensor,
    candidates: torch.Tensor,
    target_next: torch.Tensor,
    target_prev: torch.Tensor,
    order_verifier: nn.Module,
    order_pairs: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    w_listwise: float = 1.0,
    w_margin: float = 0.5,
    w_order: float = 0.2,
    margin: float = 0.07,
    temperature: float = 0.07
) -> Dict[str, torch.Tensor]:
    """
    Combined P8 loss: listwise + prev-repel + order verifier

    Args:
        query: (B, D) - predicted vectors
        candidates: (B, L, D) - candidate vectors for listwise ranking
        target_next: (B, D) - true next chunk
        target_prev: (B, D) - previous chunk
        order_verifier: OrderVerifier module
        order_pairs: tuple of (c_i, c_j, labels) for order prediction
        w_listwise: weight for listwise ranking loss
        w_margin: weight for prev-repel margin loss
        w_order: weight for order verifier loss
        margin: margin for prev-repel constraint
        temperature: temperature for listwise softmax

    Returns:
        loss_dict: dictionary with:
            - loss: total combined loss
            - loss_listwise: listwise ranking loss
            - loss_margin: prev-repel margin loss
            - loss_order: order verifier loss
            - (other metrics from sub-losses)
    """
    # Listwise ranking loss
    listwise_dict = listwise_ranking_loss(query, candidates, temperature)

    # Prev-repel margin loss
    margin_dict = prev_repel_margin_loss(query, target_next, target_prev, margin)

    # Order verifier loss
    c_i, c_j, labels = order_pairs
    order_dict = order_verifier_loss(order_verifier, c_i, c_j, labels)

    # Combined loss
    loss_total = (
        w_listwise * listwise_dict['loss'] +
        w_margin * margin_dict['loss'] +
        w_order * order_dict['loss']
    )

    # Aggregate metrics
    result = {
        'loss': loss_total,
        'loss_listwise': listwise_dict['loss'].item(),
        'loss_margin': margin_dict['loss'].item(),
        'loss_order': order_dict['loss'].item(),
        'scores_pos': listwise_dict['scores_pos'],
        'scores_neg_prev': listwise_dict['scores_neg_prev'],
        'rank_pos': listwise_dict['rank_pos'],
        'cos_next': margin_dict['cos_next'],
        'cos_prev': margin_dict['cos_prev'],
        'margin': margin_dict['margin'],
        'order_accuracy': order_dict['accuracy']
    }

    return result


def create_candidate_set(
    target_next: torch.Tensor,
    target_prev: torch.Tensor,
    hard_negatives: torch.Tensor,
    inbatch_negatives: torch.Tensor | None = None,
    max_inbatch: int = 4
) -> torch.Tensor:
    """
    Create candidate set for listwise ranking

    Candidate order:
        [0]: true next (positive)
        [1]: previous chunk (hard negative)
        [2:2+K]: hard negatives from same article
        [2+K:]: in-batch negatives (optional, limited)

    Args:
        target_next: (B, D) - true next chunks
        target_prev: (B, D) - previous chunks
        hard_negatives: (B, K, D) - same-article negatives
        inbatch_negatives: (B, M, D) - in-batch negatives (optional)
        max_inbatch: maximum number of in-batch negatives to include

    Returns:
        candidates: (B, L, D) where L = 2 + K + min(M, max_inbatch)
    """
    B, D = target_next.shape
    K = hard_negatives.shape[1] if hard_negatives is not None else 0

    candidates_list = [
        target_next.unsqueeze(1),  # (B, 1, D) - index 0
        target_prev.unsqueeze(1)   # (B, 1, D) - index 1
    ]

    # Add hard negatives
    if hard_negatives is not None and K > 0:
        candidates_list.append(hard_negatives)  # (B, K, D)

    # Add limited in-batch negatives
    if inbatch_negatives is not None:
        M = min(inbatch_negatives.shape[1], max_inbatch)
        candidates_list.append(inbatch_negatives[:, :M, :])  # (B, M, D)

    # Concatenate all candidates
    candidates = torch.cat(candidates_list, dim=1)  # (B, L, D)

    return candidates


def sample_order_pairs(
    contexts: torch.Tensor,
    targets: torch.Tensor,
    num_pairs: int = 2
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Sample pairs (c_i, c_j) for order verification

    Args:
        contexts: (B, K, D) - context sequences
        targets: (B, D) - target next chunks
        num_pairs: number of pairs to sample per batch item

    Returns:
        c_i: (B*num_pairs, D) - earlier chunks
        c_j: (B*num_pairs, D) - later chunks
        labels: (B*num_pairs,) - 1 if j>i, 0 if j<i
    """
    B, K, D = contexts.shape

    c_i_list = []
    c_j_list = []
    labels_list = []

    for _ in range(num_pairs):
        # Sample two positions: i < j (forward) or i > j (backward)
        # 50% forward, 50% backward
        is_forward = torch.rand(B) > 0.5  # (B,)

        for b in range(B):
            if is_forward[b]:
                # Forward: sample i from context, j = target
                i = torch.randint(0, K, (1,)).item()
                c_i_list.append(contexts[b, i])
                c_j_list.append(targets[b])
                labels_list.append(torch.tensor(1.0))  # j > i
            else:
                # Backward: sample j from context, i = target
                j = torch.randint(0, K, (1,)).item()
                c_i_list.append(targets[b])
                c_j_list.append(contexts[b, j])
                labels_list.append(torch.tensor(0.0))  # j < i (target is before context)

    c_i = torch.stack(c_i_list)  # (B*num_pairs, D)
    c_j = torch.stack(c_j_list)
    labels = torch.stack(labels_list)

    return c_i, c_j, labels
