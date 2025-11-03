"""
Directional & Anti-Copy Losses for 5→1 LVM Training
===================================================

Fixes the "last-frame copy" shortcut by augmenting the baseline MSE loss with:

1) Directional Margin Loss (next vs previous)
   L_dir = ReLU(m_dir − (cos(ŷ, y_next) − cos(ŷ, y_prev)))

2) Anti-Copy Hinge (next vs any context frame i∈[0..4])
   L_ac = mean_i ReLU(m_ac − (cos(ŷ, y_next) − cos(ŷ, ctx[i])))

Both are scale-matched, light-weight, and **cannot dominate** the primary MSE if you
use small weights (λ_dir ≈ 0.05, λ_ac ≈ 0.05) and small margins (m_dir≈0.05, m_ac≈0.02).

Use during training alongside standard MSE on the next vector target.

Typical config (safe defaults):
  λ_dir = 0.05, m_dir = 0.05
  λ_ac  = 0.05, m_ac  = 0.02

Optionally add ContextDrop augmentation to make copying fragile:
  - With prob p, perturb/replace the last context slot before the forward pass.

Integration snippet (inside your train loop):

    from app.lvm.losses_directional import (
        directional_margin_loss, anticopy_hinge_loss,
        mse_loss, cosine_sim_batch, context_drop
    )

    # ctx: (B,5,768); target: (B,768); prev: (B,768) = ctx[:, -2, :]
    if args.context_drop_p > 0:
        ctx_in = context_drop(ctx, p=args.context_drop_p, mode="last_to_noise")
    else:
        ctx_in = ctx

    pred = model(ctx_in)                           # (B,768)
    pred = F.normalize(pred, dim=-1)               # L2 normalize
    target = F.normalize(target, dim=-1)
    prev = F.normalize(ctx[:, -2, :], dim=-1)

    # Primary loss
    L_mse = mse_loss(pred, target)

    # Directional margin (next vs previous)
    L_dir = directional_margin_loss(pred, target, prev, margin=args.margin_dir)

    # Anti-copy hinge (next vs all context)
    L_ac = anticopy_hinge_loss(pred, ctx_in, target=target, margin=args.margin_ac)

    # Combined loss
    loss = L_mse + args.lambda_dir * L_dir + args.lambda_ac * L_ac

    # Log diagnostic
    with torch.no_grad():
        k_neg1 = cosine_sim_batch(pred, prev).mean().item()
        k_pos1 = cosine_sim_batch(pred, target).mean().item()
        margin = k_pos1 - k_neg1
        print(f"  Margin(+1 vs -1): {margin:.4f}")

Author: Claude Code + User (2025-10-31)
"""

import torch
import torch.nn.functional as F


# ============================================================================
# Core Loss Functions
# ============================================================================

def mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Standard MSE loss (primary objective).

    Args:
        pred: (B, D) predicted vectors
        target: (B, D) target vectors

    Returns:
        scalar MSE loss
    """
    return F.mse_loss(pred, target)


def cosine_sim_batch(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Batched cosine similarity (assumes inputs are already L2-normalized).

    Args:
        a: (B, D) normalized vectors
        b: (B, D) normalized vectors

    Returns:
        (B,) cosine similarities
    """
    # If not normalized, normalize here
    a_norm = F.normalize(a, dim=-1, p=2)
    b_norm = F.normalize(b, dim=-1, p=2)
    return (a_norm * b_norm).sum(dim=-1)


def directional_margin_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    prev: torch.Tensor,
    margin: float = 0.05
) -> torch.Tensor:
    """
    Directional Margin Loss: enforces cos(pred, next) > cos(pred, prev) + margin.

    L_dir = ReLU(margin - (cos(pred, next) - cos(pred, prev)))

    Args:
        pred: (B, D) model predictions (L2-normalized)
        target: (B, D) next vectors (L2-normalized)
        prev: (B, D) previous vectors (position -1, L2-normalized)
        margin: minimum required gap (default: 0.05)

    Returns:
        scalar loss
    """
    cos_next = cosine_sim_batch(pred, target)       # (B,)
    cos_prev = cosine_sim_batch(pred, prev)         # (B,)

    gap = cos_next - cos_prev                       # (B,)
    loss = F.relu(margin - gap)                     # (B,)

    return loss.mean()


def anticopy_hinge_loss(
    pred: torch.Tensor,
    context: torch.Tensor,
    target: torch.Tensor,
    margin: float = 0.02
) -> torch.Tensor:
    """
    Anti-Copy Hinge Loss: pred should be more similar to target than to ANY context frame.

    L_ac = mean_i ReLU(margin - (cos(pred, target) - cos(pred, ctx[i])))

    Args:
        pred: (B, D) model predictions (L2-normalized)
        context: (B, 5, D) context frames (L2-normalized)
        target: (B, D) next vectors (L2-normalized)
        margin: minimum required gap (default: 0.02)

    Returns:
        scalar loss
    """
    cos_target = cosine_sim_batch(pred, target)     # (B,)

    # Compute similarity to each context position
    losses = []
    for i in range(context.size(1)):
        ctx_i = context[:, i, :]                    # (B, D)
        cos_ctx_i = cosine_sim_batch(pred, ctx_i)   # (B,)
        gap = cos_target - cos_ctx_i                # (B,)
        loss_i = F.relu(margin - gap)               # (B,)
        losses.append(loss_i)

    # Average over all context positions
    total_loss = torch.stack(losses, dim=0).mean(dim=0)  # (B,)

    return total_loss.mean()


def future_margin_loss(
    pred: torch.Tensor,
    y_next: torch.Tensor,
    y_p2: torch.Tensor = None,
    y_p3: torch.Tensor = None,
    margin: float = 0.02
) -> torch.Tensor:
    """
    Future Margin Loss: pred should be more similar to +1 (next) than +2 or +3 (near future).

    Prevents k=+3 drift by explicitly anchoring predictions to immediate next position.

    L_fut = ReLU(margin - (cos(pred, y_next) - cos(pred, y_p2)))
          + ReLU(margin - (cos(pred, y_next) - cos(pred, y_p3)))

    Args:
        pred: (B, D) model predictions (L2-normalized)
        y_next: (B, D) next vectors (+1 position, L2-normalized)
        y_p2: (B, D) optional +2 position vectors (L2-normalized)
        y_p3: (B, D) optional +3 position vectors (L2-normalized)
        margin: minimum required gap (default: 0.02)

    Returns:
        scalar loss
    """
    cos_next = cosine_sim_batch(pred, y_next)  # (B,)
    loss = torch.tensor(0.0, device=pred.device)

    if y_p2 is not None:
        cos_p2 = cosine_sim_batch(pred, y_p2)
        gap_p2 = cos_next - cos_p2
        loss = loss + F.relu(margin - gap_p2).mean()

    if y_p3 is not None:
        cos_p3 = cosine_sim_batch(pred, y_p3)
        gap_p3 = cos_next - cos_p3
        loss = loss + F.relu(margin - gap_p3).mean()

    return loss


# ============================================================================
# Context Drop Augmentation
# ============================================================================

def context_drop(
    context: torch.Tensor,
    p: float = 0.2,
    mode: str = "last_to_noise"
) -> torch.Tensor:
    """
    Context Drop augmentation: randomly perturb last context position.

    Makes blind copying unreliable → forces model to use full context.

    Args:
        context: (B, 5, D) context vectors
        p: probability of applying drop (default: 0.2)
        mode: augmentation mode:
            - "last_to_noise": replace last frame with Gaussian noise
            - "last_to_zero": zero out last frame
            - "last_to_mean": replace with mean of first 4 frames

    Returns:
        (B, 5, D) augmented context (in-place safe via clone)
    """
    if p <= 0.0:
        return context

    B, L, D = context.shape
    assert L == 5, "Expected context length 5"

    # Clone to avoid in-place modification
    ctx_out = context.clone()

    # Sample mask: which samples to augment
    mask = torch.rand(B, device=context.device) < p  # (B,)

    if mask.sum() == 0:
        return ctx_out  # No samples selected

    if mode == "last_to_noise":
        # Replace last frame with Gaussian noise
        noise = torch.randn(mask.sum(), D, device=context.device)
        noise = F.normalize(noise, dim=-1, p=2)  # L2 normalize
        ctx_out[mask, -1, :] = noise

    elif mode == "last_to_zero":
        # Zero out last frame
        ctx_out[mask, -1, :] = 0.0

    elif mode == "last_to_mean":
        # Replace with mean of first 4 frames
        mean_vec = ctx_out[mask, :4, :].mean(dim=1)  # (B', D)
        mean_vec = F.normalize(mean_vec, dim=-1, p=2)
        ctx_out[mask, -1, :] = mean_vec

    else:
        raise ValueError(f"Unknown mode: {mode}")

    return ctx_out


# ============================================================================
# Diagnostic Utilities
# ============================================================================

def compute_offset_margins(
    pred: torch.Tensor,
    context: torch.Tensor,
    target: torch.Tensor
) -> dict:
    """
    Compute diagnostic margins for offset analysis.

    Returns dict with:
        - margin_pos1: cos(pred, target) - max(cos(pred, ctx[i]))
        - margin_vs_prev: cos(pred, target) - cos(pred, ctx[-2])
        - margin_vs_last: cos(pred, target) - cos(pred, ctx[-1])

    Args:
        pred: (B, D) predictions
        context: (B, 5, D) context
        target: (B, D) targets

    Returns:
        dict of scalar margins (mean over batch)
    """
    with torch.no_grad():
        cos_target = cosine_sim_batch(pred, target).mean().item()

        # Similarity to each context position
        cos_ctx = []
        for i in range(5):
            cos_i = cosine_sim_batch(pred, context[:, i, :]).mean().item()
            cos_ctx.append(cos_i)

        max_cos_ctx = max(cos_ctx)
        cos_prev = cos_ctx[-2]  # position -1 (previous)
        cos_last = cos_ctx[-1]  # position 0 (last context)

        return {
            "margin_pos1": cos_target - max_cos_ctx,
            "margin_vs_prev": cos_target - cos_prev,
            "margin_vs_last": cos_target - cos_last,
            "cos_target": cos_target,
            "cos_ctx": cos_ctx
        }


def directional_margin_loss_smooth(
    pred: torch.Tensor,
    y_next: torch.Tensor,
    y_prev: torch.Tensor,
    *,
    margin: float = 0.05,
    gamma: float = 8.0,
):
    """
    P6b: Smooth directional margin loss using softplus.

    Enforces cos(pred, next) > cos(pred, prev) + margin using smooth hinge.
    Loss = softplus(gamma * (margin - gap)) / gamma

    This is scale-friendly and avoids the hard ReLU used in earlier versions.
    Auto-scaling via λ_eff keeps this from overwhelming MSE.

    Args:
        pred: (B, D) model predictions
        y_next: (B, D) next vectors (target)
        y_prev: (B, D) previous vectors (hard negatives)
        margin: minimum required gap (default: 0.05)
        gamma: softplus sharpness (default: 8.0, higher = closer to ReLU)

    Returns:
        (loss, pos_mean, neg_mean, gap_mean) tuple
    """
    def _unit(x: torch.Tensor) -> torch.Tensor:
        """L2 normalize with safety clamp"""
        return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-8)

    p = _unit(pred)
    yn = _unit(y_next)
    yp = _unit(y_prev)

    pos = (p * yn).sum(dim=-1)   # cos(pred, next)
    neg = (p * yp).sum(dim=-1)   # cos(pred, prev)
    gap = pos - neg              # want gap >= margin

    # Smooth hinge: softplus(gamma * (margin - gap)) / gamma
    loss = F.softplus(gamma * (margin - gap)) / gamma

    return loss.mean(), pos.mean(), neg.mean(), gap.mean()


def directional_margin_loss_v21(
    pred: torch.Tensor,
    y_next: torch.Tensor,
    y_prev: torch.Tensor,
    *,
    margin_gap: float = 0.05,
    margin_ratio: float = 0.05,
    gamma: float = 4.0,
    alpha: float = 0.7,
):
    """
    P6b v2.1: Scale-aware directional margin loss with ratio term.

    Prevents "two negatives look good" failure mode by using both:
    1. Absolute gap: cos(pred, next) - cos(pred, prev)
    2. Relative ratio: gap / (|cos_pos| + |cos_neg| + eps)

    When both cosines go negative, raw gap can be large but meaningless.
    The ratio term detects this and applies proper penalty.

    Loss = α * gap_loss + (1-α) * ratio_loss
    where:
      gap_loss = softplus(γ * (m_gap - gap)) / γ
      ratio_loss = softplus(γ * (m_ratio - ratio)) / γ

    Args:
        pred: (B, D) model predictions
        y_next: (B, D) next vectors (target)
        y_prev: (B, D) previous vectors (hard negatives)
        margin_gap: minimum required absolute gap (default: 0.05)
        margin_ratio: minimum required ratio (default: 0.05)
        gamma: softplus sharpness (default: 4.0, softer than v1)
        alpha: weight for gap term vs ratio term (default: 0.7)

    Returns:
        dict with keys: loss, pos_mean, neg_mean, gap_mean, ratio_mean
    """
    def _unit(x: torch.Tensor) -> torch.Tensor:
        """L2 normalize with safety clamp"""
        return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-8)

    p = _unit(pred)
    yn = _unit(y_next)
    yp = _unit(y_prev)

    pos = (p * yn).sum(dim=-1)   # cos(pred, next)
    neg = (p * yp).sum(dim=-1)   # cos(pred, prev)
    gap = pos - neg              # absolute gap

    # Scale-aware ratio term (prevents "two negatives" victory)
    ratio = gap / (torch.abs(pos) + torch.abs(neg) + 1e-6)

    # Combined loss: weighted average of gap and ratio terms
    gap_loss = F.softplus(gamma * (margin_gap - gap)) / gamma
    ratio_loss = F.softplus(gamma * (margin_ratio - ratio)) / gamma

    loss = alpha * gap_loss + (1.0 - alpha) * ratio_loss

    return {
        'loss': loss.mean(),
        'pos_mean': pos.mean(),
        'neg_mean': neg.mean(),
        'gap_mean': gap.mean(),
        'ratio_mean': ratio.mean(),
        'gap_loss': gap_loss.mean(),
        'ratio_loss': ratio_loss.mean(),
    }


def positive_floor_penalty(
    pred: torch.Tensor,
    y_next: torch.Tensor,
    *,
    tau: float = 0.10,
):
    """
    P6b v2.1: Positive floor penalty.

    Prevents model from "winning the gap" with two negative cosines.
    Gently nudges cos(pred, y_next) to stay above threshold τ.

    penalty = ReLU(τ - cos(pred, y_next))^2

    Args:
        pred: (B, D) model predictions (will be normalized)
        y_next: (B, D) next vectors (will be normalized)
        tau: minimum threshold for cos(pred, y_next) (default: 0.10)

    Returns:
        penalty: scalar penalty term
    """
    def _unit(x: torch.Tensor) -> torch.Tensor:
        return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-8)

    p = _unit(pred)
    yn = _unit(y_next)
    cos_pos = (p * yn).sum(dim=-1)

    # ReLU(tau - cos_pos)^2
    penalty = F.relu(tau - cos_pos).pow(2)

    return penalty.mean()


def norm_regularization(pred: torch.Tensor) -> torch.Tensor:
    """
    P6b v2.1: Norm regularization (unit-sphere constraint).

    Stabilizes cosine/MSE interplay by keeping predictions near unit norm.

    penalty = (||pred||_2 - 1)^2

    Args:
        pred: (B, D) model predictions

    Returns:
        penalty: scalar penalty term
    """
    pred_norm = pred.norm(dim=-1, p=2)  # (B,)
    penalty = (pred_norm - 1.0).pow(2)
    return penalty.mean()


def orthogonality_penalty(
    pred: torch.Tensor,
    y_prev: torch.Tensor
) -> torch.Tensor:
    """
    P6b v2.2: Orthogonality penalty (anti-prev bias).

    Gently penalizes similarity to previous vector to counter backward bias.
    Uses squared cosine to make penalty smooth and avoid sharp transitions.

    penalty = (cos(pred, prev))^2

    This is VERY WEAK (κ ≈ 5e-4) and only chips away at copy-prev tendency.
    Unlike directional margin loss, this doesn't enforce ordering, just reduces alignment.

    Args:
        pred: (B, D) model predictions (will be normalized)
        y_prev: (B, D) previous vectors (will be normalized)

    Returns:
        penalty: scalar penalty term

    Usage:
        kappa = 5e-4  # Very small weight
        orth_pen = orthogonality_penalty(pred, y_prev)
        loss = loss + kappa * orth_pen
    """
    def _unit(x: torch.Tensor) -> torch.Tensor:
        return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-8)

    p = _unit(pred)
    yp = _unit(y_prev)

    # Squared cosine: smooth, differentiable, gentle
    cos_prev = (p * yp).sum(dim=-1)  # (B,)
    penalty = cos_prev.pow(2)        # (B,)

    return penalty.mean()


def log_directional_stats(
    pred: torch.Tensor,
    context: torch.Tensor,
    target: torch.Tensor,
    prefix: str = ""
) -> None:
    """
    Print diagnostic statistics for directional learning.

    Args:
        pred: (B, D) predictions
        context: (B, 5, D) context
        target: (B, D) targets
        prefix: optional prefix for print statements
    """
    stats = compute_offset_margins(pred, context, target)

    print(f"{prefix}Directional Stats:")
    print(f"{prefix}  cos(pred, target): {stats['cos_target']:.4f}")
    print(f"{prefix}  cos(pred, ctx[-2]): {stats['cos_ctx'][-2]:.4f} (prev)")
    print(f"{prefix}  cos(pred, ctx[-1]): {stats['cos_ctx'][-1]:.4f} (last)")
    print(f"{prefix}  Margin(+1 vs -1): {stats['margin_vs_prev']:.4f}")
    print(f"{prefix}  Margin(+1 vs last): {stats['margin_vs_last']:.4f}")

    if stats['margin_vs_last'] < 0:
        print(f"{prefix}  ⚠️  WARNING: Copying last context!")
    if stats['margin_vs_prev'] < 0:
        print(f"{prefix}  ⚠️  WARNING: Backward prediction!")


# ============================================================================
# P5.1 Micro-Directional Guard (Gentle Ranking Loss)
# ============================================================================

def micro_directional_loss(
    y_hat: torch.Tensor,
    y_next: torch.Tensor,
    ctx_last: torch.Tensor,
    gamma: float = 5.0,
    margin: float = 0.02
) -> torch.Tensor:
    """
    Micro-directional loss: gentle preference for next over last.

    Encourages cos(y_hat, y_next) >= cos(y_hat, ctx_last) + margin
    using a soft penalty: softplus(gamma * (cos_last - cos_next + margin))

    This is MUCH weaker than directional_margin_loss (uses softplus instead of ReLU,
    smaller gamma, smaller margin) to avoid destabilizing training while providing
    a gentle nudge away from copy-last.

    Args:
        y_hat: (B, D) model predictions (will be normalized)
        y_next: (B, D) target next vectors (will be normalized)
        ctx_last: (B, D) last context vector (will be normalized)
        gamma: softplus temperature (default: 5.0, lower = softer penalty)
        margin: minimum required gap (default: 0.02, very small)

    Returns:
        scalar loss

    Usage:
        lambda_dir = 0.001  # Very small weight
        L_micro = micro_directional_loss(pred, target, ctx[:, -1, :],
                                         gamma=5.0, margin=0.02)
        loss = L_mse + lambda_dir * L_micro
    """
    # Normalize defensively
    def _norm(x):
        return F.normalize(x, dim=-1, eps=1e-8)

    y_hat = _norm(y_hat)
    y_next = _norm(y_next)
    ctx_last = _norm(ctx_last)

    s_next = (y_hat * y_next).sum(dim=-1)  # (B,)
    s_last = (y_hat * ctx_last).sum(dim=-1)  # (B,)

    # Softplus penalty: smooth, differentiable, gentle
    penalty = F.softplus(gamma * (s_last - s_next + margin))

    return penalty.mean()
