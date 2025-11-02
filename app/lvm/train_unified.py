#!/usr/bin/env python3
"""
Unified LVM Trainer - Train Any Architecture
=============================================

Train any of the 4 LVM architectures with consistent hyperparameters:
1. LSTM Baseline (~5M params)
2. GRU Stack (~7M params)
3. Transformer (~18M params)
4. Attention Mixture Network (~2M params) [RECOMMENDED for LNSP]

Usage:
    python app/lvm/train_unified.py --model-type amn --epochs 20
    python app/lvm/train_unified.py --model-type transformer --epochs 20
    python app/lvm/train_unified.py --model-type lstm --epochs 20
    python app/lvm/train_unified.py --model-type gru --epochs 20

Key Features:
- MSE loss by default (fixed from InfoNCE bug)
- Consistent train/val split (90/10)
- Model-specific hyperparameters
- Progress logging and checkpointing
"""

import argparse
import json
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from models import create_model, MODEL_SPECS
from loss_utils import LossWeights, compute_losses
from train_helpers import (
    CycleConfig,
    compute_batch_stats,
    compute_mmd_rbf,
    mean_variance_penalty,
    maybe_cycle_penalty,
    sample_anchors,
)
from losses_directional import (
    directional_margin_loss,
    anticopy_hinge_loss,
    future_margin_loss,
    context_drop,
    compute_offset_margins,
)


# --- P2 Residual Next Wrapper ---
class ResidualNextWrapper(nn.Module):
    """
    Wraps a base LVM model to predict delta/residual from last frame.

    Instead of predicting absolute next vector:
        y_pred = model(ctx)

    Predict delta and compose with last frame:
        u = ctx[:, -1, :]              # Last frame
        delta = base_model(ctx)        # Model outputs delta
        y_pred = norm(u + alpha * delta)  # Compose

    This breaks identity copying: model must output non-zero delta to improve MSE.
    """
    def __init__(self, base: nn.Module, alpha_init: float = 0.5):
        super().__init__()
        self.base = base
        self.alpha = nn.Parameter(torch.tensor(alpha_init, dtype=torch.float32))

    def forward(self, ctx: torch.Tensor, return_raw: bool = False):
        # ctx shape: (B, 5, 768)
        u = F.normalize(ctx[:, -1, :], dim=-1, p=2)  # Last frame
        delta = self.base(ctx)  # Base model outputs delta (B, 768)
        y_pred = F.normalize(u + self.alpha * delta, dim=-1, p=2)  # Compose

        if return_raw:
            # Return both raw (unnormalized) and normalized
            y_raw = u + self.alpha * delta
            return y_raw, y_pred
        return y_pred

    def forward_with_components(self, ctx: torch.Tensor):
        """Return prediction + components for diagnostics"""
        u = F.normalize(ctx[:, -1, :], dim=-1, p=2)
        delta = self.base(ctx)
        y_pred = F.normalize(u + self.alpha * delta, dim=-1, p=2)
        return y_pred, delta, u


class VectorSequenceDataset(Dataset):
    """Dataset for autoregressive vector prediction"""

    def __init__(self, npz_path: str):
        data = np.load(npz_path, allow_pickle=True)
        if 'context_sequences' in data:
            self.contexts = torch.FloatTensor(data['context_sequences'])
        elif 'contexts' in data:
            self.contexts = torch.FloatTensor(data['contexts'])
        elif 'train_context_sequences' in data:
            self.contexts = torch.FloatTensor(data['train_context_sequences'])
        else:
            raise KeyError("Could not find context vectors in the .npz file.")

        if 'target_vectors' in data:
            self.targets = torch.FloatTensor(data['target_vectors'])
        elif 'targets' in data:
            self.targets = torch.FloatTensor(data['targets'])
        elif 'train_target_vectors' in data:
            self.targets = torch.FloatTensor(data['train_target_vectors'])
        else:
            raise KeyError("Could not find target vectors in the .npz file.")

        # Load metadata if available (for article-based splits)
        self.metadata = data.get('metadata', None)

        print(f"Loaded {len(self.contexts)} training pairs")
        print(f"Context shape: {self.contexts.shape}")
        print(f"Target shape: {self.targets.shape}")
        if self.metadata is not None:
            print(f"✅ Metadata available for article-based splits")
        else:
            print(f"⚠️  No metadata - will use random split")

    def __len__(self):
        return len(self.contexts)

    def __getitem__(self, idx):
        return self.contexts[idx], self.targets[idx]

    def get_article_indices(self):
        """Get article index for each sequence (if metadata available)"""
        if self.metadata is not None:
            return np.array([m['article_index'] for m in self.metadata])
        return None


def cosine_similarity(pred, target):
    """Compute cosine similarity between predictions and targets"""
    pred_norm = pred / (pred.norm(dim=1, keepdim=True) + 1e-8)
    target_norm = target / (target.norm(dim=1, keepdim=True) + 1e-8)
    return (pred_norm * target_norm).sum(dim=1).mean()


def train_epoch(
    model,
    dataloader,
    optimizer,
    device,
    loss_weights: LossWeights,
    *,
    anchors: torch.Tensor | None = None,
    anchor_sigma: float | None = None,
    lambda_mmd: float = 0.0,
    stats_mean: torch.Tensor | None = None,
    stats_std: torch.Tensor | None = None,
    lambda_stat: float = 0.0,
    cycle_cfg: CycleConfig | None = None,
    cycle_metrics: list[float] | None = None,
    rng: random.Random | None = None,
    lambda_dir: float = 0.0,
    margin_dir: float = 0.05,
    lambda_ac: float = 0.0,
    margin_ac: float = 0.02,
    lambda_fut: float = 0.0,
    margin_fut: float = 0.02,
    context_drop_p: float = 0.0,
    use_positional: bool = False,
    pos_scale: float = 0.03,
    rollout_h: int = 0,
    lambda_roll: float = 0.0,
    adaptive_dir: bool = False,
):
    model.train()
    total_loss = 0.0
    total_cosine = 0.0
    stats_acc = {
        "loss_info": 0.0,
        "loss_moment": 0.0,
        "loss_variance": 0.0,
        "loss_mse": 0.0,
        "loss_mmd": 0.0,
        "loss_stat": 0.0,
        "loss_cycle": 0.0,
        "loss_dir": 0.0,
        "loss_ac": 0.0,
        "loss_fut": 0.0,
        "loss_roll": 0.0,  # P4: Rollout loss (multi-step consistency)
        "margin_diagnostic": 0.0,  # Diagnostic: cos(pred, target) - cos(pred, ctx[-1])
    }
    rng = rng or random

    for batch_idx, (contexts, targets) in enumerate(dataloader):
        contexts = contexts.to(device)
        targets = targets.to(device)

        # Apply context drop augmentation if enabled
        if context_drop_p > 0.0:
            contexts = context_drop(contexts, p=context_drop_p, mode="last_to_noise")

        # Save original context for directional losses (before positional encoding)
        contexts_orig = contexts  # (B, 5, 768)

        # Add positional scalar (breaks time symmetry)
        if use_positional:
            B, T, D = contexts.shape  # T=5 (context length)
            pos = torch.linspace(0, 1, steps=T, device=device).unsqueeze(0).unsqueeze(-1)  # (1, T, 1)
            pos = pos.expand(B, T, 1) * pos_scale  # (B, T, 1)
            contexts = torch.cat([contexts, pos], dim=-1)  # (B, T, D+1) = (B, 5, 769)

        optimizer.zero_grad()

        pred_raw, pred_cos = model(contexts, return_raw=True)

        loss, stats = compute_losses(pred_raw, pred_cos, targets, loss_weights)
        for key, value in stats.items():
            if key in stats_acc:
                stats_acc[key] += value

        if lambda_mmd > 0.0 and anchors is not None and anchor_sigma is not None:
            idx = torch.randint(0, anchors.size(0), (pred_cos.size(0),), device=device)
            anchor_batch = anchors[idx]
            mmd = compute_mmd_rbf(pred_cos, anchor_batch, anchor_sigma)
            loss = loss + lambda_mmd * mmd
            stats_acc["loss_mmd"] += float(mmd.detach().item())

        if lambda_stat > 0.0 and stats_mean is not None and stats_std is not None:
            stat_penalty = mean_variance_penalty(pred_cos, stats_mean, stats_std)
            loss = loss + lambda_stat * stat_penalty
            stats_acc["loss_stat"] += float(stat_penalty.detach().item())

        if cycle_cfg is not None and cycle_cfg.enabled():
            cycle_penalty, cycle_cos = maybe_cycle_penalty(pred_raw[0], cycle_cfg, rng)
            if cycle_penalty is not None:
                loss = loss + cycle_penalty.to(device)
                stats_acc["loss_cycle"] += float(cycle_penalty.detach().item())
                if cycle_metrics is not None and cycle_cos is not None:
                    cycle_metrics.append(cycle_cos)

        # P4 Rollout Loss (makes copying fail over H steps)
        # Autoregressive prediction: [ctx → ŷ₁] → [ctx[1:],ŷ₁ → ŷ₂] → [ctx[2:],ŷ₁,ŷ₂ → ŷ₃]
        # Penalizes flat/copying trajectories by requiring forward momentum
        if rollout_h > 0 and lambda_roll > 0.0:
            import torch.nn.functional as F

            rollout_losses = []
            current_ctx = contexts_orig.clone()  # (B, 5, 768)

            for step in range(rollout_h):
                # Add positional encoding if needed
                if use_positional:
                    B, T, D = current_ctx.shape
                    pos = torch.linspace(0, 1, steps=T, device=device).unsqueeze(0).unsqueeze(-1)
                    pos = pos.expand(B, T, 1) * pos_scale
                    current_ctx_enc = torch.cat([current_ctx, pos], dim=-1)
                else:
                    current_ctx_enc = current_ctx

                # Predict next vector
                with torch.no_grad() if step > 0 else torch.enable_grad():
                    step_pred_raw, step_pred_cos = model(current_ctx_enc, return_raw=True)

                # For step 0, use actual target (ground truth for first prediction)
                # For step > 0, we don't have ground truth, so measure trajectory smoothness
                if step == 0:
                    # MSE against actual target for first step (same as base loss)
                    step_mse = F.mse_loss(step_pred_cos, targets)
                    rollout_losses.append(step_mse)
                else:
                    # Trajectory consistency: predictions should be smooth, not flat
                    # High variance = good (forward momentum), low variance = bad (copying)
                    prev_pred = F.normalize(rollout_losses[-1], dim=-1, p=2) if len(rollout_losses) > 1 else F.normalize(current_ctx[:, -1, :], dim=-1, p=2)
                    curr_pred = F.normalize(step_pred_cos, dim=-1, p=2)

                    # Penalize TOO MUCH similarity (indicates copying/flat trajectory)
                    trajectory_sim = (prev_pred * curr_pred).sum(dim=-1).mean()
                    # We want trajectories to evolve, not stay flat
                    # If similarity > 0.95, penalize (too flat)
                    flat_penalty = F.relu(trajectory_sim - 0.95) * 10.0  # Sharp penalty above 0.95
                    rollout_losses.append(flat_penalty)

                # Teacher forcing: update context with predicted vector
                # Shift context: remove first vector, append prediction
                step_pred_norm = F.normalize(step_pred_cos, dim=-1, p=2)
                current_ctx = torch.cat([current_ctx[:, 1:, :], step_pred_norm.unsqueeze(1)], dim=1)

            # Average rollout losses
            rollout_loss = sum(rollout_losses) / len(rollout_losses)
            loss = loss + lambda_roll * rollout_loss
            stats_acc["loss_roll"] += float(rollout_loss.detach().item())

        # Directional & Anti-Copy Losses (fixes "copy last context" bug)
        if lambda_dir > 0.0 or lambda_ac > 0.0 or lambda_fut > 0.0:
            import torch.nn.functional as F

            # Normalize predictions and targets
            pred_norm = F.normalize(pred_cos, dim=-1, p=2)
            target_norm = F.normalize(targets, dim=-1, p=2)

            # Use original 768D context for directional losses (not the 769D with positional)
            # Previous vector (position -1 relative to target)
            prev_vec = F.normalize(contexts_orig[:, -2, :], dim=-1, p=2)  # position 3 (second-to-last)

            if lambda_dir > 0.0:
                # P4 Adaptive directional weighting: boost lambda on high-similarity samples
                # where copy-last is most tempting (cos(ctx[-1], target) > 0.60)
                lambda_dir_eff = lambda_dir
                if adaptive_dir:
                    last_ctx = F.normalize(contexts_orig[:, -1, :], dim=-1, p=2)
                    sim_last_target = (last_ctx * target_norm).sum(dim=-1)  # (B,)
                    # Sigmoid boost: weak below 0.60, strong above 0.70
                    boost = torch.sigmoid((sim_last_target - 0.60) / 0.05)  # (B,)
                    lambda_dir_eff = lambda_dir * (1.0 + boost.mean().item())  # Scalar

                dir_loss = directional_margin_loss(pred_norm, target_norm, prev_vec, margin=margin_dir)
                loss = loss + lambda_dir_eff * dir_loss
                stats_acc["loss_dir"] += float(dir_loss.detach().item())

            if lambda_ac > 0.0:
                # Normalize context for anti-copy loss (use original 768D context)
                ctx_norm = F.normalize(contexts_orig, dim=-1, p=2)
                ac_loss = anticopy_hinge_loss(pred_norm, ctx_norm, target_norm, margin=margin_ac)
                loss = loss + lambda_ac * ac_loss
                stats_acc["loss_ac"] += float(ac_loss.detach().item())

            # TODO: Future margin loss (requires article-aware batching to get +2/+3 targets)
            # For now, this is disabled - requires dataloader to expose sequence indices
            # and article boundaries to safely look ahead to +2/+3 positions
            if lambda_fut > 0.0:
                # NOTE: Placeholder - requires implementation of article-aware sampling
                # fut_loss = future_margin_loss(pred_norm, target_norm, y_p2=None, y_p3=None, margin=margin_fut)
                # loss = loss + lambda_fut * fut_loss
                # stats_acc["loss_fut"] += float(fut_loss.detach().item())
                pass

            # Compute diagnostic margin (for logging, use original 768D context)
            with torch.no_grad():
                last_ctx = F.normalize(contexts_orig[:, -1, :], dim=-1, p=2)  # position 4 (last context)
                cos_target = (pred_norm * target_norm).sum(dim=-1).mean().item()
                cos_last = (pred_norm * last_ctx).sum(dim=-1).mean().item()
                margin_diag = cos_target - cos_last
                stats_acc["margin_diagnostic"] += margin_diag

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        with torch.no_grad():
            cosine = cosine_similarity(pred_cos, targets)
            total_cosine += cosine.item()

        if batch_idx % 100 == 0:
            log_msg = "  Batch {}/{} | Loss: {:.6f} | MSE: {:.6f} | Cosine: {:.4f}".format(
                batch_idx,
                len(dataloader),
                loss.item(),
                stats.get("loss_mse", 0.0),
                cosine.item(),
            )
            # Add directional diagnostic if enabled
            if lambda_dir > 0.0 or lambda_ac > 0.0:
                margin_avg = stats_acc["margin_diagnostic"] / (batch_idx + 1)
                log_msg += f" | Margin(+1 vs last): {margin_avg:.4f}"
            print(log_msg)

    denom = len(dataloader)
    avg_stats = {k: v / denom for k, v in stats_acc.items() if denom > 0}
    return total_loss / denom, total_cosine / denom, avg_stats


def evaluate(model, dataloader, device, use_positional: bool = False, pos_scale: float = 0.03):
    """Evaluate model

    Args:
        model: Model to evaluate
        dataloader: Validation dataloader
        device: Device to use
        use_positional: Whether to apply positional encoding (must match training)
        pos_scale: Scale factor for positional encoding
    """
    model.eval()
    total_loss = 0
    total_cosine = 0

    with torch.no_grad():
        for contexts, targets in dataloader:
            contexts = contexts.to(device)
            targets = targets.to(device)

            # Add positional scalar (same as training)
            if use_positional:
                B, T, D = contexts.shape  # T=5 (context length)
                pos = torch.linspace(0, 1, steps=T, device=device).unsqueeze(0).unsqueeze(-1)  # (1, T, 1)
                pos = pos.expand(B, T, 1) * pos_scale  # (B, T, 1)
                contexts = torch.cat([contexts, pos], dim=-1)  # (B, T, D+1) = (B, 5, 769)

            predictions = model(contexts)
            loss = nn.functional.mse_loss(predictions, targets)
            cosine = cosine_similarity(predictions, targets)

            total_loss += loss.item()
            total_cosine += cosine.item()

    return total_loss / len(dataloader), total_cosine / len(dataloader)


def get_model_config(model_type: str, input_dim: int = 768, output_dim: int = 768):
    """Get model-specific hyperparameters

    Args:
        model_type: Architecture type (lstm, gru, transformer, amn, etc.)
        input_dim: Input dimension (default 768, set to 769 if using positional encoding)
        output_dim: Output dimension (always 768 for target vectors)
    """
    configs = {
        'lstm': {
            'input_dim': input_dim,
            'hidden_dim': 512,
            'num_layers': 2,
            'dropout': 0.2,
            'output_dim': output_dim
        },
        'gru': {
            'input_dim': input_dim,
            'd_model': 512,
            'num_layers': 4,
            'dropout': 0.0,
            'output_dim': output_dim
        },
        'transformer': {
            'input_dim': input_dim,
            'd_model': 512,
            'nhead': 8,
            'num_layers': 4,
            'dropout': 0.1,
            'output_dim': output_dim
        },
        'amn': {
            'input_dim': input_dim,
            'd_model': 256,
            'hidden_dim': 512,
            'output_dim': output_dim
        },
        'hierarchical_gru': {
            'd_model': 768,
            'hidden_dim': 512,
            'chunk_size': 10,
            'num_chunks': 10,
            'local_layers': 2,
            'global_layers': 2
        },
        'memory_gru': {
            'd_model': 768,
            'hidden_dim': 512,
            'num_layers': 4,
            'memory_slots': 2048,
            'use_memory_write': True
        }
    }
    return configs.get(model_type, {})


def main():
    parser = argparse.ArgumentParser(description='Unified LVM Trainer')

    # Model selection
    parser.add_argument('--model-type', required=True,
                        choices=['lstm', 'gru', 'transformer', 'amn', 'hierarchical_gru', 'memory_gru'],
                        help='Model architecture to train')

    # Data and training
    parser.add_argument('--data', default='artifacts/lvm/training_sequences_ctx5.npz')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--device', default='mps' if torch.backends.mps.is_available() else 'cpu')
    parser.add_argument('--output-dir', default=None, help='Output directory (auto-generated if not specified)')

    # Loss weights (MSE is primary)
    parser.add_argument('--lambda-mse', type=float, default=1.0, help='MSE loss weight (PRIMARY)')
    parser.add_argument('--lambda-info', type=float, default=0.0, help='InfoNCE loss weight (disabled by default)')
    parser.add_argument('--lambda-moment', type=float, default=0.0, help='Moment matching weight')
    parser.add_argument('--lambda-variance', type=float, default=0.0, help='Variance penalty weight')
    parser.add_argument('--tau', type=float, default=0.07, help='Temperature for InfoNCE')

    # Optional regularization
    parser.add_argument('--lambda-mmd', type=float, default=0.0)
    parser.add_argument('--mmd-anchors', type=int, default=0)
    parser.add_argument('--lambda-stat', type=float, default=0.0)

    # Cycle consistency (experimental)
    parser.add_argument('--cycle-pct', type=float, default=0.0)
    parser.add_argument('--cycle-lambda', type=float, default=0.0)
    parser.add_argument('--cycle-steps', type=int, default=1)
    parser.add_argument('--cycle-timeout', type=float, default=30.0)
    parser.add_argument('--decoder-endpoint', default='http://127.0.0.1:8766/decode')
    parser.add_argument('--encoder-endpoint', default='http://127.0.0.1:8767/embed')

    # P2 Residual Prediction (architectural fix for copy-last shortcut)
    parser.add_argument('--residual-next', action='store_true',
        help='Predict delta from last frame and compose next: ŷ = norm(u + α·Δ)')
    parser.add_argument('--guards-start-epoch', type=int, default=6,
        help='Epoch to enable guard losses (directional/future/anticopy). <6 keeps warm-up pure MSE')

    # Directional & Anti-Copy Losses (tiny late guards)
    parser.add_argument('--lambda-dir', type=float, default=0.0,
        help='Weight for directional margin (next > prev). Suggested 0.002-0.01')
    parser.add_argument('--margin-dir', type=float, default=0.01, help='Directional margin threshold')
    parser.add_argument('--lambda-ac', type=float, default=0.0,
        help='Weight for anti-copy hinge (next > any ctx[i]). Suggested 0.0-0.005')
    parser.add_argument('--margin-ac', type=float, default=0.02, help='Anti-copy margin threshold')
    parser.add_argument('--context-drop-p', type=float, default=0.0, help='Context drop probability (0.0-1.0)')

    # Future Margin Loss (prevents k=+3 drift)
    parser.add_argument('--lambda-fut', type=float, default=0.0,
        help='Weight for future ranking (next > {+2,+3}). Suggested 0.002-0.005')
    parser.add_argument('--margin-fut', type=float, default=0.01, help='Future margin threshold')

    # P4 Rollout Loss (makes copying fail over multiple steps)
    parser.add_argument('--rollout-h', type=int, default=0,
        help='Rollout horizon (0=disabled, 3=predict 3 steps ahead). Punishes copy-last over multi-step')
    parser.add_argument('--lambda-roll', type=float, default=0.0,
        help='Rollout loss weight. Suggested: 0.05 early, 0.10 later')
    parser.add_argument('--rollout-start-epoch', type=int, default=4,
        help='Epoch to enable rollout loss (default 4, after basic learning)')
    parser.add_argument('--adaptive-dir', action='store_true',
        help='Make lambda_dir adaptive based on cos(ctx[-1], target). Boosts on high-similarity samples.')

    # Mini-5CAT per-epoch validation
    parser.add_argument('--mini5cat-max-samples', type=int, default=500,
        help='Run mini 5CAT each epoch on VAL; abort/backoff on negative margin')

    # Positional Encoding (breaks time symmetry)
    parser.add_argument('--use-positional', action='store_true', help='Add positional scalar to break time symmetry')
    parser.add_argument('--pos-scale', type=float, default=0.03, help='Positional encoding scale factor')
    parser.add_argument('--positional-scalar', type=float, default=0.0, help='Positional scalar weight (0.03 for P5), overrides --use-positional if > 0')

    # Curriculum Learning (P5)
    parser.add_argument('--curriculum', type=str, default='full',
                        choices=['full', 'forward_top_30', 'forward_top_70'],
                        help='Curriculum stage: full, forward_top_30 (Stage A), or forward_top_70 (Stage B)')
    parser.add_argument('--curriculum-scores', type=str,
                        help='Path to forward-distinctness scores NPZ (required for curriculum != full)')

    # Loss Scheduling (prevents early collapse)
    parser.add_argument('--warmup-epochs', type=int, default=0, help='Epochs with pure MSE before enabling guards')
    parser.add_argument('--ramp-epochs', type=int, default=0, help='Epochs for ramping guard losses (after warmup)')

    args = parser.parse_args()

    # Auto-generate output directory if not specified
    if args.output_dir is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.output_dir = f'artifacts/lvm/models/{args.model_type}_{timestamp}'

    print("=" * 80)
    print(f"Unified LVM Trainer - {args.model_type.upper()}")
    print("=" * 80)
    print(f"Model: {MODEL_SPECS[args.model_type]['name']}")
    print(f"Description: {MODEL_SPECS[args.model_type]['description']}")
    print(f"Expected params: {MODEL_SPECS[args.model_type]['params']}")
    print()
    print(f"Data: {args.data}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Device: {args.device}")
    print(f"Output: {args.output_dir}")
    print()
    print("Loss Configuration:")
    print(f"  MSE weight: {args.lambda_mse} (PRIMARY)")
    print(f"  InfoNCE weight: {args.lambda_info} (disabled)")
    print()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data (with curriculum support)
    print("Loading dataset...")

    # Determine which dataset to load based on curriculum stage
    data_path = args.data
    if args.curriculum != 'full':
        # Load curriculum-specific NPZ instead of full training data
        print(f"📚 [CURRICULUM] Stage: {args.curriculum}")

        # Derive curriculum NPZ path from training path
        train_base = Path(args.data).stem  # Remove .npz extension
        train_dir = Path(args.data).parent

        if args.curriculum == 'forward_top_30':
            curriculum_npz = train_dir / f"{train_base}_stage_a_top30.npz"
        elif args.curriculum == 'forward_top_70':
            curriculum_npz = train_dir / f"{train_base}_stage_b_top70.npz"
        else:
            curriculum_npz = Path(args.data)  # Shouldn't reach here

        if not curriculum_npz.exists():
            raise FileNotFoundError(
                f"Curriculum NPZ not found: {curriculum_npz}\n"
                f"Run tools/build_curriculum_splits.py first!"
            )

        data_path = str(curriculum_npz)
        print(f"   Loading: {data_path}")

    dataset = VectorSequenceDataset(data_path)
    targets_np = dataset.targets.numpy()

    # Optional regularization setup
    anchor_tensor = None
    anchor_sigma = None
    if args.lambda_mmd > 0.0 and args.mmd_anchors > 0:
        anchor_tensor, anchor_sigma = sample_anchors(targets_np, args.mmd_anchors)
        print(f"Anchor set prepared: {anchor_tensor.shape[0]} vectors (sigma={anchor_sigma:.4f})")

    stats_mean_tensor = None
    stats_std_tensor = None
    if args.lambda_stat > 0.0:
        mean_np, std_np = compute_batch_stats(targets_np)
        stats_mean_tensor = torch.from_numpy(mean_np)
        stats_std_tensor = torch.from_numpy(std_np)

    # Split train/val by ARTICLES (not random sequences!)
    # SKIP article split when using curriculum - curriculum NPZ is already pre-filtered
    skip_article_split = (args.curriculum != 'full')
    article_indices = dataset.get_article_indices()

    if article_indices is not None and not skip_article_split:
        # Article-based split (proper OOD validation)
        print("🎯 Using ARTICLE-BASED split (proper generalization test)")
        unique_articles = sorted(set(article_indices))
        print(f"   Total unique articles: {len(unique_articles)}")

        # Split articles 90/10
        val_article_count = max(1, int(0.1 * len(unique_articles)))
        train_articles = set(unique_articles[:-val_article_count])
        val_articles = set(unique_articles[-val_article_count:])

        print(f"   Train articles: {len(train_articles)}")
        print(f"   Val articles: {len(val_articles)}")
        print(f"   Val article range: {min(val_articles)}-{max(val_articles)}")

        # Create masks
        train_mask = np.array([art in train_articles for art in article_indices])
        val_mask = np.array([art in val_articles for art in article_indices])

        train_indices = np.where(train_mask)[0]
        val_indices = np.where(val_mask)[0]

        train_dataset = torch.utils.data.Subset(dataset, train_indices)
        val_dataset = torch.utils.data.Subset(dataset, val_indices)

        print(f"   Train sequences: {len(train_dataset)}")
        print(f"   Val sequences: {len(val_dataset)}")
        print()
    else:
        # Random split (curriculum or no metadata)
        if skip_article_split:
            print("📚 [CURRICULUM] Using full curriculum dataset with 90/10 split")
        else:
            print("⚠️  Using RANDOM split (metadata not available)")

        train_size = int(0.9 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
        print(f"   Train: {len(train_dataset)} samples")
        print(f"   Val: {len(val_dataset)} samples")
        print()

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Create model
    print("Creating model...")
    device = torch.device(args.device)

    # Adjust input dimension if using positional encoding
    # Support both --use-positional (legacy) and --positional-scalar (P5)
    use_positional_encoding = args.use_positional or (args.positional_scalar > 0)
    input_dim = 769 if use_positional_encoding else 768

    if use_positional_encoding:
        pos_weight = args.positional_scalar if args.positional_scalar > 0 else args.pos_scale
        print(f"🔢 Positional encoding ENABLED → input_dim = 769 (weight={pos_weight})")
    else:
        pos_weight = args.pos_scale  # Default value even if disabled
        print("🔢 Positional encoding DISABLED → input_dim = 768")

    model_config = get_model_config(args.model_type, input_dim=input_dim)
    base_model = create_model(args.model_type, **model_config).to(device)

    # Wrap with residual prediction if enabled
    if args.residual_next:
        print("🔄 Residual prediction ENABLED: ŷ = norm(u + α·Δ)")
        model = ResidualNextWrapper(base_model, alpha_init=0.5).to(device)
    else:
        model = base_model

    actual_params = model.count_parameters() if hasattr(model, 'count_parameters') else sum(p.numel() for p in model.parameters())
    print(f"Actual parameters: {actual_params:,}")
    print()

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    if anchor_tensor is not None:
        anchor_tensor = anchor_tensor.to(device)
    if stats_mean_tensor is not None:
        stats_mean_tensor = stats_mean_tensor.to(device)
        stats_std_tensor = stats_std_tensor.to(device)

    cycle_cfg = CycleConfig(
        pct=args.cycle_pct,
        weight=args.cycle_lambda,
        steps=args.cycle_steps,
        decoder_endpoint=args.decoder_endpoint,
        encoder_endpoint=args.encoder_endpoint,
        timeout=args.cycle_timeout,
    )
    cycle_metrics: list[float] = []
    rng = random.Random(42)

    # Training loop
    best_val_loss = float('inf')
    history = []
    loss_weights = LossWeights(
        tau=args.tau,
        mse=args.lambda_mse,
        info_nce=args.lambda_info,
        moment=args.lambda_moment,
        variance=args.lambda_variance,
    )

    print("Starting training...")
    print()

    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")

        # P4: Curriculum scheduling (warm-up → rollout → guards)
        current_epoch = epoch + 1

        # Phase 1: Pure MSE warm-up (epochs 1-3)
        if current_epoch < args.rollout_start_epoch:
            lambda_roll_sched = 0.0
            rollout_h_sched = 0
            lambda_dir_sched = 0.0
            lambda_ac_sched = 0.0
            lambda_fut_sched = 0.0
            context_drop_sched = 0.0
            if epoch == 0:
                print(f"  [Warm-up] Pure MSE (rollout starts epoch {args.rollout_start_epoch}, guards epoch {args.guards_start_epoch})")

        # Phase 2: Rollout loss active (epochs 4-5)
        elif current_epoch < args.guards_start_epoch:
            lambda_roll_sched = args.lambda_roll
            rollout_h_sched = args.rollout_h
            lambda_dir_sched = 0.0
            lambda_ac_sched = 0.0
            lambda_fut_sched = 0.0
            context_drop_sched = 0.0
            if current_epoch == args.rollout_start_epoch:
                print(f"  [Rollout ON] H={rollout_h_sched}, λ_roll={lambda_roll_sched}")

        # Phase 3: Rollout + guards (epochs 6+)
        else:
            # Increase rollout weight at epoch 7+
            lambda_roll_sched = args.lambda_roll * (2.0 if current_epoch >= 7 else 1.0)
            rollout_h_sched = args.rollout_h
            lambda_dir_sched = args.lambda_dir
            lambda_ac_sched = args.lambda_ac
            lambda_fut_sched = args.lambda_fut if current_epoch >= 10 else 0.0  # Future loss at epoch 10+
            context_drop_sched = args.context_drop_p
            if current_epoch == args.guards_start_epoch:
                print(f"  [Guards ON] λ_dir={lambda_dir_sched}, λ_fut={lambda_fut_sched}, λ_roll={lambda_roll_sched}")

        # Train
        train_loss, train_cosine, train_stats = train_epoch(
            model,
            train_loader,
            optimizer,
            device,
            loss_weights,
            anchors=anchor_tensor,
            anchor_sigma=anchor_sigma,
            lambda_mmd=args.lambda_mmd,
            stats_mean=stats_mean_tensor,
            stats_std=stats_std_tensor,
            lambda_stat=args.lambda_stat,
            cycle_cfg=cycle_cfg,
            cycle_metrics=cycle_metrics,
            rng=rng,
            lambda_dir=lambda_dir_sched,
            margin_dir=args.margin_dir,
            lambda_ac=lambda_ac_sched,
            margin_ac=args.margin_ac,
            lambda_fut=lambda_fut_sched,
            margin_fut=args.margin_fut,
            context_drop_p=context_drop_sched,
            use_positional=use_positional_encoding,
            pos_scale=pos_weight,
            rollout_h=rollout_h_sched,
            lambda_roll=lambda_roll_sched,
            adaptive_dir=args.adaptive_dir,
        )

        # Validate (use same positional encoding as training)
        val_loss, val_cosine = evaluate(model, val_loader, device,
                                        use_positional=use_positional_encoding,
                                        pos_scale=pos_weight)

        # Learning rate schedule
        scheduler.step(val_loss)

        # Log
        mse_val = train_stats.get("loss_mse", 0.0)
        info_val = train_stats.get("loss_info", 0.0)
        dir_val = train_stats.get("loss_dir", 0.0)
        ac_val = train_stats.get("loss_ac", 0.0)
        roll_val = train_stats.get("loss_roll", 0.0)
        margin_val = train_stats.get("margin_diagnostic", 0.0)

        print(
            "  Train Loss: {:.6f} | Train Cosine: {:.4f} | MSE: {:.6f}".format(
                train_loss,
                train_cosine,
                mse_val,
            )
        )
        print(f"  Val Loss: {val_loss:.6f} | Val Cosine: {val_cosine:.4f}")

        # Log rollout stats if enabled
        if args.rollout_h > 0 and lambda_roll_sched > 0.0:
            print(f"  Rollout: L_roll={roll_val:.6f} (H={rollout_h_sched}, λ={lambda_roll_sched:.3f})")

        # Log directional stats if enabled
        if args.lambda_dir > 0.0 or args.lambda_ac > 0.0:
            print(f"  Directional: L_dir={dir_val:.6f} | L_ac={ac_val:.6f} | Margin(+1 vs last)={margin_val:.4f}")
            if margin_val > 0:
                print("  ✅ Positive margin! Model predicts NEXT, not last context")
            else:
                print("  ⚠️  Negative margin - still copying last context")

        print()

        history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_cosine': train_cosine,
            'train_loss_mse': train_stats.get('loss_mse', 0.0),
            'train_loss_info': train_stats.get('loss_info', 0.0),
            'val_loss': val_loss,
            'val_cosine': val_cosine,
            'lr': optimizer.param_groups[0]['lr']
        })

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch + 1,
                'model_type': args.model_type,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_cosine': val_cosine,
                'model_config': model_config,
                'args': vars(args)
            }, output_dir / 'best_model.pt')
            print(f"  ✓ Saved best model (val_loss: {val_loss:.6f})")

    # Save final model
    torch.save({
        'epoch': args.epochs,
        'model_type': args.model_type,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
        'val_cosine': val_cosine,
        'model_config': model_config,
        'args': vars(args)
    }, output_dir / 'final_model.pt')

    # Save training history
    with open(output_dir / 'training_history.json', 'w') as f:
        json.dump({
            'model_type': args.model_type,
            'model_name': MODEL_SPECS[args.model_type]['name'],
            'history': history,
            'best_val_loss': best_val_loss,
            'final_params': actual_params,
            'trained_at': datetime.now().isoformat()
        }, f, indent=2)

    print("=" * 80)
    print("Training Complete!")
    print(f"Model: {MODEL_SPECS[args.model_type]['name']}")
    print(f"Best val loss: {best_val_loss:.6f}")
    print(f"Final val cosine: {val_cosine:.4f}")
    print(f"Models saved to: {output_dir}")
    print("=" * 80)


if __name__ == '__main__':
    main()
