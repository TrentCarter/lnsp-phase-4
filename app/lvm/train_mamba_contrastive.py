#!/usr/bin/env python3
"""
Contrastive-First Mamba Training (Phase 5.1)
=============================================

Fixes generalization failure by adding proper contrastive learning.

Key changes from base training:
1. Projection head: 768→512→256 with GELU, LayerNorm
2. InfoNCE with all in-batch negatives (τ=0.07)
3. Stop-grad on target branch
4. Combined loss: 0.7 * InfoNCE + 0.3 * AR_cosine
5. Article dropout (p=0.2) and span corruption (p=0.1)
6. Large effective batch via grad accumulation

Usage:
    python app/lvm/train_mamba_contrastive.py \
        --model-type mamba_s \
        --train-npz artifacts/lvm/train_payload_aligned.npz \
        --d-model 768 --n-layers 8 --d-state 128 \
        --batch-size 256 --grad-accum-steps 4 \
        --device mps \
        --save-dir artifacts/lvm/models/mamba_s_contrastive
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Add project root
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from app.lvm.mamba import create_model, count_parameters


class ProjectionHead(nn.Module):
    """Siamese projection head for contrastive learning."""

    def __init__(self, d_model=768, hidden_dim=512, out_dim=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden_dim, bias=True),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, out_dim, bias=False),  # No bias on final
        )

    def forward(self, x):
        h = self.mlp(x)
        return F.normalize(h, p=2, dim=-1)  # L2 normalize output


class ContrastiveLoss(nn.Module):
    """InfoNCE + AR cosine combined loss."""

    def __init__(
        self,
        projection_head: ProjectionHead,
        lambda_con: float = 0.7,
        lambda_ar: float = 0.3,
        temperature: float = 0.07,
    ):
        super().__init__()
        self.projection_head = projection_head
        self.lambda_con = lambda_con
        self.lambda_ar = lambda_ar
        self.temperature = temperature

    def forward(self, pred, target):
        """
        Args:
            pred: [B, 768] model predictions (pre-head)
            target: [B, 768] ground truth vectors

        Returns:
            total_loss, loss_dict
        """
        B = pred.shape[0]

        # 1. AR cosine loss (on raw vectors)
        loss_ar = 1.0 - F.cosine_similarity(pred, target, dim=-1).mean()

        # 2. InfoNCE contrastive loss (on projected vectors)
        # Project predictions
        h_pred = self.projection_head(pred)  # [B, 256]

        # Project targets with stop-grad
        with torch.no_grad():
            h_target = self.projection_head(target)  # [B, 256]

        # Compute all pairwise similarities (full batch negatives)
        # h_pred @ h_target.T = [B, B]
        logits = torch.mm(h_pred, h_target.t()) / self.temperature  # [B, B]

        # Labels: diagonal elements are positives (i→i)
        labels = torch.arange(B, device=pred.device)

        # InfoNCE = cross-entropy with positives on diagonal
        loss_infonce = F.cross_entropy(logits, labels)

        # 3. Combined loss
        total_loss = self.lambda_con * loss_infonce + self.lambda_ar * loss_ar

        loss_dict = {
            'infonce': loss_infonce.item(),
            'ar_cosine': loss_ar.item(),
            'total': total_loss.item(),
        }

        return total_loss, loss_dict


class VectorSequenceDataset(Dataset):
    """Dataset with article dropout and span corruption."""

    def __init__(
        self,
        npz_path: str,
        article_dropout_p: float = 0.2,
        span_corruption_p: float = 0.1,
        training: bool = True,
    ):
        self.data = np.load(npz_path)
        self.article_dropout_p = article_dropout_p
        self.span_corruption_p = span_corruption_p
        self.training = training

        # Load sequences
        if 'context_sequences' in self.data:
            self.sequences = self.data['context_sequences']
        else:
            self.sequences = self.data['sequences']

        if 'target_vectors' in self.data:
            self.targets = self.data['target_vectors']
        else:
            self.targets = self.data['targets']

        # Convert to tensors
        self.sequences = torch.from_numpy(self.sequences).float()
        self.targets = torch.from_numpy(self.targets).float()

        print(f"  Loaded {len(self.sequences)} sequences")
        print(f"  Context shape: {self.sequences.shape}")
        print(f"  Article dropout: {article_dropout_p}")
        print(f"  Span corruption: {span_corruption_p}")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx].clone()  # [context_len, 768]
        target = self.targets[idx]

        if self.training:
            # Article dropout: zero out last k positions with probability p
            if torch.rand(1).item() < self.article_dropout_p:
                k = torch.randint(1, min(4, sequence.shape[0]), (1,)).item()
                sequence[-k:] = 0.0

            # Span corruption: replace random position with another sample's vector
            if torch.rand(1).item() < self.span_corruption_p:
                pos = torch.randint(0, sequence.shape[0] - 1, (1,)).item()
                other_idx = torch.randint(0, len(self), (1,)).item()
                sequence[pos] = self.sequences[other_idx, pos]

        return {
            'sequence': sequence,
            'target': target,
        }


def get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr_ratio: float = 0.01,
):
    """Cosine LR schedule with linear warmup."""

    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        else:
            progress = float(current_step - num_warmup_steps) / float(
                max(1, num_training_steps - num_warmup_steps)
            )
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_epoch(
    model,
    projection_head,
    dataloader,
    criterion,
    optimizer,
    scheduler,
    device,
    grad_accum_steps=1,
):
    """Train for one epoch with gradient accumulation."""
    model.train()
    projection_head.train()

    total_loss = 0.0
    loss_components = {}

    optimizer.zero_grad()

    for batch_idx, batch in enumerate(dataloader):
        sequences = batch['sequence'].to(device)
        targets = batch['target'].to(device)

        # Forward pass
        outputs = model(sequences)

        # Take last timestep if sequence output
        if len(outputs.shape) == 3:
            outputs = outputs[:, -1, :]

        # L2 normalize before loss
        outputs = F.normalize(outputs, p=2, dim=-1)
        targets = F.normalize(targets, p=2, dim=-1)

        # Compute loss
        loss, loss_dict = criterion(outputs, targets)

        # Scale loss by accumulation steps
        loss = loss / grad_accum_steps

        # Backward pass
        loss.backward()

        # Update weights every grad_accum_steps
        if (batch_idx + 1) % grad_accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(
                list(model.parameters()) + list(projection_head.parameters()),
                max_norm=1.0
            )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        total_loss += loss.item() * grad_accum_steps

        # Accumulate loss components
        for k, v in loss_dict.items():
            loss_components[k] = loss_components.get(k, 0.0) + v

        # Log periodically
        if batch_idx % 100 == 0 and batch_idx > 0:
            avg_loss = total_loss / (batch_idx + 1)
            print(f"  Batch {batch_idx}/{len(dataloader)} | Loss: {avg_loss:.6f}")

    # Average losses
    n_batches = len(dataloader)
    avg_loss = total_loss / n_batches
    avg_components = {k: v / n_batches for k, v in loss_components.items()}

    return avg_loss, avg_components


@torch.no_grad()
def evaluate(model, projection_head, dataloader, criterion, device):
    """Evaluate model on validation set."""
    model.eval()
    projection_head.eval()

    total_loss = 0.0
    loss_components = {}
    all_cosines = []

    for batch in dataloader:
        sequences = batch['sequence'].to(device)
        targets = batch['target'].to(device)

        outputs = model(sequences)

        if len(outputs.shape) == 3:
            outputs = outputs[:, -1, :]

        # L2 normalize
        outputs = F.normalize(outputs, p=2, dim=-1)
        targets = F.normalize(targets, p=2, dim=-1)

        loss, loss_dict = criterion(outputs, targets)

        total_loss += loss.item()

        for k, v in loss_dict.items():
            loss_components[k] = loss_components.get(k, 0.0) + v

        # Track cosines
        cosines = F.cosine_similarity(outputs, targets, dim=-1)
        all_cosines.extend(cosines.cpu().numpy().tolist())

    n_batches = len(dataloader)
    avg_loss = total_loss / n_batches
    avg_components = {k: v / n_batches for k, v in loss_components.items()}
    avg_cosine = float(np.mean(all_cosines))

    return avg_loss, avg_components, avg_cosine


def main():
    ap = argparse.ArgumentParser(description="Contrastive-first Mamba training")

    # Model architecture
    ap.add_argument("--model-type", type=str, required=True,
                    choices=["mamba_s", "mamba_hybrid_local", "mamba_sandwich", "mamba_gr"])
    ap.add_argument("--d-model", type=int, default=768)
    ap.add_argument("--n-layers", type=int, default=8)
    ap.add_argument("--d-state", type=int, default=128)
    ap.add_argument("--conv-sz", type=int, default=4)
    ap.add_argument("--expand", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.1)

    # Hybrid/sandwich specific
    ap.add_argument("--local-attn-win", type=int, default=8)
    ap.add_argument("--local-attn-every", type=int, default=4)
    ap.add_argument("--n-heads", type=int, default=4)
    ap.add_argument("--n-layers-mamba", type=int, default=8)
    ap.add_argument("--n-layers-local", type=int, default=4)
    ap.add_argument("--gru-hidden", type=int, default=256)

    # Projection head
    ap.add_argument("--proj-hidden", type=int, default=512)
    ap.add_argument("--proj-out", type=int, default=256)

    # Contrastive loss
    ap.add_argument("--lambda-con", type=float, default=0.7)
    ap.add_argument("--lambda-ar", type=float, default=0.3)
    ap.add_argument("--temperature", type=float, default=0.07)

    # Data
    ap.add_argument("--train-npz", type=str, required=True)
    ap.add_argument("--val-split", type=float, default=0.1)
    ap.add_argument("--article-dropout", type=float, default=0.2)
    ap.add_argument("--span-corruption", type=float, default=0.1)

    # Training
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--grad-accum-steps", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=0.02)
    ap.add_argument("--warmup-steps", type=int, default=1000)
    ap.add_argument("--early-stop-patience", type=int, default=3)

    # System
    ap.add_argument("--device", type=str, default="mps",
                    choices=["cpu", "cuda", "mps"])
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--save-dir", type=Path, required=True)

    args = ap.parse_args()

    print("=" * 80)
    print("Contrastive-First Mamba Training (Phase 5.1)")
    print("=" * 80)
    print(f"Model: {args.model_type}")
    print(f"Training data: {args.train_npz}")
    print(f"Contrastive: λ_con={args.lambda_con}, λ_ar={args.lambda_ar}, τ={args.temperature}")
    print(f"Projection head: {args.d_model}→{args.proj_hidden}→{args.proj_out}")
    print(f"Regularization: article_dropout={args.article_dropout}, span_corruption={args.span_corruption}")
    print(f"Effective batch: {args.batch_size * args.grad_accum_steps}")
    print("=" * 80)

    # Create save directory
    args.save_dir.mkdir(parents=True, exist_ok=True)

    # Save args (convert Path to string for JSON serialization)
    args_dict = vars(args).copy()
    args_dict['save_dir'] = str(args_dict['save_dir'])
    with open(args.save_dir / 'args.json', 'w') as f:
        json.dump(args_dict, f, indent=2)

    # Load full dataset
    print("\nLoading full dataset...")
    full_dataset = VectorSequenceDataset(
        args.train_npz,
        article_dropout_p=args.article_dropout,
        span_corruption_p=args.span_corruption,
        training=True,
    )

    # Train/val split
    n = len(full_dataset)
    n_val = int(n * args.val_split)
    n_train = n - n_val

    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )

    # Validation dataset without augmentation
    val_dataset_clean = VectorSequenceDataset(
        args.train_npz,
        article_dropout_p=0.0,
        span_corruption_p=0.0,
        training=False,
    )
    val_dataset_clean = torch.utils.data.Subset(val_dataset_clean, val_dataset.indices)

    print(f"  Train: {len(train_dataset)} sequences")
    print(f"  Val: {len(val_dataset)} sequences")

    # Dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(args.device != 'cpu'),
    )

    val_loader = DataLoader(
        val_dataset_clean,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(args.device != 'cpu'),
    )

    # Build model
    print("\nBuilding model...")
    model_kwargs = {
        'd_model': args.d_model,
        'd_state': args.d_state,
        'conv_sz': args.conv_sz,
        'expand': args.expand,
        'dropout': args.dropout,
    }

    if 'sandwich' in args.model_type:
        model_kwargs.update({
            'n_layers_mamba': args.n_layers_mamba,
            'n_layers_local': args.n_layers_local,
            'local_attn_win': args.local_attn_win,
            'n_heads': args.n_heads,
        })
    elif 'hybrid' in args.model_type:
        model_kwargs.update({
            'n_layers': args.n_layers,
            'local_attn_win': args.local_attn_win,
            'local_attn_every': args.local_attn_every,
            'n_heads': args.n_heads,
        })
    else:
        model_kwargs['n_layers'] = args.n_layers

    if 'gr' in args.model_type:
        model_kwargs['gru_hidden'] = args.gru_hidden

    model = create_model(model_type=args.model_type, **model_kwargs)
    model.to(args.device)

    # Build projection head
    projection_head = ProjectionHead(
        d_model=args.d_model,
        hidden_dim=args.proj_hidden,
        out_dim=args.proj_out,
    )
    projection_head.to(args.device)

    total_params = count_parameters(model) + sum(p.numel() for p in projection_head.parameters())
    print(f"  Total parameters: {total_params:,}")

    # Loss function
    criterion = ContrastiveLoss(
        projection_head=projection_head,
        lambda_con=args.lambda_con,
        lambda_ar=args.lambda_ar,
        temperature=args.temperature,
    )

    # Optimizer
    all_params = list(model.parameters()) + list(projection_head.parameters())
    optimizer = torch.optim.AdamW(
        all_params,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # Scheduler
    num_training_steps = len(train_loader) * args.epochs // args.grad_accum_steps
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=num_training_steps,
    )

    print(f"  Training steps: {num_training_steps:,}")
    print(f"  Warmup steps: {args.warmup_steps:,}")

    # Training loop
    print("\n" + "=" * 80)
    print("Training")
    print("=" * 80)

    history = []
    best_val_cosine = 0.0
    patience_counter = 0

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        print("-" * 80)

        # Train
        epoch_start = time.time()
        train_loss, train_components = train_epoch(
            model, projection_head, train_loader, criterion,
            optimizer, scheduler, args.device, args.grad_accum_steps
        )

        # Validate
        val_loss, val_components, val_cosine = evaluate(
            model, projection_head, val_loader, criterion, args.device
        )

        epoch_time = time.time() - epoch_start

        # Log
        print(f"\n  Train loss: {train_loss:.6f}")
        print(f"  Val loss:   {val_loss:.6f}")
        print(f"  Val cosine: {val_cosine:.4f}")
        print(f"  Time: {epoch_time:.1f}s")

        # Record history
        history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_infonce': train_components['infonce'],
            'train_ar_cosine': train_components['ar_cosine'],
            'val_loss': val_loss,
            'val_cosine': val_cosine,
            'lr': optimizer.param_groups[0]['lr'],
            'time': epoch_time,
        })

        # Save best model
        if val_cosine > best_val_cosine:
            best_val_cosine = val_cosine
            patience_counter = 0

            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'projection_head_state_dict': projection_head.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_cosine': val_cosine,
                'val_loss': val_loss,
                'args': vars(args),
            }, args.save_dir / 'best.pt')

            print(f"  ✅ Saved best model (val_cosine={val_cosine:.4f})")
        else:
            patience_counter += 1
            print(f"  ⏸ No improvement ({patience_counter}/{args.early_stop_patience})")

            if patience_counter >= args.early_stop_patience:
                print("\n🛑 Early stopping triggered!")
                break

    # Save final model
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'projection_head_state_dict': projection_head.state_dict(),
    }, args.save_dir / 'final.pt')

    # Save history
    with open(args.save_dir / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)

    print("\n" + "=" * 80)
    print("✅ Training complete!")
    print("=" * 80)
    print(f"Best val cosine: {best_val_cosine:.4f}")
    print(f"Saved to: {args.save_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
