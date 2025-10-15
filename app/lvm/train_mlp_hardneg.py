#!/usr/bin/env python3
"""
Phase 4: MLP with Hard Negatives

Adds contrastive learning with in-batch hard negatives:
  - For each prediction, find K hardest negatives (closest wrong targets)
  - Push predictions away from negatives with margin loss
  - Cheap to compute (no vec2text calls), addresses semantic confusion

Loss = main_loss + lambda_neg * hard_negative_loss
"""

import argparse
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from torch.utils.data import Dataset, DataLoader


class SimpleMLP(nn.Module):
    """Very simple MLP: flatten context → hidden → output"""

    def __init__(self, context_size=5, vector_dim=768, hidden_dim=1024):
        super().__init__()

        input_dim = context_size * vector_dim

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, vector_dim)
        )

        # Normalize output to unit sphere
        self.normalize = True

    def forward(self, x):
        # x shape: (batch_size, context_size, vector_dim)
        batch_size = x.shape[0]

        # Flatten context
        x_flat = x.view(batch_size, -1)

        # Pass through network
        out = self.network(x_flat)

        # Normalize to unit sphere
        if self.normalize:
            out = out / (torch.norm(out, dim=1, keepdim=True) + 1e-8)

        return out


class SequenceDataset(Dataset):
    """Dataset for sequence prediction"""

    def __init__(self, contexts, targets):
        self.contexts = torch.from_numpy(contexts).float()
        self.targets = torch.from_numpy(targets).float()

    def __len__(self):
        return len(self.contexts)

    def __getitem__(self, idx):
        return self.contexts[idx], self.targets[idx]


def cosine_similarity_loss(predictions, targets):
    """1 - cosine similarity (minimize)"""
    # Normalize
    pred_norm = predictions / (torch.norm(predictions, dim=1, keepdim=True) + 1e-8)
    target_norm = targets / (torch.norm(targets, dim=1, keepdim=True) + 1e-8)

    # Cosine similarity
    cos_sim = (pred_norm * target_norm).sum(dim=1)

    # Loss: 1 - cosine (minimize)
    loss = 1.0 - cos_sim
    return loss.mean()


def compute_hard_negative_loss(predictions, targets, k=3, margin=0.5):
    """
    Compute hard negative loss with margin.

    For each prediction:
      1. Find K hardest negatives (closest non-matching targets in batch)
      2. Push prediction away from negatives with margin loss

    Args:
        predictions: (batch_size, dim) predicted vectors
        targets: (batch_size, dim) target vectors
        k: number of hard negatives per sample
        margin: minimum desired distance from negatives

    Returns:
        hard_negative_loss: scalar loss
    """
    batch_size = predictions.shape[0]

    # Normalize
    pred_norm = predictions / (torch.norm(predictions, dim=1, keepdim=True) + 1e-8)
    target_norm = targets / (torch.norm(targets, dim=1, keepdim=True) + 1e-8)

    # Compute all pairwise cosine similarities: (batch_size, batch_size)
    similarity_matrix = torch.mm(pred_norm, target_norm.t())

    # Mask diagonal (correct targets)
    mask = torch.eye(batch_size, device=predictions.device).bool()
    similarity_matrix_masked = similarity_matrix.masked_fill(mask, -float('inf'))

    # Get top-K hardest negatives (highest similarity = hardest)
    topk_similarities, topk_indices = torch.topk(similarity_matrix_masked, k=min(k, batch_size - 1), dim=1)

    # Margin loss: max(0, similarity - margin)
    # We want similarity < (1 - margin), so loss = max(0, similarity - (1 - margin))
    negative_loss = torch.relu(topk_similarities - (1.0 - margin))

    return negative_loss.mean()


def train_epoch(model, dataloader, optimizer, device, args):
    """Train for one epoch with hard negatives"""
    model.train()
    total_loss = 0.0
    total_main_loss = 0.0
    total_neg_loss = 0.0
    total_cosine = 0.0

    for contexts, targets in dataloader:
        contexts = contexts.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        # Forward
        predictions = model(contexts)

        # Main loss (always)
        main_loss = cosine_similarity_loss(predictions, targets)

        # Hard negative loss
        neg_loss = compute_hard_negative_loss(
            predictions,
            targets,
            k=args.k_negatives,
            margin=args.margin
        )

        # Combined loss
        total_loss_batch = main_loss + args.lambda_neg * neg_loss

        # Backward
        total_loss_batch.backward()
        optimizer.step()

        # Track metrics
        total_loss += total_loss_batch.item()
        total_main_loss += main_loss.item()
        total_neg_loss += neg_loss.item()

        # Compute cosine for monitoring
        with torch.no_grad():
            pred_norm = predictions / (torch.norm(predictions, dim=1, keepdim=True) + 1e-8)
            target_norm = targets / (torch.norm(targets, dim=1, keepdim=True) + 1e-8)
            cosine = (pred_norm * target_norm).sum(dim=1).mean()
            total_cosine += cosine.item()

    avg_loss = total_loss / len(dataloader)
    avg_main_loss = total_main_loss / len(dataloader)
    avg_neg_loss = total_neg_loss / len(dataloader)
    avg_cosine = total_cosine / len(dataloader)

    return avg_loss, avg_main_loss, avg_neg_loss, avg_cosine


def validate(model, dataloader, device):
    """Validate model"""
    model.eval()
    total_loss = 0.0
    total_cosine = 0.0

    with torch.no_grad():
        for contexts, targets in dataloader:
            contexts = contexts.to(device)
            targets = targets.to(device)

            # Forward
            predictions = model(contexts)

            # Loss
            loss = cosine_similarity_loss(predictions, targets)

            # Cosine
            pred_norm = predictions / (torch.norm(predictions, dim=1, keepdim=True) + 1e-8)
            target_norm = targets / (torch.norm(targets, dim=1, keepdim=True) + 1e-8)
            cosine = (pred_norm * target_norm).sum(dim=1).mean()

            total_loss += loss.item()
            total_cosine += cosine.item()

    avg_loss = total_loss / len(dataloader)
    avg_cosine = total_cosine / len(dataloader)

    return avg_loss, avg_cosine


def main():
    parser = argparse.ArgumentParser(description="Train MLP with hard negatives")
    parser.add_argument("--data", type=str, required=True, help="Path to training_sequences_ctx5.npz")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--hidden-dim", type=int, default=1024, help="Hidden dimension")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu/mps/cuda)")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory")
    parser.add_argument("--lambda-neg", type=float, default=0.1, help="Hard negative loss weight")
    parser.add_argument("--k-negatives", type=int, default=3, help="Number of hard negatives per sample")
    parser.add_argument("--margin", type=float, default=0.5, help="Margin for negative loss")
    parser.add_argument("--pretrained", type=str, default=None, help="Path to pretrained model (optional)")

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print(f"Loading data from {args.data}")
    data = np.load(args.data)
    contexts = data['context_sequences']
    targets = data['target_vectors']

    # Train/val split (90/10)
    num_samples = len(contexts)
    train_size = int(0.9 * num_samples)

    train_contexts = contexts[:train_size]
    train_targets = targets[:train_size]
    val_contexts = contexts[train_size:]
    val_targets = targets[train_size:]

    print(f"Training samples: {len(train_contexts)}")
    print(f"Validation samples: {len(val_contexts)}")

    # Create datasets
    train_dataset = SequenceDataset(train_contexts, train_targets)
    val_dataset = SequenceDataset(val_contexts, val_targets)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # Model
    device = torch.device(args.device)
    model = SimpleMLP(
        context_size=5,
        vector_dim=768,
        hidden_dim=args.hidden_dim
    ).to(device)

    # Load pretrained if specified
    if args.pretrained:
        print(f"Loading pretrained model from {args.pretrained}")
        checkpoint = torch.load(args.pretrained, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("✓ Pretrained model loaded")

    print(f"\nModel architecture:")
    print(f"  Input: 5x768D (flattened to 3840D)")
    print(f"  Hidden: {args.hidden_dim}D")
    print(f"  Output: 768D (normalized)")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print()
    print(f"Hard negative settings:")
    print(f"  Lambda (weight): {args.lambda_neg}")
    print(f"  K negatives: {args.k_negatives}")
    print(f"  Margin: {args.margin}")
    print()

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Training loop
    best_val_loss = float('inf')
    history = []

    for epoch in range(args.epochs):
        train_loss, train_main, train_neg, train_cosine = train_epoch(
            model, train_loader, optimizer, device, args
        )
        val_loss, val_cosine = validate(model, val_loader, device)

        print(f"Epoch {epoch+1}/{args.epochs}: "
              f"Train Loss: {train_loss:.4f} (main: {train_main:.4f}, neg: {train_neg:.4f}) | "
              f"Train Cosine: {train_cosine:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Cosine: {val_cosine:.4f}")

        history.append({
            'epoch': epoch + 1,
            'train_loss': float(train_loss),
            'train_main_loss': float(train_main),
            'train_neg_loss': float(train_neg),
            'train_cosine': float(train_cosine),
            'val_loss': float(val_loss),
            'val_cosine': float(val_cosine)
        })

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_cosine': val_cosine,
                'args': vars(args)
            }, output_dir / 'best_model.pt')
            print(f"  → Saved best model (val_loss: {val_loss:.4f})")

    # Save final model
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
        'val_cosine': val_cosine,
        'args': vars(args)
    }, output_dir / 'final_model.pt')

    # Save training history
    with open(output_dir / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=2)

    print(f"\nTraining complete!")
    print(f"Best val loss: {best_val_loss:.4f}")
    print(f"Models saved to: {output_dir}")


if __name__ == "__main__":
    main()
