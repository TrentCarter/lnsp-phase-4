#!/usr/bin/env python3
"""
Train two-tower retriever.

Query tower: GRU + pooling → 768D query vector
Doc tower: Identity (L2-norm only)
Loss: InfoNCE with in-batch negatives

Usage:
    python tools/train_twotower.py \
      --pairs artifacts/twotower/pairs_v1.npz \
      --query-tower gru_pool \
      --doc-tower identity \
      --bs 32 --accum 8 --epochs 20 \
      --lr 2e-5 --wd 0.01 --tau 0.07 \
      --out runs/twotower_mvp
"""

import argparse
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from tqdm import tqdm
import time


class GRUPoolQuery(nn.Module):
    """
    Query tower: Bidirectional GRU + mean pooling.
    
    Input: (batch, seq_len, 768)
    Output: (batch, 768) L2-normalized query vectors
    """
    def __init__(self, d_model=768, hidden_dim=512, num_layers=1, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.hidden_dim = hidden_dim
        
        self.gru = nn.GRU(
            input_size=d_model,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        # Project from bidirectional hidden (2*hidden) to d_model
        self.proj = nn.Linear(2 * hidden_dim, d_model)
        
    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, 768)
        Returns:
            q: (batch, 768) L2-normalized
        """
        # GRU forward
        out, _ = self.gru(x)  # (batch, seq_len, 2*hidden)
        
        # Mean pool over time
        pooled = out.mean(dim=1)  # (batch, 2*hidden)
        
        # Project and normalize
        q = self.proj(pooled)  # (batch, 768)
        q = F.normalize(q, p=2, dim=-1)  # L2 normalize
        
        return q


class IdentityDocTower(nn.Module):
    """
    Document tower: Just L2-normalize (identity transformation).
    
    Input: (batch, 768)
    Output: (batch, 768) L2-normalized
    """
    def forward(self, d):
        return F.normalize(d, p=2, dim=-1)


class TwoTowerDataset(Dataset):
    """Dataset for two-tower training pairs."""
    
    def __init__(self, X, Y):
        """
        Args:
            X: (N, seq_len, 768) - Context sequences
            Y: (N, 768) - Target vectors
        """
        self.X = torch.from_numpy(X).float()
        self.Y = torch.from_numpy(Y).float()
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


def info_nce_loss(query, doc_pos, tau=0.07):
    """
    InfoNCE loss with in-batch negatives.
    
    Args:
        query: (batch, 768) - Query vectors
        doc_pos: (batch, 768) - Positive document vectors
        tau: Temperature parameter
    
    Returns:
        loss: Scalar loss
    """
    # Compute similarities: query @ all_docs^T
    # Positive is at diagonal, rest are negatives
    logits = torch.matmul(query, doc_pos.T) / tau  # (batch, batch)
    
    # Labels: positive is at diagonal (index = own batch position)
    labels = torch.arange(query.size(0), device=query.device)
    
    # Cross-entropy loss
    loss = F.cross_entropy(logits, labels)
    
    return loss


def evaluate_recall(
    model_q,
    model_d,
    val_loader,
    bank_vectors,
    k_values=[10, 100, 500, 1000],
    device='cpu'
):
    """
    Evaluate Recall@K on validation set.
    
    Args:
        model_q: Query tower model
        model_d: Doc tower model
        val_loader: Validation DataLoader
        bank_vectors: (N_bank, 768) - Full vector bank for retrieval
        k_values: List of K values for Recall@K
        device: Device
    
    Returns:
        metrics: Dict of Recall@K values
    """
    model_q.eval()
    model_d.eval()
    
    # Encode bank once
    print("  Encoding bank...")
    bank_tensor = torch.from_numpy(bank_vectors).float().to(device)
    with torch.no_grad():
        bank_encoded = model_d(bank_tensor).cpu().numpy()  # (N_bank, 768)
    
    recalls = {k: [] for k in k_values}
    
    print("  Evaluating queries...")
    with torch.no_grad():
        for X, Y in tqdm(val_loader, desc="  Eval", leave=False):
            X = X.to(device)
            
            # Get query vectors
            queries = model_q(X).cpu().numpy()  # (batch, 768)
            targets = Y.numpy()  # (batch, 768)
            
            # For each query, find nearest neighbors in bank
            for query, target in zip(queries, targets):
                # Compute similarities to all bank vectors
                sims = np.dot(bank_encoded, query)  # (N_bank,)
                
                # Get top-K indices
                top_k_idx = np.argsort(-sims)[:max(k_values)]
                
                # Find target in bank (exact match)
                # Target should be the same as one of the bank vectors
                target_norm = target / (np.linalg.norm(target) + 1e-8)
                target_sims = np.dot(bank_encoded, target_norm)
                target_idx = np.argmax(target_sims)
                
                # Check if target is in top-K for each K
                for k in k_values:
                    recalls[k].append(1.0 if target_idx in top_k_idx[:k] else 0.0)
    
    # Aggregate
    metrics = {f'recall@{k}': np.mean(recalls[k]) * 100 for k in k_values}
    
    return metrics


def train_epoch(
    model_q,
    model_d,
    train_loader,
    optimizer,
    tau,
    accumulation_steps,
    device
):
    """Train for one epoch."""
    model_q.train()
    model_d.train()
    
    total_loss = 0.0
    optimizer.zero_grad()
    
    pbar = tqdm(train_loader, desc="  Training")
    
    for i, (X, Y) in enumerate(pbar):
        X = X.to(device)
        Y = Y.to(device)
        
        # Forward
        query = model_q(X)
        doc_pos = model_d(Y)
        
        # Loss
        loss = info_nce_loss(query, doc_pos, tau=tau)
        
        # Backward (with gradient accumulation)
        loss_scaled = loss / accumulation_steps
        loss_scaled.backward()
        
        # Accumulate
        if (i + 1) % accumulation_steps == 0:
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model_q.parameters(), 1.0)
            
            # Optimizer step
            optimizer.step()
            optimizer.zero_grad()
        
        total_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    # Final step if leftover
    if len(train_loader) % accumulation_steps != 0:
        optimizer.step()
        optimizer.zero_grad()
    
    return total_loss / len(train_loader)


def main():
    parser = argparse.ArgumentParser(description="Train two-tower retriever")
    
    # Data
    parser.add_argument('--pairs', required=True, help='Training pairs NPZ file')
    parser.add_argument('--bank', default='artifacts/wikipedia_500k_corrected_vectors.npz', help='Vector bank for eval')
    
    # Model
    parser.add_argument('--query-tower', default='gru_pool', choices=['gru_pool'], help='Query tower type')
    parser.add_argument('--doc-tower', default='identity', choices=['identity'], help='Doc tower type')
    parser.add_argument('--hidden-dim', type=int, default=512, help='GRU hidden dimension')
    parser.add_argument('--num-layers', type=int, default=1, help='GRU layers')
    
    # Training
    parser.add_argument('--bs', type=int, default=32, help='Batch size')
    parser.add_argument('--accum', type=int, default=8, help='Gradient accumulation steps')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=2e-5, help='Learning rate')
    parser.add_argument('--wd', type=float, default=0.01, help='Weight decay')
    parser.add_argument('--tau', type=float, default=0.07, help='InfoNCE temperature')
    
    # Infrastructure
    parser.add_argument('--device', default='mps', choices=['mps', 'cuda', 'cpu'], help='Device')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--out', required=True, help='Output directory')
    
    args = parser.parse_args()
    
    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Device
    if args.device == 'mps' and not torch.backends.mps.is_available():
        print("⚠️  MPS not available, using CPU")
        device = torch.device('cpu')
    else:
        device = torch.device(args.device)
    
    print("="*60)
    print("TWO-TOWER RETRIEVER TRAINING")
    print("="*60)
    print(f"Device: {device}")
    print(f"Batch size: {args.bs} × {args.accum} accum = {args.bs * args.accum} effective")
    print(f"Epochs: {args.epochs}")
    print(f"Learning rate: {args.lr}")
    print(f"Temperature: {args.tau}")
    print("="*60)
    
    # Load data
    print("\nLoading data...")
    data = np.load(args.pairs, allow_pickle=True)
    X_train = data['X_train']
    Y_train = data['Y_train']
    X_val = data['X_val']
    Y_val = data['Y_val']
    
    print(f"  Train: {len(X_train):,} pairs")
    print(f"  Val:   {len(X_val):,} pairs")
    print(f"  Context shape: {X_train.shape}")
    
    # Load bank for evaluation
    print(f"\nLoading bank from: {args.bank}")
    bank_data = np.load(args.bank, allow_pickle=True)
    bank_vectors = bank_data['vectors']
    print(f"  Bank size: {bank_vectors.shape}")
    
    # Create datasets
    train_dataset = TwoTowerDataset(X_train, Y_train)
    val_dataset = TwoTowerDataset(X_val, Y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=args.bs, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.bs, shuffle=False)
    
    # Create models
    print("\nBuilding models...")
    model_q = GRUPoolQuery(
        d_model=768,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers
    ).to(device)
    
    model_d = IdentityDocTower().to(device)
    
    print(f"  Query tower params: {sum(p.numel() for p in model_q.parameters()):,}")
    print(f"  Doc tower params: {sum(p.numel() for p in model_d.parameters()):,}")
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model_q.parameters(),  # Only query tower has trainable params
        lr=args.lr,
        weight_decay=args.wd
    )
    
    # Output directory
    out_dir = Path(args.out)
    out_dir.mkdir(exist_ok=True, parents=True)
    ckpt_dir = out_dir / 'checkpoints'
    ckpt_dir.mkdir(exist_ok=True)
    
    # Save config
    config = vars(args)
    config['device'] = str(device)
    with open(out_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    # Training loop
    print("\n" + "="*60)
    print("TRAINING")
    print("="*60)
    
    best_recall500 = 0.0
    history = []
    
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        
        # Train
        train_loss = train_epoch(
            model_q, model_d, train_loader,
            optimizer, args.tau, args.accum, device
        )
        
        print(f"  Train loss: {train_loss:.4f}")
        
        # Evaluate
        print("  Evaluating...")
        metrics = evaluate_recall(
            model_q, model_d, val_loader, bank_vectors,
            k_values=[10, 100, 500, 1000],
            device=device
        )
        
        print(f"  Recall@10:   {metrics['recall@10']:.2f}%")
        print(f"  Recall@100:  {metrics['recall@100']:.2f}%")
        print(f"  Recall@500:  {metrics['recall@500']:.2f}%")
        print(f"  Recall@1000: {metrics['recall@1000']:.2f}%")
        
        # Save history
        history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            **metrics
        })
        
        # Save checkpoint if best
        if metrics['recall@500'] > best_recall500:
            best_recall500 = metrics['recall@500']
            print(f"  ✓ New best Recall@500: {best_recall500:.2f}%")
            
            torch.save({
                'epoch': epoch,
                'model_q_state_dict': model_q.state_dict(),
                'model_d_state_dict': model_d.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'metrics': metrics,
                'config': config
            }, ckpt_dir / 'best.pt')
        
        # Save epoch checkpoint
        if epoch % 5 == 0:
            torch.save({
                'epoch': epoch,
                'model_q_state_dict': model_q.state_dict(),
                'model_d_state_dict': model_d.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'metrics': metrics,
                'config': config
            }, ckpt_dir / f'epoch_{epoch:03d}.pt')
    
    # Save training history
    with open(out_dir / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"Best Recall@500: {best_recall500:.2f}%")
    print(f"Checkpoints saved to: {ckpt_dir}")
    print("="*60)


if __name__ == '__main__':
    main()
