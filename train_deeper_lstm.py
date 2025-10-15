#!/usr/bin/env python3
"""
Deeper LSTM Trainer - Testing if more layers helps
==================================================

Same as baseline LSTM but with:
- 4 layers (vs 2)
- 1024 hidden dim (vs 512)
- ~20M params (vs 5M)

Goal: Test if deeper/wider LSTM can beat Transformer (78.60%)
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import json
from datetime import datetime
import argparse


class VectorSequenceDataset(Dataset):
    def __init__(self, npz_path: str):
        data = np.load(npz_path)
        self.contexts = torch.FloatTensor(data['context_sequences'])
        self.targets = torch.FloatTensor(data['target_vectors'])
        print(f"Loaded {len(self.contexts)} training pairs")

    def __len__(self):
        return len(self.contexts)

    def __getitem__(self, idx):
        return self.contexts[idx], self.targets[idx]


class DeeperLSTMPredictor(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=1024, num_layers=4, dropout=0.3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0
        )
        self.fc = nn.Linear(hidden_dim, input_dim)
        self.layer_norm = nn.LayerNorm(input_dim)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        output = self.fc(last_hidden)
        output = self.layer_norm(output)
        return output

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def cosine_similarity(pred, target):
    pred_norm = pred / (pred.norm(dim=1, keepdim=True) + 1e-8)
    target_norm = target / (target.norm(dim=1, keepdim=True) + 1e-8)
    return (pred_norm * target_norm).sum(dim=1).mean()


def train_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss, total_cosine = 0, 0

    for batch_idx, (contexts, targets) in enumerate(dataloader):
        contexts, targets = contexts.to(device), targets.to(device)
        optimizer.zero_grad()

        predictions = model(contexts)
        loss = nn.functional.mse_loss(predictions, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        with torch.no_grad():
            total_cosine += cosine_similarity(predictions, targets).item()

        if batch_idx % 100 == 0:
            print(f"  Batch {batch_idx}/{len(dataloader)} | Loss: {loss.item():.6f}")

    return total_loss / len(dataloader), total_cosine / len(dataloader)


def evaluate(model, dataloader, device):
    model.eval()
    total_loss, total_cosine = 0, 0

    with torch.no_grad():
        for contexts, targets in dataloader:
            contexts, targets = contexts.to(device), targets.to(device)
            predictions = model(contexts)
            loss = nn.functional.mse_loss(predictions, targets)
            total_loss += loss.item()
            total_cosine += cosine_similarity(predictions, targets).item()

    return total_loss / len(dataloader), total_cosine / len(dataloader)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='artifacts/lvm/training_sequences_ctx5.npz')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--device', default='mps' if torch.backends.mps.is_available() else 'cpu')
    parser.add_argument('--output-dir', default='artifacts/lvm/models/deeper_lstm')
    args = parser.parse_args()

    print("="*80)
    print("Deeper LSTM Trainer (4 layers, 1024 hidden)")
    print("="*80)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = VectorSequenceDataset(args.data)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    device = torch.device(args.device)
    model = DeeperLSTMPredictor(input_dim=768, hidden_dim=1024, num_layers=4).to(device)
    print(f"Parameters: {model.count_parameters():,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    best_val_loss = float('inf')
    history = []

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        train_loss, train_cosine = train_epoch(model, train_loader, optimizer, device)
        val_loss, val_cosine = evaluate(model, val_loader, device)
        scheduler.step(val_loss)

        print(f"  Train Loss: {train_loss:.6f} | Train Cosine: {train_cosine:.4f}")
        print(f"  Val Loss: {val_loss:.6f} | Val Cosine: {val_cosine:.4f}")

        history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_cosine': train_cosine,
            'val_loss': val_loss,
            'val_cosine': val_cosine
        })

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'val_loss': val_loss,
                'val_cosine': val_cosine,
                'args': vars(args)
            }, output_dir / 'best_model.pt')
            print(f"  ✓ Saved best model (val_loss: {val_loss:.6f})")

    with open(output_dir / 'training_history.json', 'w') as f:
        json.dump({
            'history': history,
            'best_val_loss': best_val_loss,
            'final_params': model.count_parameters(),
            'trained_at': datetime.now().isoformat()
        }, f, indent=2)

    print("="*80)
    print(f"Training Complete! Best val loss: {best_val_loss:.6f}")
    print("="*80)


if __name__ == '__main__':
    main()
