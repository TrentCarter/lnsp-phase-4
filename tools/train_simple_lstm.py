#!/usr/bin/env python3
"""
Simple LSTM Trainer for Debugging
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from app.lvm.models import create_model

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

def cosine_similarity(pred, target):
    pred_norm = pred / (pred.norm(dim=1, keepdim=True) + 1e-8)
    target_norm = target / (target.norm(dim=1, keepdim=True) + 1e-8)
    return (pred_norm * target_norm).sum(dim=1).mean()

def main():
    parser = argparse.ArgumentParser(description='Simple LSTM Trainer')
    parser.add_argument('--data', default='artifacts/lvm/training_sequences_ctx5.npz')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--device', default='mps' if torch.backends.mps.is_available() else 'cpu')
    args = parser.parse_args()

    print("Loading dataset...")
    dataset = VectorSequenceDataset(args.data)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    print("Creating model...")
    device = torch.device(args.device)
    model_config = {'input_dim': 768, 'd_model': 256, 'num_layers': 1, 'dropout': 0.0}
    model = create_model('lstm', **model_config).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    print("Starting training...")
    for epoch in range(args.epochs):
        total_loss = 0.0
        total_cosine = 0.0
        for contexts, targets in dataloader:
            contexts = contexts.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            
            # The model returns both raw and normalized outputs, we use the raw for MSE
            pred_raw, _ = model(contexts, return_raw=True)
            
            loss = loss_fn(pred_raw, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            with torch.no_grad():
                total_cosine += cosine_similarity(pred_raw, targets).item()

        avg_loss = total_loss / len(dataloader)
        avg_cosine = total_cosine / len(dataloader)
        print(f"Epoch {epoch+1}/{args.epochs} | Loss: {avg_loss:.6f} | Cosine: {avg_cosine:.4f}")

if __name__ == '__main__':
    main()
