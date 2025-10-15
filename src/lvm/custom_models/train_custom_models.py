import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Dataset
import os

# Import custom models
from .rwkv import CustomRWKV
# Add imports for other models here

class SequenceDataset(Dataset):
    def __init__(self, npz_path, context_length=5):
        data = np.load(npz_path)
        self.contexts = data['context_vectors']
        self.targets = data['target_vectors']
        self.context_length = context_length
        
    def __len__(self):
        return len(self.contexts)
        
    def __getitem__(self, idx):
        ctx = self.contexts[idx]
        tgt = self.targets[idx]
        return torch.tensor(ctx, dtype=torch.float32), torch.tensor(tgt, dtype=torch.float32)

def train_model(model, dataloader, epochs=10, lr=1e-3, device='cpu'):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CosineSimilarity(dim=-1)
    model.train()
    
    for epoch in range(epochs):
        total_loss = 0
        for ctx, tgt in dataloader:
            ctx, tgt = ctx.to(device), tgt.to(device)
            optimizer.zero_grad()
            pred = model(ctx)
            loss = 1 - criterion(pred, tgt).mean()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}')

# Example usage
if __name__ == "__main__":
    dataset = SequenceDataset('/path/to/artifacts/lvm/training_sequences_ctx5.npz')
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    model = CustomRWKV()
    train_model(model, dataloader, epochs=20)
    torch.save(model.state_dict(), 'artifacts/lvm/custom_models/rwkv.pth')
