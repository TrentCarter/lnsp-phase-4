#!/usr/bin/env python3
"""
Minimal reproducer to test if crash is in:
1. Data loading
2. Model forward pass
3. Loss computation
4. Backpropagation
5. Optimizer step
"""

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

print("=== CRASH CAUSE ISOLATOR ===")
print()

# Load data
print("1. Loading data...")
pairs = np.load('artifacts/twotower/pairs_v4_synth.npz')
bank_data = np.load('artifacts/wikipedia_500k_corrected_vectors.npz')

X_train = torch.from_numpy(pairs['X_train'])  # (35901, 100, 768)
Y_train = torch.from_numpy(pairs['Y_train'])  # (35901, 768)
bank_vectors = torch.from_numpy(bank_data['vectors'])  # (771115, 768)

print(f"   X_train: {X_train.shape}")
print(f"   Y_train: {Y_train.shape}")
print(f"   Bank: {bank_vectors.shape}")
print("   ✓ Data loaded")
print()

# Create simple model
print("2. Creating GRU model...")
class SimpleGRU(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = torch.nn.GRU(768, 512, 1, batch_first=True, bidirectional=True)
        self.proj = torch.nn.Linear(1024, 768)

    def forward(self, x):
        out, _ = self.gru(x)
        pooled = out.mean(dim=1)
        proj = self.proj(pooled)
        return F.normalize(proj, dim=-1)

model = SimpleGRU()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
print("   ✓ Model created")
print()

# Test forward pass
print("3. Testing forward pass...")
batch_size = 32
X_batch = X_train[:batch_size]  # (32, 100, 768)
Y_batch = Y_train[:batch_size]  # (32, 768)

try:
    q = model(X_batch)
    print(f"   Query output: {q.shape}")
    print("   ✓ Forward pass works")
except Exception as e:
    print(f"   ✗ CRASH in forward pass: {e}")
    exit(1)
print()

# Test loss computation
print("4. Testing loss computation...")
try:
    d_pos = F.normalize(Y_batch, dim=-1)
    logits = torch.matmul(q, d_pos.T) / 0.05  # (32, 32)
    labels = torch.arange(batch_size)
    loss = F.cross_entropy(logits, labels)
    print(f"   Loss: {loss.item():.4f}")
    print("   ✓ Loss computation works")
except Exception as e:
    print(f"   ✗ CRASH in loss: {e}")
    exit(1)
print()

# Test backward pass
print("5. Testing backward pass...")
try:
    loss.backward()
    print("   ✓ Backward pass works")
except Exception as e:
    print(f"   ✗ CRASH in backward: {e}")
    exit(1)
print()

# Test optimizer step
print("6. Testing optimizer step...")
try:
    optimizer.step()
    optimizer.zero_grad()
    print("   ✓ Optimizer step works")
except Exception as e:
    print(f"   ✗ CRASH in optimizer: {e}")
    exit(1)
print()

# Test multiple iterations
print("7. Testing 100 iterations...")
try:
    for i in tqdm(range(100), desc="Training"):
        # Get batch
        start_idx = (i * batch_size) % len(X_train)
        end_idx = start_idx + batch_size
        if end_idx > len(X_train):
            break

        X_batch = X_train[start_idx:end_idx]
        Y_batch = Y_train[start_idx:end_idx]

        # Forward
        q = model(X_batch)
        d_pos = F.normalize(Y_batch, dim=-1)

        # Loss
        logits = torch.matmul(q, d_pos.T) / 0.05
        labels = torch.arange(len(X_batch))
        loss = F.cross_entropy(logits, labels)

        # Backward
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if i % 20 == 0:
            print(f"   Iteration {i}: loss={loss.item():.4f}")

    print("   ✓ 100 iterations completed successfully!")
except Exception as e:
    print(f"   ✗ CRASH at iteration {i}: {e}")
    exit(1)
print()

print("=== ALL TESTS PASSED ===")
print()
print("Conclusion: Basic training loop works fine.")
print("Crash likely happens when:")
print("  - Using FAISS for hard negative mining")
print("  - Gradient accumulation over many steps")
print("  - Long-running training (memory fragmentation?)")
print()
print("Next: Run test_segfault.sh to narrow down the cause")
