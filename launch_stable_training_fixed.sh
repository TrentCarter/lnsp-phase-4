#!/usr/bin/env bash
set -euo pipefail

# 🔴 CRITICAL: macOS OpenMP fix (prevents "Abort trap: 6")
export KMP_DUPLICATE_LIB_OK=TRUE

# Stable Two-Tower Training Launch (CPU + Sync FAISS)
# Created: 2025-10-22
# Purpose: First stable training run without multiprocessing

echo "════════════════════════════════════════════════════════════"
echo "  🚀 LAUNCHING STABLE TWO-TOWER TRAINING (OpenMP Fix Applied)"
echo "════════════════════════════════════════════════════════════"
echo ""
echo "Configuration:"
echo "  Device: CPU (stable mode)"
echo "  Miner: Synchronous FAISS (no multiprocessing)"
echo "  Data: Wikipedia 771k vectors"
echo "  Epochs: 5 (short run for validation)"
echo "  Batch: 8 × 2 accum = 16 effective"
echo "  OpenMP: KMP_DUPLICATE_LIB_OK=TRUE ✓"
echo "  Output: runs/stable_sync_$(date +%Y%m%d_%H%M%S)"
echo ""
echo "Estimated time: ~2-3 hours (5 epochs)"
echo ""
echo "════════════════════════════════════════════════════════════"

# Set output directory with timestamp
OUTDIR="runs/stable_sync_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTDIR"
LOGFILE="$OUTDIR/training.log"

# Python training script
KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH=. ./.venv/bin/python3 << 'PYEOF' > "$LOGFILE" 2>&1 &
import sys
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import faiss
from pathlib import Path

# Add src to path
sys.path.insert(0, 'src')
from retrieval.miner_sync import SyncFaissMiner
from utils.memprof import log_mem, rss_mb

print("=" * 60)
print("STABLE TWO-TOWER TRAINING")
print("=" * 60)
print()

# Configuration
CONFIG = {
    'device': 'cpu',
    'batch_size': 8,
    'accum_steps': 2,
    'epochs': 5,
    'lr': 3e-4,
    'temperature': 0.07,
    'bank_size': 10000,
    'K': 500,
    'nprobe': 8,
}

print("Configuration:")
for k, v in CONFIG.items():
    print(f"  {k}: {v}")
print()

# Load data
print("Loading data...")
vectors_path = 'artifacts/wikipedia_500k_corrected_vectors.npz'
index_path = 'artifacts/wikipedia_500k_corrected_ivf_flat_ip.index'

data = np.load(vectors_path, allow_pickle=True)
bank_vectors = data['vectors']  # (771115, 768) float32
print(f"  Bank vectors: {bank_vectors.shape}")

# Load FAISS index
index = faiss.read_index(index_path)
print(f"  FAISS index loaded: {index.ntotal} vectors")
print()

# Baseline memory
baseline_rss = rss_mb()
print(f"Baseline memory: {baseline_rss:.1f} MB")
print()

# Models
class QueryTower(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(768, 768, 1, batch_first=True)
        self.ln = nn.LayerNorm(768)

    def forward(self, x):
        out, _ = self.gru(x)
        pooled = out.mean(dim=1)
        q = self.ln(pooled)
        return F.normalize(q, dim=-1)

print("Building models...")
model_q = QueryTower().to(CONFIG['device'])
print(f"  Query tower params: {sum(p.numel() for p in model_q.parameters()):,}")

# Simple dataset: random context windows
class SimpleDataset(Dataset):
    def __init__(self, vectors, n_samples=10000, ctx_len=10):
        self.vectors = vectors
        self.n_samples = n_samples
        self.ctx_len = ctx_len

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        # Random start position
        start = np.random.randint(0, len(self.vectors) - self.ctx_len)
        ctx = self.vectors[start:start+self.ctx_len]  # (10, 768)
        return torch.from_numpy(ctx.copy())

train_ds = SimpleDataset(bank_vectors, n_samples=10000)
train_loader = DataLoader(
    train_ds,
    batch_size=CONFIG['batch_size'],
    shuffle=True,
    num_workers=0,  # Stable: no multiprocessing
    pin_memory=False  # CPU-friendly
)
print(f"  Dataset: {len(train_ds)} samples")
print()

# Miner
print("Initializing synchronous FAISS miner...")
miner = SyncFaissMiner(index, nprobe=CONFIG['nprobe'])
print("  ✓ Sync miner ready")
print()

# Optimizer
opt = torch.optim.Adam(model_q.parameters(), lr=CONFIG['lr'])

# Training loop
print("=" * 60)
print("TRAINING")
print("=" * 60)
print()

for epoch in range(CONFIG['epochs']):
    model_q.train()
    t0 = time.time()
    total_loss = 0.0

    for step, ctx_batch in enumerate(train_loader):
        ctx_batch = ctx_batch.to(CONFIG['device'])  # (B, 10, 768)

        # Query encoding
        q = model_q(ctx_batch)  # (B, 768)
        q_np = q.detach().cpu().numpy().astype(np.float32)

        # Mine candidates (synchronous)
        I, D = miner.search(q_np, k=CONFIG['K'])

        # Gather doc vectors (identity tower)
        docs = bank_vectors[I]  # (B, K, 768)
        docs_t = torch.from_numpy(docs).to(CONFIG['device'])

        # InfoNCE loss (top-1 as pseudo-positive)
        sims = torch.einsum('bd,bkd->bk', q, docs_t)  # (B, K)
        logits = sims / CONFIG['temperature']

        # Top-1 per row is positive
        sorted_logits, _ = torch.sort(logits, dim=1, descending=True)
        labels = torch.zeros(logits.size(0), dtype=torch.long, device=CONFIG['device'])

        loss = F.cross_entropy(sorted_logits, labels)

        # Backprop with gradient accumulation
        loss = loss / CONFIG['accum_steps']
        loss.backward()

        if (step + 1) % CONFIG['accum_steps'] == 0:
            opt.step()
            opt.zero_grad(set_to_none=True)

        total_loss += loss.item() * CONFIG['accum_steps']

        # Logging
        if (step + 1) % 200 == 0:
            elapsed = time.time() - t0
            it_s = (step + 1) / elapsed
            print(f"Epoch {epoch+1}/{CONFIG['epochs']} | Step {step+1} | Loss {loss.item()*CONFIG['accum_steps']:.4f} | {it_s:.2f} it/s")
            log_mem(f"e{epoch+1}_s{step+1}")

            # Memory leak check
            current_rss = rss_mb()
            if current_rss > baseline_rss + 500:
                print(f"⚠️  Memory leak detected: {current_rss:.1f} MB (baseline: {baseline_rss:.1f} MB)")

    avg_loss = total_loss / len(train_loader)
    elapsed = time.time() - t0
    print(f"\nEpoch {epoch+1} complete: avg_loss={avg_loss:.4f}, time={elapsed:.1f}s")
    print()

    # Save checkpoint
    ckpt_path = Path(os.environ.get('OUTDIR', 'runs/stable_sync')) / f'epoch_{epoch+1:03d}.pt'
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'epoch': epoch + 1,
        'model_q': model_q.state_dict(),
        'optimizer': opt.state_dict(),
        'loss': avg_loss,
    }, ckpt_path)
    print(f"💾 Checkpoint saved: {ckpt_path}")
    print()

print("=" * 60)
print("TRAINING COMPLETE")
print("=" * 60)
print()
print(f"Total epochs: {CONFIG['epochs']}")
print(f"Final checkpoint: {ckpt_path}")
print(f"Memory usage: {rss_mb():.1f} MB (baseline: {baseline_rss:.1f} MB)")
print()
print("✅ Training finished successfully!")
PYEOF

# Get background PID
TRAINING_PID=$!
echo "Training launched in background (PID: $TRAINING_PID)"
echo ""
echo "Monitor with:"
echo "  tail -f $LOGFILE"
echo "  ps aux | grep $TRAINING_PID"
echo ""
echo "Kill with:"
echo "  kill $TRAINING_PID"
echo ""
echo "════════════════════════════════════════════════════════════"
echo "  ✅ Training started! Proceeding with documentation..."
echo "════════════════════════════════════════════════════════════"

# Save PID for later monitoring
echo $TRAINING_PID > /tmp/lnsp_training.pid
echo "$OUTDIR" > /tmp/lnsp_training_outdir.txt
echo "$LOGFILE" > /tmp/lnsp_training_log.txt
