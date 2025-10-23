#!/usr/bin/env python3
"""
Stable Two-Tower Training Script (Synchronous FAISS)
Created: 2025-10-22
Purpose: First stable training run without multiprocessing races
"""

import sys
import os
import time
import multiprocessing as mp
import threading
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import faiss
from pathlib import Path

# Force single-threaded BLAS (critical for background stability)
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_MAX_THREADS", "1")
os.environ.setdefault("FAISS_NUM_THREADS", "1")
os.environ.setdefault("PYTHONUNBUFFERED", "1")
os.environ.setdefault("PYTHONFAULTHANDLER", "1")

# Enable faulthandler for debugging hangs
import faulthandler
faulthandler.enable(all_threads=True)

# Import stackdump for on-demand debugging (kill -USR1 <PID>)
try:
    import tools.stackdump
except ImportError:
    sys.stderr.write("[WARNING] tools.stackdump not available\n")

# Force spawn for any multiprocessing (though we use num_workers=0)
try:
    mp.set_start_method("spawn", force=True)
except RuntimeError:
    pass

# Add src to path
sys.path.insert(0, 'src')
from retrieval.miner_sync import SyncFaissMiner
from utils.memprof import log_mem, rss_mb

# Watchdog to detect hangs
_last_step = {"t": time.time()}

def _watchdog():
    """Dump tracebacks if no progress for 120s."""
    while True:
        time.sleep(30)
        if time.time() - _last_step["t"] > 120:
            sys.stderr.write("\n[WATCHDOG] No progress for 120s — dumping stacks\n")
            sys.stderr.flush()
            faulthandler.dump_traceback(file=sys.stderr, all_threads=True)
            _last_step["t"] = time.time()  # Prevent spam

threading.Thread(target=_watchdog, daemon=True).start()

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
    'output_dir': os.environ.get('OUTDIR', 'runs/stable_sync_default'),
}

print("Configuration:")
for k, v in CONFIG.items():
    print(f"  {k}: {v}")
print()

# Create output directory
Path(CONFIG['output_dir']).mkdir(parents=True, exist_ok=True)

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

# Critical: Verify alignment
assert index.ntotal == len(bank_vectors), \
    f"Index/bank mismatch: {index.ntotal} vs {len(bank_vectors)}"
print(f"  ✓ Index/bank alignment verified")
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
            # Update watchdog tick after successful step
            _last_step["t"] = time.time()

        total_loss += loss.item() * CONFIG['accum_steps']

        # Logging with timestamp for background monitoring
        if (step + 1) % 50 == 0:
            elapsed = time.time() - t0
            it_s = (step + 1) / elapsed
            timestamp = time.strftime('%H:%M:%S')
            print(f"[{timestamp}] Epoch {epoch+1}/{CONFIG['epochs']} | Step {step+1} | Loss {loss.item()*CONFIG['accum_steps']:.4f} | {it_s:.2f} it/s", flush=True)

            # Detailed logging every 200 steps
            if (step + 1) % 200 == 0:
                log_mem(f"e{epoch+1}_s{step+1}")

                # Memory leak check
                current_rss = rss_mb()
                if current_rss > baseline_rss + 500:
                    print(f"⚠️  Memory leak detected: {current_rss:.1f} MB (baseline: {baseline_rss:.1f} MB)", flush=True)

    avg_loss = total_loss / len(train_loader)
    elapsed = time.time() - t0
    print(f"\nEpoch {epoch+1} complete: avg_loss={avg_loss:.4f}, time={elapsed:.1f}s")
    print()

    # Save checkpoint
    ckpt_path = Path(CONFIG['output_dir']) / f'epoch_{epoch+1:03d}.pt'
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
