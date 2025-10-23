#!/usr/bin/env python3
"""
Simplified single-batch probe for our Wikipedia dataset format.
Tests: DataLoader → QueryTower → FAISS → Loss → Backward

Usage:
    python -u tools/probe_simple.py --batch 8 --K 200 --timeout 120
"""

import os
import sys
import time
import argparse
import threading

# Stability env (set before imports)
os.environ.setdefault("PYTHONUNBUFFERED", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_MAX_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import faulthandler
faulthandler.enable(all_threads=True)

# Optional SIGUSR1 handler
try:
    import signal
    faulthandler.register(signal.SIGUSR1, file=sys.stderr, all_threads=True)
except Exception:
    pass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import faiss

sys.path.insert(0, 'src')
from retrieval.query_tower import QueryTower
from retrieval.miner_sync import SyncFaissMiner


class SimpleDataset(Dataset):
    """Simple dataset for context windows."""
    def __init__(self, vectors, n_samples=1000, ctx_len=10):
        self.vectors = vectors
        self.n_samples = n_samples
        self.ctx_len = ctx_len

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        start = np.random.randint(0, len(self.vectors) - self.ctx_len)
        ctx = self.vectors[start:start+self.ctx_len]
        return torch.from_numpy(ctx.copy())


def watchdog(last_tick_ref, timeout_s=120, poll=5):
    """Dump stacks if no progress."""
    while True:
        time.sleep(poll)
        if time.time() - last_tick_ref[0] > timeout_s:
            sys.stderr.write("\n[WATCHDOG] No progress — dumping stacks\n")
            sys.stderr.flush()
            faulthandler.dump_traceback(file=sys.stderr, all_threads=True)
            last_tick_ref[0] = time.time()


def main():
    p = argparse.ArgumentParser(description="Probe single training step")
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--K", type=int, default=200)
    p.add_argument("--nprobe", type=int, default=8)
    p.add_argument("--timeout", type=int, default=120)
    args = p.parse_args()

    device = torch.device("cpu")

    print("=" * 60)
    print("PROBE: Single Batch Training Step")
    print("=" * 60)
    print()

    # Load data
    print("[1/6] Loading bank vectors...")
    t0 = time.time()
    data = np.load('artifacts/wikipedia_500k_corrected_vectors.npz', allow_pickle=True)
    bank = data['vectors']
    t1 = time.time()
    print(f"      ✓ Loaded {bank.shape} in {t1-t0:.2f}s")

    print("[2/6] Loading FAISS index...")
    t0 = time.time()
    index = faiss.read_index('artifacts/wikipedia_500k_corrected_ivf_flat_ip.index')
    t1 = time.time()
    print(f"      ✓ Loaded {index.ntotal} vectors in {t1-t0:.2f}s")

    # Verify alignment
    assert index.ntotal == bank.shape[0], \
        f"Index/bank mismatch: {index.ntotal} vs {bank.shape[0]}"
    print(f"      ✓ Alignment verified")
    print()

    print("[3/6] Creating dataset and loader...")
    t0 = time.time()
    ds = SimpleDataset(bank, n_samples=100)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=0, pin_memory=False)
    t1 = time.time()
    print(f"      ✓ Created in {t1-t0:.2f}s")
    print()

    print("[4/6] Initializing model and miner...")
    t0 = time.time()
    model = QueryTower().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)
    miner = SyncFaissMiner(index, nprobe=args.nprobe)
    t1 = time.time()
    print(f"      ✓ Initialized in {t1-t0:.2f}s")
    print()

    # Start watchdog
    last_tick = [time.time()]
    threading.Thread(target=watchdog, args=(last_tick, args.timeout), daemon=True).start()
    print(f"[5/6] Watchdog started (timeout={args.timeout}s)")
    print()

    print("[6/6] Running single batch...")
    print()

    # Timed probe
    t_start = time.time()

    # Load batch
    t0 = time.time()
    batch = next(iter(dl))
    ctx = batch.to(device)
    t_load = time.time()
    print(f"  load:    {t_load-t0:.3f}s")

    # Forward pass
    t0 = time.time()
    q = model(ctx)
    t_fwd = time.time()
    print(f"  forward: {t_fwd-t0:.3f}s")

    # Retrieve
    t0 = time.time()
    q_np = q.detach().cpu().numpy().astype(np.float32)
    I, D = miner.search(q_np, k=args.K)
    t_retr = time.time()
    print(f"  retrieve: {t_retr-t0:.3f}s")

    # Loss
    t0 = time.time()
    docs = bank[I]
    docs_t = torch.from_numpy(docs).to(device)
    sims = torch.einsum('bd,bkd->bk', q, docs_t)
    logits = sims / 0.07
    sorted_logits, _ = torch.sort(logits, dim=1, descending=True)
    labels = torch.zeros(logits.size(0), dtype=torch.long, device=device)
    loss = F.cross_entropy(sorted_logits, labels)
    t_loss = time.time()
    print(f"  loss:    {t_loss-t0:.3f}s")

    # Backward
    t0 = time.time()
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    t_bwd = time.time()
    print(f"  backward: {t_bwd-t0:.3f}s")

    last_tick[0] = time.time()

    t_total = time.time() - t_start

    print()
    print("=" * 60)
    print("PROBE RESULT")
    print("=" * 60)
    print(f"Total time: {t_total:.3f}s")
    print(f"Loss: {loss.item():.4f}")
    print(f"Batch: {args.batch}, K: {args.K}, nprobe: {args.nprobe}")
    print()
    print("✅ Single batch completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[probe] Interrupted", file=sys.stderr)
        sys.exit(130)
    except Exception as e:
        print(f"\n[probe] ERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
