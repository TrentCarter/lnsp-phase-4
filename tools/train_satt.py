#!/usr/bin/env python3
"""
SATT: Sequence-Aware Two-Tower Training
Multi-task objective: L_seq (primary) + λ * L_sim (auxiliary)

Created: 2025-10-22
Purpose: Align training objective with Hit@K evaluation metric
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

# Force single-threaded BLAS
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_MAX_THREADS", "1")
os.environ.setdefault("FAISS_NUM_THREADS", "1")
os.environ.setdefault("PYTHONUNBUFFERED", "1")
os.environ.setdefault("PYTHONFAULTHANDLER", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import faulthandler
faulthandler.enable(all_threads=True)

try:
    import tools.stackdump
except ImportError:
    sys.stderr.write("[WARNING] tools.stackdump not available\n")

try:
    mp.set_start_method("spawn", force=True)
except RuntimeError:
    pass

sys.path.insert(0, 'src')
from retrieval.miner_sync import SyncFaissMiner
from utils.memprof import log_mem, rss_mb

# Watchdog
_last_step = {"t": time.time()}

def _watchdog():
    while True:
        time.sleep(30)
        if time.time() - _last_step["t"] > 120:
            sys.stderr.write("\n[WATCHDOG] No progress for 120s\n")
            sys.stderr.flush()
            faulthandler.dump_traceback(file=sys.stderr, all_threads=True)
            _last_step["t"] = time.time()

threading.Thread(target=_watchdog, daemon=True).start()

print("=" * 60)
print("SATT: SEQUENCE-AWARE TWO-TOWER TRAINING")
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
    'margin': 0.02,
    'lambda_sim': 0.3,  # Auxiliary loss weight
    'K': 500,           # Candidates for mining
    'K_hard': 10,       # Hard negatives per sample
    'nprobe': 12,       # Higher for better recall
    'ctx_len': 10,
    'warmup_steps': 2000,  # Curriculum: no hard negs initially
    'output_dir': os.environ.get('OUTDIR', 'runs/satt_default'),
}

print("Configuration:")
for k, v in CONFIG.items():
    print(f"  {k}: {v}")
print()

Path(CONFIG['output_dir']).mkdir(parents=True, exist_ok=True)

# Load data
print("Loading data...")
vectors_path = 'artifacts/wikipedia_500k_corrected_vectors.npz'
index_path = 'artifacts/wikipedia_500k_corrected_ivf_flat_ip.index'

data = np.load(vectors_path, allow_pickle=True)
bank_vectors = data['vectors']  # (771115, 768) float32
print(f"  Bank vectors: {bank_vectors.shape}")

index = faiss.read_index(index_path)
print(f"  FAISS index loaded: {index.ntotal} vectors")

assert index.ntotal == len(bank_vectors), \
    f"Index/bank mismatch: {index.ntotal} vs {len(bank_vectors)}"
print(f"  ✓ Index/bank alignment verified")
print()

baseline_rss = rss_mb()
print(f"Baseline memory: {baseline_rss:.1f} MB")
print()


# Sequence-aware dataset
class SequenceDataset(Dataset):
    """Dataset that provides (context, next_vector_id) pairs."""

    def __init__(self, vectors, n_samples=10000, ctx_len=10):
        self.vectors = vectors
        self.n_samples = n_samples
        self.ctx_len = ctx_len
        self.max_start = len(vectors) - ctx_len - 1  # Leave room for next

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        # Random start position
        start = np.random.randint(0, self.max_start)

        # Context window
        ctx = self.vectors[start:start+self.ctx_len]  # (10, 768)

        # Next vector ID (ground truth)
        next_id = start + self.ctx_len

        # Positional feature: where in sequence?
        pos_frac = self.ctx_len / (self.ctx_len + 1)  # t / |episode|

        return torch.from_numpy(ctx.copy()), torch.tensor(next_id, dtype=torch.long), torch.tensor(pos_frac, dtype=torch.float32)


# Query tower with positional encoding
class QueryTowerSATT(nn.Module):
    """Query tower with positional feature."""

    def __init__(self, hidden_size=768):
        super().__init__()
        self.gru = nn.GRU(768, hidden_size, 1, batch_first=True)
        self.ln = nn.LayerNorm(hidden_size)
        # Positional encoding projection
        self.pos_proj = nn.Linear(1, hidden_size)

    def forward(self, x, pos_frac=None):
        """
        Args:
            x: (B, T, 768) context
            pos_frac: (B,) positional fraction (optional)
        Returns:
            q: (B, 768) normalized query
        """
        out, _ = self.gru(x)
        pooled = out.mean(dim=1)  # (B, 768)

        # Add positional encoding if provided
        if pos_frac is not None:
            pos_embed = self.pos_proj(pos_frac.unsqueeze(-1))  # (B, 768)
            pooled = pooled + 0.1 * pos_embed  # Small contribution

        q = self.ln(pooled)
        return F.normalize(q, dim=-1)


def margin_infonce_loss(q, d_pos, d_neg, temperature=0.07, margin=0.02):
    """
    InfoNCE with margin for better discrimination.

    Args:
        q: (B, D) queries
        d_pos: (B, D) positive docs
        d_neg: (B, N, D) negative docs
        margin: push positives up, hard negatives down
    """
    # Positive logits with margin boost
    pos_logits = (q * d_pos).sum(-1) / temperature  # (B,)
    pos_logits = pos_logits - margin  # Harder target

    # Negative logits
    neg_logits = torch.einsum('bd,bnd->bn', q, d_neg) / temperature  # (B, N)

    # Find hardest negative per sample
    hard_neg_logits, _ = neg_logits.max(dim=1)  # (B,)
    hard_neg_logits = hard_neg_logits + margin  # Penalize hard negs more

    # Combine: positive vs all negatives (with hard neg margin)
    all_logits = torch.cat([
        pos_logits.unsqueeze(1),  # (B, 1)
        neg_logits  # (B, N)
    ], dim=1)  # (B, N+1)

    # Targets: positive is always index 0
    labels = torch.zeros(q.size(0), dtype=torch.long, device=q.device)

    return F.cross_entropy(all_logits, labels)


def sample_hard_negatives(I, D, gold_ids, bank, K_hard=10, same_doc_ratio=0.5):
    """
    Sample hard negatives excluding gold and preferring same-doc confounders.

    Args:
        I: (B, K) retrieved indices
        D: (B, K) retrieved distances
        gold_ids: (B,) ground truth next_vector_ids
        bank: (N, 768) vector bank
        K_hard: number of hard negatives per sample

    Returns:
        hard_neg_ids: (B, K_hard) indices
    """
    B, K = I.shape
    hard_neg_ids = []

    for b in range(B):
        gold_id = gold_ids[b].item()
        candidates = I[b].tolist()

        # Remove gold from candidates
        candidates = [c for c in candidates if c != gold_id]

        # Same-doc window: [gold-5, gold-1] ∪ [gold+2, gold+10]
        same_doc = []
        for offset in list(range(-5, 0)) + list(range(2, 11)):
            candidate_id = gold_id + offset
            if 0 <= candidate_id < len(bank) and candidate_id in candidates:
                same_doc.append(candidate_id)

        # Take some same-doc + some semantic neighbors
        n_same = min(len(same_doc), int(K_hard * same_doc_ratio))
        n_semantic = K_hard - n_same

        selected = []
        if same_doc:
            selected.extend(np.random.choice(same_doc, size=min(n_same, len(same_doc)), replace=False).tolist())

        # Fill remaining with top semantic neighbors
        semantic = [c for c in candidates if c not in selected][:n_semantic]
        selected.extend(semantic)

        # Pad if needed with random
        while len(selected) < K_hard:
            rand_id = np.random.randint(0, len(bank))
            if rand_id != gold_id and rand_id not in selected:
                selected.append(rand_id)

        hard_neg_ids.append(selected[:K_hard])

    return torch.tensor(hard_neg_ids, dtype=torch.long)


print("Building models...")
model_q = QueryTowerSATT().to(CONFIG['device'])
print(f"  Query tower params: {sum(p.numel() for p in model_q.parameters()):,}")

train_ds = SequenceDataset(bank_vectors, n_samples=10000, ctx_len=CONFIG['ctx_len'])
train_loader = DataLoader(
    train_ds,
    batch_size=CONFIG['batch_size'],
    shuffle=True,
    num_workers=0,
    pin_memory=False
)
print(f"  Dataset: {len(train_ds)} samples")
print()

print("Initializing FAISS miner...")
miner = SyncFaissMiner(index, nprobe=CONFIG['nprobe'])
print("  ✓ Sync miner ready")
print()

opt = torch.optim.Adam(model_q.parameters(), lr=CONFIG['lr'])

# Training loop
print("=" * 60)
print("TRAINING (DUAL LOSS: SEQUENCE + SIMILARITY)")
print("=" * 60)
print()

global_step = 0

for epoch in range(CONFIG['epochs']):
    model_q.train()
    t0 = time.time()
    total_loss = 0.0
    total_loss_seq = 0.0
    total_loss_sim = 0.0

    for step, (ctx_batch, next_ids, pos_fracs) in enumerate(train_loader):
        ctx_batch = ctx_batch.to(CONFIG['device'])  # (B, 10, 768)
        next_ids = next_ids.to(CONFIG['device'])  # (B,)
        pos_fracs = pos_fracs.to(CONFIG['device'])  # (B,)

        # Query encoding with positional feature
        q = model_q(ctx_batch, pos_fracs)  # (B, 768)
        q_np = q.detach().cpu().numpy().astype(np.float32)

        # Mine candidates
        I, D = miner.search(q_np, k=CONFIG['K'])

        # === PRIMARY LOSS: Sequence prediction ===
        # Positive: exact next vector
        d_pos_seq = bank_vectors[next_ids.cpu().numpy()]  # (B, 768)
        d_pos_seq = torch.from_numpy(d_pos_seq).to(CONFIG['device'])

        # Hard negatives: exclude gold, prefer same-doc
        use_hard_negs = global_step >= CONFIG['warmup_steps']

        if use_hard_negs:
            hard_neg_ids = sample_hard_negatives(
                torch.from_numpy(I),
                torch.from_numpy(D),
                next_ids.cpu(),
                bank_vectors,
                K_hard=CONFIG['K_hard']
            )
            d_neg = bank_vectors[hard_neg_ids.numpy()]  # (B, K_hard, 768)
        else:
            # Warmup: use random in-batch negatives
            rand_ids = np.random.randint(0, len(bank_vectors), size=(ctx_batch.size(0), CONFIG['K_hard']))
            d_neg = bank_vectors[rand_ids]  # (B, K_hard, 768)

        d_neg = torch.from_numpy(d_neg).to(CONFIG['device'])

        # Sequence loss with margin
        loss_seq = margin_infonce_loss(
            q, d_pos_seq, d_neg,
            temperature=CONFIG['temperature'],
            margin=CONFIG['margin']
        )

        # === AUXILIARY LOSS: Semantic similarity ===
        # Positive: best semantic neighbor (excluding gold)
        # Find top-1 from mined candidates that isn't the gold
        sim_pos_ids = []
        for b in range(ctx_batch.size(0)):
            gold_id = next_ids[b].item()
            candidates = I[b].tolist()
            # Take first non-gold candidate
            sim_pos = next((c for c in candidates if c != gold_id), candidates[0])
            sim_pos_ids.append(sim_pos)

        d_pos_sim = bank_vectors[sim_pos_ids]  # (B, 768)
        d_pos_sim = torch.from_numpy(d_pos_sim).to(CONFIG['device'])

        # Similarity loss (same negatives)
        loss_sim = margin_infonce_loss(
            q, d_pos_sim, d_neg,
            temperature=CONFIG['temperature'],
            margin=0.0  # No margin for auxiliary
        )

        # === TOTAL LOSS ===
        loss = loss_seq + CONFIG['lambda_sim'] * loss_sim

        # Backprop with gradient accumulation
        loss = loss / CONFIG['accum_steps']
        loss.backward()

        if (step + 1) % CONFIG['accum_steps'] == 0:
            opt.step()
            opt.zero_grad(set_to_none=True)
            _last_step["t"] = time.time()

        total_loss += loss.item() * CONFIG['accum_steps']
        total_loss_seq += loss_seq.item()
        total_loss_sim += loss_sim.item()

        global_step += 1

        # Logging
        if (step + 1) % 50 == 0:
            elapsed = time.time() - t0
            it_s = (step + 1) / elapsed
            timestamp = time.strftime('%H:%M:%S')
            hard_status = "HARD" if use_hard_negs else "WARM"
            print(f"[{timestamp}] Epoch {epoch+1}/{CONFIG['epochs']} | Step {step+1} | "
                  f"L_seq={loss_seq.item():.4f} L_sim={loss_sim.item():.4f} | "
                  f"{it_s:.2f} it/s | {hard_status}", flush=True)

            if (step + 1) % 200 == 0:
                log_mem(f"e{epoch+1}_s{step+1}")
                current_rss = rss_mb()
                if current_rss > baseline_rss + 500:
                    print(f"⚠️  Memory leak: {current_rss:.1f} MB", flush=True)

    avg_loss = total_loss / len(train_loader)
    avg_loss_seq = total_loss_seq / len(train_loader)
    avg_loss_sim = total_loss_sim / len(train_loader)
    elapsed = time.time() - t0

    print(f"\nEpoch {epoch+1} complete: "
          f"total={avg_loss:.4f}, seq={avg_loss_seq:.4f}, sim={avg_loss_sim:.4f}, "
          f"time={elapsed:.1f}s")
    print()

    # Save checkpoint
    ckpt_path = Path(CONFIG['output_dir']) / f'epoch_{epoch+1:03d}.pt'
    torch.save({
        'epoch': epoch + 1,
        'model_q': model_q.state_dict(),
        'optimizer': opt.state_dict(),
        'loss': avg_loss,
        'loss_seq': avg_loss_seq,
        'loss_sim': avg_loss_sim,
        'config': CONFIG,
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
print("✅ SATT training finished successfully!")
print()
print("Next steps:")
print("  1. Evaluate: python tools/evaluate_two_tower.py --checkpoint", ckpt_path)
print("  2. Test: python tools/test_trained_model.py --checkpoint", ckpt_path)
