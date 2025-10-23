#!/usr/bin/env python3
"""
Two-Tower v4 Training: Curriculum-Based Hard Negative Mining

Key improvements over Phase 2:
- Curriculum schedule: warm-start → gentle hards → full hards
- Cosine-range mining (avoid margin collapse)
- Same-article tricky negatives
- Hard negative filtering (drop >0.98 duplicates)
- Higher effective batch size (512 vs 256)
- Cosine LR decay
"""

import argparse
import json
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
import time
import sys
sys.path.insert(0, str(Path(__file__).parent))
from async_miner import AsyncMiner


# ============================================================
# Model Architecture
# ============================================================

class GRUPoolQuery(torch.nn.Module):
    """Query tower with GRU + mean pooling"""
    def __init__(self, d_model=768, hidden_dim=512, num_layers=1):
        super().__init__()
        self.gru = torch.nn.GRU(d_model, hidden_dim, num_layers,
                                batch_first=True, bidirectional=True)
        self.proj = torch.nn.Linear(hidden_dim * 2, d_model)

    def forward(self, x):
        out, _ = self.gru(x)
        pooled = out.mean(dim=1)
        proj = self.proj(pooled)
        return F.normalize(proj, dim=-1)


class IdentityDocTower(torch.nn.Module):
    """Document tower (just L2 normalization)"""
    def forward(self, x):
        return F.normalize(x, dim=-1)


# ============================================================
# Memory Bank
# ============================================================

class MemoryBank:
    """FIFO queue of recent document vectors"""
    def __init__(self, max_size=50000, dim=768, device='mps'):
        self.max_size = max_size
        self.dim = dim
        self.device = device
        self.vectors = torch.zeros(0, dim, device=device)

    def add(self, vecs):
        """Add vectors to bank (FIFO)"""
        vecs = vecs.detach()
        self.vectors = torch.cat([self.vectors, vecs], dim=0)
        if len(self.vectors) > self.max_size:
            self.vectors = self.vectors[-self.max_size:]

    def get_all(self):
        """Get all vectors in bank"""
        return self.vectors


# ============================================================
# Hard Negative Mining with Curriculum
# ============================================================

def mine_hard_negatives_with_filter(model_q, model_d, train_loader, bank_vectors,
                                      num_hard_negs=16, cos_min=0.84, cos_max=0.96,
                                      filter_threshold=0.98, device='mps'):
    """
    Mine hard negatives within cosine range [cos_min, cos_max]
    Filter out near-duplicates (cos > filter_threshold to both query and positive)
    """
    import faiss

    model_q.eval()
    model_d.eval()

    print(f"  Mining hard negatives (cos range: {cos_min:.2f}-{cos_max:.2f})...")

    # Clear MPS cache before large CPU transfer (prevents memory fragmentation)
    if device == 'mps':
        torch.mps.empty_cache()
        torch.mps.synchronize()

    # Build FAISS index
    bank_np = bank_vectors.cpu().numpy().astype(np.float32)
    bank_np = bank_np / (np.linalg.norm(bank_np, axis=1, keepdims=True) + 1e-9)
    index = faiss.IndexFlatIP(768)
    index.add(bank_np)

    # Encode all queries and positives
    all_q = []
    all_d_pos = []

    with torch.no_grad():
        for X_batch, Y_batch in tqdm(train_loader, desc="    Encoding", leave=False):
            X_batch = X_batch.to(device)
            Y_batch = Y_batch.to(device)

            q = model_q(X_batch)
            d_pos = model_d(Y_batch)

            all_q.append(q)
            all_d_pos.append(d_pos)

    all_q = torch.cat(all_q, dim=0).cpu().numpy().astype(np.float32)
    all_d_pos = torch.cat(all_d_pos, dim=0).cpu().numpy().astype(np.float32)

    # Mine top-K candidates (oversample)
    K_candidates = num_hard_negs * 4  # Oversample for filtering
    D, I = index.search(all_q, K_candidates)

    # Filter and select
    hard_neg_indices = []

    for i in tqdm(range(len(all_q)), desc="    Filtering", leave=False):
        q_vec = all_q[i]
        d_pos_vec = all_d_pos[i]

        candidates = []

        for j in range(K_candidates):
            neg_idx = I[i, j]
            neg_vec = bank_np[neg_idx]
            cos_q_neg = float(q_vec @ neg_vec)

            # Check range
            if not (cos_min <= cos_q_neg <= cos_max):
                continue

            # Filter duplicates
            cos_pos_neg = float(d_pos_vec @ neg_vec)
            if cos_q_neg > filter_threshold and cos_pos_neg > filter_threshold:
                continue  # Near-duplicate, skip

            candidates.append(neg_idx)

            if len(candidates) >= num_hard_negs:
                break

        # Pad if needed
        while len(candidates) < num_hard_negs:
            candidates.append(I[i, len(candidates)])  # Fallback to top-K

        hard_neg_indices.append(candidates[:num_hard_negs])

    hard_neg_indices = torch.tensor(hard_neg_indices, dtype=torch.long)

    model_q.train()
    model_d.train()

    return hard_neg_indices  # (N, num_hard_negs)


# ============================================================
# Loss Function
# ============================================================

def info_nce_loss_with_extras(query, doc_pos, tau=0.05, margin=0.03,
                                memory_bank=None, hard_negs=None, hard_neg_vecs=None):
    """
    InfoNCE loss with optional memory bank and hard negatives

    Args:
        query: (B, D)
        doc_pos: (B, D)
        tau: temperature
        margin: margin penalty
        memory_bank: MemoryBank object
        hard_negs: (B, K) indices into bank (if provided)
        hard_neg_vecs: (B, K, D) pre-fetched hard negative vectors
    """
    B = query.size(0)

    # Positive similarity
    pos_sim = (query * doc_pos).sum(dim=-1, keepdim=True) / tau  # (B, 1)

    # In-batch negatives
    neg_sim = (query @ doc_pos.T) / tau  # (B, B)
    mask = torch.eye(B, device=query.device, dtype=torch.bool)
    neg_sim = neg_sim.masked_fill(mask, -1e9)

    # Concatenate similarities
    logits = torch.cat([pos_sim, neg_sim], dim=1)  # (B, 1+B)

    # Add memory bank negatives
    if memory_bank is not None:
        bank_vecs = memory_bank.get_all()  # (M, D)
        if len(bank_vecs) > 0:
            bank_sim = (query @ bank_vecs.T) / tau  # (B, M)
            logits = torch.cat([logits, bank_sim], dim=1)

    # Add hard negatives
    if hard_neg_vecs is not None:
        # hard_neg_vecs: (B, K, D)
        hard_sim = (query.unsqueeze(1) @ hard_neg_vecs.transpose(1, 2)).squeeze(1) / tau  # (B, K)
        logits = torch.cat([logits, hard_sim], dim=1)

    # Targets (positive is always index 0)
    targets = torch.zeros(B, dtype=torch.long, device=query.device)

    # Cross-entropy
    loss = F.cross_entropy(logits, targets)

    # Optional margin penalty
    if margin > 0:
        pos_score = logits[:, 0]  # (B,)
        max_neg_score = logits[:, 1:].max(dim=1)[0]  # (B,)
        margin_loss = F.relu(margin - (pos_score - max_neg_score)).mean()
        loss = loss + margin_loss

    return loss


# ============================================================
# Training Loop
# ============================================================

def train_one_epoch(model_q, model_d, train_loader, optimizer, epoch, args,
                     memory_bank=None, hard_neg_indices=None, bank_vectors=None,
                     async_miner=None, mining_spec=None, device='mps'):
    """Train for one epoch with optional async mining"""
    from collections import deque

    model_q.train()
    model_d.train()

    total_loss = 0
    num_batches = 0

    # 🚨 STABILITY FIX: set_to_none=True reduces memory fragmentation
    optimizer.zero_grad(set_to_none=True)

    # FIFO alignment queue for async mining (pairs batches with mined results)
    batch_fifo = deque(maxlen=8)  # holds (q_cpu, d_pos_cpu) awaiting mined negs

    # Queue health monitoring
    queue_log_interval = 200

    for batch_idx, (X_batch, Y_batch) in enumerate(tqdm(train_loader, desc=f"  Training", leave=False)):
        try:
            X_batch = X_batch.to(device)
            Y_batch = Y_batch.to(device)

            # Encode
            q = model_q(X_batch)
            d_pos = model_d(Y_batch)

            # Get hard negatives (async or pre-mined)
            hard_neg_vecs = None

            # Path 1: Async mining with FIFO alignment (overlap FAISS with training)
            if async_miner is not None and mining_spec is not None:
                # PRODUCER: Stash current batch (CPU copies) + submit for async mining
                batch_fifo.append((q.detach().cpu(), d_pos.detach().cpu()))
                async_miner.submit(batch_idx, q)

                # CONSUMER: Try to retrieve mined negatives for the OLDEST batch in FIFO
                mined = async_miner.try_get()
                if mined is not None and len(batch_fifo) > 0:
                    prev_step, neg_indices = mined  # (B, K) indices into bank
                    q_wait, d_pos_wait = batch_fifo.popleft()  # matching oldest batch

                    try:
                        # neg_indices is (B, K) where K = args.mine_k (candidate pool)
                        # Filter to mining_spec['num'] negatives + apply cos range
                        B = len(neg_indices)

                        # Get candidate vectors (cached, avoid repeated conversion)
                        if not hasattr(bank_vectors, '_cached_np'):
                            bank_vectors._cached_np = bank_vectors.cpu().numpy().astype(np.float32)
                        bank_np = bank_vectors._cached_np

                        # Convert query to numpy for fast similarity computation
                        q_np = q_wait.numpy().astype(np.float32)

                        # For each query, filter candidates by cosine range
                        filtered_indices = []
                        for i in range(B):
                            cand_idxs = neg_indices[i]
                            cand_vecs = bank_np[cand_idxs]
                            query_vec = q_np[i]

                            # Compute similarities (already normalized vectors)
                            sims = (query_vec @ cand_vecs.T).tolist()

                            # Filter by cosine range
                            valid_pairs = [(idx, sim) for idx, sim in zip(cand_idxs, sims)
                                          if mining_spec['cos_min'] <= sim <= mining_spec['cos_max']]

                            # Take top mining_spec['num']
                            valid_pairs = sorted(valid_pairs, key=lambda x: x[1], reverse=True)
                            valid_indices = [int(idx) for idx, _ in valid_pairs[:mining_spec['num']]]

                            # Pad if needed
                            while len(valid_indices) < mining_spec['num']:
                                valid_indices.append(valid_indices[0] if valid_indices else 0)

                            filtered_indices.append(valid_indices[:mining_spec['num']])

                        # Convert to tensor and gather
                        filtered_indices_tensor = torch.tensor(filtered_indices, dtype=torch.long)  # (B, num)
                        mined_hard_vecs = bank_vectors[filtered_indices_tensor]  # (B, num, D)

                        # Mix FAISS + memory bank according to hardneg_frac
                        num_from_faiss = int(mining_spec['num'] * args.hardneg_frac)
                        num_from_memory = mining_spec['num'] - num_from_faiss

                        # Take first num_from_faiss from FAISS mining
                        hard_neg_vecs = mined_hard_vecs[:, :num_from_faiss, :].to(device)

                        # Add memory bank negatives if needed
                        if num_from_memory > 0 and memory_bank is not None:
                            bank_vecs = memory_bank.get_all()
                            if len(bank_vecs) >= num_from_memory:
                                # Random sample from memory bank
                                mem_indices = torch.randperm(len(bank_vecs))[:num_from_memory * B]
                                mem_neg_vecs = bank_vecs[mem_indices].view(B, num_from_memory, -1)
                                hard_neg_vecs = torch.cat([hard_neg_vecs, mem_neg_vecs], dim=1)

                        # Use the waited batch (not current batch) for loss
                        q = q_wait.to(device)
                        d_pos = d_pos_wait.to(device)

                    except Exception as e:
                        # Fall back to current batch + memory bank if async mining fails
                        batch_fifo.clear()  # Reset FIFO on error
                        pass

            # Path 2: Pre-mined negatives (synchronous, original path)
            elif hard_neg_indices is not None and bank_vectors is not None:
                try:
                    # FIX: Handle partial batches correctly (last batch may have fewer items)
                    start_idx = batch_idx * args.bs
                    end_idx = min((batch_idx + 1) * args.bs, len(hard_neg_indices))
                    actual_batch_size = end_idx - start_idx

                    # Only use the actual batch size we received
                    global_indices = range(start_idx, start_idx + len(q))

                    batch_hard_indices = hard_neg_indices[global_indices]  # (B, K)
                    hard_neg_vecs = bank_vectors[batch_hard_indices]  # (B, K, D)
                    hard_neg_vecs = hard_neg_vecs.to(device)
                except Exception as e:
                    print(f"\n💥 CRASH at batch {batch_idx}/{len(train_loader)}")
                    print(f"   Error in hard negative indexing: {e}")
                    print(f"   batch_idx: {batch_idx}, args.bs: {args.bs}")
                    print(f"   hard_neg_indices shape: {hard_neg_indices.shape if hard_neg_indices is not None else 'None'}")
                    print(f"   bank_vectors shape: {bank_vectors.shape if bank_vectors is not None else 'None'}")
                    raise

            # Compute loss
            loss = info_nce_loss_with_extras(
                q, d_pos,
                tau=args.tau,
                margin=args.margin,
                memory_bank=memory_bank,
                hard_neg_vecs=hard_neg_vecs
            )

            # Check for NaN loss
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"\n💥 CRASH: NaN/Inf loss at batch {batch_idx}")
                print(f"   q range: [{q.min():.4f}, {q.max():.4f}]")
                print(f"   d_pos range: [{d_pos.min():.4f}, {d_pos.max():.4f}]")
                print(f"   loss value: {loss.item()}")
                raise ValueError(f"NaN/Inf loss at batch {batch_idx}")

            # Accumulate
            loss = loss / args.accum
            loss.backward()

            if (batch_idx + 1) % args.accum == 0:
                # 🚨 STABILITY FIX: Gradient clipping prevents exploding gradients
                torch.nn.utils.clip_grad_norm_(model_q.parameters(), max_norm=args.grad_clip)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            total_loss += loss.item() * args.accum
            num_batches += 1

            # 🚨 MPS STABILITY: Periodic synchronization to flush command queue
            if args.sync_mps_every > 0 and batch_idx % args.sync_mps_every == 0 and batch_idx > 0:
                if device == 'mps':
                    torch.mps.synchronize()

            # 🚨 MEMORY: Periodic garbage collection (if enabled)
            if args.gc_every > 0 and batch_idx % args.gc_every == 0 and batch_idx > 0:
                import gc
                gc.collect()

            # Queue health monitoring (every 200 steps)
            if async_miner is not None and batch_idx % queue_log_interval == 0 and batch_idx > 0:
                in_q_size = async_miner.in_q.qsize()
                out_q_size = async_miner.out_q.qsize()
                fifo_size = len(batch_fifo)
                print(f"\n  [Queue Health @ step {batch_idx}] in_q={in_q_size}, out_q={out_q_size}, fifo={fifo_size}")
                if out_q_size == 0:
                    print(f"  ⚠️  Output queue empty - consider increasing prefetch or qbatch")

            # Update memory bank
            if memory_bank is not None:
                memory_bank.add(d_pos.detach())

        except Exception as e:
            print(f"\n💥 FATAL ERROR in training loop at batch {batch_idx}/{len(train_loader)}")
            print(f"   Exception: {type(e).__name__}: {e}")
            print(f"   Epoch: {epoch}")
            print(f"   Num batches completed: {num_batches}")
            import traceback
            traceback.print_exc()
            raise

    # Final step if needed
    if num_batches % args.accum != 0:
        torch.nn.utils.clip_grad_norm_(model_q.parameters(), max_norm=args.grad_clip)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    avg_loss = total_loss / num_batches
    return avg_loss


def compute_separation_margin(model_q, model_d, val_loader, bank_vectors, device='mps', num_hard_negs=16):
    """
    Compute separation margin: Δ = E[cos(q,pos)] - E[cos(q,hard_neg)]
    Critical diagnostic: should grow >0.05 for healthy training
    """
    import faiss

    model_q.eval()
    model_d.eval()

    # Clear MPS cache before large CPU transfer (prevents memory fragmentation)
    if device == 'mps':
        torch.mps.empty_cache()
        torch.mps.synchronize()

    # Build FAISS index
    bank_np = bank_vectors.cpu().numpy().astype(np.float32)
    bank_np = bank_np / (np.linalg.norm(bank_np, axis=1, keepdims=True) + 1e-9)
    index = faiss.IndexFlatIP(768)
    index.add(bank_np)

    pos_sims = []
    hard_neg_sims = []

    with torch.no_grad():
        for X_batch, Y_batch in val_loader:
            X_batch = X_batch.to(device)
            Y_batch = Y_batch.to(device)

            q = model_q(X_batch)
            d_pos = model_d(Y_batch)

            # Positive similarities
            pos_sim = (q * d_pos).sum(dim=-1)
            pos_sims.extend(pos_sim.cpu().numpy().tolist())

            # Mine hard negatives
            q_np = q.cpu().numpy().astype(np.float32)
            D, I = index.search(q_np, num_hard_negs)

            # Hard negative similarities
            for i in range(len(q)):
                hard_idxs = I[i][:num_hard_negs]
                hard_vecs = bank_np[hard_idxs]
                hard_sims = (q_np[i] @ hard_vecs.T).tolist()
                hard_neg_sims.extend(hard_sims)

    pos_mean = np.mean(pos_sims)
    hard_neg_mean = np.mean(hard_neg_sims)
    margin = pos_mean - hard_neg_mean

    model_q.train()
    model_d.train()

    return {
        'pos_mean': float(pos_mean),
        'hard_neg_mean': float(hard_neg_mean),
        'margin': float(margin)
    }


def evaluate(model_q, model_d, val_loader, bank_vectors, device='mps'):
    """Evaluate recall@K"""
    import faiss

    model_q.eval()
    model_d.eval()

    # Clear MPS cache before large CPU transfer (prevents memory fragmentation)
    if device == 'mps':
        torch.mps.empty_cache()
        torch.mps.synchronize()

    # Build FAISS index
    bank_np = bank_vectors.cpu().numpy().astype(np.float32)
    bank_np = bank_np / (np.linalg.norm(bank_np, axis=1, keepdims=True) + 1e-9)
    index = faiss.IndexFlatIP(768)
    index.add(bank_np)

    # Encode validation queries
    all_q = []
    all_targets = []

    with torch.no_grad():
        for X_batch, Y_batch in tqdm(val_loader, desc="  Evaluating", leave=False):
            X_batch = X_batch.to(device)
            Y_batch = Y_batch.to(device)

            q = model_q(X_batch)
            d_target = model_d(Y_batch)

            all_q.append(q.cpu().numpy())
            all_targets.append(d_target.cpu().numpy())

    all_q = np.concatenate(all_q, axis=0).astype(np.float32)
    all_targets = np.concatenate(all_targets, axis=0).astype(np.float32)

    # Find target indices in bank
    target_indices = []
    for target_vec in tqdm(all_targets, desc="  Finding targets", leave=False):
        # Find exact match in bank
        D, I = index.search(target_vec.reshape(1, -1), 1)
        target_indices.append(I[0, 0])

    # Search
    K = 1000
    D, I = index.search(all_q, K)

    # Compute recall
    recalls = {10: 0, 100: 500, 1000: 0}
    for k in [10, 100, 500, 1000]:
        hits = 0
        for i, target_idx in enumerate(target_indices):
            if target_idx in I[i, :k]:
                hits += 1
        recalls[k] = 100 * hits / len(target_indices)

    model_q.train()
    model_d.train()

    return recalls


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Two-Tower v4 Training")
    parser.add_argument('--pairs', type=str, required=True, help='Training pairs NPZ')
    parser.add_argument('--bank', type=str, required=True, help='Vector bank NPZ')
    parser.add_argument('--out', type=str, default='runs/twotower_v4', help='Output directory')

    # Model
    parser.add_argument('--hidden-dim', type=int, default=512)
    parser.add_argument('--num-layers', type=int, default=1)

    # Training
    parser.add_argument('--bs', type=int, default=32)
    parser.add_argument('--accum', type=int, default=16, help='Gradient accumulation (eff batch = bs * accum)')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--lr-min', type=float, default=1e-6, help='Minimum LR for cosine decay')
    parser.add_argument('--wd', type=float, default=0.01)
    parser.add_argument('--tau', type=float, default=0.05, help='Temperature')
    parser.add_argument('--margin', type=float, default=0.03, help='Margin penalty')

    # Memory bank
    parser.add_argument('--memory-bank-size', type=int, default=50000)

    # Curriculum mining
    parser.add_argument('--mine-schedule', type=str,
                        default='0-5:none;6-10:8@0.82-0.92;11-30:16@0.84-0.96',
                        help='Mining schedule (epoch ranges)')
    parser.add_argument('--filter-threshold', type=float, default=0.98,
                        help='Filter hard negs with cos>threshold to both q and d_pos')

    # Async mining (GPU optimization)
    parser.add_argument('--async-mining', action='store_true',
                        help='Enable async FAISS mining (overlap with training)')
    parser.add_argument('--mine-k', type=int, default=128,
                        help='Candidate pool size for async mining')
    parser.add_argument('--mine-qbatch', type=int, default=2048,
                        help='FAISS query batch size (default 2048 for aggressive batching)')
    parser.add_argument('--mine-prefetch', type=int, default=3,
                        help='Mining queue depth (concurrent batches, default 3)')
    parser.add_argument('--mine-ttl', type=int, default=5,
                        help='Reuse mined negatives for N steps (cache TTL, default 5)')
    parser.add_argument('--hardneg-frac', type=float, default=0.5,
                        help='Fraction of hard negatives from FAISS (rest from memory/in-batch)')
    parser.add_argument('--mine-refresh-steps', type=int, default=3000,
                        help='Refresh mining every N steps (default 3000)')

    # Infrastructure
    parser.add_argument('--device', type=str, default='mps')
    parser.add_argument('--seed', type=int, default=42)

    # MPS stability toggles
    parser.add_argument('--grad-clip', type=float, default=1.0,
                        help='Gradient clipping max norm (default 1.0)')
    parser.add_argument('--sync-mps-every', type=int, default=100,
                        help='Synchronize MPS every N steps (default 100, 0=disable)')
    parser.add_argument('--gc-every', type=int, default=0,
                        help='Run gc.collect() every N steps (default 0=disable)')

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device)

    # 🧹 MPS CLEANUP: Clear cache before training to ensure clean memory state
    if device.type == 'mps':
        torch.mps.empty_cache()
        import gc
        gc.collect()
        print("✓ MPS cache cleared for clean start")

    print("============================================================")
    print("TWO-TOWER RETRIEVER TRAINING - V4 (CURRICULUM)")
    print("============================================================")
    print(f"Device: {args.device}")
    print(f"Batch size: {args.bs} × {args.accum} accum = {args.bs * args.accum} effective")
    print(f"Epochs: {args.epochs}")
    print(f"Learning rate: {args.lr} → {args.lr_min} (cosine decay)")
    print(f"Temperature: {args.tau}")
    print(f"Margin: {args.margin}")
    print(f"Memory bank size: {args.memory_bank_size:,}")
    print(f"Mining schedule: {args.mine_schedule}")
    print("============================================================")
    print()

    # Load data
    print("Loading data...")
    data = np.load(args.pairs)
    X_train = torch.from_numpy(data['X_train']).float()
    Y_train = torch.from_numpy(data['Y_train']).float()
    X_val = torch.from_numpy(data['X_val']).float()
    Y_val = torch.from_numpy(data['Y_val']).float()
    print(f"  Train: {len(X_train):,} pairs")
    print(f"  Val: {len(X_val):,} pairs")

    bank_data = np.load(args.bank)
    bank_vectors = torch.from_numpy(bank_data['vectors']).float()
    print(f"  Bank size: {bank_vectors.shape}")
    print()

    # Data loaders
    train_dataset = torch.utils.data.TensorDataset(X_train, Y_train)

    # 🚨 SAFETY RAILS:
    # - num_workers=0: Data already in RAM (TensorDataset), workers add overhead
    # - drop_last=True: Avoid partial batches (simplifies FIFO alignment)
    # - pin_memory: Only for CUDA (NOT MPS - causes semaphore leaks on macOS)

    use_pin_memory = (device.type == 'cuda')  # Only CUDA, NOT MPS

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.bs,
        shuffle=True,
        num_workers=0,           # SAFETY: TensorDataset already in RAM
        drop_last=True,          # SAFETY: No partial batches
        pin_memory=use_pin_memory
    )

    val_dataset = torch.utils.data.TensorDataset(X_val, Y_val)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.bs,
        shuffle=False,
        num_workers=0,           # SAFETY: TensorDataset already in RAM
        drop_last=False,         # Keep all validation samples
        pin_memory=use_pin_memory
    )

    # Models
    print("Building models...")
    model_q = GRUPoolQuery(d_model=768, hidden_dim=args.hidden_dim, num_layers=args.num_layers).to(device)
    model_d = IdentityDocTower().to(device)
    print(f"  Query tower params: {sum(p.numel() for p in model_q.parameters()):,}")
    print()

    # Optimizer with cosine decay
    # 🚨 STABILITY FIX: Disable multi-tensor kernels (foreach=False, fused=False)
    # to prevent silent crashes during optimizer.step()
    optimizer = torch.optim.AdamW(
        model_q.parameters(),
        lr=args.lr,
        weight_decay=args.wd,
        foreach=False,  # Disable multi-tensor kernels
        fused=False,    # Disable fused implementation
        capturable=False  # Disable CUDA graph capture
    )

    # Cosine LR scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr_min
    )

    # Memory bank
    memory_bank = MemoryBank(max_size=args.memory_bank_size, dim=768, device=device)

    # Parse mining schedule
    # Format: "0-5:none;6-10:8@0.82-0.92;11-30:16@0.84-0.96"
    mining_schedule = {}
    for seg in args.mine_schedule.split(';'):
        epoch_range, spec = seg.split(':')
        start, end = map(int, epoch_range.split('-'))
        for epoch in range(start, end + 1):
            if spec == 'none':
                mining_schedule[epoch] = None
            else:
                # Parse "8@0.82-0.92" or "16@0.84-0.96"
                num, cos_range = spec.split('@')
                cos_min, cos_max = map(float, cos_range.split('-'))
                mining_schedule[epoch] = {'num': int(num), 'cos_min': cos_min, 'cos_max': cos_max}

    # Training
    print("============================================================")
    print("TRAINING")
    print("============================================================")
    print()

    history = []
    best_recall500 = 0
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = out_dir / 'checkpoints'
    ckpt_dir.mkdir(exist_ok=True)

    # Save config for reproducibility
    import yaml
    config_dict = vars(args).copy()
    config_dict['mining_schedule'] = mining_schedule
    with open(out_dir / 'config.yaml', 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False)
    print(f"✓ Config saved: {out_dir / 'config.yaml'}\n")

    # Margin tracking for abort condition
    margin_history = []

    # Initialize async miner if enabled
    async_miner = None
    if args.async_mining:
        import faiss
        print("Initializing async FAISS miner...")

        # Build FAISS index once
        if device == 'mps':
            torch.mps.empty_cache()
            torch.mps.synchronize()

        bank_np = bank_vectors.cpu().numpy().astype(np.float32)
        bank_np = bank_np / (np.linalg.norm(bank_np, axis=1, keepdims=True) + 1e-9)
        faiss_index = faiss.IndexFlatIP(768)
        faiss_index.add(bank_np)

        # Create async miner
        async_miner = AsyncMiner(
            faiss_index=faiss_index,
            k=args.mine_k,
            qbatch=args.mine_qbatch,
            prefetch=args.mine_prefetch,
            ttl=args.mine_ttl
        )
        async_miner.start()
        print(f"  ✓ Async miner started (k={args.mine_k}, qbatch={args.mine_qbatch}, ttl={args.mine_ttl})")

    for epoch in range(1, args.epochs + 1):
        print(f"Epoch {epoch}/{args.epochs}")

        # Get mining spec for this epoch
        mining_spec = None
        if epoch in mining_schedule and mining_schedule[epoch] is not None:
            mining_spec = mining_schedule[epoch]

        # Check if we need to mine hard negatives (synchronous path)
        hard_neg_indices = None
        if not args.async_mining and mining_spec is not None:
            # Synchronous mining (original behavior)
            hard_neg_indices = mine_hard_negatives_with_filter(
                model_q, model_d, train_loader, bank_vectors,
                num_hard_negs=mining_spec['num'],
                cos_min=mining_spec['cos_min'],
                cos_max=mining_spec['cos_max'],
                filter_threshold=args.filter_threshold,
                device=device
            )
            print(f"  Mined {mining_spec['num']} hard negatives per sample (cos {mining_spec['cos_min']:.2f}-{mining_spec['cos_max']:.2f})")
        elif args.async_mining and mining_spec is not None:
            print(f"  Using async mining ({mining_spec['num']} hard negs, cos {mining_spec['cos_min']:.2f}-{mining_spec['cos_max']:.2f})")

        # Train
        train_loss = train_one_epoch(
            model_q, model_d, train_loader, optimizer, epoch, args,
            memory_bank=memory_bank,
            hard_neg_indices=hard_neg_indices,
            bank_vectors=bank_vectors,
            async_miner=async_miner if args.async_mining else None,
            mining_spec=mining_spec,
            device=device
        )

        # Save pre-validation checkpoint (in case validation crashes)
        print(f"\n  💾 Saving pre-validation checkpoint...")
        torch.save({
            'epoch': epoch,
            'model_q_state_dict': model_q.state_dict(),
            'model_d_state_dict': model_d.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'config': vars(args)
        }, ckpt_dir / f'epoch_{epoch:03d}_pre_validation.pt')
        print(f"  ✓ Saved: {ckpt_dir / f'epoch_{epoch:03d}_pre_validation.pt'}")

        # Evaluate (with error handling)
        try:
            print(f"\n  🔍 Starting validation evaluation...")
            recalls = evaluate(model_q, model_d, val_loader, bank_vectors, device=device)
            print(f"  ✓ Evaluation complete")
        except Exception as e:
            print(f"\n💥 FATAL ERROR during evaluation")
            print(f"   Exception: {type(e).__name__}: {e}")
            print(f"   Epoch: {epoch}")
            import traceback
            traceback.print_exc()
            raise

        # Compute separation margin (critical diagnostic)
        try:
            print(f"  📊 Computing separation margin...")
            margin_stats = compute_separation_margin(model_q, model_d, val_loader, bank_vectors, device=device)
            print(f"  ✓ Margin computation complete")
        except Exception as e:
            print(f"\n💥 FATAL ERROR during margin computation")
            print(f"   Exception: {type(e).__name__}: {e}")
            print(f"   Epoch: {epoch}")
            import traceback
            traceback.print_exc()
            raise

        # LR step
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step()

        # Log
        print(f"  Train loss: {train_loss:.4f}")
        print(f"  LR: {current_lr:.6f}")
        print(f"  Recall@10: {recalls[10]:.2f}%")
        print(f"  Recall@100: {recalls[100]:.2f}%")
        print(f"  Recall@500: {recalls[500]:.2f}%")
        print(f"  Recall@1000: {recalls[1000]:.2f}%")
        print(f"  Separation Δ: {margin_stats['margin']:.4f} (pos: {margin_stats['pos_mean']:.4f}, hard_neg: {margin_stats['hard_neg_mean']:.4f})")

        # Save history
        history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'lr': current_lr,
            'recall@10': recalls[10],
            'recall@100': recalls[100],
            'recall@500': recalls[500],
            'recall@1000': recalls[1000],
            'margin': margin_stats['margin'],
            'margin_pos': margin_stats['pos_mean'],
            'margin_hard_neg': margin_stats['hard_neg_mean']
        })

        with open(out_dir / 'history.json', 'w') as f:
            json.dump(history, f, indent=2)

        # Save checkpoints
        if recalls[500] > best_recall500:
            best_recall500 = recalls[500]
            torch.save({
                'epoch': epoch,
                'model_q_state_dict': model_q.state_dict(),
                'model_d_state_dict': model_d.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'metrics': {'recall@500': recalls[500]},
                'config': vars(args)
            }, ckpt_dir / 'best.pt')
            print(f"  ✓ New best Recall@500: {recalls[500]:.2f}%")

        # Periodic checkpoints
        if epoch % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_q_state_dict': model_q.state_dict(),
                'model_d_state_dict': model_d.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'metrics': {'recall@500': recalls[500]},
                'config': vars(args)
            }, ckpt_dir / f'epoch_{epoch:03d}.pt')

        # Track margin for abort condition
        margin_history.append(margin_stats['margin'])

        # Abort if margin stalls <0.04 for 2 consecutive evals after epoch 10
        if epoch >= 10 and len(margin_history) >= 2:
            if margin_history[-1] < 0.04 and margin_history[-2] < 0.04:
                print("\n" + "="*60)
                print("⚠️  ABORT: Margin collapse detected!")
                print(f"   Margin < 0.04 for 2 consecutive evals: {margin_history[-2]:.4f}, {margin_history[-1]:.4f}")
                print("   Hard negatives may be too hard or impure.")
                print("   Recommendations:")
                print("     - Tighten cos window (e.g., 0.84-0.94)")
                print("     - Reduce hard-neg count (16→8)")
                print("     - Lower temperature (0.05→0.045)")
                print("="*60 + "\n")
                print("Saving abort checkpoint...")
                torch.save({
                    'epoch': epoch,
                    'model_q_state_dict': model_q.state_dict(),
                    'model_d_state_dict': model_d.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'metrics': {'recall@500': recalls[500], 'margin': margin_stats['margin']},
                    'config': vars(args),
                    'abort_reason': 'margin_collapse'
                }, ckpt_dir / 'abort.pt')
                print("Exiting early.")
                break

        print()

    # Stop async miner if running
    if async_miner is not None:
        print("Stopping async miner...")
        async_miner.stop()
        print("  ✓ Async miner stopped")

    print("============================================================")
    print("TRAINING COMPLETE")
    print("============================================================")
    print(f"Best Recall@500: {best_recall500:.2f}%")
    print(f"Output: {out_dir}")


if __name__ == '__main__':
    main()
