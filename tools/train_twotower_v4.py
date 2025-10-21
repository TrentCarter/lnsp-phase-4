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
                     memory_bank=None, hard_neg_indices=None, bank_vectors=None, device='mps'):
    """Train for one epoch"""
    model_q.train()
    model_d.train()

    total_loss = 0
    num_batches = 0

    optimizer.zero_grad()

    for batch_idx, (X_batch, Y_batch) in enumerate(tqdm(train_loader, desc=f"  Training", leave=False)):
        X_batch = X_batch.to(device)
        Y_batch = Y_batch.to(device)

        # Encode
        q = model_q(X_batch)
        d_pos = model_d(Y_batch)

        # Get hard negatives if available
        hard_neg_vecs = None
        if hard_neg_indices is not None and bank_vectors is not None:
            global_indices = range(batch_idx * args.bs, (batch_idx + 1) * args.bs)
            batch_hard_indices = hard_neg_indices[global_indices]  # (B, K)
            hard_neg_vecs = bank_vectors[batch_hard_indices]  # (B, K, D)
            hard_neg_vecs = hard_neg_vecs.to(device)

        # Compute loss
        loss = info_nce_loss_with_extras(
            q, d_pos,
            tau=args.tau,
            margin=args.margin,
            memory_bank=memory_bank,
            hard_neg_vecs=hard_neg_vecs
        )

        # Accumulate
        loss = loss / args.accum
        loss.backward()

        if (batch_idx + 1) % args.accum == 0:
            optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item() * args.accum
        num_batches += 1

        # Update memory bank
        if memory_bank is not None:
            memory_bank.add(d_pos.detach())

    # Final step if needed
    if num_batches % args.accum != 0:
        optimizer.step()
        optimizer.zero_grad()

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

    # Infrastructure
    parser.add_argument('--device', type=str, default='mps')
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device)

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
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.bs, shuffle=True)

    val_dataset = torch.utils.data.TensorDataset(X_val, Y_val)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.bs, shuffle=False)

    # Models
    print("Building models...")
    model_q = GRUPoolQuery(d_model=768, hidden_dim=args.hidden_dim, num_layers=args.num_layers).to(device)
    model_d = IdentityDocTower().to(device)
    print(f"  Query tower params: {sum(p.numel() for p in model_q.parameters()):,}")
    print()

    # Optimizer with cosine decay
    optimizer = torch.optim.AdamW(model_q.parameters(), lr=args.lr, weight_decay=args.wd)

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

    for epoch in range(1, args.epochs + 1):
        print(f"Epoch {epoch}/{args.epochs}")

        # Check if we need to mine hard negatives
        hard_neg_indices = None
        if epoch in mining_schedule and mining_schedule[epoch] is not None:
            spec = mining_schedule[epoch]
            hard_neg_indices = mine_hard_negatives_with_filter(
                model_q, model_d, train_loader, bank_vectors,
                num_hard_negs=spec['num'],
                cos_min=spec['cos_min'],
                cos_max=spec['cos_max'],
                filter_threshold=args.filter_threshold,
                device=device
            )
            print(f"  Mined {spec['num']} hard negatives per sample (cos {spec['cos_min']:.2f}-{spec['cos_max']:.2f})")

        # Train
        train_loss = train_one_epoch(
            model_q, model_d, train_loader, optimizer, epoch, args,
            memory_bank=memory_bank,
            hard_neg_indices=hard_neg_indices,
            bank_vectors=bank_vectors,
            device=device
        )

        # Evaluate
        recalls = evaluate(model_q, model_d, val_loader, bank_vectors, device=device)

        # Compute separation margin (critical diagnostic)
        margin_stats = compute_separation_margin(model_q, model_d, val_loader, bank_vectors, device=device)

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

    print("============================================================")
    print("TRAINING COMPLETE")
    print("============================================================")
    print(f"Best Recall@500: {best_recall500:.2f}%")
    print(f"Output: {out_dir}")


if __name__ == '__main__':
    main()
