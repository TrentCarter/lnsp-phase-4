#!/usr/bin/env python3
"""
Evaluate two-tower retriever.

Loads trained checkpoint and evaluates Recall@K metrics on validation set.

Usage:
    python tools/eval_twotower.py \
      --ckpt runs/twotower_mvp/checkpoints/best.pt \
      --bank artifacts/wikipedia_500k_corrected_vectors.npz \
      --out artifacts/evals/twotower_mvp.json
"""

import argparse
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm


# Import model definitions from training script
class GRUPoolQuery(nn.Module):
    def __init__(self, d_model=768, hidden_dim=512, num_layers=1, dropout=0.1):
        super().__init__()
        self.gru = nn.GRU(
            input_size=d_model,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.proj = nn.Linear(2 * hidden_dim, d_model)
        
    def forward(self, x):
        out, _ = self.gru(x)
        pooled = out.mean(dim=1)
        q = self.proj(pooled)
        q = F.normalize(q, p=2, dim=-1)
        return q


class IdentityDocTower(nn.Module):
    def forward(self, d):
        return F.normalize(d, p=2, dim=-1)


def load_model(ckpt_path, device='cpu'):
    """Load trained two-tower model from checkpoint."""
    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    
    config = ckpt.get('config', {})
    
    # Create models
    model_q = GRUPoolQuery(
        d_model=768,
        hidden_dim=config.get('hidden_dim', 512),
        num_layers=config.get('num_layers', 1)
    ).to(device)
    
    model_d = IdentityDocTower().to(device)
    
    # Load state dicts
    model_q.load_state_dict(ckpt['model_q_state_dict'])
    model_d.load_state_dict(ckpt['model_d_state_dict'])
    
    model_q.eval()
    model_d.eval()
    
    print(f"  Epoch: {ckpt['epoch']}")
    print(f"  Metrics: {ckpt.get('metrics', {})}")
    
    return model_q, model_d, config


def evaluate_recall(
    model_q,
    model_d,
    contexts,
    targets,
    bank_vectors,
    k_values=[1, 5, 10, 50, 100, 500, 1000],
    device='cpu',
    batch_size=16
):
    """
    Evaluate Recall@K.
    
    Args:
        model_q: Query tower
        model_d: Doc tower
        contexts: (N, seq_len, 768) - Query contexts
        targets: (N, 768) - Target vectors
        bank_vectors: (M, 768) - Full vector bank
        k_values: List of K for Recall@K
        device: Device
        batch_size: Batch size for processing
    
    Returns:
        metrics: Dict of results
    """
    print("\nEvaluating Recall@K...")
    print(f"  Query contexts: {contexts.shape}")
    print(f"  Bank size: {bank_vectors.shape}")
    
    # Encode bank once
    print("  Encoding bank...")
    bank_tensor = torch.from_numpy(bank_vectors).float().to(device)
    
    with torch.no_grad():
        bank_encoded = []
        for i in range(0, len(bank_tensor), 1000):  # Process in chunks
            batch = bank_tensor[i:i+1000]
            encoded = model_d(batch).cpu().numpy()
            bank_encoded.append(encoded)
        bank_encoded = np.concatenate(bank_encoded, axis=0)
    
    print(f"  Bank encoded: {bank_encoded.shape}")
    
    # Evaluate queries
    recalls = {k: [] for k in k_values}
    mrr_scores = []
    
    print("  Processing queries...")
    n_samples = len(contexts)
    
    for i in tqdm(range(0, n_samples, batch_size), desc="  Eval"):
        batch_ctx = contexts[i:i+batch_size]
        batch_tgt = targets[i:i+batch_size]
        
        # Encode queries
        batch_tensor = torch.from_numpy(batch_ctx).float().to(device)
        with torch.no_grad():
            queries = model_q(batch_tensor).cpu().numpy()
        
        # For each query
        for query, target in zip(queries, batch_tgt):
            # Compute similarities to all bank vectors
            sims = np.dot(bank_encoded, query)
            
            # Get top-K indices
            top_k_idx = np.argsort(-sims)[:max(k_values)]
            
            # Find target in bank (nearest neighbor to target)
            target_norm = target / (np.linalg.norm(target) + 1e-8)
            target_sims = np.dot(bank_encoded, target_norm)
            target_idx = np.argmax(target_sims)
            
            # Recall@K
            for k in k_values:
                recalls[k].append(1.0 if target_idx in top_k_idx[:k] else 0.0)
            
            # MRR
            if target_idx in top_k_idx:
                rank = np.where(top_k_idx == target_idx)[0][0] + 1
                mrr_scores.append(1.0 / rank)
            else:
                mrr_scores.append(0.0)
    
    # Aggregate
    metrics = {
        f'recall@{k}': float(np.mean(recalls[k]) * 100) for k in k_values
    }
    metrics['mrr'] = float(np.mean(mrr_scores))
    metrics['n_samples'] = int(n_samples)
    metrics['bank_size'] = int(len(bank_vectors))
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate two-tower retriever")
    parser.add_argument('--ckpt', required=True, help='Model checkpoint')
    parser.add_argument('--pairs', help='Pairs NPZ (optional, if not using --contexts/--targets)')
    parser.add_argument('--contexts', help='Context sequences NPZ')
    parser.add_argument('--targets', help='Target vectors NPZ')
    parser.add_argument('--bank', required=True, help='Vector bank NPZ')
    parser.add_argument('--device', default='mps', choices=['mps', 'cuda', 'cpu'])
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--out', required=True, help='Output JSON file')
    
    args = parser.parse_args()
    
    # Device
    if args.device == 'mps' and not torch.backends.mps.is_available():
        print("⚠️  MPS not available, using CPU")
        device = torch.device('cpu')
    else:
        device = torch.device(args.device)
    
    print("="*60)
    print("TWO-TOWER RETRIEVER EVALUATION")
    print("="*60)
    print(f"Device: {device}")
    
    # Load model
    model_q, model_d, config = load_model(args.ckpt, device)
    
    # Load validation data
    print("\nLoading validation data...")
    if args.pairs:
        data = np.load(args.pairs, allow_pickle=True)
        contexts = data['X_val']
        targets = data['Y_val']
    else:
        contexts = np.load(args.contexts)
        targets = np.load(args.targets)
    
    print(f"  Validation samples: {len(contexts):,}")
    
    # Load bank
    print(f"\nLoading bank: {args.bank}")
    bank_data = np.load(args.bank, allow_pickle=True)
    bank_vectors = bank_data['vectors']
    print(f"  Bank size: {len(bank_vectors):,}")
    
    # Evaluate
    metrics = evaluate_recall(
        model_q, model_d,
        contexts, targets,
        bank_vectors,
        k_values=[1, 5, 10, 50, 100, 500, 1000],
        device=device,
        batch_size=args.batch_size
    )
    
    # Print results
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    for k, v in sorted(metrics.items()):
        if k.startswith('recall'):
            print(f"{k:20s}: {v:6.2f}%")
        elif k == 'mrr':
            print(f"{k:20s}: {v:6.4f}")
    print("="*60)
    
    # Save
    output_path = Path(args.out)
    output_path.parent.mkdir(exist_ok=True, parents=True)
    
    with open(output_path, 'w') as f:
        json.dump({
            'metrics': metrics,
            'config': config,
            'checkpoint': str(args.ckpt),
            'bank': str(args.bank)
        }, f, indent=2)
    
    print(f"\n✓ Results saved to: {output_path}")
    
    # Success check
    if metrics['recall@500'] > 40.0:
        print("\n🎉 SUCCESS: Recall@500 > 40% (beats heuristic baseline!)")
    else:
        print(f"\n⚠️  Recall@500 = {metrics['recall@500']:.2f}% (target: >40%)")


if __name__ == '__main__':
    main()
