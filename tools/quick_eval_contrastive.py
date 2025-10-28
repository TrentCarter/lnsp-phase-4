#!/usr/bin/env python3
"""
Quick evaluation of contrastive model - generates predictions and tests retrieval.
"""
import argparse
import json
import time
from pathlib import Path

import faiss
import numpy as np
import torch
import torch.nn.functional as F

# Add project root
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.lvm.mamba import create_model


def main():
    # Load contrastive model
    print("Loading contrastive model...")
    ckpt = torch.load('artifacts/lvm/models/mamba_s_contrastive/best.pt', map_location='cpu', weights_only=False)
    
    model_args = ckpt['args']
    model = create_model(
        model_type='mamba_s',
        d_model=768,
        n_layers=8,
        d_state=128,
        conv_sz=4,
        expand=2,
        dropout=0.1,
    )
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    
    print(f"  Best val_cosine: {ckpt['val_cosine']:.4f}")
    print()
    
    # Load eval data
    print("Loading eval data...")
    eval_data = np.load('artifacts/lvm/LEAKED_EVAL_SETS/eval_v2_ready.npz', allow_pickle=True)
    contexts = torch.from_numpy(eval_data['contexts'][:1000])  # 1k sample smoke test
    truth_keys = eval_data['truth_keys'][:1000]
    
    print(f"  Eval samples: {len(contexts)}")
    print()
    
    # Generate predictions
    print("Generating predictions...")
    with torch.no_grad():
        preds = model(contexts)
        if len(preds.shape) == 3:
            preds = preds[:, -1, :]
        preds = F.normalize(preds, p=2, dim=-1).numpy()
    
    print(f"  Generated: {len(preds)} predictions")
    print()
    
    # Load payload and FAISS
    print("Loading payload and FAISS...")
    payload = np.load('artifacts/wikipedia_584k_payload.npy', allow_pickle=True).item()
    index = faiss.read_index('artifacts/wikipedia_584k_ivf_flat_ip.index')
    index.nprobe = 64
    
    # Build mapping
    article_chunk_to_id = {}
    for pid, (text, meta, vec) in payload.items():
        key = (meta['article_index'], meta['chunk_index'])
        article_chunk_to_id[key] = pid
    
    print(f"  Payload size: {len(payload)}")
    print(f"  FAISS vectors: {index.ntotal}")
    print()
    
    # Retrieval evaluation
    print("Running retrieval...")
    ranks = []
    
    for i in range(len(preds)):
        if i % 200 == 0 and i > 0:
            print(f"  Progress: {i}/{len(preds)}")
        
        pred = preds[i].reshape(1, -1).astype(np.float32)
        art_idx, chunk_idx = truth_keys[i]
        key = (int(art_idx), int(chunk_idx))
        
        if key not in article_chunk_to_id:
            ranks.append(-1)
            continue
        
        truth_id = article_chunk_to_id[key]
        
        # FAISS search
        D, I = index.search(pred, min(1000, index.ntotal))
        
        if truth_id in I[0]:
            rank = int(np.where(I[0] == truth_id)[0][0])
            ranks.append(rank)
        else:
            ranks.append(-1)
    
    ranks = np.array(ranks)
    
    # Metrics
    c20 = float(((ranks >= 0) & (ranks < 20)).mean())
    c50 = float(((ranks >= 0) & (ranks < 50)).mean())
    r1 = float((ranks == 0).mean())
    r5 = float((ranks < 5).mean())
    r10 = float((ranks < 10).mean())
    eff5 = r5 / c50 if c50 > 0 else 0.0
    
    # Results
    print()
    print("=" * 80)
    print("CONTRASTIVE MODEL RESULTS (Leaked Eval)")
    print("=" * 80)
    print(f"Samples: {len(ranks)}")
    print(f"Contain@20: {c20:.3f} ({c20*100:.1f}%)")
    print(f"Contain@50: {c50:.3f} ({c50*100:.1f}%)")
    print(f"R@1: {r1:.3f} ({r1*100:.1f}%)")
    print(f"R@5: {r5:.3f} ({r5*100:.1f}%)")
    print(f"R@10: {r10:.3f} ({r10*100:.1f}%)")
    print(f"Eff@5: {eff5:.3f}")
    print()
    
    # Interpretation
    if r5 > 0.10:
        print("✅ SUCCESS: InfoNCE + AR improves over AR-only (0% baseline)!")
        print(f"   Improvement: 0% → {r5*100:.1f}% R@5")
        print("   Contrastive learning enables transfer!")
    elif r5 > 0.05:
        print("⚠️  PARTIAL: Some improvement, but below target")
    else:
        print("❌ FAILURE: No retrieval improvement")
    
    print("=" * 80)


if __name__ == '__main__':
    main()
