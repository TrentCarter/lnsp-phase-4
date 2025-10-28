#!/usr/bin/env python3
"""
Mid-Training Checkpoint Smoke Test
===================================

Quick evaluation (1k samples) to check if training is on track.

Gates:
- Contain@50 ≥ 55% (by epoch 4)
- Eff@5 ≥ 0.65 (by epoch 4)
- R@5 ≥ 35% (by epoch 4)

If gates pass, continue training to 20 epochs.
If gates fail, stop and diagnose.

Usage:
    python tools/eval_checkpoint_smoke.py \
        --checkpoint artifacts/lvm/models/mamba_s_poc/checkpoint_epoch_4.pt \
        --eval-npz artifacts/lvm/eval_v2_payload_aligned.npz \
        --payload artifacts/wikipedia_584k_payload.npy \
        --faiss artifacts/wikipedia_584k_ivf_flat_ip.index \
        --limit 1000 \
        --epoch 4
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import faiss

# Add project root
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from app.lvm.mamba import create_model


def load_checkpoint(checkpoint_path: Path, device: str):
    """Load model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    args = checkpoint['args']
    model_type = args['model_type']

    # Build kwargs
    model_kwargs = {
        'd_model': args['d_model'],
        'd_state': args['d_state'],
        'conv_sz': args['conv_sz'],
        'expand': args['expand'],
        'dropout': args['dropout'],
    }

    if 'sandwich' in model_type:
        model_kwargs.update({
            'n_layers_mamba': args.get('n_layers_mamba', 8),
            'n_layers_local': args.get('n_layers_local', 4),
            'local_attn_win': args.get('local_attn_win', 8),
            'n_heads': args.get('n_heads', 4),
        })
    elif 'hybrid' in model_type:
        model_kwargs.update({
            'n_layers': args['n_layers'],
            'local_attn_win': args.get('local_attn_win', 8),
            'local_attn_every': args.get('local_attn_every', 4),
            'n_heads': args.get('n_heads', 4),
        })
    else:
        model_kwargs['n_layers'] = args['n_layers']

    if 'gr' in model_type:
        model_kwargs['gru_hidden'] = args.get('gru_hidden', 256)

    model_kwargs.update({
        'use_alignment_head': args.get('use_alignment_head', False),
        'alignment_alpha': args.get('alignment_alpha', 0.25),
    })

    model = create_model(model_type=model_type, **model_kwargs)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    return model, checkpoint


@torch.no_grad()
def generate_predictions(model, contexts, device, batch_size=128):
    """Generate predictions."""
    contexts_tensor = torch.from_numpy(contexts).float()
    all_preds = []

    for i in range(0, len(contexts), batch_size):
        batch = contexts_tensor[i:i+batch_size].to(device)
        preds = model(batch)

        if len(preds.shape) == 3:
            preds = preds[:, -1, :]

        preds = F.normalize(preds, p=2, dim=1)
        all_preds.append(preds.cpu().numpy())

    return np.concatenate(all_preds, axis=0)


def evaluate_retrieval(pred_vecs, truth_keys, payload, index, nprobe=64, K=50):
    """Evaluate retrieval metrics."""
    # Build reverse index
    article_chunk_to_id = {}
    for payload_id, (text, meta, vec) in payload.items():
        key = (meta['article_index'], meta['chunk_index'])
        article_chunk_to_id[key] = payload_id

    # Set nprobe
    if hasattr(index, 'nprobe'):
        index.nprobe = nprobe

    # Metrics
    hits_at_1 = 0
    hits_at_5 = 0
    hits_at_10 = 0
    hits_at_20 = 0
    contained_at_50 = 0
    latencies = []

    for i in range(len(pred_vecs)):
        pred = pred_vecs[i]
        art_idx, chunk_idx = truth_keys[i]
        truth_id = article_chunk_to_id[(int(art_idx), int(chunk_idx))]

        # FAISS search
        query = pred.reshape(1, -1).astype(np.float32)

        start = time.time()
        D, I = index.search(query, K)
        latencies.append(time.time() - start)

        # Check if truth is in top-K
        retrieved_ids = I[0].tolist()

        if truth_id in retrieved_ids:
            contained_at_50 += 1

            rank = retrieved_ids.index(truth_id)
            if rank == 0:
                hits_at_1 += 1
            if rank < 5:
                hits_at_5 += 1
            if rank < 10:
                hits_at_10 += 1
            if rank < 20:
                hits_at_20 += 1

    n = len(pred_vecs)
    latencies_ms = [1000 * lat for lat in latencies]

    return {
        'n_samples': n,
        'R@1': 100 * hits_at_1 / n,
        'R@5': 100 * hits_at_5 / n,
        'R@10': 100 * hits_at_10 / n,
        'R@20': 100 * hits_at_20 / n,
        'Contain@50': 100 * contained_at_50 / n,
        'Eff@5': 100 * hits_at_5 / contained_at_50 if contained_at_50 > 0 else 0.0,
        'latency_mean_ms': np.mean(latencies_ms),
        'latency_p50_ms': np.percentile(latencies_ms, 50),
        'latency_p95_ms': np.percentile(latencies_ms, 95),
        'latency_p99_ms': np.percentile(latencies_ms, 99),
    }


def main():
    ap = argparse.ArgumentParser(description="Mid-training checkpoint smoke test")
    ap.add_argument("--checkpoint", type=Path, required=True,
                    help="Path to checkpoint file")
    ap.add_argument("--eval-npz", type=Path, required=True,
                    help="Path to aligned eval NPZ")
    ap.add_argument("--payload", type=Path, required=True,
                    help="Path to payload NPY")
    ap.add_argument("--faiss", type=Path, required=True,
                    help="Path to FAISS index")
    ap.add_argument("--device", type=str, default="cpu",
                    choices=["cpu", "cuda", "mps"],
                    help="Device for inference")
    ap.add_argument("--limit", type=int, default=1000,
                    help="Number of samples for smoke test")
    ap.add_argument("--epoch", type=int, default=None,
                    help="Epoch number (for display)")
    ap.add_argument("--nprobe", type=int, default=64,
                    help="FAISS nprobe")
    ap.add_argument("--out", type=Path, default=None,
                    help="Optional output JSON")

    args = ap.parse_args()

    print("=" * 80)
    print("MID-TRAINING CHECKPOINT SMOKE TEST")
    print("=" * 80)
    print(f"Checkpoint: {args.checkpoint}")
    if args.epoch:
        print(f"Epoch: {args.epoch}")
    print(f"Samples: {args.limit}")
    print(f"Device: {args.device}")
    print("=" * 80)

    # Load model
    print("\nLoading checkpoint...")
    model, checkpoint = load_checkpoint(args.checkpoint, args.device)
    print(f"  Model type: {checkpoint['args']['model_type']}")
    print(f"  Training epoch: {checkpoint.get('epoch', 'unknown')}")
    print(f"  Val cosine: {checkpoint.get('val_cosine', 0.0):.4f}")

    # Load data
    print("\nLoading data...")
    eval_data = np.load(args.eval_npz, allow_pickle=True)
    contexts = eval_data['contexts'][:args.limit]
    targets = eval_data['targets'][:args.limit]
    truth_keys = eval_data['truth_keys'][:args.limit]

    payload = np.load(args.payload, allow_pickle=True).item()
    index = faiss.read_index(str(args.faiss))

    print(f"  Eval samples: {len(contexts)}")
    print(f"  Payload size: {len(payload)}")

    # Generate predictions
    print(f"\nGenerating predictions...")
    start = time.time()
    pred_vecs = generate_predictions(model, contexts, args.device)
    pred_time = time.time() - start
    print(f"  Generated {len(pred_vecs)} predictions in {pred_time:.2f}s")

    # Validation cosine
    print(f"\nValidation cosines (pred vs target):")
    cosines = [np.dot(pred_vecs[i], targets[i]) for i in range(len(pred_vecs))]
    print(f"  Mean: {np.mean(cosines):.4f}")
    print(f"  Range: [{np.min(cosines):.4f}, {np.max(cosines):.4f}]")

    # Run retrieval evaluation
    print(f"\n" + "=" * 80)
    print("RETRIEVAL EVALUATION")
    print("=" * 80)

    results = evaluate_retrieval(
        pred_vecs, truth_keys, payload, index,
        nprobe=args.nprobe, K=50
    )

    # Display results
    print(f"\nSamples: {results['n_samples']}")
    print(f"\nRetrieval Metrics:")
    print(f"  R@1:          {results['R@1']:6.2f}%")
    print(f"  R@5:          {results['R@5']:6.2f}%")
    print(f"  R@10:         {results['R@10']:6.2f}%")
    print(f"  R@20:         {results['R@20']:6.2f}%")
    print(f"  Contain@50:   {results['Contain@50']:6.2f}%")
    print(f"  Eff@5:        {results['Eff@5']:6.2f}%")
    print(f"\nLatency:")
    print(f"  Mean:         {results['latency_mean_ms']:6.2f} ms")
    print(f"  P50:          {results['latency_p50_ms']:6.2f} ms")
    print(f"  P95:          {results['latency_p95_ms']:6.2f} ms")

    # Check gates
    print(f"\n" + "=" * 80)
    print("GATE CHECK")
    print("=" * 80)

    gates_passed = []
    gates_failed = []

    # Epoch-dependent gates
    if args.epoch is not None and args.epoch >= 4:
        # Stricter gates for epoch 4+
        if results['Contain@50'] >= 55:
            gates_passed.append(f"✅ Contain@50: {results['Contain@50']:.1f}% ≥ 55%")
        else:
            gates_failed.append(f"❌ Contain@50: {results['Contain@50']:.1f}% < 55%")

        if results['Eff@5'] >= 65:
            gates_passed.append(f"✅ Eff@5: {results['Eff@5']:.1f}% ≥ 65%")
        else:
            gates_failed.append(f"⚠️  Eff@5: {results['Eff@5']:.1f}% < 65%")

        if results['R@5'] >= 35:
            gates_passed.append(f"✅ R@5: {results['R@5']:.1f}% ≥ 35%")
        else:
            gates_failed.append(f"❌ R@5: {results['R@5']:.1f}% < 35%")
    else:
        # Looser gates for early epochs
        if results['Contain@50'] >= 40:
            gates_passed.append(f"✅ Contain@50: {results['Contain@50']:.1f}% ≥ 40% (early epoch)")
        else:
            gates_failed.append(f"⚠️  Contain@50: {results['Contain@50']:.1f}% < 40%")

    # Always check latency
    if results['latency_p95_ms'] <= 2.0:
        gates_passed.append(f"✅ P95 latency: {results['latency_p95_ms']:.2f}ms ≤ 2.0ms")
    else:
        gates_failed.append(f"⚠️  P95 latency: {results['latency_p95_ms']:.2f}ms > 2.0ms")

    # Print gate results
    for gate in gates_passed:
        print(gate)
    for gate in gates_failed:
        print(gate)

    print("=" * 80)

    if len(gates_failed) == 0:
        print("✅ ALL GATES PASSED - Continue training to 20 epochs")
    elif len(gates_failed) <= 1 and args.epoch and args.epoch < 4:
        print("⚠️  PARTIAL PASS - Continue training, monitor next checkpoint")
    else:
        print("❌ GATES FAILED - Consider stopping and diagnosing")

    print("=" * 80)

    # Save results if requested
    if args.out:
        results['checkpoint'] = str(args.checkpoint)
        results['epoch'] = args.epoch
        results['val_cosine'] = float(checkpoint.get('val_cosine', 0.0))
        results['gates_passed'] = gates_passed
        results['gates_failed'] = gates_failed

        with open(args.out, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {args.out}")

    # Return exit code based on gates
    return 0 if len(gates_failed) == 0 else 1


if __name__ == "__main__":
    exit(main())
