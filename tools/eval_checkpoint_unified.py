#!/usr/bin/env python3
"""
Unified Checkpoint Evaluation (Smoke + Full)
============================================

Standardized evaluation with locked retrieval knobs and unified gates.

Gates (Unified):
- Contain@50 ≥ 60%
- Eff@5 ≥ 0.68
- R@5 ≥ 40%
- P95 ≤ 1.45ms

Locked Retrieval Knobs:
- nprobe=64
- shard-assist=ON (if available)
- MMR λ=0.7 (reranker)
- seq-bias: w_same_article=0.05, w_next_gap=0.12
- directional=0.03

Usage:
    # Smoke test (1k samples, epoch 4)
    KMP_DUPLICATE_LIB_OK=TRUE ./.venv/bin/python tools/eval_checkpoint_unified.py \
        --checkpoint artifacts/lvm/models/mamba_s_poc/best.pt \
        --eval-npz artifacts/lvm/eval_v2_payload_aligned.npz \
        --payload artifacts/wikipedia_584k_payload.npy \
        --faiss artifacts/wikipedia_584k_ivf_flat_ip.index \
        --device cpu \
        --limit 1000 \
        --epoch 4 \
        --out artifacts/lvm/smoke_epoch4.json

    # Full eval (5.2k samples)
    KMP_DUPLICATE_LIB_OK=TRUE ./.venv/bin/python tools/eval_checkpoint_unified.py \
        --checkpoint artifacts/lvm/models/mamba_s_poc/best.pt \
        --eval-npz artifacts/lvm/eval_v2_payload_aligned.npz \
        --payload artifacts/wikipedia_584k_payload.npy \
        --faiss artifacts/wikipedia_584k_ivf_flat_ip.index \
        --device cpu \
        --limit 5244 \
        --out artifacts/lvm/final_eval.json
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


def check_provenance(eval_npz, payload_path):
    """
    Check provenance alignment between eval data and payload.

    Gates:
    - embedder_id must match
    - norm must match
    - metric must match
    """
    if 'provenance' not in eval_npz:
        print("⚠️  WARNING: No provenance metadata in eval NPZ!")
        return True  # Don't fail if missing

    prov = json.loads(str(eval_npz['provenance'][0]))

    print("\nProvenance Check:")
    print(f"  embedder_id: {prov.get('embedder_id', 'MISSING')}")
    print(f"  norm: {prov.get('norm', 'MISSING')}")
    print(f"  metric: {prov.get('metric', 'MISSING')}")
    print(f"  aligned_at: {prov.get('aligned_at', 'MISSING')}")

    # Expected values
    expected = {
        'embedder_id': 'GTR-T5-base-768',
        'norm': 'l2_once',
        'metric': 'ip',
    }

    mismatches = []
    for key, expected_val in expected.items():
        actual_val = prov.get(key)
        if actual_val != expected_val:
            mismatches.append(f"{key}: expected={expected_val}, actual={actual_val}")

    if mismatches:
        print(f"\n❌ PROVENANCE MISMATCH:")
        for m in mismatches:
            print(f"  {m}")
        return False

    print(f"  ✅ Provenance matches expected values")
    return True


def verify_truth_payload_alignment(targets, truth_keys, payload, sample_size=100):
    """
    Verify truth vectors match payload vectors.

    Gates:
    - Mean cosine ≥ 0.98
    - P5 cosine ≥ 0.95
    """
    # Build reverse index
    article_chunk_to_id = {}
    for payload_id, (text, meta, vec) in payload.items():
        key = (meta['article_index'], meta['chunk_index'])
        article_chunk_to_id[key] = payload_id

    # Sample truth-to-payload cosines
    cosines = []
    indices = np.random.choice(len(targets), min(sample_size, len(targets)), replace=False)

    for i in indices:
        target = targets[i]
        art_idx, chunk_idx = truth_keys[i]
        key = (int(art_idx), int(chunk_idx))

        if key not in article_chunk_to_id:
            continue

        payload_id = article_chunk_to_id[key]
        _, _, payload_vec = payload[payload_id]

        cos = np.dot(target, payload_vec)
        cosines.append(cos)

    mean_cos = np.mean(cosines)
    p5_cos = np.percentile(cosines, 5)

    print(f"\nTruth→Payload Alignment (n={len(cosines)}):")
    print(f"  Mean cosine: {mean_cos:.4f}")
    print(f"  P5 cosine: {p5_cos:.4f}")

    if mean_cos < 0.98 or p5_cos < 0.95:
        print(f"  ❌ ALIGNMENT FAILURE: mean={mean_cos:.4f} < 0.98 or p5={p5_cos:.4f} < 0.95")
        return False, mean_cos, p5_cos

    print(f"  ✅ Alignment verified")
    return True, mean_cos, p5_cos


def load_checkpoint(checkpoint_path: Path, device: str):
    """Load model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
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
    """
    Evaluate retrieval metrics with locked knobs.

    NOTE: Full reranking (MMR, seq-bias, directional) not yet integrated.
    This uses base FAISS retrieval only. Reranker will be added in next iteration.
    """
    # Build reverse index
    article_chunk_to_id = {}
    for payload_id, (text, meta, vec) in payload.items():
        key = (meta['article_index'], meta['chunk_index'])
        article_chunk_to_id[key] = payload_id

    # Set nprobe (locked)
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
        'Contain@20': 100 * hits_at_20 / n,  # Simplified: same as R@20 without reranking
        'Contain@50': 100 * contained_at_50 / n,
        'Eff@5': 100 * hits_at_5 / contained_at_50 if contained_at_50 > 0 else 0.0,
        'latency_mean_ms': np.mean(latencies_ms),
        'latency_p50_ms': np.percentile(latencies_ms, 50),
        'latency_p95_ms': np.percentile(latencies_ms, 95),
        'latency_p99_ms': np.percentile(latencies_ms, 99),
    }


def main():
    ap = argparse.ArgumentParser(description="Unified checkpoint evaluation")
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
                    help="Number of samples")
    ap.add_argument("--epoch", type=int, default=None,
                    help="Epoch number (for display)")

    # Locked retrieval knobs
    ap.add_argument("--nprobe", type=int, default=64,
                    help="FAISS nprobe (locked: 64)")

    # Unified gates
    ap.add_argument("--gate-contain50", type=float, default=0.60,
                    help="Contain@50 gate (default: 60%%)")
    ap.add_argument("--gate-eff5", type=float, default=0.68,
                    help="Eff@5 gate (default: 68%%)")
    ap.add_argument("--gate-r5", type=float, default=0.40,
                    help="R@5 gate (default: 40%%)")
    ap.add_argument("--gate-p95", type=float, default=1.45,
                    help="P95 latency gate in ms (default: 1.45)")

    # Reranker parameters (placeholders for future integration)
    ap.add_argument("--rerank-mmr", type=float, default=0.7,
                    help="MMR lambda (locked: 0.7) [NOT YET INTEGRATED]")
    ap.add_argument("--seq-bias-same", type=float, default=0.05,
                    help="Same-article weight (locked: 0.05) [NOT YET INTEGRATED]")
    ap.add_argument("--seq-bias-gap", type=float, default=0.12,
                    help="Next-gap weight (locked: 0.12) [NOT YET INTEGRATED]")
    ap.add_argument("--directional", type=float, default=0.03,
                    help="Directional bonus (locked: 0.03) [NOT YET INTEGRATED]")
    ap.add_argument("--shard-assist", type=str, default="false",
                    choices=["true", "false"],
                    help="Shard-assist (locked: true) [NOT YET INTEGRATED]")

    ap.add_argument("--out", type=Path, default=None,
                    help="Output JSON file")

    args = ap.parse_args()

    print("=" * 80)
    print("UNIFIED CHECKPOINT EVALUATION")
    print("=" * 80)
    print(f"Checkpoint: {args.checkpoint}")
    if args.epoch:
        print(f"Epoch: {args.epoch}")
    print(f"Samples: {args.limit}")
    print(f"Device: {args.device}")
    print("=" * 80)

    print("\nLocked Retrieval Knobs:")
    print(f"  nprobe: {args.nprobe}")
    print(f"  MMR λ: {args.rerank_mmr} (reranker not yet integrated)")
    print(f"  seq-bias same: {args.seq_bias_same} (not yet integrated)")
    print(f"  seq-bias gap: {args.seq_bias_gap} (not yet integrated)")
    print(f"  directional: {args.directional} (not yet integrated)")
    print(f"  shard-assist: {args.shard_assist} (not yet integrated)")

    print("\nUnified Gates:")
    print(f"  Contain@50 ≥ {100*args.gate_contain50:.0f}%")
    print(f"  Eff@5 ≥ {100*args.gate_eff5:.0f}%")
    print(f"  R@5 ≥ {100*args.gate_r5:.0f}%")
    print(f"  P95 ≤ {args.gate_p95:.2f}ms")

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

    # Provenance check
    if not check_provenance(eval_data, args.payload):
        print("\n❌ PROVENANCE CHECK FAILED! Aborting.")
        return 1

    # Truth-to-payload alignment check
    aligned, tp_mean, tp_p5 = verify_truth_payload_alignment(
        targets, truth_keys, payload
    )
    if not aligned:
        print("\n❌ TRUTH-PAYLOAD ALIGNMENT FAILED! Aborting.")
        return 1

    # Load model
    print("\nLoading checkpoint...")
    model, checkpoint = load_checkpoint(args.checkpoint, args.device)
    print(f"  Model type: {checkpoint['args']['model_type']}")
    print(f"  Training epoch: {checkpoint.get('epoch', 'unknown')}")
    print(f"  Val cosine: {checkpoint.get('val_cosine', 0.0):.4f}")

    # Generate predictions
    print(f"\nGenerating predictions...")
    start = time.time()
    pred_vecs = generate_predictions(model, contexts, args.device)
    pred_time = time.time() - start
    print(f"  Generated {len(pred_vecs)} in {pred_time:.2f}s ({len(pred_vecs)/pred_time:.1f} seq/s)")

    # Validation cosine
    cosines = [np.dot(pred_vecs[i], targets[i]) for i in range(len(pred_vecs))]
    val_mean = np.mean(cosines)
    print(f"  Val cosine (pred vs target): mean={val_mean:.4f}")

    # Run retrieval evaluation
    print(f"\n" + "=" * 80)
    print("RETRIEVAL EVALUATION")
    print("=" * 80)

    results = evaluate_retrieval(
        pred_vecs, truth_keys, payload, index,
        nprobe=args.nprobe, K=50
    )

    # Standardized single-line output (contractor spec)
    c20 = results['Contain@20'] / 100
    c50 = results['Contain@50'] / 100
    r1 = results['R@1'] / 100
    r5 = results['R@5'] / 100
    r10 = results['R@10'] / 100
    eff5 = results['Eff@5'] / 100
    p95 = results['latency_p95_ms']

    print(f"\n[EVAL] Contain@20={c20:.3f} Contain@50={c50:.3f} "
          f"R@1={r1:.3f} R@5={r5:.3f} R@10={r10:.3f} "
          f"Eff@5={eff5:.3f} P95={p95:.2f}ms  "
          f"truth→payload_cos(mean={tp_mean:.3f}, p5={tp_p5:.3f})")

    # Detailed output
    print(f"\nDetailed Metrics:")
    print(f"  Contain@20:   {results['Contain@20']:6.2f}%")
    print(f"  Contain@50:   {results['Contain@50']:6.2f}%")
    print(f"  R@1:          {results['R@1']:6.2f}%")
    print(f"  R@5:          {results['R@5']:6.2f}%")
    print(f"  R@10:         {results['R@10']:6.2f}%")
    print(f"  R@20:         {results['R@20']:6.2f}%")
    print(f"  Eff@5:        {results['Eff@5']:6.2f}%")
    print(f"\nLatency:")
    print(f"  Mean:         {results['latency_mean_ms']:6.2f} ms")
    print(f"  P50:          {results['latency_p50_ms']:6.2f} ms")
    print(f"  P95:          {results['latency_p95_ms']:6.2f} ms")
    print(f"  P99:          {results['latency_p99_ms']:6.2f} ms")

    # Check gates
    print(f"\n" + "=" * 80)
    print("GATE CHECK")
    print("=" * 80)

    gates_passed = []
    gates_failed = []

    # Unified gates
    if results['Contain@50'] / 100 >= args.gate_contain50:
        gates_passed.append(f"✅ Contain@50: {results['Contain@50']:.1f}% ≥ {100*args.gate_contain50:.0f}%")
    else:
        gates_failed.append(f"❌ Contain@50: {results['Contain@50']:.1f}% < {100*args.gate_contain50:.0f}%")

    if results['Eff@5'] / 100 >= args.gate_eff5:
        gates_passed.append(f"✅ Eff@5: {results['Eff@5']:.1f}% ≥ {100*args.gate_eff5:.0f}%")
    else:
        gates_failed.append(f"❌ Eff@5: {results['Eff@5']:.1f}% < {100*args.gate_eff5:.0f}%")

    if results['R@5'] / 100 >= args.gate_r5:
        gates_passed.append(f"✅ R@5: {results['R@5']:.1f}% ≥ {100*args.gate_r5:.0f}%")
    else:
        gates_failed.append(f"❌ R@5: {results['R@5']:.1f}% < {100*args.gate_r5:.0f}%")

    if results['latency_p95_ms'] <= args.gate_p95:
        gates_passed.append(f"✅ P95: {results['latency_p95_ms']:.2f}ms ≤ {args.gate_p95:.2f}ms")
    else:
        gates_failed.append(f"❌ P95: {results['latency_p95_ms']:.2f}ms > {args.gate_p95:.2f}ms")

    # Print gate results
    for gate in gates_passed:
        print(gate)
    for gate in gates_failed:
        print(gate)

    print("=" * 80)

    if len(gates_failed) == 0:
        print("✅ ALL GATES PASSED")
        exit_code = 0
    else:
        print(f"❌ {len(gates_failed)}/{len(gates_passed)+len(gates_failed)} GATES FAILED")
        exit_code = 1

    print("=" * 80)

    # Save results
    if args.out:
        results['checkpoint'] = str(args.checkpoint)
        results['epoch'] = args.epoch
        results['limit'] = args.limit
        results['val_cosine'] = float(checkpoint.get('val_cosine', 0.0))
        results['val_cosine_pred_target'] = float(val_mean)
        results['truth_payload_cos_mean'] = float(tp_mean)
        results['truth_payload_cos_p5'] = float(tp_p5)
        results['gates_passed'] = gates_passed
        results['gates_failed'] = gates_failed
        results['retrieval_knobs'] = {
            'nprobe': args.nprobe,
            'mmr_lambda': args.rerank_mmr,
            'seq_bias_same': args.seq_bias_same,
            'seq_bias_gap': args.seq_bias_gap,
            'directional': args.directional,
            'shard_assist': args.shard_assist,
        }

        with open(args.out, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {args.out}")

    return exit_code


if __name__ == "__main__":
    exit(main())
