#!/usr/bin/env python3
"""
Evaluate Mamba Phase-5 Models
==============================

Generates predictions from Mamba models and evaluates retrieval performance.

Usage:
    python tools/eval_mamba_models.py \
        --model artifacts/lvm/models/mamba_s/best.pt \
        --eval-npz artifacts/lvm/eval_v2_ready_aligned.npz \
        --payload artifacts/wikipedia_584k_payload.npy \
        --faiss artifacts/wikipedia_584k_ivf_flat_ip.index \
        --out artifacts/lvm/eval_mamba_s_full.json
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# Add project root
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from app.lvm.mamba import create_model


def load_model(checkpoint_path: Path, device: str):
    """Load Mamba model from checkpoint."""
    print(f"Loading model from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    args = checkpoint['args']

    # Build kwargs based on model type
    model_type = args['model_type']

    # Common parameters
    model_kwargs = {
        'd_model': args['d_model'],
        'd_state': args['d_state'],
        'conv_sz': args['conv_sz'],
        'expand': args['expand'],
        'dropout': args['dropout'],
    }

    # Add model-type-specific parameters
    if 'sandwich' in model_type:
        # Sandwich uses n_layers_mamba and n_layers_local, NOT n_layers
        model_kwargs.update({
            'n_layers_mamba': args.get('n_layers_mamba', 8),
            'n_layers_local': args.get('n_layers_local', 4),
            'local_attn_win': args.get('local_attn_win', 8),
            'n_heads': args.get('n_heads', 4),
        })
    elif 'hybrid' in model_type:
        # Hybrid uses n_layers + local attention params
        model_kwargs.update({
            'n_layers': args['n_layers'],
            'local_attn_win': args.get('local_attn_win', 8),
            'local_attn_every': args.get('local_attn_every', 4),
            'n_heads': args.get('n_heads', 4),
        })
    else:
        # Pure SSM models (mamba_s, mamba_xl, mamba_gr) use n_layers
        model_kwargs['n_layers'] = args['n_layers']

    # Add GR-specific parameters
    if 'gr' in model_type:
        model_kwargs['gru_hidden'] = args.get('gru_hidden', 256)

    # Add alignment head parameters (optional for all)
    model_kwargs.update({
        'use_alignment_head': args.get('use_alignment_head', False),
        'alignment_alpha': args.get('alignment_alpha', 0.25),
    })

    model = create_model(model_type=model_type, **model_kwargs)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    print(f"  Model type: {args['model_type']}")
    print(f"  Val cosine: {checkpoint['val_cosine']:.4f}")
    print(f"  Epoch: {checkpoint['epoch']}")

    return model, args


@torch.no_grad()
def generate_predictions(model, contexts, device, batch_size=128):
    """
    Generate predictions for all context sequences.

    Args:
        model: Mamba model
        contexts: [N, 5, 768] context sequences
        device: Device to run on
        batch_size: Batch size for inference

    Returns:
        pred_vecs: [N, 768] predicted vectors
    """
    print(f"\nGenerating predictions for {len(contexts)} sequences...")
    print(f"  Batch size: {batch_size}")
    print(f"  Device: {device}")

    contexts_tensor = torch.from_numpy(contexts).float()
    all_preds = []

    start_time = time.time()

    for i in range(0, len(contexts), batch_size):
        batch = contexts_tensor[i:i+batch_size].to(device)

        # Forward pass
        preds = model(batch)  # [B, seq_len, 768]

        # Debug: print shape on first batch
        if i == 0:
            print(f"  First batch model output shape: {preds.shape}")

        # Take only the last prediction in the sequence
        if len(preds.shape) == 3:
            preds = preds[:, -1, :]  # [B, 768]
            if i == 0:
                print(f"  After taking last element: {preds.shape}")

        # L2 normalize
        preds = F.normalize(preds, p=2, dim=1)

        if i == 0:
            print(f"  After normalization: {preds.shape}")

        all_preds.append(preds.cpu().numpy())

        if (i // batch_size) % 10 == 0:
            elapsed = time.time() - start_time
            progress = (i + batch_size) / len(contexts)
            eta = elapsed / progress - elapsed if progress > 0 else 0
            print(f"  Progress: {i}/{len(contexts)} ({progress*100:.1f}%) | "
                  f"Elapsed: {elapsed:.1f}s | ETA: {eta:.1f}s")

    pred_vecs = np.concatenate(all_preds, axis=0)

    elapsed = time.time() - start_time
    throughput = len(contexts) / elapsed
    print(f"\n  Generated {len(pred_vecs)} predictions in {elapsed:.1f}s")
    print(f"  Final pred_vecs shape: {pred_vecs.shape} (expected: [{len(contexts)}, 768])")
    print(f"  Throughput: {throughput:.1f} seq/s")
    print(f"  Latency: {1000*elapsed/len(contexts):.2f} ms/seq")

    # Verify shape is correct
    if pred_vecs.shape != (len(contexts), 768):
        raise ValueError(f"Prediction shape mismatch! Got {pred_vecs.shape}, expected ({len(contexts)}, 768)")

    return pred_vecs


def main():
    ap = argparse.ArgumentParser(description="Evaluate Mamba models on retrieval task")
    ap.add_argument("--model", type=Path, required=True,
                    help="Path to model checkpoint (best.pt)")
    ap.add_argument("--eval-npz", type=Path, required=True,
                    help="Path to evaluation NPZ (with contexts)")
    ap.add_argument("--payload", type=Path, required=True,
                    help="Path to payload NPY file")
    ap.add_argument("--faiss", type=Path, required=True,
                    help="Path to FAISS index")
    ap.add_argument("--shards", type=Path, default=Path("artifacts/article_shards.pkl"),
                    help="Path to article shards pickle")
    ap.add_argument("--device", type=str, default="mps",
                    choices=["cpu", "cuda", "mps"],
                    help="Device to run inference on")
    ap.add_argument("--batch-size", type=int, default=128,
                    help="Batch size for inference")
    ap.add_argument("--out", type=Path, required=True,
                    help="Output JSON file for results")
    ap.add_argument("--nprobe", type=int, default=64,
                    help="FAISS nprobe parameter")

    args = ap.parse_args()

    print("=" * 80)
    print("Mamba Model Evaluation")
    print("=" * 80)
    print(f"Model: {args.model}")
    print(f"Eval NPZ: {args.eval_npz}")
    print(f"Device: {args.device}")
    print("=" * 80)

    # Load model
    model, model_args = load_model(args.model, args.device)

    # Load evaluation data
    print(f"\nLoading evaluation data from: {args.eval_npz}")
    eval_data = np.load(args.eval_npz, allow_pickle=True)
    contexts = eval_data['contexts']  # [N, 5, 768]
    last_meta = eval_data['last_meta']
    truth_keys = eval_data['truth_keys']

    print(f"  Contexts: {contexts.shape}")
    print(f"  Last meta: {len(last_meta)}")
    print(f"  Truth keys: {truth_keys.shape}")

    # Generate predictions
    pred_vecs = generate_predictions(model, contexts, args.device, args.batch_size)

    # Save temporary NPZ with predictions
    temp_npz = args.out.parent / f"{args.out.stem}_pred.npz"
    print(f"\nSaving predictions to: {temp_npz}")
    np.savez(
        temp_npz,
        pred_vecs=pred_vecs,
        last_meta=last_meta,
        truth_keys=truth_keys,
    )

    # Run evaluation using eval_retrieval_v2.py
    print("\n" + "=" * 80)
    print("Running retrieval evaluation...")
    print("=" * 80)

    import subprocess
    cmd = [
        sys.executable,
        "tools/eval_retrieval_v2.py",
        "--npz", str(temp_npz),
        "--payload", str(args.payload),
        "--faiss", str(args.faiss),
        "--out", str(args.out),
    ]

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False)

    if result.returncode != 0:
        print(f"\n⚠️  Evaluation failed with return code: {result.returncode}")
        sys.exit(1)

    # Load and display results
    print("\n" + "=" * 80)
    print("Results")
    print("=" * 80)

    with open(args.out) as f:
        results = json.load(f)

    print(json.dumps(results, indent=2))

    # Clean up temp file
    if temp_npz.exists():
        temp_npz.unlink()
        print(f"\nCleaned up: {temp_npz}")

    print("\n" + "=" * 80)
    print(f"✅ Evaluation complete! Results saved to: {args.out}")
    print("=" * 80)


if __name__ == "__main__":
    main()
