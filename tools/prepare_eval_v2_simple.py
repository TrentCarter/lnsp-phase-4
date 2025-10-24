#!/usr/bin/env python3
"""
Prepare v2-Compatible Evaluation Dataset (Simple & Fast)
=========================================================

Optimized version that:
1. Reuses existing FAISS index (no rebuild needed)
2. Efficient metadata building with progress output
3. Optimized coverage check using dict lookup
4. Unbuffered output for real-time monitoring
"""

import numpy as np
import torch
import torch.nn as nn
import faiss
from pathlib import Path
import sys

# Import AMN model
sys.path.insert(0, 'app/lvm')
from models import AttentionMixtureNetwork


def extract_lane_number(lane_name: str) -> int:
    """Extract lane number from 'lane_N' string."""
    if lane_name.startswith('lane_'):
        try:
            return int(lane_name.split('_')[1])
        except (IndexError, ValueError):
            return -1
    return -1


def load_amn_model(model_path: str, device: torch.device) -> nn.Module:
    """Load AMN production model."""
    print(f"[1/7] Loading AMN model from {model_path}...", flush=True)
    checkpoint_path = Path(model_path) / "best_model.pt"
    checkpoint = torch.load(checkpoint_path, map_location=device)
    hparams = checkpoint.get('hyperparameters', {})

    model = AttentionMixtureNetwork(
        input_dim=hparams.get('input_dim', 768),
        d_model=hparams.get('d_model', 256),
        hidden_dim=hparams.get('hidden_dim', 512)
    )

    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    print("  ✅ Model loaded", flush=True)
    return model


def run_amn_inference(model: nn.Module, contexts: np.ndarray, device: torch.device) -> np.ndarray:
    """Run AMN inference with progress output."""
    print(f"[2/7] Running AMN inference on {len(contexts):,} sequences...", flush=True)
    pred_vecs = []
    batch_size = 64

    with torch.no_grad():
        for i in range(0, len(contexts), batch_size):
            batch = contexts[i:i+batch_size]
            batch_t = torch.from_numpy(batch).float().to(device)
            preds = model(batch_t).cpu().numpy()

            # L2 normalize
            preds = preds / (np.linalg.norm(preds, axis=1, keepdims=True) + 1e-8)
            pred_vecs.append(preds)

            if (i // batch_size + 1) % 20 == 0:
                print(f"  Progress: {i + len(batch):,} / {len(contexts):,}", flush=True)

    result = np.vstack(pred_vecs).astype(np.float32)
    print(f"  ✅ Inference complete: {result.shape}", flush=True)
    return result


def build_payload_and_lane_map(vectors_path: str) -> tuple:
    """Build payload and article mapping efficiently."""
    print(f"[3/7] Loading Wikipedia vectors from {vectors_path}...", flush=True)
    data = np.load(vectors_path, allow_pickle=True)

    # Convert NumPy arrays to lists for FAST indexing (avoid memory-mapped slowness)
    print("  Converting arrays to lists for fast access...", flush=True)
    concept_texts = list(data['concept_texts'])
    vectors = list(data['vectors'])
    article_indices = list(data['article_indices'])
    chunk_indices = list(data['chunk_indices'])
    cpe_ids = list(data['cpe_ids']) if 'cpe_ids' in data else [None] * len(concept_texts)

    print(f"  Loaded {len(concept_texts):,} concepts", flush=True)

    # Build payload efficiently (now much faster with lists!)
    print("  Building payload dictionary...", flush=True)
    payload = {}
    for idx in range(len(concept_texts)):
        meta = {
            "article_index": int(article_indices[idx]),
            "chunk_index": int(chunk_indices[idx]),
            "cpe_id": cpe_ids[idx]
        }
        payload[idx] = (concept_texts[idx], meta, vectors[idx])

        if (idx + 1) % 100000 == 0:
            print(f"    Progress: {idx + 1:,} / {len(concept_texts):,}", flush=True)

    print(f"  ✅ Payload built with {len(payload):,} entries", flush=True)
    return payload, data


def match_vectors_with_faiss(
    contexts: np.ndarray,
    targets: np.ndarray,
    faiss_index: faiss.Index,
    article_indices: np.ndarray,
    chunk_indices: np.ndarray
) -> tuple:
    """Use FAISS to match context/target vectors to corpus."""
    print("[4/7] Matching test vectors to corpus using FAISS...", flush=True)

    # Normalize and prepare queries
    contexts_norm = contexts / (np.linalg.norm(contexts, axis=2, keepdims=True) + 1e-8)
    targets_norm = targets / (np.linalg.norm(targets, axis=1, keepdims=True) + 1e-8)

    last_vecs = contexts_norm[:, -1, :].astype(np.float32)
    targets_norm = targets_norm.astype(np.float32)

    # Search FAISS
    print("  Searching for last context matches...", flush=True)
    _, last_indices = faiss_index.search(last_vecs, 1)
    last_indices = last_indices[:, 0]

    print("  Searching for target matches...", flush=True)
    _, target_indices = faiss_index.search(targets_norm, 1)
    target_indices = target_indices[:, 0]

    # Build metadata arrays
    print("  Building metadata arrays...", flush=True)
    last_meta = []
    truth_keys = []

    for i in range(len(contexts)):
        last_idx = int(last_indices[i])
        target_idx = int(target_indices[i])

        last_meta.append({
            "article_index": int(article_indices[last_idx]),
            "chunk_index": int(chunk_indices[last_idx])
        })

        truth_keys.append([
            int(article_indices[target_idx]),
            int(chunk_indices[target_idx])
        ])

    print(f"  ✅ Metadata built for {len(contexts):,} sequences", flush=True)
    return np.array(last_meta, dtype=object), np.array(truth_keys, dtype=np.int32)


def check_coverage_fast(truth_keys: np.ndarray, payload: dict) -> float:
    """Fast coverage check using set lookup."""
    print("[5/7] Checking coverage...", flush=True)

    # Build set of (article_idx, chunk_idx) from payload
    payload_keys = {(meta["article_index"], meta["chunk_index"]) for _, meta, _ in payload.values()}

    # Count matches
    matches = sum(1 for article_idx, chunk_idx in truth_keys
                  if (int(article_idx), int(chunk_idx)) in payload_keys)

    coverage = matches / len(truth_keys)
    print(f"  ✅ Coverage: {coverage:.2%} ({matches}/{len(truth_keys)})", flush=True)
    return coverage


def check_split_hygiene(last_meta: np.ndarray, truth_keys: np.ndarray) -> int:
    """Check for leaks."""
    print("[6/7] Checking split hygiene...", flush=True)

    bad = sum(1 for lm, tk in zip(last_meta, truth_keys)
              if lm["article_index"] == int(tk[0]) and lm["chunk_index"] == int(tk[1]))

    print(f"  ✅ Split hygiene: {bad} leaks / {len(truth_keys)} samples", flush=True)
    return bad


def main():
    import argparse

    ap = argparse.ArgumentParser(description="Prepare v2 eval dataset (simple & fast)")
    ap.add_argument("--model", type=str, default="artifacts/lvm/production_model")
    ap.add_argument("--ood_npz", type=str, default="artifacts/lvm/wikipedia_ood_test_ctx5.npz")
    ap.add_argument("--vectors_npz", type=str, default="artifacts/wikipedia_500k_corrected_vectors.npz")
    ap.add_argument("--faiss_index", type=str, default="artifacts/wikipedia_500k_corrected_ivf_flat_ip.index")
    ap.add_argument("--out_npz", type=str, default="artifacts/lvm/wikipedia_ood_test_ctx5_v2.npz")
    ap.add_argument("--out_payload", type=str, default="artifacts/wikipedia_500k_payload.npy")
    args = ap.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}\n", flush=True)

    # Load model
    model = load_amn_model(args.model, device)

    # Load OOD test data
    print(f"\nLoading OOD test sequences...", flush=True)
    ood_data = np.load(args.ood_npz, allow_pickle=True)
    contexts = ood_data['context_sequences']
    targets = ood_data['target_vectors']
    print(f"  ✅ Loaded {len(contexts):,} test sequences\n", flush=True)

    # Run inference
    pred_vecs = run_amn_inference(model, contexts, device)
    assert pred_vecs.shape == (len(contexts), 768), f"Bad shape: {pred_vecs.shape}"
    print(f"\n✅ Contract check 1: pred_vecs.shape = {pred_vecs.shape}\n", flush=True)

    # Build payload
    payload, vec_data = build_payload_and_lane_map(args.vectors_npz)

    # Load FAISS index
    print(f"\n[4/7] Loading FAISS index from {args.faiss_index}...", flush=True)
    faiss_index = faiss.read_index(args.faiss_index)

    # Set nprobe for IVF index
    if hasattr(faiss_index, 'nprobe'):
        faiss_index.nprobe = 32  # Reasonable default for accuracy
        print(f"  Set nprobe=32 for IVF index", flush=True)

    print(f"  ✅ Loaded index with {faiss_index.ntotal:,} vectors\n", flush=True)

    # Match vectors
    last_meta, truth_keys = match_vectors_with_faiss(
        contexts, targets, faiss_index,
        vec_data['article_indices'], vec_data['chunk_indices']
    )

    # Coverage check
    print()
    coverage = check_coverage_fast(truth_keys, payload)
    if coverage < 0.95:
        print(f"  ⚠️  Warning: Coverage {coverage:.2%} < 95%!", flush=True)

    # Split hygiene
    print()
    bad = check_split_hygiene(last_meta, truth_keys)
    if bad > 0:
        print(f"  ⚠️  Warning: {bad} split leaks!", flush=True)

    # Save outputs
    print(f"\n[7/7] Saving outputs...", flush=True)
    Path(args.out_npz).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out_npz, pred_vecs=pred_vecs, last_meta=last_meta, truth_keys=truth_keys)
    print(f"  ✅ Saved {args.out_npz}", flush=True)

    Path(args.out_payload).parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out_payload, payload, allow_pickle=True)
    print(f"  ✅ Saved {args.out_payload}", flush=True)

    print("\n" + "=" * 80, flush=True)
    print("✅ v2 EVALUATION DATASET READY!", flush=True)
    print("=" * 80, flush=True)
    print(f"Test samples: {len(pred_vecs):,}", flush=True)
    print(f"Coverage: {coverage:.2%}", flush=True)
    print(f"Split leaks: {bad}", flush=True)


if __name__ == "__main__":
    main()
