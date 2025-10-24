#!/usr/bin/env python3
"""
Prepare v2-Compatible Evaluation Dataset
=========================================

Creates NPZ file for eval_retrieval_v2.py with:
- pred_vecs: [N, 768] AMN predicted vectors (L2-normalized)
- last_meta: [N] list of dicts with {"article_index": int, "chunk_index": int}
- truth_keys: [N, 2] ground truth (article_index, chunk_index) pairs

Uses synthetic metadata from TMD lanes:
- article_index = lane number (extracted from "lane_N")
- chunk_index = lane_index (position within lane)

Contract checks:
1. pred_vecs.shape == (N, 768)
2. Coverage ≥ 95%: truth_keys must exist in retrieval payload
3. Split hygiene: no target chunk in last context
"""

import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import sys
import argparse

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

    return model


def run_model_inference(model: nn.Module, contexts: np.ndarray, device: torch.device) -> np.ndarray:
    """
    Run AMN inference on context sequences.

    Args:
        model: AMN model
        contexts: [N, 5, 768] context sequences
        device: torch device

    Returns:
        [N, 768] L2-normalized predicted vectors
    """
    pred_vecs = []
    batch_size = 64

    print(f"Running AMN inference on {len(contexts):,} sequences...")
    with torch.no_grad():
        for i in range(0, len(contexts), batch_size):
            batch = contexts[i:i+batch_size]
            batch_t = torch.from_numpy(batch).float().to(device)

            # Forward pass
            preds = model(batch_t).cpu().numpy()

            # L2 normalize
            preds = preds / (np.linalg.norm(preds, axis=1, keepdims=True) + 1e-8)
            pred_vecs.append(preds)

            if (i // batch_size + 1) % 10 == 0:
                print(f"  Processed {i + len(batch):,} / {len(contexts):,}")

    return np.vstack(pred_vecs).astype(np.float32)


def build_payload_with_metadata(vectors_path: str) -> tuple:
    """
    Build payload mapping from Wikipedia vectors NPZ.

    Returns:
        (payload_dict, lane_to_article_map)
        - payload_dict: {idx: (text, meta, vec)}
        - lane_to_article_map: {lane_name: article_index}
    """
    print(f"\nLoading Wikipedia vectors from {vectors_path}...")
    data = np.load(vectors_path, allow_pickle=True)

    concept_texts = data['concept_texts']
    vectors = data['vectors']
    tmd_lanes = data['tmd_lanes']
    lane_indices = data['lane_indices']

    print(f"Loaded {len(concept_texts):,} concepts")

    # Build lane-to-article mapping
    unique_lanes = sorted(set(tmd_lanes), key=lambda x: extract_lane_number(x))
    lane_to_article = {lane: i for i, lane in enumerate(unique_lanes)}

    print(f"Found {len(unique_lanes)} unique TMD lanes")

    # Build payload with synthetic metadata
    payload = {}
    for idx in range(len(concept_texts)):
        lane = tmd_lanes[idx]
        article_idx = lane_to_article.get(lane, -1)
        chunk_idx = int(lane_indices[idx])

        meta = {
            "article_index": article_idx,
            "chunk_index": chunk_idx,
            "tmd_lane": lane,
            "cpe_id": data['cpe_ids'][idx] if 'cpe_ids' in data else None
        }

        payload[idx] = (
            concept_texts[idx],  # text
            meta,                # metadata
            vectors[idx]         # vector [768]
        )

    return payload, lane_to_article


def build_last_meta_and_truth_keys(
    contexts: np.ndarray,
    targets: np.ndarray,
    vectors: np.ndarray,
    tmd_lanes: np.ndarray,
    lane_indices: np.ndarray,
    lane_to_article: dict
) -> tuple:
    """
    Build last_meta and truth_keys from test sequences.

    Strategy:
    - Find which vectors in the corpus match the last context vector
    - Use that vector's metadata as last_meta
    - Find which vector matches the target
    - Use that vector's metadata as truth_key

    Returns:
        (last_meta, truth_keys)
    """
    print("\nBuilding metadata for test sequences...")

    # Normalize all vectors for cosine similarity
    contexts_norm = contexts / (np.linalg.norm(contexts, axis=2, keepdims=True) + 1e-8)
    targets_norm = targets / (np.linalg.norm(targets, axis=1, keepdims=True) + 1e-8)
    vectors_norm = vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-8)

    last_meta = []
    truth_keys = []

    for i in range(len(contexts)):
        # Find match for last context vector
        last_vec = contexts_norm[i, -1, :]  # Last of 5 context vectors
        sims_last = vectors_norm @ last_vec
        best_last_idx = int(np.argmax(sims_last))

        # Find match for target vector
        target_vec = targets_norm[i, :]
        sims_target = vectors_norm @ target_vec
        best_target_idx = int(np.argmax(sims_target))

        # Build last_meta
        last_lane = tmd_lanes[best_last_idx]
        last_meta.append({
            "article_index": lane_to_article.get(last_lane, -1),
            "chunk_index": int(lane_indices[best_last_idx])
        })

        # Build truth_key
        target_lane = tmd_lanes[best_target_idx]
        truth_keys.append([
            lane_to_article.get(target_lane, -1),
            int(lane_indices[best_target_idx])
        ])

        if (i + 1) % 1000 == 0:
            print(f"  Processed {i + 1:,} / {len(contexts):,}")

    return np.array(last_meta, dtype=object), np.array(truth_keys, dtype=np.int32)


def check_coverage(truth_keys: np.ndarray, payload: dict) -> float:
    """Check what fraction of truth_keys exist in payload."""
    have = 0
    for article_idx, chunk_idx in truth_keys:
        # Check if any payload entry matches this (article, chunk)
        for text, meta, vec in payload.values():
            if (meta["article_index"] == int(article_idx) and
                meta["chunk_index"] == int(chunk_idx)):
                have += 1
                break

    coverage = have / len(truth_keys)
    return coverage


def check_split_hygiene(last_meta: np.ndarray, truth_keys: np.ndarray) -> int:
    """Check for samples where target chunk appears in last context."""
    bad = 0
    for lm, tk in zip(last_meta, truth_keys):
        if (lm["article_index"] == int(tk[0]) and
            lm["chunk_index"] == int(tk[1])):
            bad += 1
    return bad


def main():
    ap = argparse.ArgumentParser(description="Prepare v2-compatible evaluation dataset")
    ap.add_argument("--model", type=str, default="artifacts/lvm/production_model",
                    help="Path to AMN model directory")
    ap.add_argument("--ood_npz", type=str, default="artifacts/lvm/wikipedia_ood_test_ctx5.npz",
                    help="OOD test sequences NPZ")
    ap.add_argument("--vectors_npz", type=str, default="artifacts/wikipedia_500k_corrected_vectors.npz",
                    help="Wikipedia vectors NPZ")
    ap.add_argument("--out_npz", type=str, default="artifacts/lvm/wikipedia_ood_test_ctx5_v2.npz",
                    help="Output v2-compatible NPZ")
    ap.add_argument("--out_payload", type=str, default="artifacts/wikipedia_500k_payload.npy",
                    help="Output payload mapping (for eval_retrieval_v2.py)")
    args = ap.parse_args()

    # Device setup
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    # Load AMN model
    print(f"\nLoading AMN model from {args.model}...")
    model = load_amn_model(args.model, device)

    # Load OOD test sequences
    print(f"\nLoading OOD test sequences from {args.ood_npz}...")
    ood_data = np.load(args.ood_npz, allow_pickle=True)
    contexts = ood_data['context_sequences']  # (N, 5, 768)
    targets = ood_data['target_vectors']      # (N, 768)
    print(f"Loaded {len(contexts):,} test sequences")

    # Run AMN inference
    pred_vecs = run_model_inference(model, contexts, device)
    assert pred_vecs.ndim == 2 and pred_vecs.shape[1] == 768, \
        f"Expected (N, 768), got {pred_vecs.shape}"
    print(f"✅ Contract check 1: pred_vecs.shape = {pred_vecs.shape}")

    # Build payload with metadata
    payload, lane_to_article = build_payload_with_metadata(args.vectors_npz)

    # Load vectors data for metadata matching
    vec_data = np.load(args.vectors_npz, allow_pickle=True)

    # Build last_meta and truth_keys
    last_meta, truth_keys = build_last_meta_and_truth_keys(
        contexts, targets,
        vec_data['vectors'],
        vec_data['tmd_lanes'],
        vec_data['lane_indices'],
        lane_to_article
    )

    # Contract check: coverage
    coverage = check_coverage(truth_keys, payload)
    print(f"\n✅ Contract check 2: Coverage = {coverage:.2%}")
    if coverage < 0.95:
        print(f"⚠️  Warning: Coverage {coverage:.2%} < 95%. Recall will be capped!")

    # Contract check: split hygiene
    bad = check_split_hygiene(last_meta, truth_keys)
    print(f"✅ Contract check 3: Split hygiene = {bad} leaks / {len(truth_keys)} samples")
    if bad > 0:
        print(f"⚠️  Warning: Found {bad} samples with target chunk in last context")

    # Save v2-compatible NPZ
    print(f"\nSaving v2-compatible NPZ to {args.out_npz}...")
    Path(args.out_npz).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out_npz,
        pred_vecs=pred_vecs,
        last_meta=last_meta,
        truth_keys=truth_keys
    )
    print(f"✅ Saved {args.out_npz}")

    # Save payload mapping
    print(f"\nSaving payload mapping to {args.out_payload}...")
    Path(args.out_payload).parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out_payload, payload, allow_pickle=True)
    print(f"✅ Saved {args.out_payload}")

    print("\n" + "=" * 80)
    print("✅ v2-Compatible Evaluation Dataset Ready!")
    print("=" * 80)
    print(f"Test samples: {len(pred_vecs):,}")
    print(f"Coverage: {coverage:.2%}")
    print(f"Split leaks: {bad} / {len(truth_keys)}")
    print("\nNext steps:")
    print(f"  1. Run baseline: python tools/eval_retrieval_v2.py --npz {args.out_npz} --payload {args.out_payload} --faiss <index> --no_mmr --out artifacts/lvm/eval_baseline_v2.json")
    print(f"  2. Run improved: python tools/eval_retrieval_v2.py --npz {args.out_npz} --payload {args.out_payload} --faiss <index> --directional_bonus 0.03 --out artifacts/lvm/eval_improved_v2.json")


if __name__ == "__main__":
    main()
