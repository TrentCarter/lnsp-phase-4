#!/usr/bin/env python3
"""
P1 Tool: Filter Sequences
==========================

Applies quality mask to filter NPZ training data.

Usage:
    ./.venv/bin/python tools/filter_sequences.py \
      --input artifacts/lvm/training_sequences_ctx5.npz \
      --mask artifacts/lvm/quality_mask_790k.npy \
      --out artifacts/lvm/training_sequences_ctx5_filtered.npz
"""

import sys
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime

def main():
    parser = argparse.ArgumentParser(description="Filter sequences using quality mask")
    parser.add_argument("--input", required=True, help="Input NPZ file")
    parser.add_argument("--mask", required=True, help="Quality mask (.npy)")
    parser.add_argument("--out", required=True, help="Output filtered NPZ")
    args = parser.parse_args()

    input_path = Path(args.input)
    mask_path = Path(args.mask)
    output_path = Path(args.out)

    if not input_path.exists():
        print(f"❌ Input file not found: {input_path}")
        return 1

    if not mask_path.exists():
        print(f"❌ Mask file not found: {mask_path}")
        return 1

    print(f"{'='*60}")
    print(f"P1: FILTER SEQUENCES")
    print(f"{'='*60}")
    print(f"Input:  {input_path}")
    print(f"Mask:   {mask_path}")
    print(f"Output: {output_path}")

    # Load mask
    print(f"\n📂 Loading mask...")
    mask = np.load(mask_path)
    print(f"   Mask shape: {mask.shape}")
    print(f"   Keep ratio: {mask.sum() / len(mask):.1%}")

    # Load data
    print(f"\n📂 Loading input data...")
    data = np.load(input_path, allow_pickle=True)

    # Extract fields
    context_seqs = data['context_sequences']
    target_vecs = data['target_vectors']
    target_texts = data['target_texts']
    target_tmds = data['target_tmds']
    target_ids = data['target_ids']
    seq_positions = data['sequence_positions']

    N_original = len(context_seqs)
    print(f"   Original sequences: {N_original:,}")

    # Verify mask length
    if len(mask) != N_original:
        print(f"❌ Mask length ({len(mask)}) doesn't match data ({N_original})")
        return 1

    # Apply mask
    print(f"\n✂️  Applying filter...")
    filtered_context = context_seqs[mask]
    filtered_target_vecs = target_vecs[mask]
    filtered_target_texts = target_texts[mask]
    filtered_target_tmds = target_tmds[mask]
    filtered_target_ids = target_ids[mask]
    filtered_positions = seq_positions[mask]

    N_filtered = len(filtered_context)
    N_removed = N_original - N_filtered

    print(f"   Kept:     {N_filtered:7,} ({100*N_filtered/N_original:5.1f}%)")
    print(f"   Removed:  {N_removed:7,} ({100*N_removed/N_original:5.1f}%)")

    # Verify shapes
    print(f"\n✅ Filtered data shapes:")
    print(f"   context_sequences:  {filtered_context.shape}")
    print(f"   target_vectors:     {filtered_target_vecs.shape}")
    print(f"   target_texts:       {filtered_target_texts.shape}")
    print(f"   target_tmds:        {filtered_target_tmds.shape}")
    print(f"   target_ids:         {filtered_target_ids.shape}")
    print(f"   sequence_positions: {filtered_positions.shape}")

    # Verify quality improvement
    print(f"\n📊 Quality verification:")

    # Compute adjacency coherence
    adjacency_cos = np.sum(filtered_target_vecs[:-1] * filtered_target_vecs[1:], axis=1)
    mean_cos = adjacency_cos.mean()
    std_cos = adjacency_cos.std()
    p50 = np.percentile(adjacency_cos, 50)

    print(f"   Mean adjacency cosine: {mean_cos:.4f}")
    print(f"   Std adjacency cosine:  {std_cos:.4f}")
    print(f"   Median (p50):          {p50:.4f}")
    print(f"   Low coherence (<0.30): {(adjacency_cos < 0.30).sum() / len(adjacency_cos):.1%}")

    # Save filtered data
    print(f"\n💾 Saving filtered data...")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        output_path,
        context_sequences=filtered_context,
        target_vectors=filtered_target_vecs,
        target_texts=filtered_target_texts,
        target_tmds=filtered_target_tmds,
        target_ids=filtered_target_ids,
        sequence_positions=filtered_positions,
        metadata=np.array([{
            'original_sequences': int(N_original),
            'filtered_sequences': int(N_filtered),
            'filter_ratio': float(N_filtered / N_original),
            'mean_adjacency_cosine': float(mean_cos),
            'std_adjacency_cosine': float(std_cos),
            'dataset_source': 'wikipedia_790k_filtered',
            'filter_date': datetime.now().isoformat(),
        }])
    )

    print(f"   ✅ Saved to: {output_path}")

    # Verify saved file
    print(f"\n🔍 Verifying saved file...")
    verify = np.load(output_path, allow_pickle=True)
    print(f"   context_sequences: {verify['context_sequences'].shape}")
    print(f"   target_vectors:    {verify['target_vectors'].shape}")
    print(f"   metadata:          {verify['metadata']}")

    print(f"\n{'='*60}")
    print(f"✅ FILTERING COMPLETE")
    print(f"{'='*60}")
    print(f"Filtered dataset: {N_filtered:,} sequences")
    print(f"Mean coherence:   {mean_cos:.4f} (baseline 584k: 0.4842)")
    print(f"Quality gap:      {100*(0.4842 - mean_cos)/0.4842:.1f}% below baseline")

    if mean_cos >= 0.45:
        print(f"\n✅ HIGH QUALITY - Ready for training")
        print(f"   Expected: Similar performance to 584k baseline")
    elif mean_cos >= 0.40:
        print(f"\n⚠️  GOOD QUALITY - Worth trying")
        print(f"   Expected: Improved over unfiltered, may need ctx=7")
    else:
        print(f"\n❌ MODERATE QUALITY - Risky")
        print(f"   Recommendation: Stick with 584k or investigate further")

    return 0


if __name__ == '__main__':
    sys.exit(main())
