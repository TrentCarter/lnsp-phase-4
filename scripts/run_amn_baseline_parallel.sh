#!/bin/bash
# Run AMN baseline evaluation in parallel with Mamba training

echo "================================================================================"
echo "AMN Baseline Evaluation (Upper Bound for Mamba)"
echo "================================================================================"
echo "AMN uses two-tower contrastive → should generalize to unseen articles"
echo "Expected: ≥0.50 cosine, ≥60% Contain@50, ≥40% R@5"
echo ""

KMP_DUPLICATE_LIB_OK=TRUE ./.venv/bin/python tools/eval_amn_baseline.py \
  --eval-npz artifacts/lvm/eval_v2_payload_aligned.npz \
  --payload artifacts/wikipedia_584k_payload.npy \
  --faiss artifacts/wikipedia_584k_ivf_flat_ip.index \
  --device cpu \
  --limit 5244 \
  --nprobe 64 \
  --out artifacts/lvm/amn_baseline_eval.json

echo ""
echo "================================================================================"
echo "AMN baseline complete!"
echo "Results: artifacts/lvm/amn_baseline_eval.json"
echo ""
echo "This bounds Mamba performance:"
echo "  - If AMN fails → data/retrieval issue (not model)"
echo "  - If AMN passes → Mamba should approach with contrastive learning"
echo "================================================================================"
