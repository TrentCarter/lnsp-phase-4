#!/bin/bash
#
# Train AMN with Article-Based Validation (PROPER OOD)
# =====================================================
#
# Uses new training data that:
# 1. Excludes articles 1500-1999 (held out for final OOD test)
# 2. Uses article-based train/val split (not random sequences)
#
# Expected results:
# - Val cosine: ~0.45-0.50 (REAL generalization to unseen articles!)
# - OOD cosine: ~0.43-0.48 (on TRULY_FIXED test set)
# - Delta < 0.10 (proves model generalizes!)

set -e

export KMP_DUPLICATE_LIB_OK=TRUE  # macOS OpenMP fix

PYTHONPATH=. ./.venv/bin/python app/lvm/train_unified.py \
  --model-type amn \
  --data artifacts/lvm/training_sequences_ctx5_584k_article_split.npz \
  --epochs 20 \
  --batch-size 256 \
  --lr 0.0005 \
  --device mps \
  --output-dir artifacts/lvm/models

echo ""
echo "============================================================"
echo "✅ Training complete!"
echo "============================================================"
echo ""
echo "Next: Verify on OOD test set:"
echo "./.venv/bin/python tools/eval_model_ood.py \\"
echo "  --model artifacts/lvm/models/amn_*/best_model.pt \\"
echo "  --ood-data artifacts/lvm/wikipedia_ood_test_ctx5_TRULY_FIXED.npz \\"
echo "  --device mps"
echo ""
echo "Expected OOD cosine: 0.43-0.48 (not negative!)"
