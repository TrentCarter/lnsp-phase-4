#!/bin/bash
#
# Train Transformer with Lighter Directional Anti-Copy Losses (V2)
# Fixes the "copy last context" bug WITHOUT overwhelming MSE objective
#
# Root cause: Model learned to copy position 4 instead of predict position 5
# V1 issue: λ=0.05 was too strong, collapsed performance (val 0.158)
# V2 solution: Reduce to λ=0.01, lighter margins, less context drop
#
# Expected outcome:
#   - Margin(+1 vs -1) should become POSITIVE (≥ +0.10)
#   - Val cosine should remain HIGH (≥ 0.50)
#   - cos(pred, ctx[-1]) should be LOWER than cos(pred, target)
#   - 5CAT offset sweep should show k=+1 as highest
#

set -e

# macOS OpenMP fix
export KMP_DUPLICATE_LIB_OK=TRUE

echo "=========================================="
echo "Training Transformer with Directional Losses V2"
echo "=========================================="
echo ""
echo "Changes from V1:"
echo "  - λ_dir: 0.05 → 0.01 (5x lighter)"
echo "  - λ_ac:  0.05 → 0.01 (5x lighter)"
echo "  - m_dir: 0.05 → 0.03 (tighter)"
echo "  - m_ac:  0.02 → 0.01 (tighter)"
echo "  - context_drop_p: 0.2 → 0.1 (less aggressive)"
echo ""
echo "Goal: Fix backward bias WITHOUT collapsing performance"
echo ""
echo "Starting training..."
echo ""

./.venv/bin/python app/lvm/train_unified.py \
  --model-type transformer \
  --data artifacts/lvm/training_sequences_ctx5_584k_clean_splits.npz \
  --epochs 20 \
  --batch-size 64 \
  --lr 0.0005 \
  --device mps \
  --output-dir artifacts/lvm/models/transformer_directional_v2 \
  --lambda-mse 1.0 \
  --lambda-dir 0.01 \
  --margin-dir 0.03 \
  --lambda-ac 0.01 \
  --margin-ac 0.01 \
  --context-drop-p 0.1 \
  --lambda-info 0.0 \
  --lambda-moment 0.0 \
  --lambda-variance 0.0 \
  --tau 0.07 \
  --lambda-mmd 0.0 \
  --mmd-anchors 0 \
  --lambda-stat 0.0 \
  --cycle-pct 0.0 \
  --cycle-lambda 0.0 \
  --cycle-steps 1 \
  --cycle-timeout 30.0

echo ""
echo "=========================================="
echo "✅ Training Complete!"
echo "=========================================="
echo "Output: artifacts/lvm/models/transformer_directional_v2/"
echo ""
echo "Next steps:"
echo "  1. Run 5CAT to verify fix:"
echo "     ./.venv/bin/python tools/tests/test_5to1_alignment.py \\"
echo "       --model artifacts/lvm/models/transformer_directional_v2/best_model.pt \\"
echo "       --val-npz artifacts/lvm/validation_sequences_ctx5_articles4000-4499_compat.npz \\"
echo "       --ood-npz artifacts/lvm/ood_sequences_ctx5_articles1500-1999.npz \\"
echo "       --articles-npz artifacts/wikipedia_584k_fresh.npz \\"
echo "       --device mps --max-samples 500"
echo ""
echo "  2. Expected results:"
echo "     - Margin(+1 vs -1): POSITIVE (≥ +0.10)"
echo "     - Val cosine: HIGH (≥ 0.50, ideally ~0.55)"
echo "     - k=+1 should be highest in offset sweep"
echo "     - R@5 should be ≥ 20% (VAL) and ≥ 10% (OOD)"
echo ""
echo "  3. If still fails:"
echo "     - Check training logs for margin evolution"
echo "     - May need even lighter weights (λ=0.005)"
echo ""
