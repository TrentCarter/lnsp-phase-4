#!/bin/bash
#
# Train Transformer with Directional Anti-Copy Losses
# Fixes the "copy last context" bug discovered via 5CAT
#
# Root cause: Model learned to copy position 4 instead of predict position 5
# Solution: Add directional margin + anti-copy hinge losses
#
# Expected outcome:
#   - Margin(+1 vs -1) should become POSITIVE (≥ +0.10)
#   - cos(pred, ctx[-1]) should be LOWER than cos(pred, target)
#   - 5CAT offset sweep should show k=+1 as highest
#

set -e

# macOS OpenMP fix
export KMP_DUPLICATE_LIB_OK=TRUE

echo "=========================================="
echo "Training Transformer with Directional Losses"
echo "=========================================="
echo ""
echo "This training run adds:"
echo "  1. Directional margin loss (next vs prev)"
echo "  2. Anti-copy hinge loss (next vs any context)"
echo "  3. Context drop augmentation (optional)"
echo ""
echo "Safe defaults:"
echo "  λ_dir = 0.05, m_dir = 0.05"
echo "  λ_ac  = 0.05, m_ac  = 0.02"
echo "  context_drop_p = 0.2"
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
  --output-dir artifacts/lvm/models/transformer_directional_fix \
  --lambda-mse 1.0 \
  --lambda-dir 0.05 \
  --margin-dir 0.05 \
  --lambda-ac 0.05 \
  --margin-ac 0.02 \
  --context-drop-p 0.2 \
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
echo "Output: artifacts/lvm/models/transformer_directional_fix/"
echo ""
echo "Next steps:"
echo "  1. Run 5CAT to verify fix:"
echo "     ./.venv/bin/python tools/tests/test_5to1_alignment.py \\"
echo "       --model artifacts/lvm/models/transformer_directional_fix/best_model.pt \\"
echo "       --val-npz artifacts/lvm/validation_sequences_ctx5_articles4000-4499_compat.npz \\"
echo "       --ood-npz artifacts/lvm/ood_sequences_ctx5_articles1500-1999.npz \\"
echo "       --articles-npz artifacts/wikipedia_584k_fresh.npz \\"
echo "       --device mps --max-samples 500"
echo ""
echo "  2. Check margin diagnostics in logs:"
echo "     Should see Margin(+1 vs -1) become POSITIVE during training"
echo ""
