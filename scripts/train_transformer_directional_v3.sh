#!/bin/bash
#
# Train Transformer with Full Directional Guardrails (V3)
# Comprehensive fix for "copy last context" bug with proper scheduling
#
# Implements all recommendations from analysis:
#   1. Scheduled ramp-up (prevents early collapse)
#   2. Positional scalar (breaks time symmetry)
#   3. Directional + anti-copy losses (lightweight)
#   4. Multi-stage training with 5CAT checkpoints
#
# Training Schedule:
#   Epochs 1-3:   Pure MSE (warm-up, λ=0, no guards)
#   Epochs 4-10:  Ramp guards (λ=0.005→0.01, gradual introduction)
#   Epochs 11-20: Full guards (λ=0.01, all features enabled)
#
# Expected 5CAT Targets:
#   Epoch 1:  val_cos ≥ 0.48, peak at k=+1
#   Epoch 3:  val_cos ≥ 0.50, margin ≥ +0.06
#   Final:    val_cos ≥ 0.54, OOD ± 0.05, margin ≥ +0.10
#

set -e

# macOS OpenMP fix
export KMP_DUPLICATE_LIB_OK=TRUE

echo "=========================================="
echo "Train Transformer with Directional Guardrails V3"
echo "=========================================="
echo ""
echo "Full Guardrails System:"
echo "  1. Warm-up (epochs 1-3): Pure MSE"
echo "     - λ_dir = 0, λ_ac = 0, context_drop = 0"
echo "     - Allows MSE to establish baseline mapping"
echo ""
echo "  2. Ramp (epochs 4-10): Gradual guard introduction"
echo "     - λ_dir ramps: 0.005 → 0.01"
echo "     - λ_ac ramps: 0.005 → 0.01"
echo "     - context_drop ramps: 0.05 → 0.10"
echo ""
echo "  3. Full (epochs 11-20): All guards at target strength"
echo "     - λ_dir = 0.01 (5x lighter than V1)"
echo "     - λ_ac = 0.01 (5x lighter than V1)"
echo "     - m_dir = 0.03 (tighter margin)"
echo "     - m_ac = 0.01 (tighter margin)"
echo "     - context_drop_p = 0.10"
echo ""
echo "  4. Positional scalar: ENABLED"
echo "     - Adds [0.0, 0.25, 0.5, 0.75, 1.0] * 0.03 to each context position"
echo "     - Breaks time-reversal symmetry (prevents backward prediction)"
echo "     - Input dim: 768 → 769"
echo ""
echo "  5. Future margin loss: DISABLED (pending article-aware batching)"
echo "     - Requires +2/+3 targets from same article"
echo "     - Infrastructure ready, awaiting dataloader update"
echo ""
echo "Target Outcomes:"
echo "  - Margin(+1 vs -1): POSITIVE (≥ +0.10)"
echo "  - Val cosine: HIGH (≥ 0.54, original was 0.558)"
echo "  - Peak offset: k=+1 (NOT k=-1 or k=+3)"
echo "  - OOD generalization: Within ±0.05 of VAL"
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
  --output-dir artifacts/lvm/models/transformer_directional_v3 \
  --lambda-mse 1.0 \
  --lambda-dir 0.01 \
  --margin-dir 0.03 \
  --lambda-ac 0.01 \
  --margin-ac 0.01 \
  --lambda-fut 0.0 \
  --context-drop-p 0.10 \
  --use-positional \
  --pos-scale 0.03 \
  --warmup-epochs 3 \
  --ramp-epochs 7 \
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
echo "Output: artifacts/lvm/models/transformer_directional_v3/"
echo ""
echo "Next Steps - Run 5CAT Validation:"
echo ""
echo "  ./.venv/bin/python tools/tests/test_5to1_alignment.py \\"
echo "    --model artifacts/lvm/models/transformer_directional_v3/best_model.pt \\"
echo "    --val-npz artifacts/lvm/validation_sequences_ctx5_articles4000-4499_compat.npz \\"
echo "    --ood-npz artifacts/lvm/ood_sequences_ctx5_articles1500-1999.npz \\"
echo "    --articles-npz artifacts/wikipedia_584k_fresh.npz \\"
echo "    --device mps --max-samples 500 | tee /tmp/5cat_v3_results.log"
echo ""
echo "Expected 5CAT Results:"
echo "  ✅ A: Offset Sweep"
echo "     - k=+1 should be HIGHEST (not k=-1 or k=+3)"
echo "     - Margin(+1 vs -1) ≥ +0.10"
echo ""
echo "  ✅ B: Retrieval Rank"
echo "     - VAL: R@1 ≥ 60%, R@5 ≥ 95%, MRR ≥ 80%"
echo "     - OOD: R@1 ≥ 55%, R@5 ≥ 92%, MRR ≥ 75%"
echo ""
echo "  ✅ C: Ablations"
echo "     - Shuffle delta: ≤ -0.15 (order matters)"
echo "     - Reverse delta: ≤ -0.15 (direction matters)"
echo ""
echo "  ✅ D: Rollout"
echo "     - VAL avg_cos@H=5: ≥ 0.45"
echo "     - OOD avg_cos@H=5: ≥ 0.42"
echo ""
echo "  ✅ E: Bins Delta"
echo "     - abs(VAL - OOD) ≤ 0.05 (generalization)"
echo ""
echo "If 5CAT shows:"
echo "  - NEGATIVE margin → guards too weak, increase λ to 0.015"
echo "  - k=+3 drift → enable future loss (requires article batching)"
echo "  - Val cosine < 0.50 → guards too strong, reduce λ to 0.007"
echo "  - Val cosine drops < 0.40 → STOP, reduce all λ by half"
echo ""
echo "Intermediate Checkpoints (recommended):"
echo "  Epoch 1:  Verify MSE baseline (should see val_cos ≥ 0.48)"
echo "  Epoch 5:  Check ramp progress (margin should start positive)"
echo "  Epoch 10: Verify full guards (margin ≥ +0.08, val_cos ≥ 0.52)"
echo "  Epoch 15: Final check (margin ≥ +0.10, val_cos ≥ 0.54)"
echo ""
