#!/usr/bin/env bash
set -euo pipefail

# P6b: NEXT Token + Smooth Directional Margin Loss (Auto-Scaled)
# ==============================================================
#
# P6b = P6 Architecture + Directional Margin Loss
# ------------------------------------------------
# 1. P6 Architecture: Predicts target_next (removes identity path)
#    - ctx[0..4] → target_next (CANNOT copy ctx[4]!)
#    - cos(ctx[4], target_next) ≈ 0.395 (much lower than ~0.8)
#
# 2. Directional Margin Loss: Explicitly enforces forward > backward
#    - Loss = softplus(gamma * (margin - (cos(pred, next) - cos(pred, prev)))) / gamma
#    - Auto-scaled λ_eff keeps directional term ≈15-35% of MSE magnitude
#    - Prevents data's backward bias from dominating training
#
# Why P6b Works (When P1-P6 Failed):
# -----------------------------------
# - P1-P5.1: Failed because data has inherent backward bias (Δ = -0.069)
# - P6: Removed identity path BUT margin still negative (-0.082)
# - P6b: Adds explicit directional loss to override backward data signal
#
# Expected Results (10 epochs):
# ------------------------------
# - ✅ Margin flips POSITIVE (≥ +0.05)
# - ✅ R@5 stays high (≥ 70%)
# - ✅ Val cosine good (≥ 0.50)
#
# Usage:
#   ./scripts/train_transformer_p6b_directional.sh [DATA] [VAL] [OOD] [ART] [DEVICE]
#
# Examples:
#   ./scripts/train_transformer_p6b_directional.sh  # Use defaults
#   ./scripts/train_transformer_p6b_directional.sh \
#     artifacts/lvm/training_sequences_ctx5_p6_next_token.npz \
#     artifacts/lvm/validation_sequences_ctx5_p6_next_token.npz \
#     artifacts/lvm/ood_sequences_ctx5_p6_next_token.npz \
#     artifacts/wikipedia_584k_fresh.npz \
#     mps

PY=${PY:-./.venv/bin/python}
TR=app/lvm/train_unified.py
TEST=tools/tests/test_5to1_alignment.py

# Default data files (P6 format: ctx → target_next)
DATA=${1:-artifacts/lvm/training_sequences_ctx5_p6_next_token.npz}
VAL=${2:-artifacts/lvm/validation_sequences_ctx5_p6_next_token.npz}
OOD=${3:-artifacts/lvm/ood_sequences_ctx5_p6_next_token.npz}
ART=${4:-artifacts/wikipedia_584k_fresh.npz}
DEVICE=${5:-mps}

echo "============================================="
echo "P6b: NEXT Token + Directional Margin Loss"
echo "============================================="
echo "Training Data: $DATA"
echo "Val Data: $VAL"
echo "OOD Data: $OOD"
echo "Device: $DEVICE"
echo ""
echo "Key Features:"
echo "  - P6 architecture (predicts target_next, not target)"
echo "  - Smooth directional loss (softplus, auto-scaled)"
echo "  - Ramped margin: 0.02 → 0.05 over 5 epochs"
echo "  - Auto λ_eff: keeps dir term ≈15-35% of MSE"
echo ""

# Create output directory
BASE_DIR="artifacts/lvm/models/transformer_p6b_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BASE_DIR"

# Training arguments
$PY $TR \
  --model-type transformer \
  --data "$DATA" \
  --epochs 12 \
  --batch-size 32 \
  --lr 5e-4 \
  --device "$DEVICE" \
  --p6b-directional \
  --fivecat-every-epoch 1 \
  --fivecat-max-samples 2000 \
  --output-dir "$BASE_DIR"

TRAIN_STATUS=$?

# Run full 5CAT evaluation
echo ""
echo "============================================="
echo "P6b: Full 5CAT Evaluation (5000 samples)"
echo "============================================="
$PY $TEST \
  --model "$BASE_DIR/best_model.pt" \
  --val-npz "$VAL" \
  --ood-npz "$OOD" \
  --articles-npz "$ART" \
  --device "$DEVICE" \
  --max-samples 5000 | tee "$BASE_DIR/5cat_results.json"

# Check training status
if [ $TRAIN_STATUS -ne 0 ]; then
    echo ""
    echo "❌ P6b Training FAILED"
    echo "   Check logs above for details"
    exit 1
fi

echo ""
echo "============================================="
echo "✅ P6b Training Complete!"
echo "============================================="
echo "Model: $BASE_DIR/best_model.pt"
echo ""
echo "📊 Expected Results:"
echo "   - Margin: +0.05 to +0.10 (POSITIVE!)"
echo "   - R@5: ≥ 70% (high retrieval accuracy)"
echo "   - Val cosine: ≥ 0.50 (good similarity)"
echo ""
echo "📋 What to Check:"
echo "   1. Margin is POSITIVE in 5CAT results above"
echo "   2. λ_eff stayed in range [0.001 - 0.02] (auto-scaling worked)"
echo "   3. Directional term was ≈15-35% of MSE (frac_of_mse in logs)"
echo ""
echo "🎯 Success Criteria:"
echo "   ✅ A: Offset Sweep → margin(+1) ≥ +0.05"
echo "   ✅ B: Retrieval → R@5 ≥ 70%"
echo "   ✅ D: Rollout → avg_cos ≥ 0.45"
echo "   ✅ Pass 3/5 gates minimum"
echo ""
echo "📁 Saved to: $BASE_DIR"
echo "============================================="
