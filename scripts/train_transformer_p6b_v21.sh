#!/usr/bin/env bash
set -euo pipefail

# P6b v2.1: Comprehensive Guardrails (Production-Ready)
# ======================================================
#
# **All 6 Critical Fixes Implemented**:
#   1. Scale-aware directional loss (ratio + gap)
#   2. Positive floor penalty (prevents "two negatives" victory)
#   3. Norm regularization (unit-sphere constraint)
#   4. Adaptive λ guard (ρ-cap at 25%)
#   5. Skip/attenuate on bad signs (collapse detection)
#   6. Multi-guard logging (ρ, neg_cos_rate, ratio, etc.)
#
# **Why v2.1 vs v2**:
#   - v2: Gentler ramp, softer gamma, lower clamp
#   - v2.1: ALL of v2 + 6 additional guardrails
#   - v2.1 = Production-ready, collapse-resistant
#
# **Expected Behavior**:
#   - NO collapse (multiple safety nets)
#   - NO negative cosines (positive floor + skip logic)
#   - ρ stays ≤ 0.25 (adaptive guard enforces 25% cap)
#   - Margin flips positive by epoch 6-8
#
# **Usage**:
#   ./scripts/train_transformer_p6b_v21.sh [DATA] [VAL] [OOD] [ART] [DEVICE]
#
# **Examples**:
#   ./scripts/train_transformer_p6b_v21.sh  # Use defaults (fresh start)

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
echo "P6b v2.1: Comprehensive Guardrails"
echo "============================================="
echo "Training Data: $DATA"
echo "Val Data: $VAL"
echo "OOD Data: $OOD"
echo "Device: $DEVICE"
echo ""
echo "🛡️ Guardrails Active (6/6):"
echo "  1. ✅ Scale-aware loss (ratio + gap)"
echo "  2. ✅ Positive floor (τ=0.10)"
echo "  3. ✅ Norm regularization (η=1e-3)"
echo "  4. ✅ Adaptive λ guard (ρ≤0.25)"
echo "  5. ✅ Sign-based skip (collapse detection)"
echo "  6. ✅ Enhanced logging (ρ, ratio, neg_cos_rate)"
echo ""
echo "📊 Ramp Schedule (Gentle):"
echo "  Epochs 1-2: margin=0.02, frac=0.10"
echo "  Epochs 3-5: margin=0.03, frac=0.15"
echo "  Epochs 6-8: margin=0.04, frac=0.20"
echo "  Epochs 9-12: margin=0.05, frac=0.25"
echo ""

# Create output directory
BASE_DIR="artifacts/lvm/models/transformer_p6b_v21_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BASE_DIR"

echo "🚀 Starting P6b v2.1 Training..."
echo ""

# Training arguments (enable --p6b-v21 flag)
$PY $TR \
  --model-type transformer \
  --data "$DATA" \
  --epochs 12 \
  --batch-size 32 \
  --lr 5e-4 \
  --device "$DEVICE" \
  --p6b-v21 \
  --fivecat-every-epoch 1 \
  --fivecat-max-samples 2000 \
  --output-dir "$BASE_DIR"

TRAIN_STATUS=$?

# Run full 5CAT evaluation
echo ""
echo "============================================="
echo "P6b v2.1: Full 5CAT Evaluation"
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
    echo "❌ P6b v2.1 Training FAILED"
    echo "   Check logs above for details"
    exit 1
fi

echo ""
echo "============================================="
echo "✅ P6b v2.1 Training Complete!"
echo "============================================="
echo "Model: $BASE_DIR/best_model.pt"
echo ""
echo "📊 Expected Results:"
echo "   - Margin: +0.03 to +0.05 (POSITIVE!)"
echo "   - R@5: ≥ 70% (high accuracy)"
echo "   - Val cosine: ≥ 0.48 (stable)"
echo "   - NO collapse warnings in logs ✅"
echo ""
echo "🔍 Guardrail Diagnostics:"
echo "   1. Check for 'P6b v2.1 SKIP' messages (should be minimal)"
echo "   2. Verify ρ stayed ≤ 0.25 (adaptive guard working)"
echo "   3. Check pos_mu stayed ≥ 0.0 (positive floor working)"
echo "   4. Review ratio_mu (should be positive and climbing)"
echo ""
echo "📁 Saved to: $BASE_DIR"
echo "============================================="
