#!/usr/bin/env bash
set -euo pipefail

# P6b v2.2: ρ-Controller with Controlled Stronger Pressure
# =========================================================
#
# **Improvements over v2.1**:
#   - ρ-controller: Makes ρ a TARGET (not just a cap), actively pushes to 35-50%
#   - Stronger margins: 0.06-0.07 (vs 0.05 in v2.1)
#   - Stronger anchors: pos_floor τ=0.12, β=2e-3 (was τ=0.10, β=1e-3)
#   - Orthogonality penalty: κ=5e-4 (NEW) - chips away at backward bias
#   - Higher λ_max: 0.03 (was 0.02) - allows stronger directional pressure
#
# **Why v2.2 vs v2.1**:
#   - v2.1: Improved margin (-0.082 → -0.047) but too conservative
#   - v2.2: Pushes directional pressure higher while keeping v2.1's stability
#   - v2.2 = Surgical escalation to flip margin positive
#
# **Expected Behavior**:
#   - Epochs 3-4: margin -0.04 → -0.02 (gentle climb)
#   - Epochs 5-6: margin ≈ 0.00 → +0.01 (FLIP POSITIVE!)
#   - Epochs 7-9: margin +0.02...+0.04 (stabilize in target band)
#   - R@5: ≥ 70% throughout (accuracy maintained)
#   - Val cosine: ≥ 0.48 (healthy similarity)
#   - ρ: Climbs to 0.35-0.50 (controlled by ρ-controller)
#
# **Usage**:
#   ./scripts/train_transformer_p6b_v22.sh [DATA] [VAL] [OOD] [ART] [DEVICE]
#
# **Examples**:
#   ./scripts/train_transformer_p6b_v22.sh  # Use defaults (fresh start)

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
echo "P6b v2.2: ρ-Controller with Stronger Pressure"
echo "============================================="
echo "Training Data: $DATA"
echo "Val Data: $VAL"
echo "OOD Data: $OOD"
echo "Device: $DEVICE"
echo ""
echo "🛡️ Guardrails Active (6/6 from v2.1 + NEW enhancements):"
echo "  1. ✅ Scale-aware loss (ratio + gap)"
echo "  2. ✅ Positive floor (τ=0.12, β=2e-3) [STRONGER]"
echo "  3. ✅ Norm regularization (η=1e-3)"
echo "  4. ✅ ρ-controller (target ρ=0.35, cap ρ=0.50) [NEW LOGIC]"
echo "  5. ✅ Sign-based skip (collapse detection)"
echo "  6. ✅ Enhanced logging (ρ, ρ_tgt, ρ_cap, ratio)"
echo "  7. ✅ Orthogonality penalty (κ=5e-4) [NEW]"
echo ""
echo "📊 ρ Schedule (Epoch-Gated):"
echo "  Epochs 1-2: ρ_target=0.15, ρ_cap=0.35, margin=0.02"
echo "  Epochs 3-4: ρ_target=0.25, ρ_cap=0.45, margin=0.04"
echo "  Epochs 5-8: ρ_target=0.35, ρ_cap=0.50, margin=0.06"
echo "  Epochs 9-12: ρ_target=0.35, ρ_cap=0.50, margin=0.07"
echo ""
echo "🎯 Controller Logic:"
echo "  - If ρ < ρ_target * 0.8: Increase λ_eff"
echo "  - If ρ > ρ_cap: Decrease λ_eff (safety)"
echo "  - λ_max raised to 0.03 (was 0.02)"
echo ""

# Create output directory
BASE_DIR="artifacts/lvm/models/transformer_p6b_v22_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BASE_DIR"

echo "🚀 Starting P6b v2.2 Training..."
echo ""

# Training arguments (enable --p6b-v22 flag)
$PY $TR \
  --model-type transformer \
  --data "$DATA" \
  --epochs 12 \
  --batch-size 32 \
  --lr 5e-4 \
  --device "$DEVICE" \
  --p6b-v22 \
  --fivecat-every-epoch 1 \
  --fivecat-max-samples 2000 \
  --output-dir "$BASE_DIR"

TRAIN_STATUS=$?

# Run full 5CAT evaluation
echo ""
echo "============================================="
echo "P6b v2.2: Full 5CAT Evaluation"
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
    echo "❌ P6b v2.2 Training FAILED"
    echo "   Check logs above for details"
    exit 1
fi

echo ""
echo "============================================="
echo "✅ P6b v2.2 Training Complete!"
echo "============================================="
echo "Model: $BASE_DIR/best_model.pt"
echo ""
echo "📊 Expected Results:"
echo "   - Margin: +0.03 to +0.05 (POSITIVE!)"
echo "   - R@5: ≥ 70% (high accuracy)"
echo "   - Val cosine: ≥ 0.48 (stable)"
echo "   - ρ: 0.35-0.50 (controlled climb)"
echo ""
echo "🔍 Controller Diagnostics:"
echo "   1. Check ρ vs ρ_target in logs (should track closely)"
echo "   2. Verify λ_eff adjusted dynamically (controller active)"
echo "   3. Check margin evolution (should flip positive E5-6)"
echo "   4. Review skip rate (should be minimal, <1%)"
echo ""
echo "📈 Key Differences from v2.1:"
echo "   - ρ climbed to 0.35-0.50 (vs capped at 0.25)"
echo "   - Stronger pos_floor (τ=0.12 vs 0.10)"
echo "   - Orthogonality penalty added (κ=5e-4)"
echo "   - Margin increased to 0.06-0.07 (vs 0.05)"
echo ""
echo "📁 Saved to: $BASE_DIR"
echo "============================================="
