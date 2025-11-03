#!/usr/bin/env bash
set -euo pipefail

# P6b v2.3: "Goldilocks" Balanced Pressure with Survival Gates
# ==============================================================
#
# **What went wrong with v2.2**:
#   - Orthogonal escape: Model predicted vectors far from target (cosine 0.44 → 0.18)
#   - Negative cosines to prev: pos=0.19, neg=-0.086 (extreme anti-prev bias)
#   - R@5 100% but margin flip was FAKE (gap positive but predictions wrong)
#   - Root cause: Directional pressure too strong (ρ=0.35-0.50), overwhelmed MSE loss
#
# **v2.3 Solution - "Goldilocks" Zone**:
#   1. **Directional-when-confident gate** (CRITICAL FIX):
#      - Scale directional loss by cos(pred, target_next)
#      - If cos < 0.30: directional OFF (no pressure when misaligned)
#      - If cos > 0.45: directional FULL (apply pressure when aligned)
#      - Prevents gap objective from dragging predictions off-target
#
#   2. **Lower ρ targets** (balanced pressure):
#      - E1-3: ρ=0.15 (cap 0.30) - same as v2.2
#      - E4-6: ρ=0.20 (cap 0.35) - lower than v2.2's 0.25
#      - E7-12: ρ=0.25 (cap 0.40) - much lower than v2.2's 0.35
#
#   3. **Weaker auxiliary penalties**:
#      - pos_floor: τ=0.10, β=1e-3 (back to v2.1, not v2.2's τ=0.12, β=2e-3)
#      - orth_pen: κ=1e-4 (weakened 80% from v2.2's 5e-4)
#      - λ_max: 0.018 (reduced from v2.2's 0.03)
#
#   4. **All v2.1 guardrails kept**:
#      - Scale-aware loss (gap + ratio)
#      - Sign-based skip (collapse detection)
#      - Norm regularization
#
# **Expected Behavior**:
#   - Epochs 1-3: margin ≈ -0.03, ρ ≈ 0.15, cosine stable ≥ 0.42
#   - Epochs 4-6: margin ≈ -0.01, ρ ≈ 0.20, cosine stable ≥ 0.40
#   - Epochs 7-9: margin ≈ +0.01, ρ ≈ 0.25, cosine stable ≥ 0.40
#   - Epochs 10-12: margin +0.01 to +0.03 (slight positive), R@5 ≥ 70%, cosine ≥ 0.40
#
# **Go/No-Go Gates**:
#   - Go: margin ≥ 0 by E8 AND cosine ≥ 0.40 AND R@5 ≥ 70%
#   - No-Go: cosine < 0.35 for >1k steps OR epoch drop >0.05 → rollback + warm 2k with λ=0
#
# **Usage**:
#   ./scripts/train_transformer_p6b_v23.sh [DATA] [VAL] [OOD] [ART] [DEVICE]
#
# **Examples**:
#   ./scripts/train_transformer_p6b_v23.sh  # Use defaults (fresh start)

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
echo "P6b v2.3: \"Goldilocks\" with Survival Gates"
echo "============================================="
echo "Training Data: $DATA"
echo "Val Data: $VAL"
echo "OOD Data: $OOD"
echo "Device: $DEVICE"
echo ""
echo "🛡️ Survival Gates Active (8/8):"
echo "  1. ✅ Directional-when-confident (scale by cos alignment) [NEW!]"
echo "  2. ✅ Scale-aware loss (ratio + gap)"
echo "  3. ✅ Positive floor (τ=0.10, β=1e-3)"
echo "  4. ✅ Norm regularization (η=1e-3)"
echo "  5. ✅ ρ-controller (target ρ=0.25, cap ρ=0.40)"
echo "  6. ✅ Sign-based skip (collapse detection)"
echo "  7. ✅ Enhanced logging (conf, c_pt, ρ, ρ_tgt)"
echo "  8. ✅ Orthogonality penalty (κ=1e-4, weakened 80%)"
echo ""
echo "📊 ρ Schedule (Epoch-Gated - Balanced Goldilocks):"
echo "  Epochs 1-3: ρ_target=0.15, ρ_cap=0.30, margin=0.02"
echo "  Epochs 4-6: ρ_target=0.20, ρ_cap=0.35, margin=0.03"
echo "  Epochs 7-12: ρ_target=0.25, ρ_cap=0.40, margin=0.04"
echo ""
echo "🎯 Directional-When-Confident Gate:"
echo "  - If cos(pred, target_next) < 0.30: scale=0 (OFF)"
echo "  - If cos(pred, target_next) > 0.45: scale=1 (FULL)"
echo "  - Linear ramp between 0.30-0.45"
echo "  - Prevents orthogonal escape (v2.2's failure mode)"
echo ""
echo "📈 Expected Results:"
echo "  - Margin: +0.01 to +0.03 (slightly positive, sustainable)"
echo "  - Cosine: ≥ 0.40 throughout (NO COLLAPSE)"
echo "  - R@5: ≥ 70% (good retrieval)"
echo "  - ρ: 0.15 → 0.20 → 0.25 (controlled climb)"
echo ""
echo "✅ Go/No-Go Gates:"
echo "  - Go: margin ≥ 0 by E8, cosine ≥ 0.40, R@5 ≥ 70%"
echo "  - No-Go: cosine < 0.35 for >1k steps OR drop >0.05 → rollback"
echo ""

# Create output directory
BASE_DIR="artifacts/lvm/models/transformer_p6b_v23_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BASE_DIR"

echo "🚀 Starting P6b v2.3 Training..."
echo ""

# Training arguments (enable --p6b-v23 flag)
$PY $TR \
  --model-type transformer \
  --data "$DATA" \
  --epochs 12 \
  --batch-size 32 \
  --lr 5e-4 \
  --device "$DEVICE" \
  --p6b-v23 \
  --fivecat-every-epoch 1 \
  --fivecat-max-samples 2000 \
  --output-dir "$BASE_DIR"

TRAIN_STATUS=$?

# Run full 5CAT evaluation
echo ""
echo "============================================="
echo "P6b v2.3: Full 5CAT Evaluation"
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
    echo "❌ P6b v2.3 Training FAILED"
    echo "   Check logs above for details"
    exit 1
fi

echo ""
echo "============================================="
echo "✅ P6b v2.3 Training Complete!"
echo "============================================="
echo "Model: $BASE_DIR/best_model.pt"
echo ""
echo "📊 Target Results:"
echo "   - Margin: +0.01 to +0.03 (slightly positive)"
echo "   - Cosine: ≥ 0.40 (NO collapse!)"
echo "   - R@5: ≥ 70% (good retrieval)"
echo "   - ρ: 0.25-0.30 (balanced pressure)"
echo ""
echo "🔍 Key Diagnostics:"
echo "   1. Check conf (confidence scale) in logs"
echo "      - Should be ≥ 0.5 most of the time (well-aligned)"
echo "      - If < 0.3 often: directional gate working (turning OFF when misaligned)"
echo "   2. Check c_pt (cos pred→target)"
echo "      - Should stay ≥ 0.40 throughout (NO v2.2-style collapse)"
echo "   3. Check ρ tracking ρ_tgt"
echo "      - Should climb smoothly: 0.15 → 0.20 → 0.25"
echo "   4. Check skip rate"
echo "      - Should be minimal (<1%)"
echo ""
echo "📈 Comparison to v2.2:"
echo "   - Lower ρ targets (0.25 vs 0.35)"
echo "   - Directional-when-confident gate (NEW!)"
echo "   - Weaker penalties (β=1e-3 vs 2e-3, κ=1e-4 vs 5e-4)"
echo "   - Should prevent orthogonal escape (v2.2's failure)"
echo ""
echo "📁 Saved to: $BASE_DIR"
echo "============================================="
