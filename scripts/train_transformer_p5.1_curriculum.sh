#!/usr/bin/env bash
set -euo pipefail

# P5.1: Enhanced Curriculum with Positional Ramp, Attention Bias, Last-Slot Noise, Micro-Directional Guard
# Strategy: Reshape learning landscape BEFORE model commits to copy-last
#
# Key Enhancements over P5:
# 1. Positional Ramp: 0.00 → 0.10 over epochs 1-3 (vs P5's fixed 0.03)
# 2. Attention Bias: 0.0 → 0.6 over epochs 1-4 (makes copying mechanically harder)
# 3. Last-Slot Noise: p=0.15, σ=0.03, swap_p=0.05 (corrupts copy-last path)
# 4. Micro-Directional Guard: λ=0.001, γ=5.0, m=0.02 (gentle nudge, not hammer)
# 5. Mini-5CAT every epoch (early detection of backward bias)
# 6. Strict gates: Margin ≥ +0.02, R@5 ≥ 60% (abort if Stage A fails)
#
# Usage:
#   ./scripts/train_transformer_p5.1_curriculum.sh [TRAIN_NPZ] [VAL_NPZ] [OOD_NPZ] [ART_NPZ] [DEVICE]
#
# Example:
#   ./scripts/train_transformer_p5.1_curriculum.sh \
#     artifacts/lvm/training_sequences_ctx5_584k_clean_splits.npz \
#     artifacts/lvm/validation_sequences_ctx5_articles4000-4499_compat.npz \
#     artifacts/lvm/ood_sequences_ctx5_articles1500-1999.npz \
#     artifacts/wikipedia_584k_fresh.npz \
#     cpu

PY=${PY:-./.venv/bin/python}
TR=app/lvm/train_unified.py
TEST=tools/tests/test_5to1_alignment.py

TRAIN_NPZ=${1:-artifacts/lvm/training_sequences_ctx5_584k_clean_splits.npz}
VAL_NPZ=${2:-artifacts/lvm/validation_sequences_ctx5_articles4000-4499_compat.npz}
OOD_NPZ=${3:-artifacts/lvm/ood_sequences_ctx5_articles1500-1999.npz}
ART_NPZ=${4:-artifacts/wikipedia_584k_fresh.npz}
DEVICE=${5:-cpu}

echo "============================================="
echo "P5.1: Enhanced Curriculum Training"
echo "============================================="
echo "Training Data: $TRAIN_NPZ"
echo "Val Data: $VAL_NPZ"
echo "OOD Data: $OOD_NPZ"
echo "Articles: $ART_NPZ"
echo "Device: $DEVICE"
echo ""

echo "[P5.1] Computing forward-advantage metrics…"
$PY tools/compute_forward_distinctness.py --npz "$TRAIN_NPZ"

SCORES_NPZ="${TRAIN_NPZ%.*}_forward_scores.npz"

# Curriculum thresholds (consultant-recommended)
TAU_SIM_A=0.66    # Stage A: sim_prev threshold
TAU_ADV_A=0.08    # Stage A: advantage threshold
TAU_SIM_B=0.58    # Stage B: sim_prev threshold
TAU_ADV_B=0.05    # Stage B: advantage threshold

echo "[P5.1] Building curriculum splits with forward-advantage thresholds…"
echo "   Stage A: sim_prev ≥ $TAU_SIM_A AND adv_prev ≥ $TAU_ADV_A"
echo "   Stage B: sim_prev ≥ $TAU_SIM_B OR  adv_prev ≥ $TAU_ADV_B"
$PY tools/build_curriculum_splits.py \
  --train-npz "$TRAIN_NPZ" \
  --scores-npz "$SCORES_NPZ" \
  --tau-sim-A "$TAU_SIM_A" \
  --tau-adv-A "$TAU_ADV_A" \
  --tau-sim-B "$TAU_SIM_B" \
  --tau-adv-B "$TAU_ADV_B"

BASE_DIR="artifacts/lvm/models/transformer_p5.1_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BASE_DIR"

# P5.1 Parameters (as recommended by consultant)
POS_MAX=0.10              # Stronger than P5's 0.03 (6.7x increase)
POS_RAMP=3                # Ramp from 0 to 0.10 over 3 epochs
ATTN_BIAS_MAX=0.6         # Negative bias on last context position
ATTN_BIAS_RAMP=4          # Ramp from 0 to 0.6 over 4 epochs
NOISE_P=0.15              # Last-slot corruption probability
NOISE_SIGMA=0.03          # Gaussian noise sigma
NOISE_SWAP_P=0.05         # Swap with second-to-last probability
LAMBDA_MICRO=0.001        # Micro-directional guard weight (1/5 of P3's 0.005)
GAMMA=5.0                 # Softplus temperature (gentle penalty)
MARGIN=0.02               # Micro-directional margin
DIR_RAMP=6                # Ramp lambda_micro from 0 to max over 6 epochs

# Stage A Gates (strict)
GATE_MARGIN=0.02          # Minimum margin to pass (+0.02)
GATE_R5=0.60              # Minimum R@5 to pass (60%)
GATE_ROLLOUT=0.46         # Minimum rollout coherence

echo "============================================="
echo "P5.1 Configuration:"
echo "  Positional: 0 → $POS_MAX over $POS_RAMP epochs"
echo "  Attn Bias: 0 → $ATTN_BIAS_MAX over $ATTN_BIAS_RAMP epochs"
echo "  Last-Slot Noise: p=$NOISE_P, σ=$NOISE_SIGMA, swap=$NOISE_SWAP_P"
echo "  Micro-Dir Guard: λ=0 → $LAMBDA_MICRO over $DIR_RAMP epochs, γ=$GAMMA, m=$MARGIN"
echo "  Stage A Gates: Margin≥$GATE_MARGIN, R@5≥$GATE_R5"
echo "============================================="
echo ""

echo "[P5.1] Stage A (top30, epochs 1–4)…"
$PY $TR --model-type transformer --data "$TRAIN_NPZ" --epochs 4 --batch-size 32 --device "$DEVICE" \
  --positional-scalar $POS_MAX \
  --positional-ramp-epochs $POS_RAMP \
  --curriculum forward_top_30 --curriculum-scores "$SCORES_NPZ" \
  --attn-last-bias-max $ATTN_BIAS_MAX \
  --attn-last-bias-warmup-epochs $ATTN_BIAS_RAMP \
  --last-slot-noise-p $NOISE_P \
  --last-slot-noise-sigma $NOISE_SIGMA \
  --last-slot-swap-p $NOISE_SWAP_P \
  --lambda-dir $LAMBDA_MICRO \
  --dir-gamma $GAMMA \
  --dir-margin $MARGIN \
  --dir-warmup-epochs $DIR_RAMP \
  --fivecat-every-epoch 1 \
  --fivecat-max-samples 2000 \
  --gate-min-margin $GATE_MARGIN \
  --gate-min-r-at5 $GATE_R5 \
  --gate-min-rollout $GATE_ROLLOUT \
  --output-dir "$BASE_DIR/stageA"

STAGE_A_STATUS=$?

echo "[P5.1] Full 5CAT after Stage A…"
$PY $TEST --model "$BASE_DIR/stageA/best_model.pt" \
  --val-npz "$VAL_NPZ" --ood-npz "$OOD_NPZ" --articles-npz "$ART_NPZ" \
  --device "$DEVICE" --max-samples 2000 | tee "$BASE_DIR/stageA_5cat.json"

# Check if Stage A passed
if [ $STAGE_A_STATUS -ne 0 ]; then
    echo ""
    echo "❌ Stage A FAILED (training exited with error)"
    echo "   Check mini-5CAT logs above for details"
    echo ""
    echo "📋 Next Steps:"
    echo "   Option 1: Increase positional/bias strength (--positional-scalar 0.15, --attn-last-bias-max 0.8)"
    echo "   Option 2: Switch to P6 (NEXT token architecture)"
    echo ""
    exit 3
fi

echo ""
echo "✅ Stage A Complete!"
echo "   Model: $BASE_DIR/stageA/best_model.pt"
echo ""
echo "📋 Next Steps:"
echo "   1. Check 5CAT results above (expect Margin ≥ +0.02, R@5 ≥ 60%)"
echo "   2. If passed, proceed to Stage B (top 70% curriculum)"
echo "   3. If failed, try stronger settings or P6"
echo ""

# Optional: Continue to Stage B if Stage A passed
# Uncomment to run full 3-stage curriculum automatically

# echo "[P5.1] Stage B (top70, epochs 5–10)…"
# $PY $TR --model-type transformer --data "$TRAIN_NPZ" --epochs 10 --batch-size 32 --device "$DEVICE" \
#   --positional-scalar $POS_MAX \
#   --curriculum forward_top_70 --curriculum-scores "$SCORES_NPZ" \
#   --attn-last-bias-max $ATTN_BIAS_MAX \
#   --last-slot-noise-p $NOISE_P \
#   --lambda-dir $LAMBDA_MICRO \
#   --fivecat-every-epoch 1 \
#   --resume "$BASE_DIR/stageA/best_model.pt" \
#   --output-dir "$BASE_DIR/stageB"
#
# echo "[P5.1] 5CAT after Stage B…"
# $PY $TEST --model "$BASE_DIR/stageB/best_model.pt" \
#   --val-npz "$VAL_NPZ" --ood-npz "$OOD_NPZ" --articles-npz "$ART_NPZ" \
#   --device "$DEVICE" --max-samples 2000 | tee "$BASE_DIR/stageB_5cat.json"
#
# echo "[P5.1] Stage C (full, epochs 11–20)…"
# $PY $TR --model-type transformer --data "$TRAIN_NPZ" --epochs 20 --batch-size 32 --device "$DEVICE" \
#   --positional-scalar $POS_MAX \
#   --curriculum full \
#   --attn-last-bias-max $ATTN_BIAS_MAX \
#   --last-slot-noise-p $NOISE_P \
#   --lambda-dir $LAMBDA_MICRO \
#   --fivecat-every-epoch 1 \
#   --resume "$BASE_DIR/stageB/best_model.pt" \
#   --output-dir "$BASE_DIR/stageC"
#
# echo "[P5.1] Final 5CAT after Stage C…"
# $PY $TEST --model "$BASE_DIR/stageC/best_model.pt" \
#   --val-npz "$VAL_NPZ" --ood-npz "$OOD_NPZ" --articles-npz "$ART_NPZ" \
#   --device "$DEVICE" --max-samples 5000 | tee "$BASE_DIR/stageC_5cat.json"

echo "[P5.1] DONE → $BASE_DIR"
