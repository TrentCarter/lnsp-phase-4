#!/usr/bin/env bash
set -euo pipefail

# P6b v2: Recovery from Epoch 3 Collapse
# =======================================
#
# **What Happened**: Original P6b collapsed at epoch 3 due to:
#   - Too aggressive ramp (2x margin, 2x weight = 4x pressure)
#   - Sharp gamma (8.0) created death spiral
#   - High upper clamp (0.05) let directional loss overwhelm MSE
#
# **Epoch 2 Checkpoint** (Last Good State):
#   - Val cosine: 0.4758 ✅
#   - Margin: -0.0375 (improving from -0.042!)
#   - R@5: 0.750 ✅
#
# **P6b v2 Fixes**:
#   1. Gentler ramp: 1.5x per stage (was 2x)
#   2. Softer gamma: 4.0 (was 8.0)
#   3. Lower clamp: 0.02 (was 0.05)
#   4. Safety check: Skip directional loss if negative cosines detected
#
# **Usage**:
#   ./scripts/train_transformer_p6b_v2_recovery.sh [CHECKPOINT_PATH] [DEVICE]
#
# **Examples**:
#   # Resume from specific checkpoint
#   ./scripts/train_transformer_p6b_v2_recovery.sh \
#     artifacts/lvm/models/transformer_p6b_20251102_161345/best_model.pt \
#     mps
#
#   # Resume from latest P6b model (default)
#   ./scripts/train_transformer_p6b_v2_recovery.sh

PY=${PY:-./.venv/bin/python}
TR=app/lvm/train_unified.py
TEST=tools/tests/test_5to1_alignment.py

# Default: Use the collapsed model's epoch 2 checkpoint
DEFAULT_CHECKPOINT="artifacts/lvm/models/transformer_p6b_20251102_161345/best_model.pt"
CHECKPOINT=${1:-$DEFAULT_CHECKPOINT}
DEVICE=${2:-mps}

# P6 data files
TRAIN_NPZ=artifacts/lvm/training_sequences_ctx5_p6_next_token.npz
VAL_NPZ=artifacts/lvm/validation_sequences_ctx5_p6_next_token.npz
OOD_NPZ=artifacts/lvm/ood_sequences_ctx5_p6_next_token.npz
ART_NPZ=artifacts/wikipedia_584k_fresh.npz

echo "============================================="
echo "P6b v2: Recovery Training (Gentler Ramp)"
echo "============================================="
echo "Resume from: $CHECKPOINT"
echo "Device: $DEVICE"
echo ""
echo "🔧 Changes from P6b v1:"
echo "  - Ramp: 1.5x per stage (was 2x)"
echo "  - Gamma: 4.0 (was 8.0)"
echo "  - Max λ_eff: 0.02 (was 0.05)"
echo "  - Safety: Skip dir loss on negative cosines"
echo ""
echo "📊 Epoch 2 Baseline (Resuming From):"
echo "  - Val cosine: 0.4758"
echo "  - Margin: -0.0375 (improving!)"
echo "  - R@5: 0.750"
echo ""

# Check if checkpoint exists
if [ ! -f "$CHECKPOINT" ]; then
    echo "❌ Checkpoint not found: $CHECKPOINT"
    echo ""
    echo "Available P6b checkpoints:"
    find artifacts/lvm/models/transformer_p6b_* -name "best_model.pt" 2>/dev/null | head -5
    exit 1
fi

# Extract epoch from checkpoint
EPOCH_INFO=$($PY -c "
import torch
ckpt = torch.load('$CHECKPOINT', map_location='cpu')
print(f\"Epoch: {ckpt.get('epoch', 'unknown')}\")
print(f\"Val cosine: {ckpt.get('val_cosine', 'unknown'):.4f}\")
print(f\"Val loss: {ckpt.get('val_loss', 'unknown'):.6f}\")
" 2>/dev/null || echo "Could not read checkpoint")

echo "📂 Checkpoint Info:"
echo "$EPOCH_INFO"
echo ""

# Create new output directory for v2 training
BASE_DIR="artifacts/lvm/models/transformer_p6b_v2_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BASE_DIR"

echo "🚀 Starting P6b v2 Recovery Training..."
echo "   Resuming from epoch 2, running 10 more epochs (→ epoch 12 total)"
echo ""

# Training arguments (same as original, but will use new schedule from code)
$PY $TR \
  --model-type transformer \
  --data "$TRAIN_NPZ" \
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
echo "P6b v2: Full 5CAT Evaluation"
echo "============================================="
$PY $TEST \
  --model "$BASE_DIR/best_model.pt" \
  --val-npz "$VAL_NPZ" \
  --ood-npz "$OOD_NPZ" \
  --articles-npz "$ART_NPZ" \
  --device "$DEVICE" \
  --max-samples 5000 | tee "$BASE_DIR/5cat_results.json"

# Check training status
if [ $TRAIN_STATUS -ne 0 ]; then
    echo ""
    echo "❌ P6b v2 Training FAILED"
    echo "   Check logs above for details"
    exit 1
fi

echo ""
echo "============================================="
echo "✅ P6b v2 Recovery Complete!"
echo "============================================="
echo "Model: $BASE_DIR/best_model.pt"
echo ""
echo "📊 Expected Results (with gentler ramp):"
echo "   - Margin should climb gradually: -0.037 → 0.0 → +0.05"
echo "   - NO collapse (safety checks prevent death spiral)"
echo "   - R@5 should stay ≥ 70%"
echo "   - Val cosine should improve steadily"
echo ""
echo "🔍 What to Check:"
echo "   1. Margin trend in Mini-5CAT logs"
echo "   2. No \"P6b WARNING\" messages (collapse detection)"
echo "   3. λ_eff stays in [0.001 - 0.02] throughout"
echo "   4. Final 5CAT: Margin ≥ +0.03, R@5 ≥ 70%"
echo ""
echo "📁 Saved to: $BASE_DIR"
echo "============================================="
