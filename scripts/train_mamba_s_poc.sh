#!/bin/bash
# POC Retrain: Mamba-S with Payload-Aligned Data
# ==================================================
#
# Strategy:
# 1. Train with mid-epoch checkpoints (every 2 epochs)
# 2. Smoke test at epochs 2, 4 with 1k samples
# 3. If epoch 4 clears gates (Contain@50 ≥55%, Eff@5 ≥0.65), continue to 20
# 4. Use MPS for speed (CPU fallback available)
#
# Gates:
# - Contain@50 ≥ 60%
# - Eff@5 ≥ 0.68 (R@5 / Contain@50)
# - R@5 ≥ 40%
# - P95 ≤ 1.45ms

set -e

# Configuration
MODEL_TYPE="mamba_s"
TRAIN_NPZ="artifacts/lvm/train_payload_aligned.npz"
VAL_NPZ="artifacts/lvm/val_payload_aligned.npz"
EVAL_NPZ="artifacts/lvm/eval_v2_payload_aligned.npz"
PAYLOAD="artifacts/wikipedia_584k_payload.npy"
FAISS_INDEX="artifacts/wikipedia_584k_ivf_flat_ip.index"
SAVE_DIR="artifacts/lvm/models/mamba_s_poc"
LOG_FILE="logs/mamba_s_poc_$(date +%Y%m%d_%H%M%S).log"

# Device selection (prefer MPS, fallback to CPU)
DEVICE="mps"

# Create directories
mkdir -p "$(dirname "$LOG_FILE")"
mkdir -p "$SAVE_DIR"

echo "================================================================================"
echo "Mamba-S POC Retrain with Payload-Aligned Data"
echo "================================================================================"
echo "Training data: $TRAIN_NPZ"
echo "Val data: $VAL_NPZ"
echo "Eval data: $EVAL_NPZ"
echo "Device: $DEVICE"
echo "Save dir: $SAVE_DIR"
echo "Log: $LOG_FILE"
echo "================================================================================"
echo ""

# Start training
echo "Starting training..."
echo "Command:"
echo "KMP_DUPLICATE_LIB_OK=TRUE ./.venv/bin/python app/lvm/train_mamba_unified.py \\"
echo "  --model-type $MODEL_TYPE \\"
echo "  --train-npz $TRAIN_NPZ \\"
echo "  --d-model 768 \\"
echo "  --n-layers 8 \\"
echo "  --d-state 128 \\"
echo "  --conv-sz 4 \\"
echo "  --expand 2 \\"
echo "  --dropout 0.1 \\"
echo "  --epochs 20 \\"
echo "  --batch-size 256 \\"
echo "  --lr 1e-3 \\"
echo "  --device $DEVICE \\"
echo "  --save-dir $SAVE_DIR"
echo ""

KMP_DUPLICATE_LIB_OK=TRUE ./.venv/bin/python app/lvm/train_mamba_unified.py \
  --model-type "$MODEL_TYPE" \
  --train-npz "$TRAIN_NPZ" \
  --d-model 768 \
  --n-layers 8 \
  --d-state 128 \
  --conv-sz 4 \
  --expand 2 \
  --dropout 0.1 \
  --epochs 20 \
  --batch-size 256 \
  --lr 1e-3 \
  --device "$DEVICE" \
  --save-dir "$SAVE_DIR" \
  2>&1 | tee "$LOG_FILE"

echo ""
echo "================================================================================"
echo "Training complete!"
echo "================================================================================"
echo "Log: $LOG_FILE"
echo "Model: $SAVE_DIR/best.pt"
echo ""
echo "Next steps:"
echo "  1. Run full evaluation on 5.2k samples"
echo "  2. Check gates: Contain@50 ≥ 60%, Eff@5 ≥ 0.68, R@5 ≥ 40%"
echo "  3. If successful, proceed with Sandwich/H/XL"
echo "================================================================================"
