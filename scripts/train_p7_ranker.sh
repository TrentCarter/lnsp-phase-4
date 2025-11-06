#!/bin/bash
# Train P7 "Directional Ranker" LVM with semantic anchoring
#
# Usage:
#   ./scripts/train_p7_ranker.sh [--context 5] [--margin 0.07] [--lambda 0.8]
#
# Recommended grid search:
#   Context: 3, 5, 7
#   Margin: 0.05, 0.07, 0.10
#   Lambda: 0.6, 0.8, 0.9 (learnable, clamped)

set -e

# Default parameters (can override via CLI)
CONTEXT=5
MARGIN=0.07
LAMBDA=0.8
W_RANK=1.0
W_MARGIN=0.5
W_TEACHER=0.2
EPOCHS=10
BATCH_SIZE=64
LR=5e-4
DEVICE="mps"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --context)
            CONTEXT="$2"
            shift 2
            ;;
        --margin)
            MARGIN="$2"
            shift 2
            ;;
        --lambda)
            LAMBDA="$2"
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Ensure we're in project root
cd /Users/trentcarter/Artificial_Intelligence/AI_Projects/lnsp-phase-4

# Training and validation data (arXiv)
TRAIN_NPZ="artifacts/lvm/arxiv_train_sequences.npz"
VAL_NPZ="artifacts/lvm/arxiv_val_sequences.npz"

# Check data exists
if [ ! -f "$TRAIN_NPZ" ]; then
    echo "❌ Training data not found: $TRAIN_NPZ"
    exit 1
fi

if [ ! -f "$VAL_NPZ" ]; then
    echo "❌ Validation data not found: $VAL_NPZ"
    exit 1
fi

# Experiment name with key hyperparameters
EXP_NAME="p7_ranker_c${CONTEXT}_m${MARGIN}_l${LAMBDA}"

echo "=========================================="
echo "P7 DIRECTIONAL RANKER TRAINING"
echo "=========================================="
echo "Context length: $CONTEXT"
echo "Margin: $MARGIN"
echo "Anchor λ: $LAMBDA (learnable)"
echo "Epochs: $EPOCHS"
echo "Batch size: $BATCH_SIZE"
echo "Device: $DEVICE"
echo ""
echo "Loss weights:"
echo "  w_rank: $W_RANK"
echo "  w_margin: $W_MARGIN"
echo "  w_teacher: $W_TEACHER (warmup only)"
echo ""
echo "Training data: $TRAIN_NPZ"
echo "Validation data: $VAL_NPZ"
echo "Experiment name: $EXP_NAME"
echo "=========================================="
echo ""

# Run training (unbuffered output for real-time monitoring)
PYTHONUNBUFFERED=1 ./.venv/bin/python app/lvm/train_p7_ranker.py \
    --train-npz "$TRAIN_NPZ" \
    --val-npz "$VAL_NPZ" \
    --context-length "$CONTEXT" \
    --model-type transformer \
    --d-model 512 \
    --nhead 8 \
    --num-layers 4 \
    --dropout 0.1 \
    --anchor-lambda "$LAMBDA" \
    --anchor-learnable \
    --w-rank "$W_RANK" \
    --w-margin "$W_MARGIN" \
    --w-teacher "$W_TEACHER" \
    --margin "$MARGIN" \
    --temperature 0.07 \
    --gate-threshold 0.03 \
    --gate-weak-weight 0.25 \
    --floor-threshold 0.20 \
    --batch-size "$BATCH_SIZE" \
    --epochs "$EPOCHS" \
    --lr "$LR" \
    --warmup-epochs 2 \
    --device "$DEVICE" \
    --exp-name "$EXP_NAME"

echo ""
echo "=========================================="
echo "Training complete!"
echo "=========================================="
