#!/bin/bash
##
# Retrain All LVM Models with 232k Dataset
#
# This script retrains all 5 LVM models (AMN, LSTM, GRU, Transformer, GraphMERT)
# with the larger 232k training dataset for comparison with 80k baseline.
#
# Usage:
#   ./tools/retrain_all_models_232k.sh
##

set -e

echo "================================================================================"
echo "Retrain All LVM Models with 232k Dataset"
echo "================================================================================"
echo ""
echo "Training data: artifacts/lvm/training_sequences_ctx5.npz"
echo "Training samples: 232,600"
echo "Device: MPS (Apple Silicon)"
echo ""

# Configuration
TRAINING_DATA="artifacts/lvm/training_sequences_ctx5.npz"
DEVICE="mps"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BASE_OUTPUT_DIR="artifacts/lvm/models"

# Verify training data exists
if [ ! -f "$TRAINING_DATA" ]; then
    echo "❌ Training data not found: $TRAINING_DATA"
    exit 1
fi

# Create output directory
mkdir -p "$BASE_OUTPUT_DIR"

echo "================================================================================"
echo "Training Schedule"
echo "================================================================================"
echo ""
echo "1. AMN (fastest - ~15 mins)"
echo "2. LSTM (~20 mins)"
echo "3. GRU (~25 mins)"
echo "4. Transformer (~35 mins)"
echo "5. GraphMERT-LVM (~55 mins)"
echo ""
echo "Total estimated time: ~2.5 hours"
echo ""
read -p "Continue with training? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 1
fi
echo ""

# ============================================================================
# Model 1: AMN (Additive Memory Network)
# ============================================================================

echo "================================================================================"
echo "1/5: Training AMN with 232k dataset"
echo "================================================================================"
echo ""

AMN_OUTPUT_DIR="${BASE_OUTPUT_DIR}/amn_232k_${TIMESTAMP}"

./.venv/bin/python app/lvm/train_unified.py \
    --model-type amn \
    --data "$TRAINING_DATA" \
    --epochs 20 \
    --batch-size 32 \
    --lr 0.0005 \
    --device "$DEVICE" \
    --output-dir "$AMN_OUTPUT_DIR"

echo ""
echo "✓ AMN training complete!"
echo "  Model saved to: $AMN_OUTPUT_DIR"
echo ""

# ============================================================================
# Model 2: LSTM
# ============================================================================

echo "================================================================================"
echo "2/5: Training LSTM with 232k dataset"
echo "================================================================================"
echo ""

LSTM_OUTPUT_DIR="${BASE_OUTPUT_DIR}/lstm_232k_${TIMESTAMP}"

./.venv/bin/python app/lvm/train_unified.py \
    --model-type lstm \
    --data "$TRAINING_DATA" \
    --epochs 20 \
    --batch-size 32 \
    --device "$DEVICE" \
    --output-dir "$LSTM_OUTPUT_DIR"

echo ""
echo "✓ LSTM training complete!"
echo "  Model saved to: $LSTM_OUTPUT_DIR"
echo ""

# ============================================================================
# Model 3: GRU
# ============================================================================

echo "================================================================================"
echo "3/5: Training GRU with 232k dataset"
echo "================================================================================"
echo ""

GRU_OUTPUT_DIR="${BASE_OUTPUT_DIR}/gru_232k_${TIMESTAMP}"

./.venv/bin/python app/lvm/train_unified.py \
    --model-type gru \
    --data "$TRAINING_DATA" \
    --epochs 20 \
    --batch-size 32 \
    --device "$DEVICE" \
    --output-dir "$GRU_OUTPUT_DIR"

echo ""
echo "✓ GRU training complete!"
echo "  Model saved to: $GRU_OUTPUT_DIR"
echo ""

# ============================================================================
# Model 4: Transformer
# ============================================================================

echo "================================================================================"
echo "4/5: Training Transformer with 232k dataset"
echo "================================================================================"
echo ""

TRANSFORMER_OUTPUT_DIR="${BASE_OUTPUT_DIR}/transformer_232k_${TIMESTAMP}"

./.venv/bin/python app/lvm/train_unified.py \
    --model-type transformer \
    --data "$TRAINING_DATA" \
    --epochs 20 \
    --batch-size 32 \
    --device "$DEVICE" \
    --output-dir "$TRANSFORMER_OUTPUT_DIR"

echo ""
echo "✓ Transformer training complete!"
echo "  Model saved to: $TRANSFORMER_OUTPUT_DIR"
echo ""

# ============================================================================
# Model 5: GraphMERT-LVM (Neurosymbolic)
# ============================================================================

echo "================================================================================"
echo "5/5: Training GraphMERT-LVM with 232k dataset"
echo "================================================================================"
echo ""

GRAPHMERT_OUTPUT_DIR="${BASE_OUTPUT_DIR}/graphmert_lvm_232k_${TIMESTAMP}"

# Note: GraphMERT training needs early stopping at ~8 epochs based on 80k results
./.venv/bin/python app/lvm/train_graphmert_lvm_benchmark.py \
    --data "$TRAINING_DATA" \
    --epochs 12 \
    --batch-size 32 \
    --device "$DEVICE" \
    --output-dir "$GRAPHMERT_OUTPUT_DIR"

echo ""
echo "✓ GraphMERT-LVM training complete!"
echo "  Model saved to: $GRAPHMERT_OUTPUT_DIR"
echo ""

# ============================================================================
# Summary
# ============================================================================

echo "================================================================================"
echo "TRAINING COMPLETE!"
echo "================================================================================"
echo ""
echo "All 5 models trained with 232k dataset:"
echo ""
echo "  1. AMN:            $AMN_OUTPUT_DIR"
echo "  2. LSTM:           $LSTM_OUTPUT_DIR"
echo "  3. GRU:            $GRU_OUTPUT_DIR"
echo "  4. Transformer:    $TRANSFORMER_OUTPUT_DIR"
echo "  5. GraphMERT-LVM:  $GRAPHMERT_OUTPUT_DIR"
echo ""
echo "Next step: Run comparison benchmark!"
echo ""
echo "  ./tools/compare_80k_vs_232k_models.sh"
echo ""
echo "================================================================================"
