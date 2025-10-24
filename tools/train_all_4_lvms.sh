#!/bin/bash
#
# Train All 4 LVM Models Sequentially
# =====================================
#
# Trains LSTM, AMN, Transformer, and GRU models on fresh Wikipedia data.
# Estimated time: 8-16 hours total (2-4 hours per model)
#
# Usage:
#   bash tools/train_all_4_lvms.sh

set -e  # Exit on error

# CRITICAL: macOS OpenMP fix
export KMP_DUPLICATE_LIB_OK=TRUE

# Configuration
DATA_FILE="artifacts/lvm/wikipedia_fresh_sequences_ctx5.npz"
EPOCHS=20
BATCH_SIZE=32
DEVICE="mps"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="logs/lvm_training_${TIMESTAMP}"

# Create log directory
mkdir -p "${LOG_DIR}"

echo "================================================================================"
echo "TRAINING ALL 4 LVM MODELS - COMPREHENSIVE BENCHMARK"
echo "================================================================================"
echo "Data: ${DATA_FILE}"
echo "Epochs: ${EPOCHS}"
echo "Batch size: ${BATCH_SIZE}"
echo "Device: ${DEVICE}"
echo "Logs: ${LOG_DIR}"
echo ""
echo "Models to train:"
echo "  1. LSTM     (⭐ Production recommended)"
echo "  2. AMN      (⚡ Fastest)"
echo "  3. GRU      (Middle ground)"
echo "  4. Transformer (🎯 Best accuracy)"
echo ""
echo "Estimated time: 8-16 hours total"
echo "================================================================================"
echo ""

# Function to train a model
train_model() {
    local model_type=$1
    local model_name=$2
    local emoji=$3

    echo ""
    echo "================================================================================"
    echo "${emoji} TRAINING MODEL: ${model_name} (${model_type})"
    echo "================================================================================"
    echo "Start time: $(date)"
    echo ""

    # Train the model
    ./.venv/bin/python app/lvm/train_unified.py \
        --model-type "${model_type}" \
        --data "${DATA_FILE}" \
        --epochs "${EPOCHS}" \
        --batch-size "${BATCH_SIZE}" \
        --lambda-mse 1.0 \
        --device "${DEVICE}" \
        2>&1 | tee "${LOG_DIR}/${model_type}_training.log"

    echo ""
    echo "✅ ${model_name} training complete!"
    echo "End time: $(date)"
    echo ""
}

# Train all 4 models
train_model "lstm" "LSTM Baseline" "⭐"
train_model "amn" "Attention Mixture Network" "⚡"
train_model "gru" "GRU Stack" "📊"
train_model "transformer" "Transformer" "🎯"

echo ""
echo "================================================================================"
echo "✅ ALL 4 MODELS TRAINED SUCCESSFULLY!"
echo "================================================================================"
echo "End time: $(date)"
echo ""
echo "Training logs: ${LOG_DIR}/"
echo ""
echo "Next steps:"
echo "  1. Create OOD test set:"
echo "     ./.venv/bin/python tools/create_ood_test_set.py"
echo ""
echo "  2. Run comprehensive benchmark:"
echo "     ./.venv/bin/python tools/benchmark_all_lvms_comprehensive.py"
echo ""
echo "  3. View results:"
echo "     cat artifacts/lvm/benchmark_results_${TIMESTAMP}.md"
echo ""
