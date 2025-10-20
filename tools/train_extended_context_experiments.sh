#!/bin/bash
#
# Train Extended Context Experiments (Hierarchical + Memory + Baseline)
# ========================================================================
#
# Implements the 3-experiment plan from Extended Context PRD:
#   A. Hierarchical GRU (100-vector context, two-level processing)
#   B. Memory-Augmented GRU (100-vector context + external memory)
#   C. Baseline GRU (100-vector context, standard architecture)
#
# Prerequisites:
#   1. Wikipedia ingestion completed (~630k concepts)
#   2. Extended context data exported (100-vector sequences)
#   3. Fair comparison test set prepared
#
# Usage:
#   ./tools/train_extended_context_experiments.sh
#
# Output:
#   - 3 trained models in artifacts/lvm/models_extended_context/
#   - Training logs in logs/extended_context_training.log
#   - Comparison results in artifacts/lvm/extended_context_comparison.json
#
# Created: 2025-10-19 (Extended Context Experiments)

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${YELLOW}======================================${NC}"
echo -e "${YELLOW}Extended Context Experiments${NC}"
echo -e "${YELLOW}======================================${NC}"
echo ""

# Configuration
DATA_DIR="artifacts/lvm/data_extended"
OUTPUT_DIR="artifacts/lvm/models_extended_context"
LOG_FILE="logs/extended_context_training.log"
EPOCHS=20
BATCH_SIZE=32
DEVICE="mps"  # Change to "cpu" if MPS not available

# Check prerequisites
echo -e "${YELLOW}Checking prerequisites...${NC}"

if [ ! -f "$DATA_DIR/training_sequences_ctx100.npz" ]; then
    echo -e "${RED}✗ Extended context data not found!${NC}"
    echo ""
    echo "Please run data export first:"
    echo "  ./.venv/bin/python tools/export_lvm_training_data_extended.py \\"
    echo "    --input artifacts/fw600k_vectors_tmd.npz \\"
    echo "    --context-length 100 \\"
    echo "    --output-dir $DATA_DIR"
    echo ""
    exit 1
fi

echo -e "${GREEN}✓ Extended context data found${NC}"

# Create output directories
mkdir -p "$OUTPUT_DIR"
mkdir -p "logs"

# Log start time
START_TIME=$(date +%s)
echo "Training started: $(date)" > "$LOG_FILE"

echo ""
echo -e "${YELLOW}Training Configuration:${NC}"
echo "  Data: $DATA_DIR/training_sequences_ctx100.npz"
echo "  Epochs: $EPOCHS"
echo "  Batch size: $BATCH_SIZE"
echo "  Device: $DEVICE"
echo "  Output: $OUTPUT_DIR"
echo ""

# ============================================================================
# Experiment C: Baseline GRU (100-vector context)
# ============================================================================

echo -e "${YELLOW}======================================${NC}"
echo -e "${YELLOW}Experiment C: Baseline GRU${NC}"
echo -e "${YELLOW}======================================${NC}"
echo ""
echo "Standard GRU with extended 100-vector context"
echo "Purpose: Establish baseline for extended context"
echo ""

PYTHONPATH=. ./.venv/bin/python app/lvm/train_unified.py \
    --model-type gru \
    --data "$DATA_DIR/training_sequences_ctx100.npz" \
    --epochs $EPOCHS \
    --batch-size $BATCH_SIZE \
    --device $DEVICE \
    --output-dir "$OUTPUT_DIR/baseline_gru_ctx100" \
    2>&1 | tee -a "$LOG_FILE"

echo ""
echo -e "${GREEN}✓ Baseline GRU training complete${NC}"
echo ""

# ============================================================================
# Experiment A: Hierarchical GRU
# ============================================================================

echo -e "${YELLOW}======================================${NC}"
echo -e "${YELLOW}Experiment A: Hierarchical GRU${NC}"
echo -e "${YELLOW}======================================${NC}"
echo ""
echo "Two-level processing: 10 chunks of 10 vectors"
echo "Purpose: Test hierarchical attention for extended context"
echo ""

PYTHONPATH=. ./.venv/bin/python app/lvm/train_unified.py \
    --model-type hierarchical_gru \
    --data "$DATA_DIR/training_sequences_ctx100.npz" \
    --epochs $EPOCHS \
    --batch-size $BATCH_SIZE \
    --device $DEVICE \
    --output-dir "$OUTPUT_DIR/hierarchical_gru_ctx100" \
    2>&1 | tee -a "$LOG_FILE"

echo ""
echo -e "${GREEN}✓ Hierarchical GRU training complete${NC}"
echo ""

# ============================================================================
# Experiment B: Memory-Augmented GRU
# ============================================================================

echo -e "${YELLOW}======================================${NC}"
echo -e "${YELLOW}Experiment B: Memory-Augmented GRU${NC}"
echo -e "${YELLOW}======================================${NC}"
echo ""
echo "GRU + External Memory Bank (2,048 slots)"
echo "Purpose: Test persistent knowledge with memory augmentation"
echo ""

PYTHONPATH=. ./.venv/bin/python app/lvm/train_unified.py \
    --model-type memory_gru \
    --data "$DATA_DIR/training_sequences_ctx100.npz" \
    --epochs $EPOCHS \
    --batch-size $BATCH_SIZE \
    --device $DEVICE \
    --output-dir "$OUTPUT_DIR/memory_gru_ctx100" \
    2>&1 | tee -a "$LOG_FILE"

echo ""
echo -e "${GREEN}✓ Memory-Augmented GRU training complete${NC}"
echo ""

# ============================================================================
# Summary
# ============================================================================

END_TIME=$(date +%s)
TOTAL_TIME=$((END_TIME - START_TIME))
HOURS=$((TOTAL_TIME / 3600))
MINUTES=$(((TOTAL_TIME % 3600) / 60))

echo -e "${GREEN}======================================${NC}"
echo -e "${GREEN}All Experiments Complete!${NC}"
echo -e "${GREEN}======================================${NC}"
echo ""
echo "Total training time: ${HOURS}h ${MINUTES}m"
echo ""

echo -e "${YELLOW}Trained Models:${NC}"
echo "  1. Baseline GRU (ctx100):     $OUTPUT_DIR/baseline_gru_ctx100/"
echo "  2. Hierarchical GRU (ctx100): $OUTPUT_DIR/hierarchical_gru_ctx100/"
echo "  3. Memory GRU (ctx100):       $OUTPUT_DIR/memory_gru_ctx100/"
echo ""

echo -e "${YELLOW}Model Comparison:${NC}"
echo "  View training histories:"
echo "    cat $OUTPUT_DIR/*/training_history.json | jq '.final_val_cosine'"
echo ""

echo -e "${YELLOW}Best Model Selection:${NC}"
for model_dir in "$OUTPUT_DIR"/*; do
    if [ -f "$model_dir/training_history.json" ]; then
        MODEL_NAME=$(basename "$model_dir")
        VAL_COSINE=$(cat "$model_dir/training_history.json" | jq -r '.history[-1].val_cosine')
        echo "  $MODEL_NAME: Val Cosine = $VAL_COSINE"
    fi
done
echo ""

echo -e "${YELLOW}Next Steps:${NC}"
echo "  1. Compare with previous 5-vector models"
echo "  2. Run inference benchmarks"
echo "  3. Visualize attention patterns (Hierarchical GRU)"
echo "  4. Analyze memory usage (Memory GRU)"
echo ""
echo "  Run comparison:"
echo "    ./.venv/bin/python tools/compare_context_windows.py \\"
echo "      --ctx5-models artifacts/lvm/models_367k/ \\"
echo "      --ctx100-models $OUTPUT_DIR/"
echo ""

echo -e "${GREEN}Training log saved to: $LOG_FILE${NC}"
echo ""
