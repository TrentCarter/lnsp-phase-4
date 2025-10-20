#!/bin/bash
# Train All 4 LVM Models with 367k Dataset
# Expected runtime: ~3-4 hours total (45-60min each)
# Created: 2025-10-18

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Configuration
DATA="artifacts/lvm/data/training_sequences_ctx5.npz"
OUTPUT_DIR="artifacts/lvm/models_367k"
EPOCHS=20
BATCH_SIZE=64
DEVICE="mps"  # Change to "cpu" if needed

echo -e "${YELLOW}========================================${NC}"
echo -e "${YELLOW}Training All 4 LVM Models (367k Dataset)${NC}"
echo -e "${YELLOW}========================================${NC}"
echo ""
echo "Dataset: $DATA"
echo "Output: $OUTPUT_DIR"
echo "Epochs: $EPOCHS"
echo "Batch size: $BATCH_SIZE"
echo "Device: $DEVICE"
echo ""
echo "Estimated time: ~3-4 hours total"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Start timestamp
START_TIME=$(date +%s)

echo -e "${YELLOW}1/4 - Training AMN (Attention Memory Network)...${NC}"
PYTHONPATH=. ./.venv/bin/python app/lvm/train_unified.py \
  --model-type amn \
  --data "$DATA" \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --device "$DEVICE" \
  --output-dir "$OUTPUT_DIR/amn" | tee "$OUTPUT_DIR/amn_training.log"

echo ""
echo -e "${GREEN}✓ AMN training complete${NC}"
echo ""

echo -e "${YELLOW}2/4 - Training LSTM...${NC}"
PYTHONPATH=. ./.venv/bin/python app/lvm/train_unified.py \
  --model-type lstm \
  --data "$DATA" \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --device "$DEVICE" \
  --output-dir "$OUTPUT_DIR/lstm" | tee "$OUTPUT_DIR/lstm_training.log"

echo ""
echo -e "${GREEN}✓ LSTM training complete${NC}"
echo ""

echo -e "${YELLOW}3/4 - Training GRU...${NC}"
PYTHONPATH=. ./.venv/bin/python app/lvm/train_unified.py \
  --model-type gru \
  --data "$DATA" \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --device "$DEVICE" \
  --output-dir "$OUTPUT_DIR/gru" | tee "$OUTPUT_DIR/gru_training.log"

echo ""
echo -e "${GREEN}✓ GRU training complete${NC}"
echo ""

echo -e "${YELLOW}4/4 - Training Transformer...${NC}"
PYTHONPATH=. ./.venv/bin/python app/lvm/train_unified.py \
  --model-type transformer \
  --data "$DATA" \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --device "$DEVICE" \
  --output-dir "$OUTPUT_DIR/transformer" | tee "$OUTPUT_DIR/transformer_training.log"

echo ""
echo -e "${GREEN}✓ Transformer training complete${NC}"
echo ""

# End timestamp
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
HOURS=$((DURATION / 3600))
MINUTES=$(((DURATION % 3600) / 60))

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}All 4 Models Trained Successfully!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "Total training time: ${HOURS}h ${MINUTES}m"
echo ""
echo "Models saved to:"
echo "  - $OUTPUT_DIR/amn/"
echo "  - $OUTPUT_DIR/lstm/"
echo "  - $OUTPUT_DIR/gru/"
echo "  - $OUTPUT_DIR/transformer/"
echo ""
echo "Training logs:"
echo "  - $OUTPUT_DIR/amn_training.log"
echo "  - $OUTPUT_DIR/lstm_training.log"
echo "  - $OUTPUT_DIR/gru_training.log"
echo "  - $OUTPUT_DIR/transformer_training.log"
echo ""
echo "Next step: Fair comparison (232k vs 340k vs 367k)!"
echo "  ./.venv/bin/python tools/fair_comparison_all_datasets.py"
echo ""
