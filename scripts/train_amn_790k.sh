#!/bin/bash
# Train AMN on 790k Wikipedia Dataset
# =====================================
#
# This script trains the AMN (Attention Mixture Network) model on 790,391 concepts
# from Wikipedia articles 1-15,023.
#
# Duration: 6-8 hours on Apple Silicon M1 Max
# Output: artifacts/lvm/models/amn_790k_<timestamp>/
#
# Usage:
#   bash scripts/train_amn_790k.sh
#
# Recovery (if crashed):
#   # Check last checkpoint
#   ls -lht artifacts/lvm/models/amn_790k_*/
#   # Resume from checkpoint
#   export RESUME_CHECKPOINT=artifacts/lvm/models/amn_790k_<timestamp>/epoch_XX.pt
#   bash scripts/train_amn_790k.sh

set -e  # Exit on error

# ============================================================================
# CRITICAL: macOS OpenMP Fix
# ============================================================================
# Prevents "Abort trap: 6" crashes from duplicate OpenMP libraries
export KMP_DUPLICATE_LIB_OK=TRUE

# ============================================================================
# Configuration
# ============================================================================
PROJECT_ROOT="/Users/trentcarter/Artificial_Intelligence/AI_Projects/lnsp-phase-4"
cd "$PROJECT_ROOT"

# Data files (trainer does its own 90/10 split, so use unsplit file)
DATA_FILE="artifacts/lvm/training_sequences_ctx5.npz"
MODEL_DIR="artifacts/lvm/models/amn_790k_$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$MODEL_DIR/training.log"

# Training hyperparameters (matching 584k training)
EPOCHS=20
BATCH_SIZE=32
LEARNING_RATE=0.0005
DEVICE="mps"  # Apple Silicon

# ============================================================================
# Pre-flight Checks
# ============================================================================
echo "=" | awk '{for(i=1;i<=80;i++) printf "="; print ""}'
echo "AMN TRAINING - 790K DATASET"
echo "=" | awk '{for(i=1;i<=80;i++) printf "="; print ""}'
echo ""

# Check data file exists
if [ ! -f "$DATA_FILE" ]; then
    echo "❌ ERROR: Training data not found!"
    echo "   Expected: $DATA_FILE"
    echo "   Run data preparation first."
    exit 1
fi

# Check Python environment
if [ ! -f ".venv/bin/python" ]; then
    echo "❌ ERROR: Virtual environment not found!"
    echo "   Expected: .venv/bin/python"
    exit 1
fi

echo "✅ Pre-flight checks passed"
echo ""
echo "📊 Configuration:"
echo "   Model: AMN (Attention Mixture Network)"
echo "   Data: $DATA_FILE"
echo "   Output: $MODEL_DIR"
echo "   Epochs: $EPOCHS"
echo "   Batch Size: $BATCH_SIZE"
echo "   Learning Rate: $LEARNING_RATE"
echo "   Device: $DEVICE"
echo ""

# Create model directory
mkdir -p "$MODEL_DIR"

# ============================================================================
# Training
# ============================================================================
echo "🚀 Starting training..."
echo "   Log: $LOG_FILE"
echo ""
echo "   ⏱️  Estimated duration: 6-8 hours"
echo "   💾 Checkpoints saved every epoch"
echo "   🛑 Press Ctrl+C to stop gracefully"
echo ""

# Check if resuming from checkpoint
RESUME_FLAG=""
if [ -n "$RESUME_CHECKPOINT" ]; then
    echo "🔄 Resuming from checkpoint: $RESUME_CHECKPOINT"
    RESUME_FLAG="--resume $RESUME_CHECKPOINT"
fi

# Start training (PROVEN: MSE-only like 584k successful model)
./.venv/bin/python app/lvm/train_unified.py \
    --model-type amn \
    --data "$DATA_FILE" \
    --output-dir "$MODEL_DIR" \
    --epochs 20 \
    --batch-size $BATCH_SIZE \
    --lr $LEARNING_RATE \
    --device $DEVICE \
    --lambda-mse 1.0 \
    --lambda-info 0.0 \
    $RESUME_FLAG \
    2>&1 | tee "$LOG_FILE"

# ============================================================================
# Post-Training
# ============================================================================
echo ""
echo "=" | awk '{for(i=1;i<=80;i++) printf "="; print ""}'
echo "✅ TRAINING COMPLETE!"
echo "=" | awk '{for(i=1;i<=80;i++) printf "="; print ""}'
echo ""
echo "📁 Model saved to: $MODEL_DIR"
echo "📊 Training log: $LOG_FILE"
echo ""

# Find best checkpoint
BEST_MODEL=$(find "$MODEL_DIR" -name "best_model.pt" 2>/dev/null || echo "")
if [ -n "$BEST_MODEL" ]; then
    echo "🏆 Best model: $BEST_MODEL"

    # Read final metrics from checkpoint
    python -c "
import torch
ckpt = torch.load('$BEST_MODEL', map_location='cpu', weights_only=False)
print(f\"   Val Cosine: {ckpt.get('val_cosine', 'N/A'):.4f}\")
print(f\"   Epoch: {ckpt.get('epoch', 'N/A')}\")
" 2>/dev/null || echo "   (metrics not available)"
fi

echo ""
echo "📝 Next steps:"
echo "   1. Evaluate OOD performance:"
echo "      ./.venv/bin/python tools/eval_amn_ood.py --model $MODEL_DIR/best_model.pt"
echo ""
echo "   2. Update recovery log:"
echo "      # Mark AMN training as complete in artifacts/lvm/TRAINING_RECOVERY_LOG.md"
echo ""
echo "   3. Start next model:"
echo "      bash scripts/train_lstm_790k.sh"
echo ""
echo "=" | awk '{for(i=1;i<=80;i++) printf "="; print ""}'
