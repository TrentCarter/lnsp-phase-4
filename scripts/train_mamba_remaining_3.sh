#!/bin/bash
# Complete Training for Mamba-XL, Mamba-Sandwich, and Mamba-GR
# These models only completed 1/20 epochs and need to be fully trained
# according to PRD_5_Mamba_Models.md specifications

set -e

export KMP_DUPLICATE_LIB_OK=TRUE  # Critical for macOS OpenMP fix

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="./logs/mamba_phase5"
mkdir -p "$LOG_DIR"

echo "================================================================================
Launching Mamba Phase-5 Remaining Models
================================================================================
Timestamp: $TIMESTAMP
Models: Mamba-XL, Mamba-Sandwich, Mamba-GR
Strategy: Restart from scratch (only 1 epoch wasted)
Estimated time: 6-8 hours total (parallel on MPS)
================================================================================
"

# Backup incomplete checkpoints
echo "Backing up incomplete checkpoints..."
for model in mamba_xl mamba_sandwich mamba_gr; do
    if [ -d "./artifacts/lvm/models/$model" ]; then
        mv "./artifacts/lvm/models/$model" "./artifacts/lvm/models/${model}_epoch1_backup_${TIMESTAMP}"
        echo "  Backed up: $model -> ${model}_epoch1_backup_${TIMESTAMP}"
    fi
done

echo ""
echo "Launching trainings in parallel..."
echo ""

# Model C: Mamba-XL (Deeper/Wider Pure SSM)
echo "1/3 Launching Mamba-XL..."
nohup ./.venv/bin/python app/lvm/train_mamba_unified.py \
    --model-type mamba_xl \
    --d-model 384 \
    --n-layers 16 \
    --d-state 192 \
    --conv-sz 4 \
    --expand 2 \
    --epochs 20 \
    --batch-size 768 \
    --device mps \
    > "$LOG_DIR/mamba_xl_${TIMESTAMP}.log" 2>&1 &
MAMBA_XL_PID=$!
echo "  PID: $MAMBA_XL_PID"
echo "  Log: $LOG_DIR/mamba_xl_${TIMESTAMP}.log"

sleep 5

# Model D: Mamba-Sandwich (Attn→SSM→Attn)
echo ""
echo "2/3 Launching Mamba-Sandwich..."
nohup ./.venv/bin/python app/lvm/train_mamba_unified.py \
    --model-type mamba_sandwich \
    --d-model 320 \
    --n-layers-mamba 8 \
    --n-layers-local 4 \
    --local-attn-win 8 \
    --d-state 160 \
    --conv-sz 4 \
    --epochs 20 \
    --batch-size 896 \
    --device mps \
    > "$LOG_DIR/mamba_sandwich_${TIMESTAMP}.log" 2>&1 &
MAMBA_SANDWICH_PID=$!
echo "  PID: $MAMBA_SANDWICH_PID"
echo "  Log: $LOG_DIR/mamba_sandwich_${TIMESTAMP}.log"

sleep 5

# Model E: Mamba-GR (SSM + GRU Gate)
# NOTE: Fixed d_state from 128 → 144 per PRD specification
echo ""
echo "3/3 Launching Mamba-GR (FIXED: d_state=144)..."
nohup ./.venv/bin/python app/lvm/train_mamba_unified.py \
    --model-type mamba_gr \
    --d-model 288 \
    --n-layers 10 \
    --d-state 144 \
    --conv-sz 4 \
    --gru-hidden 256 \
    --epochs 20 \
    --batch-size 1024 \
    --device mps \
    > "$LOG_DIR/mamba_gr_${TIMESTAMP}.log" 2>&1 &
MAMBA_GR_PID=$!
echo "  PID: $MAMBA_GR_PID"
echo "  Log: $LOG_DIR/mamba_gr_${TIMESTAMP}.log"

echo ""
echo "================================================================================
All 3 trainings launched in background!
================================================================================

Process IDs:
  Mamba-XL:        $MAMBA_XL_PID
  Mamba-Sandwich:  $MAMBA_SANDWICH_PID
  Mamba-GR:        $MAMBA_GR_PID

Monitor progress:
  tail -f $LOG_DIR/mamba_xl_${TIMESTAMP}.log
  tail -f $LOG_DIR/mamba_sandwich_${TIMESTAMP}.log
  tail -f $LOG_DIR/mamba_gr_${TIMESTAMP}.log

Check status:
  ps aux | grep train_mamba_unified

Estimated completion: $(date -v+8H '+%Y-%m-%d %H:%M:%S')

================================================================================
"

# Save PIDs for easy monitoring
echo "$MAMBA_XL_PID" > /tmp/mamba_xl.pid
echo "$MAMBA_SANDWICH_PID" > /tmp/mamba_sandwich.pid
echo "$MAMBA_GR_PID" > /tmp/mamba_gr.pid

echo "PIDs saved to /tmp/mamba_*.pid for reference"
echo ""
