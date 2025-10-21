#!/bin/bash
# Watch Phase 1 training, generate report when complete, then launch Phase 2

set -e

PHASE1_PID=94937
PHASE1_DIR="runs/twotower_v3_phase1"
REPORT_FILE="TWO_TOWER_V3_PHASE1_REPORT.md"

echo "=== TWO-TOWER TRAINING WATCHER ==="
echo "Started: $(date)"
echo "Watching PID: $PHASE1_PID"
echo ""

# Wait for Phase 1 to complete
echo "Waiting for Phase 1 training to complete..."
while ps -p $PHASE1_PID > /dev/null 2>&1; do
    sleep 30
    CURRENT_TIME=$(date +%H:%M:%S)
    EPOCHS=$(grep -c "^Epoch" "$PHASE1_DIR/training.log" 2>/dev/null || echo "?")
    echo "  [$CURRENT_TIME] Still running... (Epochs completed: $EPOCHS)"
done

echo ""
echo "✓ Phase 1 training completed!"
echo "  Finished: $(date)"
echo ""

# Give it a moment to flush final logs
sleep 5

# Check final results
echo "=== PHASE 1 FINAL RESULTS ==="
if [ -f "$PHASE1_DIR/history.json" ]; then
    echo "History file found. Extracting final metrics..."
    ./.venv/bin/python3 -c "
import json
import sys

with open('$PHASE1_DIR/history.json') as f:
    history = json.load(f)

if history:
    final = history[-1]
    print(f\"  Final epoch: {final['epoch']}\")
    print(f\"  Train loss: {final['train_loss']:.4f}\")
    print(f\"  Recall@10: {final['recall@10']:.2f}%\")
    print(f\"  Recall@100: {final['recall@100']:.2f}%\")
    print(f\"  Recall@500: {final['recall@500']:.2f}%\")
    print(f\"  Recall@1000: {final['recall@1000']:.2f}%\")

    # Find best recall@500
    best = max(history, key=lambda x: x['recall@500'])
    print(f\"\\n  Best Recall@500: {best['recall@500']:.2f}% (Epoch {best['epoch']})\")
"
else
    echo "No history file found yet. Checking training log..."
    grep -E "Recall@500|New best" "$PHASE1_DIR/training.log" | tail -10
fi
echo ""

# Generate comprehensive report
echo "=== GENERATING COMPREHENSIVE REPORT ==="
echo "Report will be saved to: $REPORT_FILE"
echo ""

# This is a placeholder - will be filled by Claude
echo "NOTE: Report generation will be done by Claude Code"
echo ""

# Prompt user for Phase 2 launch
echo "=== READY TO LAUNCH PHASE 2 ==="
echo ""
echo "Phase 1 is complete. To launch Phase 2 overnight training, run:"
echo ""
echo "  ./launch_phase2_overnight.sh"
echo ""
echo "This will start Phase 2 with:"
echo "  - Hard negative mining (every 2 epochs)"
echo "  - Memory bank (20k vectors)"
echo "  - 50 epochs total"
echo "  - Initialized from Phase 1 best checkpoint"
echo ""

echo "Watcher completed: $(date)"
