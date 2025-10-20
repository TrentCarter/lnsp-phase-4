#!/bin/bash
# Autonomous Improved Training Pipeline
# ======================================
#
# Phase 1: Quick test (2 epochs) to validate improvements
# Phase 2: If test passes, launch all 3 full training runs (20 epochs each)
#
# Expected runtime: 4-6 hours total
# Safe for 3-hour meeting - will run in background

set -e

LOG_DIR="/tmp/lnsp_improved_training"
mkdir -p $LOG_DIR

echo "=========================================="
echo "AUTONOMOUS IMPROVED TRAINING PIPELINE"
echo "=========================================="
echo ""
echo "Phase 1: Quick validation test (2 epochs)"
echo "Phase 2: Full training (3 models × 20 epochs)"
echo ""
echo "Logs: $LOG_DIR"
echo ""

# ==========================================
# PHASE 1: QUICK TEST (2 epochs)
# ==========================================

echo "🔬 PHASE 1: Running 2-epoch validation test..."
echo ""

TEST_LOG="$LOG_DIR/test_2epoch.log"

./.venv/bin/python -m app.lvm.train_improved \
  --model-type memory_gru \
  --data artifacts/lvm/data_extended/training_sequences_ctx100.npz \
  --epochs 2 \
  --batch-size 16 \
  --lr 0.0005 \
  --device mps \
  --coherence-threshold 0.0 \
  --lambda-mse 1.0 \
  --lambda-cosine 0.5 \
  --lambda-infonce 0.1 \
  --temperature 0.07 \
  --output-dir artifacts/lvm/models_improved/test_2epoch \
  2>&1 | tee $TEST_LOG

# Check if test completed successfully
if [ ${PIPESTATUS[0]} -ne 0 ]; then
    echo ""
    echo "❌ Test failed! Check log: $TEST_LOG"
    exit 1
fi

echo ""
echo "✅ Test completed successfully!"
echo ""

# Extract final validation metrics
FINAL_METRICS=$(tail -50 $TEST_LOG | grep -A 5 "TRAINING COMPLETE" || echo "Metrics not found")
echo "Test Results:"
echo "$FINAL_METRICS"
echo ""

# ==========================================
# PHASE 2: FULL TRAINING (20 epochs each)
# ==========================================

echo ""
echo "=========================================="
echo "🚀 PHASE 2: Launching full training runs"
echo "=========================================="
echo ""
echo "Training 3 models with improved trainer:"
echo "  1. Baseline GRU (control)"
echo "  2. Hierarchical GRU (Experiment A)"
echo "  3. Memory GRU (Experiment B)"
echo ""
echo "Each model: 20 epochs (~1.5 hours)"
echo "Total time: ~4.5 hours"
echo ""

# Model 1: Baseline GRU
echo "📊 Starting Baseline GRU training..."
./.venv/bin/python -m app.lvm.train_improved \
  --model-type gru \
  --data artifacts/lvm/data_extended/training_sequences_ctx100.npz \
  --epochs 20 \
  --batch-size 16 \
  --lr 0.0005 \
  --device mps \
  --coherence-threshold 0.0 \
  --lambda-mse 1.0 \
  --lambda-cosine 0.5 \
  --lambda-infonce 0.1 \
  --output-dir artifacts/lvm/models_improved/baseline_gru_final \
  > $LOG_DIR/baseline_gru.log 2>&1 &

BASELINE_PID=$!
echo "  Started (PID: $BASELINE_PID)"
echo "  Log: $LOG_DIR/baseline_gru.log"
echo ""

sleep 10  # Stagger starts to avoid memory spikes

# Model 2: Hierarchical GRU
echo "📊 Starting Hierarchical GRU training..."
./.venv/bin/python -m app.lvm.train_improved \
  --model-type hierarchical_gru \
  --data artifacts/lvm/data_extended/training_sequences_ctx100.npz \
  --epochs 20 \
  --batch-size 16 \
  --lr 0.0005 \
  --device mps \
  --coherence-threshold 0.0 \
  --lambda-mse 1.0 \
  --lambda-cosine 0.5 \
  --lambda-infonce 0.1 \
  --output-dir artifacts/lvm/models_improved/hierarchical_gru_final \
  > $LOG_DIR/hierarchical_gru.log 2>&1 &

HIERARCHICAL_PID=$!
echo "  Started (PID: $HIERARCHICAL_PID)"
echo "  Log: $LOG_DIR/hierarchical_gru.log"
echo ""

sleep 10

# Model 3: Memory GRU
echo "📊 Starting Memory GRU training..."
./.venv/bin/python -m app.lvm.train_improved \
  --model-type memory_gru \
  --data artifacts/lvm/data_extended/training_sequences_ctx100.npz \
  --epochs 20 \
  --batch-size 16 \
  --lr 0.0005 \
  --device mps \
  --coherence-threshold 0.0 \
  --lambda-mse 1.0 \
  --lambda-cosine 0.5 \
  --lambda-infonce 0.1 \
  --output-dir artifacts/lvm/models_improved/memory_gru_final \
  > $LOG_DIR/memory_gru.log 2>&1 &

MEMORY_PID=$!
echo "  Started (PID: $MEMORY_PID)"
echo "  Log: $LOG_DIR/memory_gru.log"
echo ""

# Save PIDs for monitoring
echo "$BASELINE_PID" > $LOG_DIR/baseline_gru.pid
echo "$HIERARCHICAL_PID" > $LOG_DIR/hierarchical_gru.pid
echo "$MEMORY_PID" > $LOG_DIR/memory_gru.pid

echo ""
echo "=========================================="
echo "✅ ALL TRAININGS LAUNCHED"
echo "=========================================="
echo ""
echo "Running in background. PIDs:"
echo "  Baseline GRU:     $BASELINE_PID"
echo "  Hierarchical GRU: $HIERARCHICAL_PID"
echo "  Memory GRU:       $MEMORY_PID"
echo ""
echo "Monitor progress:"
echo "  tail -f $LOG_DIR/baseline_gru.log"
echo "  tail -f $LOG_DIR/hierarchical_gru.log"
echo "  tail -f $LOG_DIR/memory_gru.log"
echo ""
echo "Check status:"
echo "  ps -p $BASELINE_PID $HIERARCHICAL_PID $MEMORY_PID"
echo ""
echo "Expected completion: ~4.5 hours from now"
echo ""

# Create a monitoring script
cat > $LOG_DIR/monitor.sh << 'EOF'
#!/bin/bash
LOG_DIR="/tmp/lnsp_improved_training"

echo "=== Training Status ==="
echo ""

for model in baseline_gru hierarchical_gru memory_gru; do
    PID=$(cat $LOG_DIR/${model}.pid 2>/dev/null)
    if [ -n "$PID" ] && ps -p $PID > /dev/null 2>&1; then
        echo "✅ ${model}: Running (PID: $PID)"
        # Show last progress line
        tail -20 $LOG_DIR/${model}.log | grep -E "(Epoch|Val:|Hit@)" | tail -3
    elif [ -f "$LOG_DIR/${model}.log" ]; then
        echo "✓ ${model}: Complete"
        tail -10 $LOG_DIR/${model}.log | grep -E "Best|Final|COMPLETE"
    else
        echo "⏳ ${model}: Not started"
    fi
    echo ""
done

echo "Full logs:"
echo "  tail -f $LOG_DIR/*.log"
EOF

chmod +x $LOG_DIR/monitor.sh

echo "Quick status check:"
echo "  $LOG_DIR/monitor.sh"
echo ""
echo "Autonomous training pipeline launched! 🚀"
echo ""
