#!/bin/bash
# Run P7 hyperparameter grid search (4 experiments)
#
# Recommended grid:
#   1. Baseline: context=5, margin=0.07, lambda=0.8
#   2. Strong anchoring: context=5, margin=0.07, lambda=0.6
#   3. Higher margin: context=5, margin=0.10, lambda=0.8
#   4. Larger context: context=7, margin=0.07, lambda=0.8
#
# Total time: ~10-12 hours on MPS (2.5-3 hrs per experiment)

set -e

# Ensure we're in project root
cd /Users/trentcarter/Artificial_Intelligence/AI_Projects/lnsp-phase-4

# Create results directory
RESULTS_DIR="artifacts/lvm/p7_grid_results_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULTS_DIR"

echo "=========================================="
echo "P7 DIRECTIONAL RANKER - GRID SEARCH"
echo "=========================================="
echo "Running 4 experiments:"
echo "  1. Baseline (ctx=5, m=0.07, λ=0.8)"
echo "  2. Strong anchor (ctx=5, m=0.07, λ=0.6)"
echo "  3. Higher margin (ctx=5, m=0.10, λ=0.8)"
echo "  4. Larger context (ctx=7, m=0.07, λ=0.8)"
echo ""
echo "Results will be saved to: $RESULTS_DIR"
echo "Estimated total time: 10-12 hours"
echo "=========================================="
echo ""

# Function to run experiment and log results
run_experiment() {
    local name=$1
    local context=$2
    local margin=$3
    local lambda=$4

    echo ""
    echo "=========================================="
    echo "EXPERIMENT: $name"
    echo "  Context: $context"
    echo "  Margin: $margin"
    echo "  Lambda: $lambda"
    echo "  Start time: $(date)"
    echo "=========================================="

    # Run training
    if ./scripts/train_p7_ranker.sh \
        --context "$context" \
        --margin "$margin" \
        --lambda "$lambda" \
        --epochs 10 \
        --device mps > "$RESULTS_DIR/${name}.log" 2>&1; then

        echo "✅ COMPLETED: $name"
        echo "   End time: $(date)"

        # Find the model directory (most recent with matching params)
        MODEL_DIR=$(ls -td artifacts/lvm/models/p7_ranker_c${context}_m${margin}_l${lambda}_* 2>/dev/null | head -1)

        if [ -n "$MODEL_DIR" ]; then
            # Copy training history to results
            if [ -f "$MODEL_DIR/training_history.json" ]; then
                cp "$MODEL_DIR/training_history.json" "$RESULTS_DIR/${name}_history.json"
            fi

            # Extract final metrics
            if [ -f "$MODEL_DIR/training_history.json" ]; then
                echo "   Model dir: $MODEL_DIR" >> "$RESULTS_DIR/${name}_summary.txt"
                tail -20 "$MODEL_DIR/training_history.json" >> "$RESULTS_DIR/${name}_summary.txt"
            fi
        fi

        return 0
    else
        echo "❌ FAILED: $name"
        echo "   Check log: $RESULTS_DIR/${name}.log"
        return 1
    fi
}

# Track start time
GRID_START=$(date +%s)

# Experiment 1: Baseline
run_experiment "exp1_baseline" 5 0.07 0.8 || echo "⚠️  Experiment 1 failed, continuing..."

# Experiment 2: Strong anchor
run_experiment "exp2_strong_anchor" 5 0.07 0.6 || echo "⚠️  Experiment 2 failed, continuing..."

# Experiment 3: Higher margin
run_experiment "exp3_higher_margin" 5 0.10 0.8 || echo "⚠️  Experiment 3 failed, continuing..."

# Experiment 4: Larger context
run_experiment "exp4_larger_context" 7 0.07 0.8 || echo "⚠️  Experiment 4 failed, continuing..."

# Calculate total time
GRID_END=$(date +%s)
GRID_DURATION=$((GRID_END - GRID_START))
GRID_HOURS=$((GRID_DURATION / 3600))
GRID_MINS=$(( (GRID_DURATION % 3600) / 60 ))

echo ""
echo "=========================================="
echo "GRID SEARCH COMPLETE"
echo "=========================================="
echo "Total time: ${GRID_HOURS}h ${GRID_MINS}m"
echo "Results saved to: $RESULTS_DIR"
echo ""
echo "Summary of experiments:"
echo "  1. Baseline:       $RESULTS_DIR/exp1_baseline_summary.txt"
echo "  2. Strong anchor:  $RESULTS_DIR/exp2_strong_anchor_summary.txt"
echo "  3. Higher margin:  $RESULTS_DIR/exp3_higher_margin_summary.txt"
echo "  4. Larger context: $RESULTS_DIR/exp4_larger_context_summary.txt"
echo ""
echo "Next steps:"
echo "  1. Compare results: ./scripts/compare_p7_results.sh $RESULTS_DIR"
echo "  2. Select best model based on margin (target: ≥ +0.20)"
echo "  3. Validate with directional test"
echo "  4. Deploy if passing ship criteria"
echo "=========================================="

# Create comparison script
cat > "$RESULTS_DIR/compare_results.sh" << 'EOF'
#!/bin/bash
# Quick comparison of P7 grid results

echo "P7 Grid Results Comparison"
echo "=========================================="

for exp in exp1_baseline exp2_strong_anchor exp3_higher_margin exp4_larger_context; do
    if [ -f "${exp}_history.json" ]; then
        echo ""
        echo "$exp:"
        # Extract final epoch metrics
        python3 -c "
import json
with open('${exp}_history.json', 'r') as f:
    history = json.load(f)
    if history:
        final = history[-1]
        print(f\"  Epoch: {final.get('epoch', 'N/A')}\")
        val = final.get('val', {})
        print(f\"  Margin: {val.get('margin', 'N/A'):.4f}\")
        print(f\"  cos_next: {val.get('cos_next', 'N/A'):.4f}\")
        print(f\"  cos_prev: {val.get('cos_prev', 'N/A'):.4f}\")
        print(f\"  cos_anchor: {val.get('cos_anchor', 'N/A'):.4f}\")
        print(f\"  anchor_λ: {final.get('anchor_lambda', 'N/A'):.3f}\")
" 2>/dev/null || echo "  [Could not parse history]"
    else
        echo ""
        echo "$exp: [No results]"
    fi
done

echo ""
echo "=========================================="
EOF

chmod +x "$RESULTS_DIR/compare_results.sh"

echo ""
echo "To compare results now, run:"
echo "  cd $RESULTS_DIR && ./compare_results.sh"
