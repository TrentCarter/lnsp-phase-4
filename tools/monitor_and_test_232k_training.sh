#!/bin/bash
##
# Autonomous 80k vs 232k Training Monitor & Tester
#
# This script runs autonomously and:
# 1. Checks training progress every 30 minutes
# 2. Tests each model as it completes (80k vs 232k)
# 3. Shows comparison results
# 4. Continues until all 5 models are done
##

set -e

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="artifacts/lvm/80k_vs_232k_results"
LOG_FILE="logs/monitor_232k_${TIMESTAMP}.log"

mkdir -p "$RESULTS_DIR"
mkdir -p logs

echo "================================================================================" | tee -a "$LOG_FILE"
echo "Autonomous 80k vs 232k Training Monitor Started" | tee -a "$LOG_FILE"
echo "================================================================================" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "Started: $(date)" | tee -a "$LOG_FILE"
echo "Results dir: $RESULTS_DIR" | tee -a "$LOG_FILE"
echo "Log file: $LOG_FILE" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Track completed models
declare -A tested_models
tested_models[amn]=0
tested_models[lstm]=0
tested_models[gru]=0
tested_models[transformer]=0
tested_models[graphmert]=0

# Model order
models=("amn" "lstm" "gru" "transformer" "graphmert")
model_80k_paths=(
    "artifacts/lvm/models/amn_20251016_133427/best_model.pt"
    "artifacts/lvm/models/lstm_20251016_133934/best_model.pt"
    "artifacts/lvm/models/gru_20251016_134451/best_model.pt"
    "artifacts/lvm/models/transformer_20251016_135606/best_model.pt"
    "artifacts/lvm/models/graphmert_lvm_80k_full/benchmark_model.pt"
)

# Function to find latest 232k model
find_232k_model() {
    local model_type=$1
    local pattern="artifacts/lvm/models/${model_type}_232k_*/best_model.pt"

    if [ "$model_type" = "graphmert" ]; then
        pattern="artifacts/lvm/models/graphmert_lvm_232k_*/benchmark_model.pt"
    fi

    local found=$(ls -t $pattern 2>/dev/null | head -1)
    echo "$found"
}

# Function to test a single model comparison
test_model_comparison() {
    local model_name=$1
    local model_80k=$2
    local model_232k=$3

    echo "" | tee -a "$LOG_FILE"
    echo "================================================================================" | tee -a "$LOG_FILE"
    echo "Testing $model_name: 80k vs 232k" | tee -a "$LOG_FILE"
    echo "================================================================================" | tee -a "$LOG_FILE"
    echo "$(date)" | tee -a "$LOG_FILE"
    echo "" | tee -a "$LOG_FILE"

    # Quick test with 5 examples
    local test_output="${RESULTS_DIR}/${model_name}_comparison.txt"

    echo "80k model: $model_80k" | tee -a "$test_output"
    echo "232k model: $model_232k" | tee -a "$test_output"
    echo "" | tee -a "$test_output"

    # Run quick comparison test
    ./.venv/bin/python -c "
import torch
import numpy as np
import sys
sys.path.insert(0, 'app/lvm')

# Test texts
test_texts = [
    'Artificial intelligence is transforming modern technology.',
    'The quick brown fox jumps over the lazy dog.',
    'Machine learning models learn patterns from data.',
    'Climate change is a pressing global challenge.',
    'The human brain contains billions of neurons.'
]

print('Model: $model_name')
print('='*80)
print()

# Load models
try:
    if '$model_name' == 'graphmert':
        from graphmert_lvm_768d import GraphMERTLVM768D

        # 80k model
        ckpt_80k = torch.load('$model_80k', map_location='cpu')
        model_80k = GraphMERTLVM768D(d_model=768, n_layers=12, n_heads=8, d_ff=2048, dropout=0.1, lambda_decay=0.6)
        model_80k.load_state_dict(ckpt_80k['model_state_dict'])
        model_80k.eval()

        # 232k model
        ckpt_232k = torch.load('$model_232k', map_location='cpu')
        model_232k = GraphMERTLVM768D(d_model=768, n_layers=12, n_heads=8, d_ff=2048, dropout=0.1, lambda_decay=0.6)
        model_232k.load_state_dict(ckpt_232k['model_state_dict'])
        model_232k.eval()

        val_cosine_80k = max([e.get('val_cosine', 0) for e in ckpt_80k.get('history', [])])
        val_cosine_232k = max([e.get('val_cosine', 0) for e in ckpt_232k.get('history', [])])
    else:
        from models import create_model

        # 80k model
        ckpt_80k = torch.load('$model_80k', map_location='cpu')
        model_80k = create_model(ckpt_80k['model_type'], **ckpt_80k.get('model_config', {}))
        model_80k.load_state_dict(ckpt_80k['model_state_dict'])
        model_80k.eval()
        val_cosine_80k = ckpt_80k.get('val_cosine', 0)

        # 232k model
        ckpt_232k = torch.load('$model_232k', map_location='cpu')
        model_232k = create_model(ckpt_232k['model_type'], **ckpt_232k.get('model_config', {}))
        model_232k.load_state_dict(ckpt_232k['model_state_dict'])
        model_232k.eval()
        val_cosine_232k = ckpt_232k.get('val_cosine', 0)

    print(f'80k  Training Val Cosine: {val_cosine_80k:.4f}')
    print(f'232k Training Val Cosine: {val_cosine_232k:.4f}')
    print()

    # Quick inference test
    test_vector = torch.randn(1, 5, 768)

    with torch.no_grad():
        out_80k = model_80k(test_vector)
        out_232k = model_232k(test_vector)

    print(f'✓ Both models loaded and tested successfully')
    print(f'  80k output shape: {out_80k.shape}')
    print(f'  232k output shape: {out_232k.shape}')
    print()

    # Summary
    improvement = ((val_cosine_232k - val_cosine_80k) / val_cosine_80k) * 100 if val_cosine_80k > 0 else 0
    print('SUMMARY:')
    print(f'  Training improvement: {improvement:+.2f}%')
    print(f'  80k → 232k: {val_cosine_80k:.4f} → {val_cosine_232k:.4f}')

    if improvement > 0:
        print(f'  ✅ 232k model is BETTER (+{improvement:.2f}%)')
    elif improvement < -1:
        print(f'  ⚠️  232k model is WORSE ({improvement:.2f}%)')
    else:
        print(f'  ≈  Similar performance')

except Exception as e:
    print(f'✗ Error testing model: {str(e)[:200]}')
    import traceback
    traceback.print_exc()
" 2>&1 | tee -a "$test_output" | tee -a "$LOG_FILE"

    echo "" | tee -a "$LOG_FILE"
    echo "✓ Results saved to: $test_output" | tee -a "$LOG_FILE"
    echo "" | tee -a "$LOG_FILE"
}

# Main monitoring loop
iteration=0
all_done=false

while [ "$all_done" = false ]; do
    iteration=$((iteration + 1))

    echo "================================================================================" | tee -a "$LOG_FILE"
    echo "Check #$iteration - $(date)" | tee -a "$LOG_FILE"
    echo "================================================================================" | tee -a "$LOG_FILE"
    echo "" | tee -a "$LOG_FILE"

    # Check each model
    models_completed=0

    for i in "${!models[@]}"; do
        model="${models[$i]}"
        model_80k="${model_80k_paths[$i]}"

        # Skip if already tested
        if [ "${tested_models[$model]}" -eq 1 ]; then
            models_completed=$((models_completed + 1))
            continue
        fi

        # Check if 232k model exists
        model_232k=$(find_232k_model "$model")

        if [ -n "$model_232k" ] && [ -f "$model_232k" ]; then
            echo "✓ $model 232k model found: $model_232k" | tee -a "$LOG_FILE"

            # Test it!
            test_model_comparison "$model" "$model_80k" "$model_232k"

            # Mark as tested
            tested_models[$model]=1
            models_completed=$((models_completed + 1))
        else
            echo "⏳ $model still training..." | tee -a "$LOG_FILE"
        fi
    done

    echo "" | tee -a "$LOG_FILE"
    echo "Progress: $models_completed/5 models completed and tested" | tee -a "$LOG_FILE"
    echo "" | tee -a "$LOG_FILE"

    # Check if all done
    if [ "$models_completed" -eq 5 ]; then
        all_done=true
        echo "================================================================================" | tee -a "$LOG_FILE"
        echo "🎉 ALL MODELS COMPLETED AND TESTED!" | tee -a "$LOG_FILE"
        echo "================================================================================" | tee -a "$LOG_FILE"
        echo "" | tee -a "$LOG_FILE"
        echo "Finished: $(date)" | tee -a "$LOG_FILE"
        echo "" | tee -a "$LOG_FILE"
        echo "Results directory: $RESULTS_DIR" | tee -a "$LOG_FILE"
        echo "" | tee -a "$LOG_FILE"
        echo "Individual comparisons:" | tee -a "$LOG_FILE"
        ls -lh "$RESULTS_DIR"/*.txt | tee -a "$LOG_FILE"
        break
    fi

    # Wait 30 minutes before next check
    echo "Waiting 30 minutes before next check..." | tee -a "$LOG_FILE"
    echo "" | tee -a "$LOG_FILE"
    sleep 1800  # 30 minutes
done

echo "" | tee -a "$LOG_FILE"
echo "Monitor script completed!" | tee -a "$LOG_FILE"
