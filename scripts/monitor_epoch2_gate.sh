#!/bin/bash
# Monitor training and alert when epoch 2 completes

echo "Monitoring Mamba-S contrastive training for Epoch 2 gate..."
echo "Target: val_cosine ≥ 0.50 (was 0.22 with AR-only)"
echo ""

while true; do
    # Check if history.json exists and has epoch 2
    if [ -f "artifacts/lvm/models/mamba_s_contrastive/history.json" ]; then
        EPOCHS=$(python3 -c "import json; print(len(json.load(open('artifacts/lvm/models/mamba_s_contrastive/history.json'))))" 2>/dev/null || echo "0")
        
        if [ "$EPOCHS" -ge 2 ]; then
            echo ""
            echo "=========================================="
            echo "EPOCH 2 COMPLETE - RUNNING GATE CHECK"
            echo "=========================================="
            python3 tools/check_contrastive_sanity.py --epoch 2
            exit $?
        else
            echo "$(date +%H:%M:%S) - Training in progress: $EPOCHS/2 epochs completed"
        fi
    else
        echo "$(date +%H:%M:%S) - Waiting for training to start..."
    fi
    
    sleep 120  # Check every 2 minutes
done
