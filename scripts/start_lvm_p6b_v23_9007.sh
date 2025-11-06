#!/bin/bash
# Start P6b v2.3 (arXiv-trained) on port 9007

# Kill existing process on 9007
lsof -ti :9007 | xargs kill -9 2>/dev/null

# Set Python path and activate venv
cd /Users/trentcarter/Artificial_Intelligence/AI_Projects/lnsp-phase-4

# Configure model
export LVM_MODEL_TYPE="transformer"
export LVM_MODEL_PATH="artifacts/lvm/models/transformer_p6b_v23_arxiv_20251104_200153/best_model.pt"
export LVM_DEVICE="mps"

# Configure encoder/decoder services (ports 7001/7002)
export ENCODER_URL="http://localhost:7001/encode"
export DECODER_URL="http://localhost:7002/decode"

# Launch service
echo "🚀 Starting P6b v2.3 (arXiv-trained) on port 9007..."
echo "   Model: $LVM_MODEL_PATH"
echo "   Device: $LVM_DEVICE"
echo "   Training: 87k arXiv sequences, Δ=+0.064"
echo "   Results: Val cosine=0.619, R@5=78.7%"
echo ""

./.venv/bin/python -c "
import os
import sys
sys.path.insert(0, '.')
from app.api.lvm_inference import config

# Override config
config.model_type = '$LVM_MODEL_TYPE'
config.model_path = '$LVM_MODEL_PATH'
config.device = '$LVM_DEVICE'
config.encoder_url = '$ENCODER_URL'
config.decoder_url = '$DECODER_URL'
" 2>&1 | head -5

./.venv/bin/uvicorn app.api.lvm_inference:app --host 127.0.0.1 --port 9007
