#!/usr/bin/env python3
"""
Launch P1 Baseline Transformer on port 9007
"""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import and configure before uvicorn starts
from app.api import lvm_inference

# Configure for P1 baseline
lvm_inference.config.model_type = "transformer"
lvm_inference.config.model_path = "artifacts/lvm/models/transformer_baseline_p1/best_model.pt"
lvm_inference.config.device = "mps"
lvm_inference.config.encoder_url = "http://localhost:7001/encode"
lvm_inference.config.decoder_url = "http://localhost:7002/decode"

print("="*60)
print("🚀 Launching P1 Baseline Transformer")
print("="*60)
print(f"Model: {lvm_inference.config.model_type}")
print(f"Path: {lvm_inference.config.model_path}")
print(f"Device: {lvm_inference.config.device}")
print(f"Port: 9007")
print("="*60)
print()

# Now start uvicorn
import uvicorn
uvicorn.run(
    "app.api.lvm_inference:app",
    host="127.0.0.1",
    port=9007,
    log_level="info"
)
