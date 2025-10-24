#!/usr/bin/env python3
"""
Simple LVM Test Script

Quick test to verify model loading and basic inference works.
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import sys

# Import LVM models
sys.path.insert(0, 'app/lvm')
from models import create_model

def test_single_model():
    """Test loading and running a single model."""

    print("🔬 Simple LVM Test")
    print("=" * 50)

    # Device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    # Model paths
    model_paths = {
        'amn': 'artifacts/lvm/production_model',
        'transformer': 'artifacts/lvm/fallback_accuracy',
        'gru': 'artifacts/lvm/fallback_secondary'
    }

    # Test each model
    for model_name, model_path in model_paths.items():
        print(f"\n📦 Testing {model_name.upper()}...")

        model_dir = Path(model_path)
        if not model_dir.exists():
            print(f"  ⚠️  Model path not found: {model_path}")
            continue

        checkpoint_path = model_dir / "best_model.pt"
        if not checkpoint_path.exists():
            checkpoint_path = model_dir / "final_model.pt"

        if not checkpoint_path.exists():
            print(f"  ⚠️  No checkpoint found")
            continue

        try:
            # Load checkpoint
            print(f"  Loading checkpoint from {checkpoint_path}...")
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

            # Create model
            hparams = checkpoint.get('hyperparameters', {})
            print(f"  Hyperparameters: {hparams}")

            model = create_model(model_name, **hparams)
            model.load_state_dict(checkpoint['model_state_dict'])
            model = model.to(device)
            model.eval()

            print(f"  ✅ Model loaded: {model.count_parameters():,} parameters")

            # Test inference
            print("  Running test inference...")
            test_input = torch.randn(1, 5, 768).to(device)

            with torch.no_grad():
                output = model(test_input)

            print(f"  ✅ Inference successful: {output.shape}")
            print(f"  Output norm: {output.norm().item():.4f}")

        except Exception as e:
            print(f"  ❌ Failed: {e}")
            continue

    print(f"\n{'='*50}")
    print("Test complete!")

if __name__ == "__main__":
    test_single_model()
