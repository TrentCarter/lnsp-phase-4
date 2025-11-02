#!/bin/bash
#
# Train LVM Models with 5CAT Validation
# ======================================
#
# Trains LVM models on clean 584k data with integrated 5→1 Causal Alignment Testing.
# Runs 5CAT every 5 epochs to detect backward prediction bias early.
#
# Usage:
#   ./scripts/train_with_5cat_validation.sh transformer
#   ./scripts/train_with_5cat_validation.sh amn
#   ./scripts/train_with_5cat_validation.sh gru
#   ./scripts/train_with_5cat_validation.sh lstm
#
# Features:
# - Uses CLEAN 584k data (coherence 0.457, strong temporal signal)
# - Runs 5CAT validation every 5 epochs
# - Alerts if backward bias detected (margin < 0)
# - Early stopping if bias is severe
# - Saves best model based on Val + 5CAT metrics

set -e

# Parse arguments
MODEL_TYPE="${1:-transformer}"
EPOCHS="${2:-20}"
DEVICE="${3:-mps}"

echo "============================================"
echo "Train LVM with 5CAT Validation"
echo "============================================"
echo "Model:  $MODEL_TYPE"
echo "Epochs: $EPOCHS"
echo "Device: $DEVICE"
echo ""

# Check device availability (use venv Python)
if [ "$DEVICE" = "mps" ]; then
    if ! ./.venv/bin/python3 -c "import torch; assert torch.backends.mps.is_available()" 2>/dev/null; then
        echo "⚠️  MPS not available, falling back to CPU"
        DEVICE="cpu"
    else
        echo "✅ Using MPS (Apple Silicon GPU) - Training will be 20-40x faster!"
    fi
elif [ "$DEVICE" = "cpu" ]; then
    echo "⚠️  Using CPU (SLOW) - Consider using 'mps' for 20-40x speedup"
fi

# Paths
TRAIN_DATA="artifacts/lvm/training_sequences_ctx5_584k_clean_splits.npz"
VAL_DATA="artifacts/lvm/validation_sequences_ctx5_articles4000-4499_compat.npz"
OOD_DATA="artifacts/lvm/ood_sequences_ctx5_articles1500-1999.npz"
ARTICLES_DATA="artifacts/wikipedia_584k_fresh.npz"

# Verify data files exist
echo "📥 Verifying data files..."
for file in "$TRAIN_DATA" "$VAL_DATA" "$OOD_DATA" "$ARTICLES_DATA"; do
    if [ ! -f "$file" ]; then
        echo "❌ ERROR: Data file not found: $file"
        exit 1
    fi
    echo "   ✅ $file"
done
echo ""

# Create output directory
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="artifacts/lvm/models/${MODEL_TYPE}_5cat_${TIMESTAMP}"
mkdir -p "$OUTPUT_DIR"
LOG_FILE="$OUTPUT_DIR/training.log"

echo "📁 Output directory: $OUTPUT_DIR"
echo "📝 Log file: $LOG_FILE"
echo ""

# macOS OpenMP fix
export KMP_DUPLICATE_LIB_OK=TRUE

# Training function with 5CAT integration
echo "🚀 Starting training with 5CAT validation..."
echo "   Training will pause every 5 epochs for 5CAT testing"
echo ""

PYTHONPATH=. ./.venv/bin/python -u << EOF | tee "$LOG_FILE"
import sys
import json
import subprocess
from pathlib import Path
from datetime import datetime

# Training configuration
MODEL_TYPE = "$MODEL_TYPE"
EPOCHS = $EPOCHS
DEVICE = "$DEVICE"
TRAIN_DATA = "$TRAIN_DATA"
VAL_DATA = "$VAL_DATA"
OOD_DATA = "$OOD_DATA"
ARTICLES_DATA = "$ARTICLES_DATA"
OUTPUT_DIR = Path("$OUTPUT_DIR")

print("=" * 80)
print(f"TRAINING {MODEL_TYPE.upper()} WITH 5CAT VALIDATION")
print("=" * 80)
print()
print(f"Training data: {TRAIN_DATA}")
print(f"Validation data: {VAL_DATA}")
print(f"OOD test data: {OOD_DATA}")
print(f"Articles data: {ARTICLES_DATA}")
print(f"Output: {OUTPUT_DIR}")
print(f"Device: {DEVICE}")
print()

# Import training modules
sys.path.insert(0, 'app/lvm')
from models import create_model
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np

class VectorSequenceDataset(Dataset):
    def __init__(self, npz_path: str):
        data = np.load(npz_path, allow_pickle=True)
        # Handle different key formats
        if 'context_sequences' in data:
            self.contexts = torch.FloatTensor(data['context_sequences'])
            self.targets = torch.FloatTensor(data['target_vectors'])
        elif 'contexts' in data:
            self.contexts = torch.FloatTensor(data['contexts'])
            self.targets = torch.FloatTensor(data['targets'])
        else:
            raise KeyError(f"Cannot find context/target keys in {npz_path}")

        print(f"   Loaded {len(self.contexts):,} sequences")

    def __len__(self):
        return len(self.contexts)

    def __getitem__(self, idx):
        return self.contexts[idx], self.targets[idx]

def cosine_similarity(pred, target):
    pred_norm = pred / (pred.norm(dim=1, keepdim=True) + 1e-8)
    target_norm = target / (target.norm(dim=1, keepdim=True) + 1e-8)
    return (pred_norm * target_norm).sum(dim=1).mean()

def run_5cat_test(model_path: str, epoch: int):
    """Run 5CAT validation and return key metrics"""
    print()
    print("=" * 80)
    print(f"🧪 Running 5CAT Validation (Epoch {epoch})")
    print("=" * 80)

    cmd = [
        "./.venv/bin/python",
        "tools/tests/test_5to1_alignment.py",
        "--model", str(model_path),
        "--val-npz", VAL_DATA,
        "--ood-npz", OOD_DATA,
        "--articles-npz", ARTICLES_DATA,
        "--device", DEVICE,
        "--max-samples", "1000",
        "--horizon", "3"
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        output = result.stdout

        # Parse JSON output from 5CAT
        if "5→1 Causal Alignment Test — Summary" in output:
            # Extract JSON section
            json_start = output.find("{", output.find("Summary"))
            json_end = output.rfind("}")
            if json_start != -1 and json_end != -1:
                results_json = output[json_start:json_end+1]
                results = json.loads(results_json)

                val_margin = results['VAL'].get('A:margin(+1)', float('nan'))
                ood_margin = results['OOD'].get('A:margin(+1)', float('nan'))
                val_rollout = results['VAL'].get('D:avg_cos@H=3', float('nan'))
                ood_rollout = results['OOD'].get('D:avg_cos@H=3', float('nan'))

                print()
                print("📊 5CAT Results:")
                print(f"   VAL Margin: {val_margin:+.4f} (need ≥+0.12)")
                print(f"   OOD Margin: {ood_margin:+.4f} (need ≥+0.10)")
                print(f"   VAL Rollout: {val_rollout:.4f} (need ≥0.45)")
                print(f"   OOD Rollout: {ood_rollout:.4f} (need ≥0.42)")
                print()

                # Check for backward bias
                if val_margin < -0.05:
                    print("🚨 WARNING: BACKWARD BIAS DETECTED!")
                    print(f"   Margin is {val_margin:.4f} (negative)")
                    print("   Model is learning to predict PREVIOUS vector instead of NEXT!")
                    return False, val_margin, ood_margin, val_rollout, ood_rollout
                elif val_margin < 0:
                    print("⚠️  CAUTION: Slight backward tendency detected")
                    print(f"   Margin is {val_margin:.4f} (should be positive)")
                elif val_margin < 0.05:
                    print("⚠️  Weak forward signal")
                    print(f"   Margin is {val_margin:.4f} (target: ≥0.12)")
                else:
                    print("✅ Good forward prediction!")
                    print(f"   Margin is {val_margin:+.4f}")

                return True, val_margin, ood_margin, val_rollout, ood_rollout

        print("⚠️  Could not parse 5CAT results")
        return None, None, None, None, None

    except subprocess.TimeoutExpired:
        print("⚠️  5CAT test timed out (skipping)")
        return None, None, None, None, None
    except Exception as e:
        print(f"⚠️  5CAT test failed: {e}")
        return None, None, None, None, None

# Load datasets
print("📥 Loading datasets...")
train_dataset = VectorSequenceDataset(TRAIN_DATA)
val_dataset = VectorSequenceDataset(VAL_DATA)

train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False, num_workers=0)

print()

# Create model
print(f"🏗️  Creating {MODEL_TYPE} model...")
device = torch.device(DEVICE)
model = create_model(MODEL_TYPE, input_dim=768, d_model=256, hidden_dim=512).to(device)
params = sum(p.numel() for p in model.parameters())
print(f"   Parameters: {params:,}")
print()

# Optimizer and scheduler
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

# Training loop
print("🏃 Starting training...")
print()

best_val_loss = float('inf')
best_5cat_score = float('-inf')
training_history = []

for epoch in range(1, EPOCHS + 1):
    print(f"Epoch {epoch}/{EPOCHS}")

    # Train
    model.train()
    train_loss = 0.0
    train_cosine = 0.0
    n_train_batches = 0

    for batch_idx, (contexts, targets) in enumerate(train_loader):
        contexts, targets = contexts.to(device), targets.to(device)

        optimizer.zero_grad()
        pred = model(contexts)

        loss = torch.nn.functional.mse_loss(pred, targets)
        cos = cosine_similarity(pred, targets)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_cosine += cos.item()
        n_train_batches += 1

        if batch_idx % 100 == 0 and batch_idx > 0:
            print(f"  Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.6f} | Cosine: {cos.item():.4f}")

    train_loss /= n_train_batches
    train_cosine /= n_train_batches

    # Validate
    model.eval()
    val_loss = 0.0
    val_cosine = 0.0
    n_val_batches = 0

    with torch.no_grad():
        for contexts, targets in val_loader:
            contexts, targets = contexts.to(device), targets.to(device)
            pred = model(contexts)

            loss = torch.nn.functional.mse_loss(pred, targets)
            cos = cosine_similarity(pred, targets)

            val_loss += loss.item()
            val_cosine += cos.item()
            n_val_batches += 1

    val_loss /= n_val_batches
    val_cosine /= n_val_batches

    print(f"  Train Loss: {train_loss:.6f} | Train Cosine: {train_cosine:.4f}")
    print(f"  Val Loss: {val_loss:.6f} | Val Cosine: {val_cosine:.4f}")

    # Save checkpoint
    checkpoint_path = OUTPUT_DIR / f"checkpoint_epoch{epoch}.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_type': MODEL_TYPE,
        'model_config': {'input_dim': 768, 'd_model': 256, 'hidden_dim': 512},
        'val_cosine': val_cosine,
        'val_loss': val_loss,
        'train_cosine': train_cosine,
        'train_loss': train_loss,
        'epoch': epoch,
        'optimizer_state_dict': optimizer.state_dict(),
    }, checkpoint_path)

    # Save best model (based on val loss)
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model_path = OUTPUT_DIR / "best_model.pt"
        torch.save({
            'model_state_dict': model.state_dict(),
            'model_type': MODEL_TYPE,
            'model_config': {'input_dim': 768, 'd_model': 256, 'hidden_dim': 512},
            'val_cosine': val_cosine,
            'val_loss': val_loss,
            'train_cosine': train_cosine,
            'train_loss': train_loss,
            'epoch': epoch,
        }, best_model_path)
        print(f"  ✅ Saved best model (val_loss: {val_loss:.6f})")

    # Run 5CAT every 5 epochs
    cat_metrics = {}
    if epoch % 5 == 0 or epoch == EPOCHS:
        success, val_margin, ood_margin, val_rollout, ood_rollout = run_5cat_test(checkpoint_path, epoch)

        if success is not None:
            cat_metrics = {
                'val_margin': val_margin,
                'ood_margin': ood_margin,
                'val_rollout': val_rollout,
                'ood_rollout': ood_rollout
            }

            # Compute 5CAT score (higher = better)
            # Prioritize: positive margin (80%) + rollout (20%)
            if val_margin is not None and val_rollout is not None:
                cat_score = val_margin * 0.8 + val_rollout * 0.2
                cat_metrics['cat_score'] = cat_score

                # Save best 5CAT model
                if cat_score > best_5cat_score:
                    best_5cat_score = cat_score
                    best_5cat_path = OUTPUT_DIR / "best_5cat_model.pt"
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'model_type': MODEL_TYPE,
                        'model_config': {'input_dim': 768, 'd_model': 256, 'hidden_dim': 512},
                        'val_cosine': val_cosine,
                        'val_loss': val_loss,
                        'epoch': epoch,
                        '5cat_val_margin': val_margin,
                        '5cat_ood_margin': ood_margin,
                        '5cat_val_rollout': val_rollout,
                        '5cat_ood_rollout': ood_rollout,
                        '5cat_score': cat_score,
                    }, best_5cat_path)
                    print(f"  🏆 Saved best 5CAT model (score: {cat_score:.4f})")

            # Early stopping if severe backward bias
            if success is False and val_margin < -0.10:
                print()
                print("🛑 EARLY STOPPING: Severe backward bias detected!")
                print(f"   Margin: {val_margin:.4f} (threshold: -0.10)")
                print("   Model is learning wrong direction - stopping training")
                break

    # Record history
    history_entry = {
        'epoch': epoch,
        'train_loss': train_loss,
        'train_cosine': train_cosine,
        'val_loss': val_loss,
        'val_cosine': val_cosine,
        **cat_metrics
    }
    training_history.append(history_entry)

    # Save history
    with open(OUTPUT_DIR / "training_history.json", 'w') as f:
        json.dump(training_history, f, indent=2)

    scheduler.step(val_loss)
    print()

# Final 5CAT test on best model
print("=" * 80)
print("🏁 Final 5CAT Validation on Best Model")
print("=" * 80)
run_5cat_test(OUTPUT_DIR / "best_model.pt", "FINAL")

print()
print("=" * 80)
print("✅ TRAINING COMPLETE!")
print("=" * 80)
print(f"Output directory: {OUTPUT_DIR}")
print(f"Best val loss: {best_val_loss:.6f}")
if best_5cat_score > float('-inf'):
    print(f"Best 5CAT score: {best_5cat_score:.4f}")
print()
print("Models saved:")
print(f"  - best_model.pt        (lowest val loss)")
if (OUTPUT_DIR / "best_5cat_model.pt").exists():
    print(f"  - best_5cat_model.pt   (best 5CAT metrics)")
print()
print("Next steps:")
print(f"  1. Review training history: cat {OUTPUT_DIR}/training_history.json")
print(f"  2. Run full 5CAT test: ./.venv/bin/python tools/tests/test_5to1_alignment.py \\\\")
print(f"       --model {OUTPUT_DIR}/best_5cat_model.pt \\\\")
print(f"       --val-npz {VAL_DATA} \\\\")
print(f"       --ood-npz {OOD_DATA} \\\\")
print(f"       --articles-npz {ARTICLES_DATA} \\\\")
print(f"       --device {DEVICE} --max-samples 5000")
print()

EOF

echo ""
echo "============================================"
echo "Training script completed"
echo "============================================"
echo ""
echo "📁 Output: $OUTPUT_DIR"
echo "📝 Log: $LOG_FILE"
echo ""
