#!/usr/bin/env python3
"""
Phase 2: Decode Quality Diagnosis

Tests vec2text round-trip compatibility on validation predictions.
Measures cycle consistency: cosine(pred, encode(decode(pred)))

This will determine if the problem is:
1. LVM prediction quality (vectors off-manifold)
2. Vec2text decode compatibility (wrong vector space)
3. Training data corruption (wrong encoder used)
"""

import numpy as np
import torch
import torch.nn as nn
import sys
import requests
from pathlib import Path

# Add app/lvm to path for model imports
sys.path.insert(0, str(Path(__file__).parent.parent / "app" / "lvm"))
from train_lstm_baseline import LSTMVectorPredictor


def cosine_similarity(a, b):
    """Compute cosine similarity between two vectors."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def vec2text_decode(vector):
    """Decode 768D vector to text using vec2text API."""
    try:
        response = requests.post(
            "http://localhost:8766/decode",
            json={
                "vectors": [vector.tolist()],
                "subscribers": "jxe",  # Use JXE decoder only for speed
                "steps": 1,
                "device": "cpu"
            },
            timeout=30
        )
        if response.status_code == 200:
            result = response.json()
            # Extract text from nested response structure
            # Response: {"results": [{"index": 0, "subscribers": {"gtr → jxe": {"output": "..."}}}]}
            first_result = result["results"][0]
            jxe_output = first_result["subscribers"].get("gtr → jxe", {})
            return jxe_output.get("output", "[NO TEXT]")
        else:
            return f"[ERROR: {response.status_code}]"
    except Exception as e:
        return f"[ERROR: {str(e)}]"


def vec2text_encode(text):
    """Encode text to 768D vector using vec2text-compatible encoder."""
    try:
        response = requests.post(
            "http://localhost:8767/embed",
            json={"texts": [text]},
            timeout=10
        )
        if response.status_code == 200:
            return np.array(response.json()["embeddings"][0])
        else:
            return None
    except Exception as e:
        return None


def main():
    print("=" * 80)
    print("Phase 2: Decode Quality Diagnosis")
    print("=" * 80)
    print()

    # Check API availability
    print("Checking API availability...")
    try:
        enc_resp = requests.get("http://localhost:8767/health", timeout=5)
        if enc_resp.status_code != 200:
            print("✗ Vec2text-compatible encoder (port 8767) not available")
            print("  Start with: VEC2TEXT_FORCE_CPU=1 ./.venv/bin/uvicorn app.api.vec2text_embedding_server:app --host 127.0.0.1 --port 8767 &")
            sys.exit(1)
        print("✓ Vec2text-compatible encoder (port 8767) available")
    except:
        print("✗ Vec2text-compatible encoder (port 8767) not available")
        sys.exit(1)

    try:
        dec_resp = requests.get("http://localhost:8766/health", timeout=5)
        if dec_resp.status_code != 200:
            print("✗ Vec2text decoder (port 8766) not available")
            print("  Start with: VEC2TEXT_FORCE_CPU=1 ./.venv/bin/uvicorn app.api.vec2text_server:app --host 127.0.0.1 --port 8766 &")
            sys.exit(1)
        print("✓ Vec2text decoder (port 8766) available")
    except:
        print("✗ Vec2text decoder (port 8766) not available")
        sys.exit(1)

    print()

    # Load model
    model_path = "artifacts/lvm/models/lstm_29k/best_model.pt"
    print(f"Loading model: {model_path}")

    checkpoint = torch.load(model_path, map_location='cpu')
    model = LSTMVectorPredictor(
        input_dim=768,
        hidden_dim=512,
        num_layers=2
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print("✓ Model loaded\n")

    # Load validation data
    data_path = "artifacts/lvm/training_sequences_ctx5.npz"
    print(f"Loading validation data: {data_path}")
    data = np.load(data_path)

    # Split into train/val (same as training: 90/10)
    total_samples = len(data['context_sequences'])
    train_size = int(0.9 * total_samples)

    val_contexts = data['context_sequences'][train_size:]
    val_targets = data['target_vectors'][train_size:]

    # Sample 20 validation examples (reduced from 100 for speed)
    num_samples = min(20, len(val_contexts))
    indices = np.random.choice(len(val_contexts), num_samples, replace=False)

    print(f"✓ Loaded {num_samples} validation samples\n")

    # Run diagnosis
    print("=" * 80)
    print(f"Testing vec2text round-trip on {num_samples} samples")
    print("=" * 80)
    print()
    print("NOTE: Each decode takes ~10 seconds. Total time: ~3-4 minutes")
    print()

    cycle_scores = []
    pred_target_scores = []
    decode_risky = []

    print("Progress: ", end="", flush=True)
    decode_errors = []
    encode_errors = []

    for i, idx in enumerate(indices):
        if i % 10 == 0:
            print(f"{i}...", end="", flush=True)

        context = val_contexts[idx]
        target = val_targets[idx]

        # Generate prediction
        with torch.no_grad():
            context_tensor = torch.from_numpy(context).unsqueeze(0).float()
            pred = model(context_tensor)[0].numpy()

        # Normalize prediction
        pred = pred / np.linalg.norm(pred)

        # Measure prediction quality
        pred_target_cos = cosine_similarity(pred, target)
        pred_target_scores.append(pred_target_cos)

        # Test vec2text round-trip
        decoded_text = vec2text_decode(pred)

        if decoded_text.startswith("[ERROR"):
            decode_errors.append((i, decoded_text))
            continue

        if decoded_text == "[NO TEXT]":
            decode_errors.append((i, "No text extracted from API"))
            continue

        reencoded_vec = vec2text_encode(decoded_text)

        if reencoded_vec is None:
            encode_errors.append((i, decoded_text))
            continue

        # Normalize re-encoded vector
        reencoded_vec = reencoded_vec / np.linalg.norm(reencoded_vec)

        # Measure cycle consistency
        cycle_cos = cosine_similarity(pred, reencoded_vec)
        cycle_scores.append(cycle_cos)

        # Flag decode-risky samples
        if cycle_cos < 0.7:
            decode_risky.append({
                'idx': i,
                'cycle_cos': cycle_cos,
                'pred_target_cos': pred_target_cos,
                'decoded_text': decoded_text
            })

    print(" Done!\n")

    # Summary statistics
    print("=" * 80)
    print("CYCLE CONSISTENCY ANALYSIS")
    print("=" * 80)
    print()

    # Report errors first
    if len(decode_errors) > 0:
        print(f"⚠ Decode errors: {len(decode_errors)}/{num_samples}")
        for idx, error in decode_errors[:3]:
            print(f"  Sample {idx}: {error}")
        print()

    if len(encode_errors) > 0:
        print(f"⚠ Re-encode errors: {len(encode_errors)}/{num_samples}")
        for idx, text in encode_errors[:3]:
            print(f"  Sample {idx}: Failed to re-encode '{text[:50]}...'")
        print()

    if len(cycle_scores) > 0:
        mean_cycle = np.mean(cycle_scores)
        median_cycle = np.median(cycle_scores)
        good_pct = 100 * np.mean([s >= 0.8 for s in cycle_scores])
        risky_pct = 100 * np.mean([s < 0.7 for s in cycle_scores])

        print(f"Cycle Consistency (pred → decode → encode → compare):")
        print(f"  Mean:   {mean_cycle:.3f}")
        print(f"  Median: {median_cycle:.3f}")
        print(f"  % >= 0.8 (good):  {good_pct:.1f}%")
        print(f"  % < 0.7 (risky):  {risky_pct:.1f}%")
        print()

        print(f"Prediction Quality (pred → target):")
        print(f"  Mean:   {np.mean(pred_target_scores):.3f}")
        print(f"  Median: {np.median(pred_target_scores):.3f}")
        print()

        # Interpretation
        print("=" * 80)
        print("INTERPRETATION")
        print("=" * 80)
        print()

        if mean_cycle >= 0.8:
            print("✓ CYCLE CONSISTENCY IS GOOD (>= 0.8)")
            print("  → Vec2text decoder is compatible with predicted vectors")
            print()
            if np.mean(pred_target_scores) < 0.3:
                print("✗ PREDICTION QUALITY IS POOR (< 0.3)")
                print("  → Problem: LVM is not learning to predict target vectors accurately")
                print("  → Diagnosis: Model architecture or training approach issue")
                print()
                print("Recommended fixes:")
                print("  1. Increase model capacity (more layers, hidden_dim)")
                print("  2. Adjust InfoNCE temperature")
                print("  3. Try different loss function (MSE + cosine)")
                print("  4. Increase training epochs or learning rate")
        elif mean_cycle >= 0.7:
            print("⚠ CYCLE CONSISTENCY IS MODERATE (0.7-0.8)")
            print("  → Some predicted vectors are off-manifold")
            print("  → Proceed to Phase 3: Light Cycle Consistency (10% batches)")
            print()
            print("Recommended fix:")
            print("  Add cycle loss to 10-20% of training batches")
            print("  Loss: lambda_cycle * (1 - cos(pred, encode(decode(pred))))")
        else:
            print("✗ CYCLE CONSISTENCY IS POOR (< 0.7)")
            print("  → Predicted vectors are off-manifold for vec2text decoder")
            print("  → MUST proceed to Phase 3: Light Cycle Consistency")
            print()
            print("Recommended fix:")
            print("  Add cycle loss to 20-30% of training batches")
            print("  Loss: lambda_cycle * (1 - cos(pred, encode(decode(pred))))")
            print("  Start with lambda_cycle = 0.2")

        print()

        # Show decode-risky samples
        if len(decode_risky) > 0:
            print("=" * 80)
            print(f"DECODE-RISKY SAMPLES ({len(decode_risky)} samples with cycle < 0.7)")
            print("=" * 80)
            print()

            for sample in decode_risky[:5]:  # Show first 5
                print(f"Sample {sample['idx']}:")
                print(f"  Cycle consistency: {sample['cycle_cos']:.3f}")
                print(f"  Pred→Target cos:   {sample['pred_target_cos']:.3f}")
                print(f"  Decoded text: '{sample['decoded_text']}'")
                print()
    else:
        print("✗ FATAL: Could not complete any vec2text round-trips")
        print("  → Check that encoder (port 8767) and decoder (port 8766) are working")
        print("  → Verify they use compatible vector spaces")

    print("=" * 80)
    print("Phase 2 Complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
