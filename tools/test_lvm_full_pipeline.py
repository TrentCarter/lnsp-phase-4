#!/usr/bin/env python3
"""
Test LVM Full Pipeline: Text → 768D → LVM → 768D → Text

Tests the complete pipeline:
1. Encode context (5 sentences) to 768D vectors
2. LVM predicts next vector from context
3. Decode predicted vector to text
4. Compare with expected next sentence
"""

import sys
import torch
import torch.nn as nn
import numpy as np
import requests
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.lvm.train_mamba2 import Mamba2VectorPredictor


# Test sequences (each is 6 sentences: 5 context + 1 target)
TEST_SEQUENCES = [
    {
        "name": "Cooking Recipe",
        "context": [
            "Heat the oven to 350 degrees.",
            "Mix flour, sugar, and eggs in a bowl.",
            "Add vanilla extract and milk.",
            "Pour the batter into a greased pan.",
            "Bake for 30 minutes until golden brown."
        ],
        "expected": "Let the cake cool before frosting."
    },
    {
        "name": "Morning Routine",
        "context": [
            "The alarm went off at 6 AM.",
            "I got out of bed and stretched.",
            "I brushed my teeth and washed my face.",
            "I made coffee and toast for breakfast.",
            "I checked my phone for messages."
        ],
        "expected": "Then I got dressed for work."
    },
    {
        "name": "Scientific Process",
        "context": [
            "The hypothesis was formulated carefully.",
            "Materials were gathered for the experiment.",
            "Control and test groups were established.",
            "Data was collected over three weeks.",
            "Statistical analysis showed significant results."
        ],
        "expected": "The findings were published in a journal."
    },
    {
        "name": "Travel Story",
        "context": [
            "We arrived at the airport early.",
            "Security screening took about 20 minutes.",
            "We found our gate and waited to board.",
            "The flight took off on schedule.",
            "We landed in Paris after 8 hours."
        ],
        "expected": "We took a taxi to our hotel."
    },
    {
        "name": "Simple Conversation",
        "context": [
            "Hello, how are you today?",
            "I'm doing well, thank you.",
            "Would you like some coffee?",
            "Yes, that would be nice.",
            "I'll make a fresh pot."
        ],
        "expected": "How do you take it?"
    }
]


def encode_text(text: str) -> np.ndarray:
    """Encode text to 768D vector using vec2text-compatible GTR-T5 encoder API"""
    try:
        response = requests.post(
            "http://127.0.0.1:8767/embed",  # FIXED: Use port 8767 (vec2text-compatible)
            json={"texts": [text]},
            timeout=10
        )
        response.raise_for_status()
        data = response.json()
        return np.array(data["embeddings"][0], dtype=np.float32)
    except Exception as e:
        print(f"Error encoding text: {e}")
        raise


def decode_vector(vector: np.ndarray, steps: int = 1) -> str:
    """Decode 768D vector to text using vec2text decoder API"""
    try:
        response = requests.post(
            "http://127.0.0.1:8766/decode",
            json={
                "vectors": [vector.tolist()],
                "steps": steps,
                "subscribers": "ielab"
            },
            timeout=30
        )
        response.raise_for_status()
        data = response.json()
        # Response format: {"results": [{"index": 0, "subscribers": {"gtr → ielab": {"output": "text", ...}}}], "count": 1}
        result = data["results"][0]
        subscriber_results = result["subscribers"]
        # Get the first subscriber's output (should be "gtr → ielab")
        first_key = next(iter(subscriber_results.keys()))
        return subscriber_results[first_key]["output"]
    except Exception as e:
        print(f"Error decoding vector: {e}")
        raise


def load_lvm_model(checkpoint_path: str, device: str = "cpu", model_type: str = "mamba2"):
    """Load trained LVM model"""
    if model_type == "lstm":
        from app.lvm.train_lstm_baseline import LSTMVectorPredictor
        model = LSTMVectorPredictor(input_dim=768, hidden_dim=512, num_layers=2)
    elif model_type == "gru":
        from app.lvm.train_lstm_baseline import LSTMVectorPredictor
        model = LSTMVectorPredictor(input_dim=768, hidden_dim=512, num_layers=2, use_gru=True)
    else:  # mamba2
        model = Mamba2VectorPredictor(input_dim=768, d_model=512, num_layers=4)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model


def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """Compute cosine similarity between two vectors"""
    v1_norm = v1 / (np.linalg.norm(v1) + 1e-8)
    v2_norm = v2 / (np.linalg.norm(v2) + 1e-8)
    return float(np.dot(v1_norm, v2_norm))


def test_sequence(model, sequence_data: dict, device: str = "cpu"):
    """Test one sequence through the full pipeline"""
    print(f"\n{'='*80}")
    print(f"Test: {sequence_data['name']}")
    print(f"{'='*80}")

    # 1. Encode context (5 sentences) to vectors
    print("\n1. CONTEXT (5 sentences):")
    context_vectors = []
    for i, sentence in enumerate(sequence_data['context'], 1):
        print(f"   {i}. {sentence}")
        vector = encode_text(sentence)
        context_vectors.append(vector)

    # 2. Encode expected target to vector (for comparison)
    print(f"\n2. EXPECTED TARGET:")
    print(f"   {sequence_data['expected']}")
    expected_vector = encode_text(sequence_data['expected'])

    # 3. LVM predicts next vector from context
    print(f"\n3. LVM PREDICTION:")
    context_tensor = torch.FloatTensor(np.stack(context_vectors)).unsqueeze(0).to(device)
    with torch.no_grad():
        predicted_vector = model(context_tensor).cpu().numpy()[0]

    # 4. Compute similarity between predicted and expected
    similarity = cosine_similarity(predicted_vector, expected_vector)
    print(f"   Cosine similarity to expected: {similarity:.4f}")

    # 5. Decode predicted vector to text
    print(f"\n4. DECODING PREDICTED VECTOR:")
    print(f"   (This may take ~10-20 seconds...)")
    predicted_text = decode_vector(predicted_vector, steps=1)

    # 6. Also decode expected vector for comparison
    expected_decoded = decode_vector(expected_vector, steps=1)

    # 7. Results
    print(f"\n5. RESULTS:")
    print(f"   Expected:  '{sequence_data['expected']}'")
    print(f"   Expected→: '{expected_decoded}'")
    print(f"   Predicted: '{predicted_text}'")
    print(f"   Similarity: {similarity:.2%}")

    return {
        "name": sequence_data['name'],
        "expected": sequence_data['expected'],
        "expected_decoded": expected_decoded,
        "predicted": predicted_text,
        "similarity": similarity
    }


def main():
    import sys
    print("="*80)
    print("LVM Full Pipeline Test")
    print("="*80)
    print("Pipeline: Text → 768D (GTR-T5) → LVM → 768D → Text (vec2text)")
    print()

    # Load model (allow command-line override)
    if len(sys.argv) > 1:
        checkpoint_path = sys.argv[1]
        model_type = sys.argv[2] if len(sys.argv) > 2 else "mamba2"
    else:
        checkpoint_path = "artifacts/lvm/models/lstm_vec2text/best_model.pt"
        model_type = "lstm"

    print(f"Loading model from: {checkpoint_path}")
    print(f"Model type: {model_type}")
    model = load_lvm_model(checkpoint_path, device="cpu", model_type=model_type)
    print("✓ Model loaded")

    # Check APIs
    print("\nChecking API availability...")
    try:
        # Check encoder (vec2text-compatible)
        response = requests.get("http://127.0.0.1:8767/health", timeout=5)
        print("✓ Vec2text-compatible GTR-T5 encoder API (port 8767) available")
    except:
        print("✗ Vec2text-compatible encoder API not available")
        print("  Start with: VEC2TEXT_FORCE_CPU=1 ./.venv/bin/uvicorn app.api.vec2text_embedding_server:app --host 127.0.0.1 --port 8767")
        return

    try:
        # Check decoder
        response = requests.get("http://127.0.0.1:8766/health", timeout=5)
        print("✓ vec2text decoder API (port 8766) available")
    except:
        print("✗ vec2text decoder API not available")
        print("  Start with: ./.venv/bin/uvicorn app.api.vec2text_server:app --host 127.0.0.1 --port 8766")
        return

    # Run tests
    results = []
    for i, sequence_data in enumerate(TEST_SEQUENCES, 1):
        print(f"\n\n{'#'*80}")
        print(f"# TEST {i}/5")
        print(f"{'#'*80}")
        result = test_sequence(model, sequence_data, device="cpu")
        results.append(result)

    # Summary
    print("\n\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"\n{'Test':<20} {'Similarity':<12} {'Status'}")
    print("-"*80)

    total_similarity = 0
    for result in results:
        status = "✓ Good" if result['similarity'] > 0.5 else "✗ Poor"
        print(f"{result['name']:<20} {result['similarity']:<12.2%} {status}")
        total_similarity += result['similarity']

    avg_similarity = total_similarity / len(results)
    print("-"*80)
    print(f"{'Average':<20} {avg_similarity:<12.2%}")
    print()

    # Detailed comparison
    print("\n" + "="*80)
    print("DETAILED COMPARISON")
    print("="*80)
    for result in results:
        print(f"\n{result['name']}:")
        print(f"  Expected:  {result['expected']}")
        print(f"  Predicted: {result['predicted']}")
        print(f"  Similarity: {result['similarity']:.2%}")

    print("\n" + "="*80)
    print("Test Complete!")
    print("="*80)


if __name__ == "__main__":
    main()
