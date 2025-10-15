#!/usr/bin/env python3
"""
Test simple MLP model on 5 diverse test sequences
"""

import sys
import torch
import numpy as np
import requests
from pathlib import Path

# Add to path
sys.path.insert(0, str(Path(__file__).parent.parent / "app" / "lvm"))
from train_simple_mlp import SimpleMLP


# Test sequences (same as LSTM test)
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


def encode_text(text):
    """Encode text using vec2text-compatible encoder"""
    response = requests.post(
        "http://localhost:8767/embed",
        json={"texts": [text]},
        timeout=10
    )
    if response.status_code == 200:
        return np.array(response.json()["embeddings"][0])
    return None


def decode_vector(vector):
    """Decode vector using vec2text"""
    response = requests.post(
        "http://localhost:8766/decode",
        json={
            "vectors": [vector.tolist()],
            "subscribers": "ielab",
            "steps": 1,
            "device": "cpu"
        },
        timeout=30
    )
    if response.status_code == 200:
        result = response.json()
        return result["results"][0]["subscribers"]["gtr → ielab"]["output"]
    return None


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def main():
    print("=" * 80)
    print("Simple MLP Full Pipeline Test")
    print("=" * 80)
    print("Pipeline: Text → 768D (GTR-T5) → MLP → 768D → Text (vec2text)")
    print()

    # Load model (can specify via command line arg)
    import sys
    model_path = sys.argv[1] if len(sys.argv) > 1 else "artifacts/lvm/models/mlp_simple/best_model.pt"
    print(f"Loading model from: {model_path}")

    checkpoint = torch.load(model_path, map_location='cpu')
    model = SimpleMLP(context_size=5, vector_dim=768, hidden_dim=1024)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print("✓ Model loaded\n")

    # Check APIs
    print("Checking API availability...")
    try:
        requests.get("http://localhost:8767/health", timeout=5)
        print("✓ Vec2text-compatible GTR-T5 encoder API (port 8767) available")
    except:
        print("✗ Encoder not available")
        return

    try:
        requests.get("http://localhost:8766/health", timeout=5)
        print("✓ vec2text decoder API (port 8766) available")
    except:
        print("✗ Decoder not available")
        return

    print()

    # Run tests
    results = []

    for i, test in enumerate(TEST_SEQUENCES):
        print("#" * 80)
        print(f"# TEST {i+1}/{len(TEST_SEQUENCES)}")
        print("#" * 80)
        print()
        print("=" * 80)
        print(f"Test: {test['name']}")
        print("=" * 80)
        print()

        # Encode context
        print("1. CONTEXT (5 sentences):")
        context_vecs = []
        for j, sent in enumerate(test['context']):
            print(f"   {j+1}. {sent}")
            vec = encode_text(sent)
            if vec is not None:
                context_vecs.append(vec)

        if len(context_vecs) != 5:
            print("✗ Failed to encode context")
            continue

        context_array = np.array(context_vecs)

        print()
        print("2. EXPECTED TARGET:")
        print(f"   {test['expected']}")
        print()

        # Encode expected
        expected_vec = encode_text(test['expected'])
        if expected_vec is None:
            print("✗ Failed to encode expected")
            continue

        # Generate prediction
        with torch.no_grad():
            context_tensor = torch.from_numpy(context_array).unsqueeze(0).float()
            pred_vec = model(context_tensor)[0].numpy()

        # Normalize
        pred_vec = pred_vec / np.linalg.norm(pred_vec)

        similarity = cosine_similarity(pred_vec, expected_vec)
        print("3. MLP PREDICTION:")
        print(f"   Cosine similarity to expected: {similarity:.4f}")
        print()

        # Decode
        print("4. DECODING PREDICTED VECTOR:")
        print("   (This may take ~10-20 seconds...)")
        print()

        decoded = decode_vector(pred_vec)

        if decoded is None:
            print("✗ Decode failed")
            continue

        print("5. RESULTS:")
        print(f"   Expected:  '{test['expected']}'")
        print(f"   Predicted: '{decoded}'")
        print(f"   Similarity: {similarity * 100:.2f}%")
        print()
        print()

        results.append({
            'name': test['name'],
            'expected': test['expected'],
            'predicted': decoded,
            'similarity': similarity
        })

    # Summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print()
    print(f"{'Test':<20} {'Similarity':<12} {'Status':<10}")
    print("-" * 80)

    total_sim = 0
    for r in results:
        status = "✓ Good" if r['similarity'] >= 0.7 else "○ Fair" if r['similarity'] >= 0.4 else "✗ Poor"
        print(f"{r['name']:<20} {r['similarity']*100:>6.2f}%      {status}")
        total_sim += r['similarity']

    avg_sim = total_sim / len(results) if results else 0
    print("-" * 80)
    print(f"{'Average':<20} {avg_sim*100:>6.2f}%     ")
    print()
    print()

    # Detailed comparison
    print("=" * 80)
    print("DETAILED COMPARISON")
    print("=" * 80)
    print()

    for r in results:
        print(f"{r['name']}:")
        print(f"  Expected:  {r['expected']}")
        print(f"  Predicted: {r['predicted']}")
        print(f"  Similarity: {r['similarity']*100:.2f}%")
        print()

    print("=" * 80)
    print("Test Complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
