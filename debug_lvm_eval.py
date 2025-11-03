#!/usr/bin/env python3
"""
Test script for LVM evaluation logic debugging
"""

import sys
import os
import numpy as np
import requests
import json

# Add the src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

def test_encoder_decoder_services():
    """Test if encoder/decoder services are working"""
    print("=== Testing Encoder/Decoder Services ===")

    # Test encoder
    try:
        response = requests.post(
            "http://localhost:7001/encode",
            json={"texts": ["entity -> continuant -> independent continuant -> material entity"]},
            timeout=10
        )
        if response.status_code == 200:
            data = response.json()
            print("✅ Encoder working")
            print(f"   Response keys: {list(data.keys())}")
            if "embeddings" in data:
                embedding = data["embeddings"][0]
                print(f"   Vector shape: {len(embedding)}")
            return True
        else:
            print(f"❌ Encoder failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Encoder error: {e}")
        return False

    # Test decoder
    try:
        test_vector = np.random.randn(768).tolist()
        response = requests.post(
            "http://localhost:7002/decode",
            json={"vectors": [test_vector], "subscriber": "ielab", "steps": 3},
            timeout=30
        )
        if response.status_code == 200:
            data = response.json()
            print("✅ Decoder working")
            print(f"   Response keys: {list(data.keys())}")
            if "results" in data:
                decoded = data["results"][0]
                print(f"   Decoded: {repr(decoded[:50])}...")
            return True
        else:
            print(f"❌ Decoder failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Decoder error: {e}")
        return False

def test_data_processing():
    """Test the data processing logic"""
    print("\n=== Testing Data Processing ===")

    # Import the function
    from lvm_eval.routes import get_test_data

    # Test different modes
    for mode in ['both', 'in_distribution', 'out_of_distribution']:
        print(f"\nTesting mode: {mode}")
        try:
            data = get_test_data(mode)
            print(f"  ✅ Got {len(data)} samples")

            # Show first few samples
            for i, sample in enumerate(data[:3]):
                print(f"  Sample {i}:")
                print(f"    Text: {repr(sample.get('text', '')[:60])}...")
                print(f"    Expected: {repr(sample.get('expected_text', '')[:60])}...")

        except Exception as e:
            print(f"  ❌ Error: {e}")

def test_next_token_logic():
    """Test the next-token prediction logic in isolation"""
    print("\n=== Testing Next-Token Prediction Logic ===")

    # Test cases
    test_cases = [
        "entity -> continuant -> independent continuant -> material entity -> material information bearer",
        "Plants release oxygen as a byproduct of photosynthesis.",
        "Machine learning enables predictive analytics."
    ]

    for i, full_text in enumerate(test_cases):
        print(f"\nTest case {i+1}: {repr(full_text)}")

        # Apply the same logic as in routes.py
        if ' -> ' in full_text:
            concepts = full_text.split(' -> ')
            num_concepts_param = 5

            if len(concepts) >= num_concepts_param:
                # Use first (num_concepts-1) as context, last one as target
                context_concepts = concepts[:num_concepts_param-1]
                target_concept = concepts[num_concepts_param-1]

                input_text = ' -> '.join(context_concepts)
                expected_text = target_concept

                print(f"  ✅ Ontology chain: {len(concepts)} concepts")
                print(f"    Context: {repr(input_text)}")
                print(f"    Target: {repr(expected_text)}")
            else:
                print(f"  ❌ Not enough concepts ({len(concepts)} < {num_concepts_param})")
        else:
            print(f"  ❌ Not an ontology chain")

def test_evaluation_logic():
    """Test the complete evaluation logic"""
    print("\n=== Testing Complete Evaluation Logic ===")

    try:
        # Import the evaluation function
        from lvm_eval.routes import evaluate_single_model, get_test_data

        # Get test data
        test_data = get_test_data('both')[:2]  # Just 2 samples for testing
        print(f"Using {len(test_data)} test samples")

        # Test evaluation
        result = evaluate_single_model(
            '/Users/trentcarter/Artificial_Intelligence/AI_Projects/lnsp-phase-4/artifacts/lvm/models/final_model.pt',
            test_data,
            1,  # progress base
            1   # total models
        )

        print(f"Evaluation result: {result.get('status')}")
        print(f"Test cases in result: {len(result.get('test_cases', []))}")

        # Show details
        for i, tc in enumerate(result.get('test_cases', [])):
            print(f"\nTest case {i}:")
            print(f"  Input: {repr(tc.get('input', 'MISSING')[:60])}...")
            print(f"  Expected: {repr(tc.get('expected', 'MISSING')[:60])}...")
            print(f"  Output: {repr(tc.get('output', 'MISSING')[:60])}...")
            print(f"  Cosine: {tc.get('cosine_similarity', 'MISSING')}")
            print(f"  Status: {tc.get('status', 'MISSING')}")

    except Exception as e:
        print(f"❌ Evaluation error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("LVM Evaluation Debug Test")
    print("=" * 50)

    # Run all tests
    encoder_ok = test_encoder_decoder_services()
    test_data_processing()
    test_next_token_logic()
    test_evaluation_logic()

    print("\n" + "=" * 50)
    if encoder_ok:
        print("✅ Services appear to be working")
    else:
        print("❌ Services need to be started")

    print("\nNext steps:")
    print("1. Check if services are running: curl http://localhost:7001/health")
    print("2. Check logs: tail -f /tmp/lvm_api_logs/*.log")
    print("3. Test manual encoding: curl -X POST http://localhost:7001/encode -H 'Content-Type: application/json' -d '{\"texts\": [\"test\"]}'")
