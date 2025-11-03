#!/usr/bin/env python3
"""Test model output directly to diagnose issues"""

import torch
import numpy as np
import requests

def test_model():
    # Load model
    model_path = "artifacts/lvm/models/transformer_p4_rollout/best_model.pt"
    checkpoint = torch.load(model_path, map_location='cpu')
    
    print(f"Model trained for {checkpoint.get('epoch', '?')} epochs")
    print(f"Validation loss: {checkpoint.get('val_loss', '?')}")
    print(f"Validation cosine similarity: {checkpoint.get('val_cosine', '?')}")
    
    # Test encoding/decoding pipeline
    test_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is a subset of artificial intelligence.",
        "Neural networks process information in layers."
    ]
    
    print("\n=== Testing GTR-T5 Encoding/Decoding Pipeline ===")
    for text in test_texts:
        # Encode
        response = requests.post('http://localhost:7001/encode', json={'text': text})
        if response.status_code == 200:
            vector = np.array(response.json()['vector'])
            print(f"\nOriginal: {text}")
            print(f"Vector shape: {vector.shape}, Mean: {vector.mean():.4f}, Std: {vector.std():.4f}")
            
            # Decode
            response = requests.post('http://localhost:7002/decode', 
                                    json={'vector': vector.tolist()})
            if response.status_code == 200:
                decoded = response.json()['text']
                print(f"Decoded:  {decoded}")
            else:
                print(f"Decode failed: {response.status_code}")
        else:
            print(f"Encode failed: {response.status_code}")
    
    print("\n=== Testing Model Output Distribution ===")
    # Create random input to see what the model outputs
    from lvm_eval.routes import SimpleTransformerModel
    
    model_config = checkpoint.get('model_config', {})
    model = SimpleTransformerModel(**model_config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Test with random input
    random_input = torch.randn(1, 3, 768)  # Batch=1, Seq=3, Features=768
    with torch.no_grad():
        output = model(random_input)
        print(f"Random input shape: {random_input.shape}")
        print(f"Model output shape: {output.shape}")
        print(f"Output mean: {output.mean().item():.4f}")
        print(f"Output std: {output.std().item():.4f}")
        print(f"Output min: {output.min().item():.4f}")
        print(f"Output max: {output.max().item():.4f}")
        
        # Check if output is in reasonable range for embeddings
        if output.std().item() < 0.01:
            print("WARNING: Output has very low variance - model may be collapsed!")
        if abs(output.mean().item()) > 10:
            print("WARNING: Output mean is very high - may be unstable!")

if __name__ == "__main__":
    test_model()
