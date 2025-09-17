#!/usr/bin/env python3
"""
Simple test of Text-Vector-Text pipeline without dependency conflicts
"""

import torch
from sentence_transformers import SentenceTransformer
from tabulate import tabulate
import time

def simple_vec2text_mock(vector, steps=1):
    """Mock vec2text for testing - returns a placeholder"""
    # In a real implementation, this would use the actual vec2text model
    # For now, we'll return a simple transformation
    return f"[Decoded with {steps} steps from vector with norm {torch.norm(vector).item():.3f}]"

def main():
    print("\n" + "="*80)
    print("Simple Text-Vector-Text Pipeline Test")
    print("="*80 + "\n")
    
    # Test inputs
    test_texts = [
        "The cat sits on the mat",
        "Linear algebra basics",
        "Machine learning models"
    ]
    
    # Load GTR-T5 encoder
    print("Loading GTR-T5 encoder...")
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    encoder = SentenceTransformer("sentence-transformers/gtr-t5-base", device=device)
    print(f"✅ Encoder loaded on {device}\n")
    
    # Encode texts
    print("Encoding texts to vectors...")
    start_time = time.time()
    vectors = encoder.encode(
        test_texts,
        normalize_embeddings=True,
        convert_to_tensor=True,
        device=device
    )
    encoding_time = time.time() - start_time
    print(f"✅ Encoded {len(test_texts)} texts in {encoding_time:.2f}s")
    print(f"   Vector shape: {vectors.shape}")
    print(f"   Vector norms: {[torch.norm(vectors[i]).item() for i in range(len(test_texts))]}\n")
    
    # Process through mock vec2text
    results = []
    for i, text in enumerate(test_texts):
        vector = vectors[i]
        
        # Mock decoding
        decoded = simple_vec2text_mock(vector, steps=3)
        
        # Re-encode for cosine similarity
        re_encoded = encoder.encode([decoded], normalize_embeddings=True, convert_to_tensor=True)
        cosine = torch.cosine_similarity(vector, re_encoded.squeeze(), dim=0).item()
        
        results.append({
            'input': text,
            'output': decoded,
            'cosine': cosine
        })
    
    # Display results
    print("Results:")
    print("-" * 80)
    
    table_data = []
    for i, result in enumerate(results):
        table_data.append([
            i,
            result['input'][:40] + "..." if len(result['input']) > 40 else result['input'],
            f"{result['cosine']:.3f}",
            result['output'][:50] + "..." if len(result['output']) > 50 else result['output']
        ])
    
    headers = ['#', 'Input Text', 'Cosine', 'Output Text']
    print(tabulate(table_data, headers=headers, tablefmt='grid'))
    
    print("\n✅ Test completed successfully!")
    print(f"   Average cosine similarity: {sum(r['cosine'] for r in results) / len(results):.3f}")

if __name__ == '__main__':
    main()