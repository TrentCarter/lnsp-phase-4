#!/usr/bin/env python3
"""
Simple vec2text test: text → 768D → text
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path

# Force CPU-only
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

def test_simple_vec2text():
    """Test the full text → 768D → text pipeline"""

    print("🔬 SIMPLE VEC2TEXT TEST")
    print("=" * 50)

    # Simple test texts
    test_texts = [
        "Hello world",
        "The quick brown fox",
        "Machine learning is powerful"
    ]

    for test_text in test_texts:
        print(f"\n📝 Testing: '{test_text}'")
        print("-" * 40)

        try:
            # Step 1: Encode with GTR-T5 (standalone encoder)
            from sentence_transformers import SentenceTransformer

            print("🔄 Encoding to 768D...")
            encoder = SentenceTransformer('sentence-transformers/gtr-t5-base', device='cpu')
            encoder.eval()

            vector_768d = encoder.encode([test_text], convert_to_tensor=True)
            print(f"✅ Vector shape: {vector_768d.shape}")
            print(f"✅ Vector norm: {torch.norm(vector_768d):.4f}")

            # Step 2: Decode with vec2text
            print("🔄 Decoding back to text...")

            from vec2text.api import load_pretrained_corrector

            corrector = load_pretrained_corrector('gtr-base')

            # Generate text from vector
            tokens = corrector.inversion_trainer.generate(
                inputs={'frozen embeddings': vector_768d.unsqueeze(0)},
                generation_kwargs={'min_length': 3, 'max_length': 30}
            )

            decoded_text = corrector.tokenizer.batch_decode(tokens, skip_special_tokens=True)[0]

#!/usr/bin/env python3
"""
Simple vec2text test: text → 768D → text
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path

# Force CPU-only
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

def test_simple_vec2text():
    """Test the full text → 768D → text pipeline"""

    print("🔬 SIMPLE VEC2TEXT TEST")
    print("=" * 50)

    # Simple test texts
    test_texts = [
        "Hello world",
        "The quick brown fox",
        "Machine learning is powerful"
    ]

    for test_text in test_texts:
        print(f"\n📝 Testing: '{test_text}'")
        print("-" * 40)

        try:
            # Step 1: Encode with GTR-T5 (standalone encoder)
            from sentence_transformers import SentenceTransformer

            print("🔄 Encoding to 768D...")
            encoder = SentenceTransformer('sentence-transformers/gtr-t5-base', device='cpu')
            encoder.eval()

            vector_768d = encoder.encode([test_text], convert_to_tensor=True)
            print(f"✅ Vector shape: {vector_768d.shape}")
            print(f"✅ Vector norm: {torch.norm(vector_768d):.4f}")

            # Step 2: Decode with vec2text
            print("🔄 Decoding back to text...")

            from vec2text.api import load_pretrained_corrector

            corrector = load_pretrained_corrector('gtr-base')

            # Generate text from vector
            tokens = corrector.inversion_trainer.generate(
                inputs={'frozen_embeddings': vector_768d.unsqueeze(0)},
                generation_kwargs={'min_length': 3, 'max_length': 30}
            )

            decoded_text = corrector.tokenizer.batch_decode(tokens, skip_special_tokens=True)[0]

            print(f"📝 Decoded: '{decoded_text}'")

            # Step 3: Calculate quality metrics
            decoded_vector = encoder.encode([decoded_text], convert_to_tensor=True)
            cosine_sim = torch.cosine_similarity(vector_768d, decoded_vector, dim=-1).item()

            print(f"🎯 Cosine similarity: {cosine_sim:.4f}")

            # Quality assessment
            if cosine_sim > 0.85:
                print("✅ EXCELLENT - Working perfectly!")
            elif cosine_sim > 0.7:
                print("⚠️  GOOD - Acceptable quality")
            elif cosine_sim > 0.5:
                print("🤔 OK - Some semantic preservation")
            else:
                print("❌ BROKEN - Producing gibberish")

            # Check for obvious gibberish indicators
            if len(decoded_text.strip()) < 3:
                print("❌ WARNING: Very short output - likely gibberish")
            elif decoded_text.count('.') > 3:
                print("❌ WARNING: Too many periods - likely gibberish")
            elif any(word in decoded_text.lower() for word in ['corán', 'heresy', 'assembly', 'torn']):
                print("❌ WARNING: Contains known gibberish words")

        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_simple_vec2text()
