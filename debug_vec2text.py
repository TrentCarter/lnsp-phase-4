#!/usr/bin/env python3
"""
Direct test of vec2text decoding to debug gibberish output
"""

import os
import sys
import time
import torch
import numpy as np
from pathlib import Path

# Force CPU-only
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
os.environ['VEC2TEXT_FORCE_CPU'] = '1'

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.vect_text_vect.vec2text_processor import Vec2TextProcessor, Vec2TextConfig

def test_vec2text_directly():
    """Test vec2text decoding directly"""

    print("🔬 Direct Vec2Text Decoding Test")
    print("=" * 50)

    # Test vector (from encoding server)
    test_vector = [
        -0.021062346175312996, 0.027753053233027458, -0.023571845144033432,
        0.022744543850421906, 0.0034527014940977097, 0.02531748265028,
        0.01733117364346981, -0.0003801722778007388, -0.01680866815149784,
        0.02071741782128811, -0.032021842896938324, 0.00853264145553112,
        0.031971149146556854, -0.029292484745383263, 0.0569167360663414,
        -0.0567718930542469, -0.018872594460844994, -0.03293171525001526,
        0.02639967016875744, 0.05489058792591095, 0.005565948318690062,
        -0.0006955786957405508, 0.009523146785795689, 0.06351199001073837,
        -0.021541712805628777, -0.04287438839673996, -0.01574205979704857,
        0.012043065391480923, -0.01127548236399889, 0.01626155525445938,
        0.003931837622076273, -0.03197656199336052, 0.010873091407120228,
        0.03915588930249214, -0.013014577329158783, 0.026780923828482628,
        -0.014013775624334812, 0.018446803092956543, -0.107211634516716,
        -0.011973668821156025, -0.05471318960189819, 0.0023521136026
    ]

    # Pad to 768 dimensions
    while len(test_vector) < 768:
        test_vector.append(0.0)

    print(f"✅ Test vector: {len(test_vector)} dimensions")

    # Test with different configurations
    configs_to_test = [
        {"steps": 1, "seed": 42, "name": "1 step, seed=42"},
        {"steps": 5, "seed": 42, "name": "5 steps, seed=42"},
        {"steps": 10, "seed": 42, "name": "10 steps, seed=42"},
        {"steps": 5, "seed": None, "name": "5 steps, no seed"},
    ]

    for config in configs_to_test:
        print(f"\n🧪 Testing: {config['name']}")
        print("-" * 30)

        try:
            # Create processor
            vec2text_config = Vec2TextConfig(
                teacher_model='sentence-transformers/gtr-t5-base',
                device='cpu',
                random_seed=config['seed'],
                debug=True
            )

            processor = Vec2TextProcessor(vec2text_config)
            print("✅ Processor loaded")

            # Convert vector to tensor
            vector_tensor = torch.tensor([test_vector], dtype=torch.float32)

            # Decode
            start_time = time.time()
            results = processor.decode_embeddings(
                vector_tensor,
                num_iterations=config['steps'],
                beam_width=1,
                prompts=[" "]
            )
            decode_time = time.time() - start_time

            # Extract results
            result = results[0]
            decoded_text = result.get('final_text', 'ERROR')
            cosine = result.get('final_cosine', 0.0)

            print(f"⏱️  Decode time: {decode_time:.2f}s")
            print(f"📝 Decoded: '{decoded_text}'")
            print(f"🎯 Cosine similarity: {cosine:.4f}")

            if cosine > 0.8:
                print("✅ GOOD: High similarity!")
            elif cosine > 0.6:
                print("⚠️  OK: Moderate similarity")
            else:
                print("❌ BAD: Low similarity (gibberish)")

        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_vec2text_directly()
