#!/usr/bin/env python3
"""
Test if vec2text quality improves with more iterations.
"""

import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from vec2text.api import load_pretrained_corrector

print("\n" + "=" * 80)
print("Vec2Text Iteration Quality Test")
print("=" * 80)
print()

# Test text
test_text = "The quick brown fox jumps over the lazy dog."
print(f"Original text: {test_text}")
print()

# Load models
print("Loading models...")
teacher = SentenceTransformer('sentence-transformers/gtr-t5-base', device='cpu')
teacher.eval()
corrector = load_pretrained_corrector("gtr-base")
corrector.inversion_trainer.model.to('cpu')
corrector.model.to('cpu')
corrector.inversion_trainer.model.eval()
corrector.model.eval()
print("✓ Models loaded")
print()

# Encode
with torch.no_grad():
    embedding = teacher.encode([test_text], convert_to_tensor=True)
    embedding = embedding.to('cpu').float()
    embedding = F.normalize(embedding, dim=-1)

# Test with different iteration counts
iterations_to_test = [1, 5, 10, 20]

print("Testing different iteration counts...")
print()

for num_iters in iterations_to_test:
    print(f"{'=' * 80}")
    print(f"Testing with {num_iters} iterations")
    print(f"{'=' * 80}")

    try:
        gen_kwargs = {
            'min_length': 1,
            'max_length': 128,
            'do_sample': False,
            'num_beams': 1,
        }

        with torch.no_grad():
            refined_tokens = corrector.generate(
                inputs={"frozen_embeddings": embedding},
                generation_kwargs=gen_kwargs,
                num_recursive_steps=num_iters,
                sequence_beam_width=1,
            )

        decoded = corrector.tokenizer.batch_decode(refined_tokens, skip_special_tokens=True)[0].strip()

        print(f"Decoded:  '{decoded}'")

        # Calculate metrics
        with torch.no_grad():
            decoded_emb = teacher.encode([decoded], convert_to_tensor=True)
            decoded_emb = decoded_emb.to('cpu').float()
            decoded_emb = F.normalize(decoded_emb, dim=-1)
            cosine_sim = F.cosine_similarity(embedding, decoded_emb, dim=-1).item()

        orig_words = set(test_text.lower().split())
        decoded_words = set(decoded.lower().split())
        overlap = len(orig_words & decoded_words)
        total = len(orig_words)

        print(f"Cosine similarity: {cosine_sim:.4f}")
        print(f"Word overlap: {overlap}/{total} ({100*overlap/total:.1f}%)")
        print()

    except Exception as e:
        print(f"✗ Failed with {num_iters} iterations: {e}")
        print()

print("=" * 80)
print("CONCLUSION")
print("=" * 80)
print()
print("If quality improves significantly with more iterations:")
print("  → Vec2text works but needs many iterations")
print("  → Update our code to use 20+ iterations")
print()
print("If quality stays poor regardless of iterations:")
print("  → Vec2text models may be fundamentally broken")
print("  → Need to try different models or approach")
print()
