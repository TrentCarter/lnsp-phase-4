#!/usr/bin/env python3
"""
Attempt to get vec2text working by:
1. Forcing CPU-only (avoid device conflicts)
2. Using more iterations
3. Testing multiple texts
"""

import os
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer

# Force CPU-only to avoid device conflicts
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

print("\n" + "=" * 80)
print("Vec2Text Fix Attempt - CPU-Only, More Iterations")
print("=" * 80)
print()

# Test texts
test_texts = [
    "The quick brown fox jumps over the lazy dog.",
    "Machine learning is a subset of artificial intelligence.",
    "Python is a popular programming language.",
]

print("Step 1: Loading models (CPU-only)...")
try:
    # Load teacher embedder
    teacher = SentenceTransformer('sentence-transformers/gtr-t5-base')
    teacher = teacher.to('cpu')
    teacher.eval()
    print("✓ Teacher embedder loaded (CPU)")

    # Load vec2text corrector
    from vec2text.api import load_pretrained_corrector
    corrector = load_pretrained_corrector("gtr-base")

    # Force everything to CPU
    corrector.inversion_trainer.model = corrector.inversion_trainer.model.to('cpu')
    corrector.model = corrector.model.to('cpu')
    corrector.inversion_trainer.model.eval()
    corrector.model.eval()

    # Also move embedder to CPU
    if hasattr(corrector.inversion_trainer, 'embedder_model'):
        corrector.inversion_trainer.embedder_model = corrector.inversion_trainer.embedder_model.to('cpu')

    print("✓ Vec2text corrector loaded (CPU)")

except Exception as e:
    print(f"✗ Model loading failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print()
print("=" * 80)
print("Testing Round-Trip with 20 Iterations")
print("=" * 80)
print()

for i, test_text in enumerate(test_texts, 1):
    print(f"\n{'─' * 80}")
    print(f"Test {i}/{len(test_texts)}")
    print(f"{'─' * 80}")
    print(f"Original: {test_text}")

    try:
        # Encode with teacher
        with torch.no_grad():
            embedding = teacher.encode([test_text], convert_to_tensor=True, device='cpu')
            embedding = embedding.to('cpu').float()
            embedding = F.normalize(embedding, dim=-1)

        print(f"  → Encoded to {embedding.shape} on {embedding.device}")

        # Decode with vec2text (20 iterations)
        gen_kwargs = {
            'min_length': 1,
            'max_length': 128,
            'do_sample': False,
            'num_beams': 1,
        }

        with torch.no_grad():
            # Move embedding explicitly to CPU
            embedding_cpu = embedding.to('cpu')

            decoded_tokens = corrector.generate(
                inputs={"frozen_embeddings": embedding_cpu},
                generation_kwargs=gen_kwargs,
                num_recursive_steps=20,  # More iterations
                sequence_beam_width=1,
            )

        decoded_text = corrector.tokenizer.batch_decode(decoded_tokens, skip_special_tokens=True)[0].strip()
        print(f"Decoded:  {decoded_text}")

        # Calculate metrics
        with torch.no_grad():
            decoded_emb = teacher.encode([decoded_text], convert_to_tensor=True, device='cpu')
            decoded_emb = decoded_emb.to('cpu').float()
            decoded_emb = F.normalize(decoded_emb, dim=-1)
            cosine_sim = F.cosine_similarity(embedding_cpu, decoded_emb, dim=-1).item()

        # Word overlap
        orig_words = set(test_text.lower().split())
        decoded_words = set(decoded_text.lower().split())
        overlap = len(orig_words & decoded_words)
        total = len(orig_words)

        print(f"\n  Cosine similarity: {cosine_sim:.4f}")
        print(f"  Word overlap:      {overlap}/{total} ({100*overlap/total:.1f}%)")

        if overlap >= total * 0.7:  # 70% or more
            print(f"  ✓ GOOD reconstruction")
        elif overlap >= total * 0.3:  # 30-70%
            print(f"  ○ PARTIAL reconstruction")
        else:
            print(f"  ✗ POOR reconstruction (gibberish)")

    except Exception as e:
        print(f"\n  ✗ Round-trip failed: {e}")
        import traceback
        traceback.print_exc()

print("\n" + "=" * 80)
print("RESULT")
print("=" * 80)
print()
print("If round-trip works with 20 iterations:")
print("  → Vec2text CAN work, just needs more iterations")
print("  → Update production code to use 20+ iterations")
print()
print("If round-trip still produces gibberish:")
print("  → Vec2text models are fundamentally incompatible")
print("  → Need to try alternative models or approach")
print()
