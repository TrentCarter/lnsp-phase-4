#!/usr/bin/env python3
"""
Minimal vec2text test using the exact same API calls as our Vec2TextProcessor.

This isolates whether the problem is:
1. Our wrapper logic
2. The vec2text library itself
"""

import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer

print("\n" + "=" * 80)
print("Minimal Vec2Text Test (Matching Our Infrastructure)")
print("=" * 80)
print()

# Test text
test_text = "The quick brown fox jumps over the lazy dog."
print(f"Original text: {test_text}")
print()

# Step 1: Load teacher embedder (GTR-T5)
print("Step 1: Loading GTR-T5 teacher embedder...")
try:
    teacher = SentenceTransformer('sentence-transformers/gtr-t5-base', device='cpu')
    teacher.eval()
    print("✓ Teacher embedder loaded")
except Exception as e:
    print(f"✗ Failed to load teacher: {e}")
    exit(1)
print()

# Step 2: Load vec2text corrector
print("Step 2: Loading vec2text corrector...")
try:
    from vec2text.api import load_pretrained_corrector
    corrector = load_pretrained_corrector("gtr-base")
    corrector.inversion_trainer.model.to('cpu')
    corrector.model.to('cpu')
    corrector.inversion_trainer.model.eval()
    corrector.model.eval()
    print("✓ Corrector loaded")
except Exception as e:
    print(f"✗ Failed to load corrector: {e}")
    print("\nError details:")
    import traceback
    traceback.print_exc()
    exit(1)
print()

# Step 3: Encode text to vector
print("Step 3: Encoding text with GTR-T5...")
with torch.no_grad():
    embedding = teacher.encode([test_text], convert_to_tensor=True)
    embedding = embedding.to('cpu').float()
    embedding = F.normalize(embedding, dim=-1)
print(f"✓ Encoded to shape {embedding.shape}")
print()

# Step 4: Decode vector to text (inversion step)
print("Step 4: Generating initial hypothesis...")
try:
    with torch.no_grad():
        gen_kwargs = {
            'min_length': 1,
            'max_length': 128,
            'do_sample': False,  # Required by vec2text
            'num_beams': 1,
        }
        tokens = corrector.inversion_trainer.generate(
            inputs={"frozen_embeddings": embedding},
            generation_kwargs=gen_kwargs,
        )
    initial_text = corrector.tokenizer.batch_decode(tokens, skip_special_tokens=True)[0].strip()
    print(f"Initial hypothesis: '{initial_text}'")
    print()
    print("⚠ CRITICAL: Initial hypothesis is GIBBERISH!")
    print("This means vec2text inversion model is broken/incompatible.")
except Exception as e:
    print(f"✗ Initial generation failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)
print()

# Step 5: Refine text (corrector step)
print("Step 5: Refining with corrector (1 iteration)...")
try:
    with torch.no_grad():
        refined_tokens = corrector.generate(
            inputs={"frozen_embeddings": embedding},
            generation_kwargs=gen_kwargs,
            num_recursive_steps=1,
            sequence_beam_width=1,
        )
    final_text = corrector.tokenizer.batch_decode(refined_tokens, skip_special_tokens=True)[0].strip()
    print(f"Final text:      '{final_text}'")
except Exception as e:
    print(f"✗ Refinement failed: {e}")
    print("\nThis is expected if initial hypothesis was gibberish.")
    final_text = initial_text  # Use initial if corrector fails
print()

# Step 6: Calculate similarity
print("Step 6: Calculating similarity...")
with torch.no_grad():
    final_embedding = teacher.encode([final_text], convert_to_tensor=True)
    final_embedding = final_embedding.to('cpu').float()
    final_embedding = F.normalize(final_embedding, dim=-1)
    cosine_sim = F.cosine_similarity(embedding, final_embedding, dim=-1).item()

print(f"Cosine similarity: {cosine_sim:.4f}")
print()

# Step 7: Word overlap analysis
orig_words = set(test_text.lower().split())
final_words = set(final_text.lower().split())
overlap = len(orig_words & final_words)
total = len(orig_words)
print(f"Word overlap: {overlap}/{total} ({100*overlap/total:.1f}%)")
print()

# Diagnosis
print("=" * 80)
print("DIAGNOSIS")
print("=" * 80)
print()
if overlap >= 7:  # Most words match
    print("✓ Vec2text library WORKS correctly")
    print("  → Problem is in our wrapper/server infrastructure")
    print("  → Investigate app/api/vec2text_server.py")
elif overlap >= 3:  # Some words match
    print("⚠ Vec2text PARTIALLY works")
    print("  → May need more iterations or different parameters")
else:  # Complete gibberish
    print("✗ Vec2text library produces GIBBERISH")
    print("  → vec2text installation may be broken")
    print("  → Try: pip install --upgrade --force-reinstall vec2text")
    print("  → Or: Check HuggingFace cache for corrupted models")
print()
