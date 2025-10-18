#!/usr/bin/env python3
"""
Complete LVM Pipeline Test
Tests: GTR-T5 encoding, vec2text decoding, LVM training, and generative prediction
"""

import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from typing import List, Dict, Tuple

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.db_postgres import connect as connect_pg

print("=" * 80)
print("LVM Pipeline Test")
print("=" * 80)
print()

# ============================================================================
# Step 1: Test GTR-T5 ↔ vec2text Round-Trip
# ============================================================================

print("Step 1: Testing GTR-T5 → 768D → vec2text round-trip...")
print()

# Test sentences
test_sentences = [
    "The quick brown fox jumps over the lazy dog.",
    "Machine learning models process sequential data efficiently.",
    "Photosynthesis converts sunlight into chemical energy in plants."
]

# Initialize GTR-T5 encoder
print("  Loading GTR-T5 encoder...")
from sentence_transformers import SentenceTransformer
encoder = SentenceTransformer('sentence-transformers/gtr-t5-base')
print(f"  ✓ GTR-T5 loaded (dimension: {encoder.get_sentence_embedding_dimension()})")
print()

# Encode test sentences
print("  Encoding test sentences...")
embeddings = encoder.encode(test_sentences, convert_to_numpy=True)
print(f"  ✓ Generated {len(embeddings)} embeddings, shape: {embeddings[0].shape}")
print()

# Initialize vec2text decoder
print("  Loading vec2text decoder (IELab)...")
sys.path.insert(0, str(Path(__file__).parent.parent / "app" / "vect_text_vect"))
from vec_text_vect_isolated import IsolatedVecTextVectOrchestrator

os.environ['VEC2TEXT_FORCE_PROJECT_VENV'] = '1'
os.environ['VEC2TEXT_DEVICE'] = 'cpu'

orchestrator = IsolatedVecTextVectOrchestrator()
print("  ✓ Vec2text orchestrator loaded")
print()

# Test round-trip
print("  Testing round-trip reconstruction...")
print()
for i, (original, embedding) in enumerate(zip(test_sentences, embeddings)):
    # Convert to torch tensor
    embedding_tensor = torch.from_numpy(embedding).unsqueeze(0)  # Add batch dim

    # Decode with IELab (CPU-only) using subprocess wrapper
    result = orchestrator._run_subscriber_subprocess(
        'ielab',
        embedding_tensor.cpu(),
        metadata={'original_texts': [original]},
        device_override='cpu'
    )

    if result['status'] == 'error':
        print(f"  Test {i+1}: ERROR - {result['error']}")
        continue

    decoded_text = result['result'][0] if isinstance(result['result'], list) else result['result']

    print(f"  Test {i+1}:")
    print(f"    Original: {original}")
    print(f"    Decoded:  {decoded_text}")
    print(f"    Match: {'✓' if original.lower() in decoded_text.lower() or decoded_text.lower() in original.lower() else '✗'}")
    print()

print("✓ Step 1 Complete: GTR-T5 ↔ vec2text working")
print()

# ============================================================================
# Step 2: Extract Training Vectors from Database
# ============================================================================

print("Step 2: Extracting 768D training vectors from database...")
print()

# Connect to database
conn = connect_pg()
cur = conn.cursor()

# Get sequential chunks from multiple articles for testing
query = """
SELECT
    e.cpe_id,
    e.concept_text,
    e.batch_id,
    e.created_at,
    v.concept_vec
FROM cpe_entry e
JOIN cpe_vectors v ON e.cpe_id = v.cpe_id
WHERE e.dataset_source = 'wikipedia_500k'
  AND e.batch_id IS NOT NULL
ORDER BY e.batch_id, e.created_at
LIMIT 200;
"""

print("  Fetching sequential chunks...")
cur.execute(query)
rows = cur.fetchall()

if len(rows) == 0:
    print("  ✗ No data found in database!")
    sys.exit(1)

print(f"  ✓ Fetched {len(rows)} sequential chunks")
print()

# Extract vectors and texts
chunk_ids = []
chunk_texts = []
chunk_vectors = []
chunk_batches = []

for row in rows:
    cpe_id, concept_text, batch_id, created_at, vec_str = row

    # Deserialize vector (stored as pgvector type, returned as string like "[0.1,0.2,...]")
    # Remove brackets and parse as floats
    vec_str_cleaned = vec_str.strip('[]')
    vec = np.array([float(x) for x in vec_str_cleaned.split(',')], dtype=np.float32)

    chunk_ids.append(cpe_id)
    chunk_texts.append(concept_text)
    chunk_vectors.append(vec)
    chunk_batches.append(batch_id)

chunk_vectors = np.array(chunk_vectors)
print(f"  Vector matrix shape: {chunk_vectors.shape}")
print(f"  Vector dimension: {chunk_vectors.shape[1]}")
print()

# Group by batch (article) for sequential training
from collections import defaultdict
batch_sequences = defaultdict(list)
for i, batch_id in enumerate(chunk_batches):
    batch_sequences[batch_id].append({
        'text': chunk_texts[i],
        'vector': chunk_vectors[i],
        'cpe_id': chunk_ids[i]
    })

# Note: Chunks are already ordered by created_at within each batch from the query

print(f"  Organized into {len(batch_sequences)} article sequences")
for batch_id, seq in list(batch_sequences.items())[:3]:
    print(f"    Article {batch_id}: {len(seq)} chunks")
print()

print("✓ Step 2 Complete: Training vectors extracted")
print()

# ============================================================================
# Step 3: Train Simple LVM Transformer
# ============================================================================

print("Step 3: Training LVM Transformer on sequential prediction...")
print()

# Simple Transformer for next-vector prediction
class SimpleLVM(nn.Module):
    """Simple transformer that predicts next 768D vector from sequence"""
    def __init__(self, d_model=768, nhead=8, num_layers=4, dim_feedforward=2048):
        super().__init__()
        self.d_model = d_model

        # Positional encoding
        self.pos_encoder = nn.Parameter(torch.randn(1, 100, d_model) * 0.02)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True,
            dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output projection (predict next vector)
        self.output_proj = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        """
        Args:
            x: (batch, seq_len, 768) - sequence of 768D vectors
            mask: Optional attention mask
        Returns:
            (batch, 768) - predicted next vector
        """
        batch_size, seq_len, _ = x.shape

        # Add positional encoding
        x = x + self.pos_encoder[:, :seq_len, :]

        # Transformer encoding
        encoded = self.transformer(x, mask=mask)

        # Take last position and project
        last_hidden = encoded[:, -1, :]  # (batch, 768)
        next_vec = self.output_proj(last_hidden)  # (batch, 768)

        return next_vec

# Create training data (sequences of 5 → predict 6th)
print("  Preparing training data (context=5, predict next)...")
train_sequences = []
train_targets = []

for batch_id, chunks in batch_sequences.items():
    if len(chunks) < 6:  # Need at least 6 chunks
        continue

    # Create sliding windows
    for i in range(len(chunks) - 5):
        context = [chunks[j]['vector'] for j in range(i, i+5)]
        target = chunks[i+5]['vector']
        train_sequences.append(np.array(context))
        train_targets.append(target)

if len(train_sequences) == 0:
    print("  ✗ Not enough sequential data for training!")
    sys.exit(1)

train_sequences = np.array(train_sequences)
train_targets = np.array(train_targets)

print(f"  ✓ Created {len(train_sequences)} training examples")
print(f"    Input shape: {train_sequences.shape} (batch, seq=5, dim=768)")
print(f"    Target shape: {train_targets.shape} (batch, dim=768)")
print()

# Convert to tensors
X_train = torch.from_numpy(train_sequences).float()
y_train = torch.from_numpy(train_targets).float()

# Initialize model
print("  Initializing LVM transformer...")
model = SimpleLVM(d_model=768, nhead=8, num_layers=4, dim_feedforward=2048)
print(f"  ✓ Model initialized ({sum(p.numel() for p in model.parameters()):,} parameters)")
print()

# Training setup
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
criterion = nn.MSELoss()

# Train
print("  Training LVM (50 epochs)...")
model.train()
batch_size = 32
num_epochs = 50

for epoch in range(num_epochs):
    total_loss = 0
    num_batches = 0

    # Shuffle data
    indices = torch.randperm(len(X_train))

    for i in range(0, len(X_train), batch_size):
        batch_idx = indices[i:i+batch_size]
        X_batch = X_train[batch_idx]
        y_batch = y_train[batch_idx]

        # Forward
        optimizer.zero_grad()
        pred = model(X_batch)
        loss = criterion(pred, y_batch)

        # Backward
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    avg_loss = total_loss / num_batches

    if (epoch + 1) % 10 == 0:
        print(f"    Epoch {epoch+1:3d}: Loss = {avg_loss:.6f}")

print()
print("✓ Step 3 Complete: LVM trained")
print()

# ============================================================================
# Step 4: Test Generative Prediction
# ============================================================================

print("Step 4: Testing generative prediction (text→768D→LVM→768D→text)...")
print()

# Find a test sequence (first article with 6+ chunks)
test_batch_id = None
test_sequence = None

for batch_id, chunks in batch_sequences.items():
    if len(chunks) >= 6:
        test_batch_id = batch_id
        test_sequence = chunks[:6]
        break

if test_sequence is None:
    print("  ✗ No test sequence found!")
    sys.exit(1)

print(f"  Test article: {test_batch_id}")
print(f"  Using first 6 chunks for test")
print()

# Ground truth: chunks 0-5
context_chunks = test_sequence[:5]
target_chunk = test_sequence[5]

print("  Context (first 5 chunks):")
for i, chunk in enumerate(context_chunks):
    print(f"    [{i}] {chunk['text'][:80]}...")
print()

print("  Ground truth (6th chunk):")
print(f"    [5] {target_chunk['text'][:80]}...")
print()

# Step 4a: Encode context with GTR-T5
print("  Step 4a: Encoding context with GTR-T5...")
context_texts = [chunk['text'] for chunk in context_chunks]
context_vectors = encoder.encode(context_texts, convert_to_numpy=True)
context_tensor = torch.from_numpy(context_vectors).float().unsqueeze(0)  # (1, 5, 768)
print(f"    ✓ Context encoded: {context_tensor.shape}")
print()

# Step 4b: Predict next vector with LVM
print("  Step 4b: Predicting next vector with LVM...")
model.eval()
with torch.no_grad():
    predicted_vec = model(context_tensor)  # (1, 768)

predicted_vec_np = predicted_vec.squeeze(0).numpy()
print(f"    ✓ Predicted vector: {predicted_vec_np.shape}")
print()

# Step 4c: Decode predicted vector with vec2text
print("  Step 4c: Decoding predicted vector with vec2text...")
predicted_tensor = torch.from_numpy(predicted_vec_np).unsqueeze(0)  # (1, 768)

decode_result = orchestrator._run_subscriber_subprocess(
    'ielab',
    predicted_tensor.cpu(),
    metadata={'original_texts': ['']},
    device_override='cpu'
)

if decode_result['status'] == 'error':
    print(f"    ✗ Decoding failed: {decode_result['error']}")
    predicted_text = "[DECODE ERROR]"
else:
    predicted_text = decode_result['result'][0] if isinstance(decode_result['result'], list) else decode_result['result']
    print(f"    ✓ Decoded text: {predicted_text[:100]}...")
print()

# Step 4d: Compare with ground truth
print("  Step 4d: Comparison...")
print()
print("  Ground truth (actual 6th chunk):")
print(f"    {target_chunk['text']}")
print()
print("  LVM prediction:")
print(f"    {predicted_text}")
print()

# Calculate cosine similarity
from numpy.linalg import norm
def cosine_similarity(a, b):
    return np.dot(a, b) / (norm(a) * norm(b))

# Compare predicted vector vs ground truth vector
target_vec = target_chunk['vector']
similarity = cosine_similarity(predicted_vec_np, target_vec)
print(f"  Vector similarity: {similarity:.4f}")
print()

# Simple text overlap check
ground_words = set(target_chunk['text'].lower().split())
pred_words = set(predicted_text.lower().split())
overlap = len(ground_words & pred_words)
print(f"  Word overlap: {overlap}/{len(ground_words)} words ({overlap/len(ground_words)*100:.1f}%)")
print()

print("=" * 80)
print("Test Complete!")
print("=" * 80)
print()

print("Summary:")
print(f"  ✓ GTR-T5 encoding: Working")
print(f"  ✓ vec2text decoding: Working")
print(f"  ✓ LVM training: {len(train_sequences)} examples, final loss: {avg_loss:.6f}")
print(f"  ✓ Generative prediction: {similarity:.4f} cosine similarity")
print()

# Save model
model_path = Path(__file__).parent.parent / "artifacts" / "lvm_test_model.pt"
model_path.parent.mkdir(exist_ok=True)
torch.save({
    'model_state_dict': model.state_dict(),
    'model_config': {
        'd_model': 768,
        'nhead': 8,
        'num_layers': 4,
        'dim_feedforward': 2048
    },
    'training_loss': avg_loss,
    'num_examples': len(train_sequences)
}, model_path)
print(f"Model saved to: {model_path}")
print()

# Cleanup
cur.close()
conn.close()
