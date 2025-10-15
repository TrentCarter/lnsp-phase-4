#!/usr/bin/env python3
import numpy as np
import torch
import torch.nn as nn
import requests
from pathlib import Path
import sys

sys.path.insert(0, '.')
from app.lvm.train_lstm_baseline import LSTMVectorPredictor
from app.lvm.train_transformer import TransformerVectorPredictor

VEC2TEXT_URL = "http://127.0.0.1:8766"

def compute_cosine(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def decode_vector(vector):
    """Decode vector using vec2text API."""
    response = requests.post(
        f"{VEC2TEXT_URL}/decode",
        json={"vectors": [vector.tolist()], "steps": 1, "subscribers": "jxe"},
        timeout=60
    )
    result = response.json()
    # Response format: {"results": [{"index": 0, "subscribers": {"gtr → jxe": {"output": "...", "cosine": 0.x}}}], "count": 1}
    return result["results"][0]["subscribers"]["gtr → jxe"]["output"]

def test_model(model_name, model_path, model_class):
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    
    # Load model
    checkpoint = torch.load(model_path, map_location=device)
    if model_class == LSTMVectorPredictor:
        model = model_class(input_dim=768, hidden_dim=512, num_layers=2).to(device)
    else:
        model = model_class(input_dim=768, d_model=512, nhead=8, num_layers=4).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Load data
    npz = np.load("artifacts/lvm/training_sequences_ctx5.npz")
    texts_npz = np.load("artifacts/lvm/wikipedia_42113_ordered.npz")
    contexts = npz['context_sequences']
    targets = npz['target_vectors']
    texts = texts_npz['texts']
    
    print(f"\n{'='*80}")
    print(f"{model_name}")
    print(f"{'='*80}")
    print(f"Val Loss: {checkpoint['val_loss']:.6f} | Val Cosine: {checkpoint['val_cosine']*100:.2f}%\n")
    
    # Test 5 examples
    indices = [100, 2000, 4000, 6000, 8000]
    for i, idx in enumerate(indices, 1):
        if idx >= len(contexts):
            continue
        
        # Get context and predict
        context = torch.FloatTensor(contexts[idx]).unsqueeze(0).to(device)
        target = targets[idx]
        ground_truth = texts[idx + 5]
        
        with torch.no_grad():
            prediction = model(context).cpu().numpy()[0]
        
        # Decode
        pred_decoded = decode_vector(prediction)
        gt_decoded = decode_vector(target)
        
        # Cosine
        pred_cosine = compute_cosine(prediction, target)
        
        print(f"Example {i} (idx={idx})")
        print(f"  Ground Truth: {ground_truth[:80]}")
        print(f"  GT→Vec2Text:  {gt_decoded[:80]}")
        print(f"  LVM→Vec2Text: {pred_decoded[:80]}")
        print(f"  Pred Cosine:  {pred_cosine:.4f}")
        print()
    
    return checkpoint['val_loss'], checkpoint['val_cosine']

print("="*80)
print("LVM + Vec2Text End-to-End Test")
print("="*80)

lstm_loss, lstm_cos = test_model("LSTM", "artifacts/lvm/models/lstm_baseline/best_model.pt", LSTMVectorPredictor)
trans_loss, trans_cos = test_model("Transformer", "artifacts/lvm/models/transformer/best_model.pt", TransformerVectorPredictor)

print(f"\n{'='*80}")
print("FINAL RESULTS")
print(f"{'='*80}\n")
print(f"| Model       | Params | Val Loss | Val Cosine |")
print(f"|-------------|--------|----------|------------|")
print(f"| LSTM        | 5.1M   | {lstm_loss:.6f} | {lstm_cos*100:6.2f}%   |")
print(f"| Transformer | 17.6M  | {trans_loss:.6f} | {trans_cos*100:6.2f}%   |")
