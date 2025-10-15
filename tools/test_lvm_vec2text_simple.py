#!/usr/bin/env python3
"""Simple LVM + Vec2Text Test"""
import numpy as np
import torch
import torch.nn as nn
import requests
from pathlib import Path
import sys

sys.path.insert(0, '.')
from app.lvm.train_lstm_baseline import LSTMVectorPredictor
from app.lvm.train_transformer import TransformerVectorPredictor

VEC2TEXT_DECODER_URL = "http://127.0.0.1:8766"

def compute_cosine(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def decode_vector(vector):
    """Decode vector to text."""
    try:
        response = requests.post(
            f"{VEC2TEXT_DECODER_URL}/decode",
            json={"vectors": [vector.tolist()], "steps": 1, "subscribers": "jxe"},
            timeout=30
        )
        result = response.json()
        return result.get("decoded_texts", result.get("texts", [None]))[0]
    except Exception as e:
        return f"ERROR: {e}"

def load_and_test(model_name, model_path, model_class):
    """Load model and test."""
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Create model
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
    
    # Test indices
    indices = [100, 2000, 4000, 6000, 8000]
    
    print(f"\n{'='*80}")
    print(f"{model_name} Model Results")
    print(f"{'='*80}")
    print(f"Val Loss: {checkpoint['val_loss']:.6f}")
    print(f"Val Cosine: {checkpoint['val_cosine']*100:.2f}%")
    
    results = []
    for i, idx in enumerate(indices):
        if idx >= len(contexts):
            continue
            
        # Predict
        context = torch.FloatTensor(contexts[idx]).unsqueeze(0).to(device)
        target = targets[idx]
        ground_truth = texts[idx + 5]
        
        with torch.no_grad():
            prediction = model(context).cpu().numpy()[0]
        
        # Metrics
        pred_cosine = compute_cosine(prediction, target)
        
        # Decode
        pred_decoded = decode_vector(prediction)
        gt_decoded = decode_vector(target)
        
        results.append({
            'gt': ground_truth,
            'gt_dec': gt_decoded,
            'pred_dec': pred_decoded,
            'cos': pred_cosine
        })
        
        print(f"\n--- Test {i+1} ---")
        print(f"Ground Truth:   {ground_truth[:70]}")
        print(f"GT→Vec2Text:    {gt_decoded[:70]}")
        print(f"Pred→Vec2Text:  {pred_decoded[:70]}")
        print(f"Pred Cosine:    {pred_cosine:.4f}")
    
    return checkpoint['val_loss'], checkpoint['val_cosine'], results

print("="*80)
print("LVM + Vec2Text Pipeline Test")
print("="*80)

lstm_loss, lstm_cos, lstm_res = load_and_test(
    "LSTM",
    "artifacts/lvm/models/lstm_baseline/best_model.pt",
    LSTMVectorPredictor
)

trans_loss, trans_cos, trans_res = load_and_test(
    "Transformer",
    "artifacts/lvm/models/transformer/best_model.pt",
    TransformerVectorPredictor
)

print(f"\n{'='*80}")
print("Comparison Table")
print(f"{'='*80}")
print(f"\n| Model       | Params | Val Loss | Val Cosine |")
print(f"|-------------|--------|----------|------------|")
print(f"| LSTM        | 5.1M   | {lstm_loss:.6f} | {lstm_cos*100:6.2f}%   |")
print(f"| Transformer | 17.6M  | {trans_loss:.6f} | {trans_cos*100:6.2f}%   |")
print()
