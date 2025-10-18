#!/usr/bin/env python3
"""
Test Full LVM Pipeline: Text → Vector → LVM → Vector → Text
=============================================================

Complete end-to-end pipeline:
1. Input: 5 text chunks (context)
2. Encode: Text → 768D vectors (using vec2text-compatible encoder)
3. LVM: 5x768D → 1x768D (predict next vector)
4. Decode: 768D → Text (using vec2text decoder)
5. Output: Predicted next chunk text

This measures the TOTAL latency including encoding and decoding.

Usage:
    python tools/test_full_lvm_pipeline.py --model-type amn
"""

import argparse
import sys
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch

sys.path.insert(0, 'app/lvm')
sys.path.insert(0, 'app/vect_text_vect')

from models import create_model, load_model
from vec_text_vect_isolated import IsolatedVecTextVectOrchestrator


class FullLVMPipeline:
    """Complete text→vec→LVM→vec→text pipeline"""

    def __init__(self, lvm_model, lvm_device, orchestrator):
        self.lvm_model = lvm_model
        self.lvm_device = lvm_device
        self.orchestrator = orchestrator

    def encode_texts(self, texts: List[str]) -> np.ndarray:
        """Encode texts to 768D vectors"""
        return self.orchestrator.encode_texts(texts)

    def decode_vector(self, vector: np.ndarray, subscriber: str = 'jxe') -> str:
        """Decode 768D vector back to text"""
        # Use vec2text decoder (jxe or ielab)
        result = self.orchestrator.vec_to_text(
            vector,
            subscribers=[subscriber],
            steps=1
        )
        return result[subscriber]['decoded_text']

    def predict_next_chunk(
        self,
        context_texts: List[str]
    ) -> Tuple[str, dict]:
        """
        Given 5 context chunks, predict the next chunk.

        Returns:
            (predicted_text, timing_dict)
        """
        timings = {}

        # 1. Encode context texts to vectors
        start = time.perf_counter()
        context_vectors = self.encode_texts(context_texts)  # [5, 768]
        timings['encode_ms'] = (time.perf_counter() - start) * 1000

        # 2. Run LVM to predict next vector
        start = time.perf_counter()
        context_tensor = torch.FloatTensor(context_vectors).unsqueeze(0)  # [1, 5, 768]
        context_tensor = context_tensor.to(self.lvm_device)

        with torch.no_grad():
            predicted_vector = self.lvm_model(context_tensor)  # [1, 768]

        predicted_vector = predicted_vector.cpu().numpy()[0]  # [768]
        timings['lvm_ms'] = (time.perf_counter() - start) * 1000

        # 3. Decode predicted vector to text
        start = time.perf_counter()
        predicted_text = self.decode_vector(predicted_vector, subscriber='jxe')
        timings['decode_ms'] = (time.perf_counter() - start) * 1000

        timings['total_ms'] = sum(timings.values())

        return predicted_text, timings


def test_pipeline_with_real_data():
    """Test pipeline with real Wikipedia-like chunks"""

    # Example: 5 consecutive Wikipedia chunks about AI
    context_chunks = [
        "Artificial intelligence (AI) is intelligence demonstrated by machines.",
        "The term artificial intelligence was coined in 1956 by John McCarthy.",
        "AI research has been highly successful in developing effective techniques.",
        "Machine learning algorithms build a model based on sample data.",
        "Deep learning is part of a broader family of machine learning methods."
    ]

    expected_continuation = "Neural networks are computing systems inspired by biological neural networks."

    return context_chunks, expected_continuation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-type', default='amn', choices=['amn', 'lstm', 'gru', 'transformer'])
    parser.add_argument('--model-dir', help='Path to trained model directory')
    parser.add_argument('--device', default='mps' if torch.backends.mps.is_available() else 'cpu')
    parser.add_argument('--num-trials', type=int, default=10, help='Number of trials for averaging')
    args = parser.parse_args()

    print("=" * 80)
    print("Full LVM Pipeline Test: Text → Vec → LVM → Vec → Text")
    print("=" * 80)
    print()

    # Find model directory
    if args.model_dir:
        model_dir = Path(args.model_dir)
    else:
        # Find latest trained model of this type
        models_base = Path('artifacts/lvm/models')
        matching = sorted(models_base.glob(f'{args.model_type}_*'))
        if not matching:
            print(f"❌ No trained {args.model_type.upper()} model found!")
            print(f"   Run training first: python app/lvm/train_unified.py --model-type {args.model_type}")
            return
        model_dir = matching[-1]

    print(f"📦 Loading Model: {model_dir.name}")
    device = torch.device(args.device)

    # Load LVM model
    checkpoint_path = model_dir / 'best_model.pt'
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    model_config = checkpoint.get('model_config', {})
    lvm_model = create_model(args.model_type, **model_config)
    lvm_model.load_state_dict(checkpoint['model_state_dict'])
    lvm_model.to(device)
    lvm_model.eval()

    print(f"   ✓ LVM Model Loaded ({lvm_model.count_parameters():,} parameters)")

    # Load vec2text encoder/decoder
    print(f"\n📦 Loading Vec2Text Orchestrator...")
    orchestrator = IsolatedVecTextVectOrchestrator()
    print(f"   ✓ Vec2Text Ready (encoder + decoder)")

    # Create pipeline
    pipeline = FullLVMPipeline(lvm_model, device, orchestrator)

    # Get test data
    context_chunks, expected_continuation = test_pipeline_with_real_data()

    print("\n" + "=" * 80)
    print("🧪 Testing Pipeline with Real Data")
    print("=" * 80)

    print("\n📝 Context (5 chunks):")
    for i, chunk in enumerate(context_chunks, 1):
        print(f"   {i}. {chunk}")

    print(f"\n🎯 Expected continuation:\n   → {expected_continuation}")

    # Warmup run
    print(f"\n⏳ Warming up...")
    _, _ = pipeline.predict_next_chunk(context_chunks)
    print("   ✓ Warmup complete")

    # Multiple trials
    print(f"\n⏱️  Running {args.num_trials} trials...")
    all_timings = []
    predictions = []

    for trial in range(args.num_trials):
        predicted_text, timings = pipeline.predict_next_chunk(context_chunks)
        all_timings.append(timings)
        predictions.append(predicted_text)

        if trial == 0:
            # Print first prediction
            print(f"\n🤖 LVM Prediction:\n   → {predicted_text}\n")

    # Calculate average timings
    avg_timings = {
        key: np.mean([t[key] for t in all_timings])
        for key in all_timings[0].keys()
    }

    print("=" * 80)
    print("📊 Performance Results (averaged over {} trials)".format(args.num_trials))
    print("=" * 80)
    print()

    print("⏱️  Latency Breakdown:")
    print(f"   1. Text Encoding:      {avg_timings['encode_ms']:>8.2f} ms  ({avg_timings['encode_ms']/avg_timings['total_ms']*100:.1f}%)")
    print(f"   2. LVM Inference:      {avg_timings['lvm_ms']:>8.2f} ms  ({avg_timings['lvm_ms']/avg_timings['total_ms']*100:.1f}%)")
    print(f"   3. Vector Decoding:    {avg_timings['decode_ms']:>8.2f} ms  ({avg_timings['decode_ms']/avg_timings['total_ms']*100:.1f}%)")
    print(f"   ───────────────────────────────────")
    print(f"   TOTAL PIPELINE:        {avg_timings['total_ms']:>8.2f} ms")
    print()

    print("📈 Throughput:")
    queries_per_sec = 1000 / avg_timings['total_ms']
    print(f"   {queries_per_sec:.1f} complete predictions/sec")
    print()

    print("💾 Model Details:")
    print(f"   Model: {args.model_type.upper()}")
    print(f"   Parameters: {lvm_model.count_parameters():,}")
    print(f"   Val Cosine: {checkpoint.get('val_cosine', 'N/A')}")
    print()

    print("=" * 80)
    print("📋 Summary")
    print("=" * 80)
    print()
    print(f"✅ Full pipeline works end-to-end!")
    print(f"✅ Total latency: {avg_timings['total_ms']:.2f} ms per query")
    print(f"✅ Throughput: {queries_per_sec:.1f} predictions/sec")
    print()
    print(f"⚠️  Bottleneck: {'Encoding' if avg_timings['encode_ms'] > avg_timings['decode_ms'] and avg_timings['encode_ms'] > avg_timings['lvm_ms'] else 'Decoding' if avg_timings['decode_ms'] > avg_timings['lvm_ms'] else 'LVM'}")
    print()

    # Quality assessment (manual for now)
    print("🔍 Quality Assessment (Manual):")
    print("   Check if predicted text is semantically related to context.")
    print("   For production, compare predicted vector cosine similarity to ground truth.")
    print()

    print("=" * 80)


if __name__ == '__main__':
    main()
