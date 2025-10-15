# Custom LVM Models for 8k Wikipedia Dataset

## Overview
This directory contains custom implementations of 9 LVM models, each with a novel compatibility layer to bridge sentence-transformers and vec2text embeddings. Instead of retraining on new data, we'll use an adapter mechanism.

## Models
1. RWKV (169M) - With custom recurrence tweaks
2. RetNet (125M) - O(1) inference optimized
3. DistilGPT-2 (82M) - Lightweight transformer
4. Hyena (125M) - Implicit long convolution
5. Performer (100M) - FA2 attention
6. Linformer (100M) - Low-rank attention
7. S4 (100M) - Structured state space
8. GRU Stacked (12M) - Multi-layer GRU
9. Hybrid Mamba-Attn (150M) - Custom hybrid

## Training Script
See `train_custom_models.py` for unified training.

## Compatibility Layer
Each model includes an `EmbeddingAdapter` to map sentence-transformers vectors to vec2text-compatible space.
