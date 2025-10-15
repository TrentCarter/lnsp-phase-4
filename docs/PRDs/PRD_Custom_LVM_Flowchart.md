# Custom RWKV Model Flowchart

```
┌─────────────────────────────────────────────────────────────────────┐
│                      Custom RWKV LVM Model Flow                     │
└─────────────────────────────────────────────────────────────────────┘

Input: Context Sequence (5x768D vectors from Wikipedia 8k dataset)
┌─────────────────────────────────────────────────────────────────────┐
│  Context Vectors (sentence-transformers GTR-T5)                   │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐│
│  │ Vec 1       │  │ Vec 2       │  │ Vec 3       │  │ Vec 4       ││
│  │ [768D]      │  │ [768D]      │  │ [768D]      │  │ [768D]      ││
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘│
│  Target Vector (sentence-transformers GTR-T5)                       │
│  ┌─────────────┐                                                    │
│  │ Next Vec    │  ← Ground Truth (sentence-transformers)           │
│  │ [768D]      │                                                    │
│  └─────────────┘                                                    │
└─────────────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Embedding Adapter Layer (Custom Compatibility Bridge)             │
│  - Affine Transform: Linear(768→768)                               │
│  - LayerNorm                                                       │
│  - Maps sentence-transformers → vec2text-compatible space        │
└─────────────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────────────┐
│  RWKV RNN Layers (6 layers, hidden=512)                           │
│  - GRU-based recurrence with custom recurrence tweaks             │
│  - Processes adapted embeddings                                   │
└─────────────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Projection Head                                                   │
│  - Linear(512→768)                                                 │
│  - Dropout(0.1)                                                    │
│  - L2 Normalization (norm=1.0 for vec2text compatibility)         │
└─────────────────────────────────────────────────────────────────────┘
        │
        ▼
Output: Predicted Vector (768D, L2-normed, vec2text-compatible)
┌─────────────────────────────────────────────────────────────────────┐
│  Predicted Vector                                                  │
│  ┌─────────────┐                                                    │
│  │ [768D]      │  ← Use with vec2text decoder for text generation │
│  └─────────────┘                                                    │
└─────────────────────────────────────────────────────────────────────┘

Training: Cosine Similarity Loss (1 - cos(pred, target))
Dataset: 8k Wikipedia sequences (custom 5-step contexts)
```
