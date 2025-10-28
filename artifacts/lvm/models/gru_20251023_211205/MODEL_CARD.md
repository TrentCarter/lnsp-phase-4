# Model Card: GRU Stack

## Model Information
- **Name**: GRU Stack (4-layer Gated Recurrent Unit)
- **Type**: Latent Vector Model (LVM) - Tokenless Vector-Native
- **Parameters**: 7,079,936 (~7.1M)
- **Architecture**: 4-layer stacked GRU (d_model=512)
- **Status**: **✅ PRODUCTION FALLBACK (Secondary)**

## Training Details
- **Dataset**: Wikipedia 500k (543,556 sequences, articles 1-8470)
- **Training Date**: October 23, 2025
- **Context Size**: 5 vectors (768D each)
- **Prediction Target**: Next vector (768D)
- **Loss Function**: MSE (Mean Squared Error)
- **Optimizer**: AdamW (lr=0.0005, weight_decay=0.01)
- **Epochs**: 20
- **Batch Size**: 32
- **Device**: Apple M1 Max (MPS)

## Performance Metrics

### In-Distribution (Wikipedia 500k val set)
- **Cosine Similarity**: 0.5920 (2nd best in-dist)
- **MSE Loss**: 0.0011
- **Latency (P50)**: 2.11 ms

### Out-of-Distribution (New Wikipedia articles 8471-8970)
- **Cosine Similarity**: 0.6295 (2nd best OOD)
- **Delta**: +0.0375 (excellent generalization)

## Key Strengths
1. **🥈 Runner-Up Performance**: 2nd best on both in-dist (0.5920) and OOD (0.6295)
2. **⚖️ Balanced Trade-off**: Good accuracy with moderate latency (2.11ms)
3. **📊 Stable Architecture**: Well-understood GRU mechanics
4. **🔄 Sequential Processing**: Natural fit for temporal/causal data

## Use Cases
- **Tertiary fallback** when both AMN and Transformer(opt) remain low-confidence
- Queries requiring sequential reasoning
- Medium-complexity domains (between AMN speed and Transformer accuracy)
- Backup model for production resilience

## Escalation Criteria (Use GRU)
- Both AMN AND Transformer(opt) show low confidence
- Sequential/temporal reasoning required
- Medium-stakes queries (not top-tier critical)
- When Transformer latency (2.65ms) is too high but AMN (0.62ms) insufficient

## Limitations
- **3.4x slower than AMN**: 2.11ms vs 0.62ms
- **In-dist accuracy**: 0.5920 vs Transformer(opt) 0.5864 (+0.0056 better)
- **OOD performance**: 0.6295 vs AMN 0.6375 (-0.0080 worse)
- **Middle-ground model**: Not fastest, not most accurate

## Deployment Configuration
- **Symlink**: `artifacts/lvm/fallback_secondary`
- **Load Path**: `artifacts/lvm/models/gru_20251023_211205/best_model.pt`
- **Trigger Conditions**:
  - AMN confidence < 0.70 AND Transformer(opt) confidence < 0.70
  - Sequential reasoning detected
  - Resilience/backup needed
- **Expected Usage**: < 5% of queries

## Comparison to Other Models

| Model | In-Dist | OOD | Latency | Use Case |
|-------|---------|-----|---------|----------|
| AMN | 0.5597 | **0.6375** 🏆 | **0.62ms** ⚡ | Primary |
| Transformer(opt) | **0.5864** 🏆 | 0.6257 | 2.65ms | Accuracy fallback |
| **GRU** | 0.5920 | 0.6295 | 2.11ms | Secondary fallback |

## When to Choose GRU Over Others
- **vs AMN**: When AMN fails confidence check but full Transformer overhead not justified
- **vs Transformer(opt)**: When Transformer also low-confidence or latency budget tighter than 2.65ms
- **Resilience**: If Transformer(opt) unavailable/fails, GRU is solid backup

## Model Card Version
- **Version**: 1.0
- **Created**: 2025-10-24
- **Last Updated**: 2025-10-24
- **Contact**: LNSP Development Team
