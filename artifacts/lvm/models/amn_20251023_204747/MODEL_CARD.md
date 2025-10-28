# Model Card: AMN (Attention Mixture Network)

## Model Information
- **Name**: Attention Mixture Network (AMN)
- **Type**: Latent Vector Model (LVM) - Tokenless Vector-Native
- **Parameters**: 1,472,768 (~1.5M)
- **Architecture**: Lightweight attention-based mixture model
- **Status**: **✅ PRODUCTION PRIMARY**

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
- **Cosine Similarity**: 0.5597 ± 0.0198
- **MSE Loss**: 0.0011
- **Latency (P50)**: 0.62 ms ⚡

### Out-of-Distribution (New Wikipedia articles 8471-8970)
- **Cosine Similarity**: **0.6375** 🏆 (BEST)
- **Delta**: +0.0778 (significantly better than training distribution!)

## Key Strengths
1. **🏆 Best OOD Generalization**: 0.6375 cosine on unseen articles
2. **⚡ Fastest Inference**: 0.62 ms per query (5x faster than Transformer)
3. **🔹 Smallest Footprint**: 1.5M params (12x smaller than Transformer)
4. **💯 Production-Ready**: Stable, efficient, excellent generalization

## Use Cases
- **Primary model** for 95% of production queries
- Real-time inference (<1ms latency requirement)
- Resource-constrained environments
- Queries where speed > marginal accuracy gains

## Escalation Criteria (Use Fallback Instead)
- Low confidence (cosine to nearest neighbors < 0.70)
- High-stakes lanes: sci-fact, math-derivation, med-guideline, policy/procedure
- Long context (>384 token equivalent)
- Margin between top predictions < 0.02

## Limitations
- **In-distribution accuracy**: 0.5597 (3.7% below GRU baseline)
- May underperform on very specific/technical domains
- Best used as fast first-pass with accuracy fallback available

## Deployment Configuration
- **Symlink**: `artifacts/lvm/production_model`
- **Load Path**: `artifacts/lvm/models/amn_20251023_204747/best_model.pt`
- **SLO Target**: P95 latency ≤ 1.2 ms
- **Fallback Rate**: ≤ 20% of queries
- **Monitoring**: Shadow eval 1-5% queries, alert if OOD delta drops > 0.03 over 24h

## Model Card Version
- **Version**: 1.0
- **Created**: 2025-10-24
- **Last Updated**: 2025-10-24
- **Contact**: LNSP Development Team
