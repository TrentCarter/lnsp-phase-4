# Wikipedia Temporal Flow Analysis
**Date**: 2025-11-02T22:57:05.863460
**Samples**: 5000

## Overall Directional Bias
- **Forward** (ctx[-1] → target_next): 0.3876 ± 0.1372
- **Backward** (ctx[-1] → target_prev): 0.4572 ± 0.1769
- **Δ (Forward - Backward)**: -0.0696 ± 0.2064

⚠️ **BACKWARD BIAS CONFIRMED**: Δ = -0.0696 (backward 7.0% stronger)

## Per-Article Analysis
- **Mean Δ per article**: -0.0766
- **Std Δ per article**: 0.0263
- **Range**: [-0.1762, -0.0490]
- **% articles with backward bias**: 100.0%

## By Chunk Position
- **early (0-4)**: Δ = -0.0693 (n=88)
- **mid (5-9)**: Δ = -0.0498 (n=109)
- **late (10+)**: Δ = -0.0700 (n=4803)

## Offset Curve (cos vs offset k)
Shows cosine similarity between ctx[-1] (position 4) and earlier positions:
- **k=-4** (position 0): cos = 0.3583
- **k=-3** (position 1): cos = 0.3769
- **k=-2** (position 2): cos = 0.4047
- **k=-1** (position 3): cos = 0.4688
- **k=0** (position 4): cos = 1.0000

## Reversed Order Test
- **Original order**: Δ = -0.0634
- **Reversed order**: Δ = -0.0395
- **Improvement**: 0.0238

⚠️ Reversing chunk order does NOT help significantly.

## Worst Examples (Most Backward-Biased)
| Index | Article ID | Chunk ID | Δ | Forward | Backward |
|-------|------------|----------|---|---------|----------|
| 1025 | 5 | 1 | -0.919 | 0.062 | 0.980 |
| 4996 | 23 | 1 | -0.895 | 0.093 | 0.988 |
| 4029 | 19 | 1 | -0.865 | 0.110 | 0.975 |
| 2811 | 11 | 5 | -0.823 | 0.124 | 0.947 |
| 4186 | 19 | 158 | -0.809 | 0.186 | 0.995 |
| 2606 | 10 | 93 | -0.799 | 0.156 | 0.955 |
| 3160 | 15 | 36 | -0.792 | 0.069 | 0.861 |
| 510 | 4 | 33 | -0.780 | 0.208 | 0.988 |
| 1240 | 5 | 216 | -0.770 | 0.221 | 0.992 |
| 2807 | 11 | 1 | -0.765 | 0.171 | 0.936 |

## Conclusions
1. ❌ **Wikipedia data has STRONG backward bias** (Δ < -0.05)
2. 🔬 **Root cause**: Likely explanatory structure (later chunks reference earlier concepts)
3. 💡 **Solutions**:
   - Try reversing chunk order within articles
   - Use different data sources (scientific papers, tutorials, stories)
   - Synthetically generate forward-flowing sequences
   - Accept backward bias and train with MUCH stronger directional pressure
