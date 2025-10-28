# Release v0 — Production Retriever & Reranker
**Date:** Oct 28, 2025
**Owner:** Retrieval Platform

## Goal
Ship a stable, fast retriever + lightweight reranker that meets gates on the clean, article-disjoint eval:
- **Ship gate:** R@5 ≥ 0.30 **or** MRR ≥ 0.20
- **Truth index for eval:** FAISS FLAT IP
- **Serving index:** FAISS IVF-Flat (+ optional re-rank)

## Stack (v0)
- **Embeddings:** GTR-T5 768D (vec2text-compatible)
- **Retriever:** FAISS IVF-Flat (cosine via IP on L2-norm), per-lane nlist≈√N, nprobe=8 (tunable)
- **Reranker:** Vector-only MLP (2 layers) over features:
  - cosine(q,p), margin vs best, per-article local ctx stats, light diversity prior
- **Observability:** heartbeat JSON; metrics emitted every eval; clean split enforced by default

## Interfaces & Artifacts
- **Encoders:** `svc-vec2text-enc:8767` (Text→768D), `svc-vec2text-dec:8766` (768D→Text)
- **Indexes:**
  - truth: `artifacts/faiss/p_flat_ip.faiss`
  - serving: `artifacts/faiss/p_ivf.faiss` (+ sidecar `p_norms.npy` if needed)
- **Reranker model:** `artifacts/lvm/reranker/mlp_v0.pt`
- **Release bundle:** `artifacts/releases/retriever_v0/{model.pt,index.faiss,metrics.json,VERSION}`

## Procedure (condensed)
1. **Embed P & Q** (L2-normalize).
2. **Build FLAT IP** index for eval; **build IVF** for serving.
3. **Run eval (clean split)** → write `metrics.json`.
4. **If gate <pass> as-is** → tag and ship.
   **If shallow ordering but containment ≥82%** → train/apply reranker, re-evaluate, then ship.
5. **Publish release bundle** and model card.

## Gates & SLOs
- **Ship gate:** R@5 ≥ 0.30 **or** MRR ≥ 0.20 (clean)
- **Containment watch:** Contain@50 ≥ 0.82 preferred
- **Latency SLO (serving):** P95 ≤ 8 ms @ nprobe=8 on median lane; re-rank adds ≤ 2 ms

## Backout
- Revert to previous release bundle and FLAT truth index.

## Model Selection (v0 Primary)
**Primary:** AMN_v0
- OOD: 0.6375 (best generalization)
- Latency: 0.62 ms (fastest)
- Memory: 5.8 MB (smallest)

**Fallback:** GRU_v0
- In-Dist: 0.5920 (best accuracy)
- OOD: 0.6295 (excellent generalization)
- Latency: 2.11 ms (acceptable for batch)

See model cards in `docs/ModelCards/` for full details.

## Two-Tower Status
**Archived:** Mamba two-tower parked after Epoch 5 polish training
- Epoch 4: R@5 17.2%, Contain@50 76.8%
- Epoch 5: R@5 18.4%, Contain@50 76.6%
- Kill-switch triggered: Below minimum thresholds for v0 ship

See `artifacts/archive/twotower_mamba_2025-10-28/SUMMARY.md` for details.
