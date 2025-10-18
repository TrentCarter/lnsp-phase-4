# worldRAG + LVM Architecture (v1)

**Author:** Trent Carter + ChatGPT • **Date:** 2025‑10‑16 • **Target:** LNSP / worldRAG blueprint

---

## Legend

* [Q] = Query vector (768D)
* [TMD] = 16‑bit packed + 16D dense
* [F784] = Fused vector [16D TMD ⊕ 768D Concept]
* CPE = Concept‑Probe‑Expected triple
* Cos(·,·) = cosine similarity

---

## High‑Level Architecture (ASCII)

```
                                   ┌──────────────────────────────────────────────────────┐
                                   │                      CONTROL PLANE                   │
                                   │  • Ingestion P1–P12  • Nightly compaction           │
                                   │  • Index retrain     • Lane health + SLOs           │
                                   └──────────────────────────────────────────────────────┘
                                                        ▲                    ▲
                                                        │                    │
                                           telemetry + metrics        admin APIs
                                                        │                    │
                                                        │                    │
User ──prompt──▶ Host LLM ──────────────────────────────────────────────────────────────────────────┐
                      │                                                                              │
                      │ (A) Query Understanding                                                     │
                      ▼                                                                              │
            ┌────────────────────┐                                                                   │
            │  TMD Classifier    │  → tmd_bits (uint16) + tmd_dense (16D) + lane_index              │
            └─────────┬──────────┘                                                                   │
                      │                                                                              │
                      │ (B) Vectorization                                                            │
                      ▼                                                                              │
            ┌────────────────────┐                                                                   │
            │   Encoder (768D)   │  Q = embed(query)                                                │
            │ (GTR‑T5 / Stella)  │                                                                   │
            └─────────┬──────────┘                                                                   │
                      │                                                                              │
                      │ (C) Lane Routing                                                             │
                      ▼                                                                              │
            ┌────────────────────┐            soft/hard route by lane_index                           │
            │  Lane Router       │────────────────────────────────────────────────────┐               │
            └─────────┬──────────┘                                                    │               │
                      │                                                                │               │
          ┌───────────┴────────────────────────────────────────────────────────────────▼───────────┐
          │                                       RETRIEVAL FABRIC                                  │
          │                                                                                         │
          │   ┌──────────────────────────────┐     ┌──────────────────────────────┐                 │
          │   │  Vector Index (per‑lane)    │     │      Graph DB (Neo4j)        │                 │
          │   │  Faiss / pgvector (F784)    │     │  edges: REL{type,confidence} │                 │
          │   └───────────┬─────────────────┘     └───────────┬─────────────────┘                 │
          │               │ (1) ANN top‑K by Cos(Q⊕TMD, F784)                │ (2) expand hops    │
          │               ▼                                                  ▼                    │
          │     ┌──────────────────┐                              ┌──────────────────┐            │
          │     │  K CPE_ID hits   │──────────────┬──────────────▶│  Neighbor IDs    │            │
          │     └──────────────────┘              │               └──────────────────┘            │
          │                                        │ (3) hydrate                                   │
          │                                        ▼                                               │
          │                            ┌────────────────────────────┐                              │
          │                            │  Text/Meta DB (Postgres)   │ (mission, concept, probe,   │
          │                            │  + cpe_vectors (pgvector)  │  expected, tmd_bits, etc.)  │
          │                            └────────────────────────────┘                              │
          │                                                                                         │
          └─────────────────────────────────────────────────────────────────────────────────────────┘
                                                          │
                                                          │ (4) Echo Validation + Rank Fusion
                                                          ▼
                                       ┌────────────────────────────────────────────┐
                                       │   Echo Validator (P13)                     │
                                       │   • Cos(question_vec, concept_vec) ≥ τ     │
                                       │   • Drop low‑echo, rescore by:              │
                                       │     w1·cos + w2·echo + w3·graph_degree     │
                                       └──────────────────────────┬─────────────────┘
                                                                  │ top‑K context packs (vectors + text)
                                                                  ▼
                                 ┌───────────────────────────────────────────────────────────────────┐
                                 │                LVM (Vector‑Native Reasoner)                       │
                                 │  Mamba/MoE over vectors: consumes context pack (F784 + graph pri) │
                                 │  • Compositional reasoning in latent space                        │
                                 │  • Produces answer vector(s)                                      │
                                 └─────────────────────┬─────────────────────────────────────────────┘
                                                       │ (optional decode for humans / host LLM)
                                                       ▼
                                       ┌────────────────────────────────────────────┐
                                       │  Vec2Text & Response Synthesizer          │
                                       │  • Decode vector answers to text          │
                                       │  • Host LLM finalizes style/format        │
                                       └────────────────────────────────────────────┘
```

---

## Retrieval Algorithm (precise order)

1. Classify query → (domain, task, modifier) → `tmd_bits`, `lane_index`, `tmd_dense`.
2. Encode to 768D; fuse with `tmd_dense` if doing query‑time fusion.
3. Lane‑scoped ANN search on fused 784D; nprobe tuned per lane.
4. Graph walk 1–2 hops, conf≥0.6, to enrich evidence set.
5. Hydrate CPE text + vectors; compute Echo score vs. probe/expected.
6. Rank: `score = w1·cos + w2·echo + w3·deg + w4·recency(optional)`.
7. Hand top‑K (vectors + light text) to LVM; generate latent answer; decode if needed.

---

## Storage & IDs (inter‑DB linking)

* Universal key: **CPE_ID** (UUID) across Postgres, Faiss/pgvector, Neo4j.
* Vector policy (lean): keep **fused 784D** (+ optional question_vec). Rebuild pure 768D on demand.
* Lane indices: SMALLINT 0..32767; `tmd_bits` kept as uint16 plus learned 16D `tmd_dense`.

---

## Observability (minimum viable SLOs)

* **Recall@K (lane‑scoped)**, **Echo pass‑rate τ=0.82**, **Latency p95** per step (encode, ANN, hydrate, echo, LVM),
* **Per‑lane drift** (centroid shift), **hard‑negative rate**, **graph confidence distribution**.

---

## Three Novel Upgrades (high‑impact, implementable)

### 1) Adaptive Semantic‑GPS Router (ASGR)

**Goal:** Replace hard lane gating with a learnable *multi‑lane mixture* that preserves precision but boosts recall.

* **Mechanism:** learn `π(lane|Q)` via a small MLP over (Q, tmd_dense); route to top‑m lanes (m∈{2..4}) with soft quotas; entropy regularizer to avoid collapse.
* **Benefit:** +5–12% Recall@K in cross‑domain queries; keeps strict filtering via `tmd_bits` as a prior.
* **Ops:** train on historical retrieval hits; update weekly. Fallback to hard gate if π is flat.

### 2) Echo‑Weighted Contrastive Tuning (EWCT) for the LVM

**Goal:** Continually align the LVM to *what actually retrieves well*.

* **Mechanism:** Positive pairs = (Q, CPE) that passed Echo; Hard negatives = near‑misses (high cos, low Echo). InfoNCE loss with lane‑temperature τ_lane.
* **Benefit:** Reduces hallucination and stabilizes reasoning chains; improves Echo pass‑rate 2–4 pts over 2 weeks.
* **Ops:** Nightly micro‑batches; per‑lane sampling caps to avoid popularity bias.

### 3) Vector‑Delta Patching (VDP) for Knowledge Maintenance

**Goal:** Cheap updates and compositional synthesis without re‑embedding the world.

* **Mechanism:** Store small Δ‑vectors between related CPEs and time‑versioned facts (e.g., `v_new = v_old + Δ_t`). Compose deltas at query time.
* **Benefit:** 3–6× less churn on reindex; enables “what changed since T?” queries; supports temporal reasoning without full re‑ingest.
* **Ops:** Track Δ magnitude and sparsity; prune low‑impact deltas during compaction.

---

## Integration Notes

* Keep `tmd_bits` as deterministic routing + `tmd_dense` as learnable feature; do **not** conflate.
* ANN config: IVF lists ≈ √N per lane; autotune nprobe; shard by lane before count.
* Echo τ=0.82 default; per‑lane overrides allowed; schedule re‑interrogation if fail‑rate >7%/10k.

---

## Next Actions (to ship v1)

1. Stand up Postgres + pgvector tables; create Faiss per‑lane indexes.
2. Implement ASGR (tiny MLP
