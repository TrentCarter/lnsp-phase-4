# PRD: TMDT — Hierarchical Domain Strategy for the **D** in TMD

**Doc Status:** Draft → Ready for implementation
**Owner:** [You / LNSP Core]
**Contributors:** [Architect], [Programmer], [Consultant]
**Last Updated:** 2025‑10‑27

---

## 1) Executive Summary

We introduce **TMDT (Task–Modifier–Domain Trees)**: a hierarchical domain system that extends the **D** in TMD from a flat 4‑bit (16‑way) label into a **variable‑depth path** (L0…L4, cap at 5). TMDT improves retrieval and generation by routing queries to the right **semantic neighborhood**, shrinking candidate sets from millions to thousands. We keep the **L0 domain** packed in the existing TMD bit layout (4b), and represent deeper levels as a compact **path embedding** used for **FiLM conditioning** of the 768D latent and for **path‑aware retrieval/reranking**.
This design provides **interpretability**, **low latency**, and **taxonomy agility** via versioning and a curation workflow.

---

## 2) Goals & Non‑Goals

### Goals

* **Better routing**: Gate queries by **top domain (L0)** and optionally L1 for tighter candidate pools.
* **Lower latency**: Per‑top‑domain **IVF sub‑indexes**; split to per‑subdomain shards as needed.
* **Explainability**: Human‑readable **slug paths** and versioned taxonomy.
* **Compatibility**: Preserve current **TMD bit‑packing** (4b Domain | 5b Task | 6b Modifier).
* **Model leverage**: Use **FiLM** conditioning from a **domain path embedding** at **input** (no output forcing).

### Non‑Goals

* Forcing the model to **predict** hierarchical domains at output time (optional telemetry head only).
* Infinite depth: we cap at **5 levels** to avoid sparsity and feature bloat.

---

## 3) Locked Scope & Decisions

* **Top‑level domains (L0=16):**
  0. Science

  1. Mathematics
  2. Technology
  3. Engineering
  4. Medicine
  5. Psychology
  6. Philosophy
  7. History
  8. Literature
  9. Art
  10. Economics
  11. Law
  12. Politics
  13. Education
  14. Environment
  15. Software
      *Note:* **Software** is the L0 home for programming; **Technology** remains broader (hardware, systems, protocols).

* **Depth:** Variable with **hard cap L0…L4 (5 levels)**. Typical depth 2–3.

* **Routing strictness:** **ε‑exploration** budget (default 10%, anneal toward 3%) to nearest neighbor domains by LCA distance.

* **Conditioning:** **FiLM** over 768D from a 32‑64D path embedding (concat retained for ablations only).

* **Output side:** **Input‑only** conditioning; optional **aux path head** ON at train time, OFF at inference.

* **Index topology:** Start **IVF per L0**; split to **per‑L1 shards** when a top domain passes **3M vectors** or P95 latency target is breached after nprobe tuning.

* **Curation:** **Staging taxonomy** → human approval → **version bump**. Weekly change budget.

---

## 4) Taxonomy Model

### 4.1 Structure

* **Nodes**: Domains (with `level ∈ {0..4}`)
* **Edges**: `PARENT_OF` (tree; DAG avoided by policy)
* **Branching**: Unconstrained (0..N children)
* **Slug path**: e.g., `software/python/collections/dicts/sorting`
* **Versioning**: `taxonomy_version` integer; artifacts pin to a version.

### 4.2 Encoding

* **L0**: 4‑bit field in TMD (unchanged).
* **Deeper levels**: Encoded via **PathKey** and **Path Embedding** (see §7).

---

## 5) Storage & Artifacts

### 5.1 Postgres (authoritative tables)

```sql
CREATE TABLE domains (
  id BIGSERIAL PRIMARY KEY,
  slug TEXT NOT NULL,            -- unique within a version per level
  name TEXT NOT NULL,
  level SMALLINT NOT NULL CHECK (level BETWEEN 0 AND 4),
  parent_id BIGINT REFERENCES domains(id),
  taxonomy_version INTEGER NOT NULL,
  meta JSONB DEFAULT '{}'::jsonb,
  active BOOLEAN DEFAULT TRUE,
  created_at TIMESTAMPTZ DEFAULT now()
);

CREATE UNIQUE INDEX ux_domains_slug_version_level
  ON domains (taxonomy_version, level, slug);

CREATE TABLE domain_paths (
  doc_id BIGINT NOT NULL,
  path_slug TEXT NOT NULL,       -- e.g., "software/python/collections/dicts/sorting"
  pathkey BIGINT NOT NULL,       -- §7.1
  taxonomy_version INTEGER NOT NULL,
  score REAL DEFAULT 1.0,        -- confidence / mapping score
  source TEXT,                   -- mapper name / rule
  updated_at TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX ix_domain_paths_pathkey ON domain_paths (pathkey);
CREATE INDEX ix_domain_paths_version ON domain_paths (taxonomy_version);
CREATE INDEX ix_domain_paths_docid ON domain_paths (doc_id);

-- optional learned / centroid embeddings per node
CREATE TABLE domain_embeddings (
  domain_id BIGINT PRIMARY KEY REFERENCES domains(id),
  emb VECTOR(64) NOT NULL,       -- learned or centroid
  taxonomy_version INTEGER NOT NULL
);
```

### 5.2 Neo4j (curation/analytics)

```
(:Domain {id, slug, level, version})-[:PARENT_OF]->(:Domain)
```

### 5.3 Vector Store (FAISS)

* **Per L0 IVF** index: `indexes/<version>/<L0>.ivf`
* Optional **per‑L1 shards**: `indexes/<version>/<L0>/<L1>.ivf`

### 5.4 Artifacts (on disk)

```
artifacts/
  taxonomy/<version>/domains.jsonl
  mappings/<version>/wikicat_to_domain.tsv
  indexes/<version>/<L0>/ivf_flat.index
```

---

## 6) Retrieval & Serving Flow

### 6.1 Gate → Retrieve → Rerank

1. **Gate**: Tiny MLP over the **768D** latent predicts **L0** (+ optional L1). Select top‑1 plus ε neighbors by LCA.
2. **Retrieve**: Query the **gated IVF**; if L1 predicted and shard exists, retrieve there first.
3. **Rerank**: **pMMR** (path‑aware MMR): cosine + diversity − λ·redundancy + **path prior** by tree distance.

### 6.2 Path‑aware MMR (pMMR)

Let query vector **q**, candidate vectors **cᵢ**, cosine sim `s(q,cᵢ)`. Diversity term uses tree distance via **LCA depth**.
Score:

```
J(cᵢ) = s(q,cᵢ) + α·PathPrior(path(q), path(cᵢ)) − λ·max_{cⱼ∈S} s(cᵢ, cⱼ)
```

Where `PathPrior = β·(2·LCAdepth / (depth(q)+depth(cᵢ)))`. Tunables: α, β, λ.

### 6.3 ε‑Exploration

Allocate 10% of the retrieval budget to nearest neighbor L0s (by LCA distance), anneal toward 3% over a session as in‑domain confidence stabilizes.

---

## 7) Path Encoding & Conditioning

### 7.1 PathKey (64‑bit mixed‑radix)

* **Layout**: 5 levels × 12 bits = 60 bits for node IDs, plus 4 bits **level mask**.
* **Properties**: constant‑time **prefix** checks, fast joins, compact storage.
* **Example**: `software (id=12) / python (17) / collections (5) / dicts (2) / sorting (9)` → pack as:
  `depth=5 (mask=11111b); ids=[12,17,5,2,9]` → 64‑bit `pathkey`.

### 7.2 H‑ADDR (Hierarchical Address Embedding)

A compact **24–48D** compositional hash of the slug path using per‑level salts (Bloom‑style). Pros: tiny, fast, debuggable. Use as a drop‑in conditioner input when table lookups are cold/stale.

### 7.3 Path Embedding & FiLM

* **Per‑node embeddings** (e.g., 16–32D).
* **Encoder**: sum of node embeddings **or** a tiny GRU over slug tokens → **32–64D** path vector.
* **FiLM**: `γ, β = MLP(path_vec)`; apply to the 768D input: `x' = γ ⊙ x + β`.
* **Ablation**: concat `[x; path_vec]` (adds 32–64 dims) for comparison.

### 7.4 Auxiliary Head (Telemetry)

Train‑time only: predict the **path** with **hierarchical label smoothing** (parents get partial credit). Disabled at inference.

---

## 8) Training

### 8.1 Objectives

* Main task loss unchanged (next‑vector / contrastive / InfoNCE, etc.).
* **Domain classifier** (L0±L1) CE loss.
* Optional **aux path loss** with hierarchical label smoothing.

### 8.2 Hierarchical Negative Curriculum

Start with **cousins** (same L0, different L1) as negatives → then **siblings** → finally **remote** domains.

```python
for epoch in range(E):
    if epoch < E1: neg_pool = cousins(q)
    elif epoch < E2: neg_pool = siblings(q)
    else: neg_pool = remote_domains(q)
    loss = info_nce(q, pos, sample(neg_pool, k)) + λ_cls*CE(l0_pred, l0) + λ_aux*aux_path_loss
```

### 8.3 Data Labeling

* Semi‑automatic mapping from sources (Wikipedia cats, ACM CCS, MeSH, etc.) to slug paths.
* Write **`domain_paths`** with `score` (confidence).
* Curators approve in **staging taxonomy**; promotion bumps `taxonomy_version`.

---

## 9) Evaluation & SLOs

* **Domain@1 / Domain@3** (L0 classifier accuracy)
* **Within‑domain Contain@50** (containment of ground‑truth hits in gated pool)
* **R@5 / R@10 (within gate)**
* **Cross‑domain leakage rate** (fraction of top‑k outside intended path)
* **Latency P95** end‑to‑end (target: ≤ current budget)
* **Taxonomy drift alerts** (unusual label distribution shifts)

**Ablations**: FiLM vs Concat; H‑ADDR vs learned path table; pMMR on/off; ε‑exploration on/off.

---

## 10) Index Lifecycle & Migration

1. **Create**: Build `domains.jsonl`, `domain_embeddings`, and per‑L0 IVF under `taxonomy_version N`.
2. **Warm**: Sanity‑check containment and latency; dry‑run traffic mirroring.
3. **Promote**: Flip serving to version **N**; keep **N‑1** hot for rollback.
4. **Split**: When L0 > 3M vectors or P95 breached, split into L1 shards and update router.
5. **Rollback**: If leakage or latency regress, switch back to **N‑1**; open incident.

---

## 11) APIs & Interfaces

### 11.1 Classification (internal)

```
POST /v1/domain/classify
{ "q_vec": [768 floats] } → { "l0": int, "p": float, "l1?": int, "confidences": {l0: p, ...}, "version": N }
```

### 11.2 Search (gated retrieval)

```
POST /v1/search
{
  "q_vec": [768],
  "tmd": {"domain": L0, "task": T, "modifier": M},
  "epsilon": 0.10,
  "max_k": 100
}
→ { "hits": [...], "gate": { "l0": L0, "l1?": L1 }, "version": N }
```

### 11.3 Mapping (authoring)

```
POST /v1/taxonomy/map
{ "doc_id": 123, "path_slug": "software/python/...", "score": 0.92, "source": "wikicat" }
```

---

## 12) Security, Safety, Governance

* **Version pinning** in artifacts; reject mismatches.
* **Change budgets** for merges/splits; audit trails in Postgres & Neo4j.
* **Leakage alarms** and auto‑backoff of ε‑exploration on incident.

---

## 13) Risks & Mitigations

* **Taxonomy churn** → pin by `taxonomy_version`; keep N‑1 hot.
* **Mis‑routing** → ε‑exploration + pMMR path prior + telemetry head for triage.
* **Shard imbalance** → monitor per‑shard P95; auto‑rebalance thresholds.
* **Over‑fitting to path** → ablate FiLM strength; regularize path embeddings.

---

## 14) Rollout Plan

* **P1**: Implement schema, pathkey, and authoring endpoints; backfill L0/L1 for top corpora.
* **P2**: Train L0 classifier + FiLM conditioner; enable per‑L0 IVF routing (ε=0.10).
* **P3**: Enable **pMMR** and **H‑ADDR**; ship telemetry head (train‑time only).
* **P4**: Latency & leakage hardening; split hot L1s where needed.
* **P5**: CI gates, drift alerts, and versioned promotion playbook.

---

## 15) Acceptance Criteria

* ΔContain@50 and R@5 within gated pools **improve** over baseline by agreed pp.
* **Domain@1 ≥ 0.95** on held‑out eval.
* **P95 latency** ≤ target with ε‑exploration enabled.
* Full **rollback** verified (N ↔ N‑1) with zero data loss.

---

## 16) Appendix

### 16.1 TMD Bit Layout (existing)

```
[Domain 4b | Task 5b | Modifier 6b]  -- stored in TMD field, Domain=0..15
```

### 16.2 Example Paths

* `software/python/collections/dicts/sorting`
* `medicine/oncology/diagnosis/pathology/immuno`

### 16.3 Glossary

* **LCA**: Lowest Common Ancestor in the domain tree.
* **FiLM**: Feature‑wise Linear Modulation, scaling and shifting features by a conditioning vector.
* **H‑ADDR**: Hierarchical Address Embedding, compositional hash of a path.
* **pMMR**: Path‑aware Maximal Marginal Relevance reranking.
