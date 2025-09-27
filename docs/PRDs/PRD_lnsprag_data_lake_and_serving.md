# LNSPRAG Data Lake & Serving PRD (v0.1) — 2025‑09‑26

> PRD for the permanent CPESH data lake and VecRAG/GraphRAG serving stack used by: (1) classic RAG, (2) training a concept‑vector LVM (e.g., Mamba), (3) vector→text decoding, and (4) RL‑VecRAG continual learning.

---

## 1) Objectives & Non‑Goals

### Objectives

* Persist **all CPESH** (Concept, Probe, Expected + soft/hard negatives) with full **TMD** routing and timestamps as **permanent training data**.
* Provide **tiered storage** that scales from 10M → 10B entries without deletion, with fast read paths for serving and training.
* Expose **serving APIs** (VecRAG + GraphRAG) with lane‑aware routing, CPESH gating, and dynamic Faiss nlist.
* Offer **operator tooling**: warm rotation, status/telemetry, pruning by policy (never delete; only tier/curate), snapshots.

### Non‑Goals

* No cloud dependency; local‑first. No auto‑deletion of CPESH. No managed vector DB requirement (Faiss primary; Postgres/Neo4j optional mirrors).

---

## 2) Terminology

* **CPESH**: Concept‑Probe‑Expected with **S**oft/**H**ard negatives.
* **TMD**: Task‑Modifier‑Domain → `tmd_bits:uint16`, `lane_index:uint15`.
* **VecRAG / LNSPRAG**: Vector‑first RAG with lane routing + optional CPESH gating.
* **Active / Warm / Cold** tiers: hot JSONL, rotated Parquet segments, and long‑term Parquet lake respectively.

---

## 3) Primary Use Cases

1. **Classic RAG**: Query → lane prefilter → Faiss ANN → (optional) CPESH‑boosted rerank → hydrate text.
2. **LVM Training (vector‑only)**: Sample across tiers for diverse CPESH training triples (concept/probe/expected + negatives) with TMD conditioning.
3. **Vector→Text Decoder**: Use CPESH as supervised pairs to decode latent vectors back to text (concept / expected).
4. **RL‑VecRAG**: Online reinforcement loop that updates the LVM using retrieval outcomes and echo metrics.

---

## 4) Data Model (Authoritative)

### 4.1 CPESH Core Record (logical)

```jsonc
{
  "cpe_id": "uuid",
  "concept_text": "string",
  "probe_question": "string",
  "expected_answer": "string",
  "soft_negative": "string|null",
  "hard_negative": "string|null",
  "mission_text": "string",
  "source_chunk": "string",
  "content_type": "enum",
  "dataset_source": "string",
  "chunk_position": {"doc_id":"str","start":int,"end":int},
  "relations_text": [{"subj":"","pred":"","obj":""}],
  "tmd_bits": "uint16",
  "tmd_lane": "str",
  "lane_index": "uint15",
  "echo_score": 0.0,
  "validation_status": "pending|passed|failed",
  "created_at": "ISO8601",
  "last_accessed": "ISO8601",
  "access_count": 0
}
```

### 4.2 Vector Attachments (lean vs full)

* **Lean** (recommended): `fused_vec(784D)`, `question_vec(768D)`, `fused_norm`.
* **Full** (optional): `concept_vec(768D)`, `tmd_dense(16D)` retained for analysis/debug & research.

### 4.3 Graph Attachments

* Nodes keyed by `cpe_id`. Edges carry `type`, `confidence`, optional provenance. Lane fields mirrored for fast subgraph pulls.

---

## 5) Tiered Storage Design

### 5.1 Layout

```
artifacts/
  cpesh_active.jsonl                 # hot append (< ~1M lines or 100MB)
  cpesh_segments/                    # warm/cold segments
    seg_YYYYMMDD_HHMMSS.parquet      # ZSTD Parquet, columnar
  cpesh_manifest.jsonl               # segment metadata & lineage
  cpesh_index.db                     # SQLite/DuckDB: id→segment, quick lookups
```

### 5.2 Policies

* **Rotation trigger**: Active → new Parquet segment when size/line thresholds exceed policy.
* **Retention**: Keep **all** segments. Never delete CPESH; only compress and index.
* **Indexing**: `cpesh_index.db` stores `(cpe_id, segment_path, row_offset, lane_index, created_at)`.
* **Sampling API**: balanced sampler across Active/Warm/Cold for training batches.

---

## 6) Vector Serving & Indexing

### 6.1 Faiss index

* **Dynamic nlist** with 40× training rule and scale defaults; auto‑downshift when unsafe.
* **nprobe selection** via CPESH gating: `nprobe_cpesh=8`, `nprobe_default=16` (lane overrides allowed).
* **Per‑lane sharding** (optional at scale): one shard per `lane_index` range or per segment family.

### 6.2 NPZ contract (768D mode)

* Required arrays: `vectors( N×768 )`, `ids`, `doc_ids`, `concept_texts`, `tmd_dense( N×16 )`, `lane_indices`.
* CI validation: dims, shapes, non‑zero norms, key presence.

---

## 7) APIs & CLIs

### 7.1 HTTP

* `GET /health/faiss` → loaded, type, metric, nlist, nprobe, ntotal.
* `GET /metrics/gating` → `{total, used_cpesh, by_nprobe, latency_hist}`.
* `POST /search` → lane‑aware ANN with optional CPESH boost; decision logged.
* `GET /cpesh/segments` → segment/manifest listing with sizes and date ranges.

### 7.2 CLI/Make targets

* `make lnsp-status` → ASCII dashboard (index, segments, gating stats, latency slices).
* `make cpesh-rotate` → materialize Active → Parquet segment, update manifest/index.
* `make gating-snapshot` → capture gating usage/latency into eval snapshots.

---

## 8) Observability & SLOs

* **Artifacts**: `index_meta.json`, `gating_decisions.jsonl`, `cpesh_manifest.jsonl`.
* **Dashboards**: lnsprag\_status tables for index & data lake utilization.
* **SLOs (10k dial‑plan)**: Hit\@1 ≥ 45%, Hit\@3 ≥ 55%; warm P50 ≤ 80ms, P95 ≤ 450ms at `nprobe ≤ 16`.

---

## 9) Operations Runbook

1. Ingest batch → write CPESH to Active JSONL; vectors to Faiss; graph to Neo4j.
2. Nightly: **rotate** Active → Parquet; refresh `cpesh_index.db`.
3. Re‑train IVF centroids if `ntrain < 40×nlist` or drift detected; rebuild index as needed.
4. Evaluate: gating grid, nprobe sweep, lane health; snapshot SLO metrics.
5. RL‑VecRAG: export reward signals + training batches from segments.

---

## 10) Backward Compatibility & Migration

* Existing JSONL cache is treated as **Active**; rotation produces Parquet segments and a manifest.
* Legacy entries without timestamps are patched at read time; `created_at` backfilled, `last_accessed` on hit.

---

## 11) Security & Governance

* No PII expected; add optional redaction hooks at ingest.
* Provenance and dataset source tracked per record.
* Append‑only cold lake; audit via manifest + per‑segment checksums.

---

## 12) Acceptance Criteria (DoD)

* `cpesh_rotate.py` writes Parquet + updates manifest; `lnsprag_status` shows segment table.
* `/metrics/gating` reflects decision counts; `gating_decisions.jsonl` non‑empty on queries.
* `index_meta.json` records `requested_nlist` and `max_safe_nlist` with applied `nlist`.
* NPZ schema validator rejects invalid bundles (CI).
* `make lnsp-status` renders all three sections: Index, CPESH tiers, Gating/Latency stats.

---

## 13) Open Questions

* Per‑lane vs global IVF training cadence at ≥100M?
* When to promote/query cold segments directly (duckdb over Parquet) vs preload?
* Unified row‑level quality metric beyond `echo_score` for training curriculum?

---

## 14) Appendix: Column Dictionary (selected)

* `cpe_id (UUID)`: universal key across text/vector/graph.
* `tmd_bits (uint16)`: packed Task|Modifier|Domain.
* `lane_index (int16)`: 0..32767; fast filter & shard key.
* `created_at, last_accessed (ISO8601)`: audit & recency signals.
* `access_count (int64)`: usage telemetry.
* `fused_vec (784D)`: concat of `tmd_dense(16D)` + `concept_vec(768D)` (normed).
* `relations_text (JSON[])`: raw triples as extracted.

— End PRD —
