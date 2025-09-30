GWOM: GraphRAG + WikiSearch + Ontology Model (GWOM)
Requirements Document (v0.1)
Date: 2025-09-27Authors: Trent Carter + ChatGPT-5

1. Purpose
Define a hybrid framework to generate ordered concept sequences suitable for training vector-native generative models (e.g., LNSP/Mamba). The model must go beyond static concept storage by producing narrative chains of related concepts (prepositions) using three complementary methods:
1. GraphRAG Walks – leverage weighted edges from CPESH/GraphRAG to form sequential paths.
2. WikiSearch Anchoring – map CPESH concepts to Wikipedia entries, extract adjacent concepts, validate via vector similarity.
3. Ontology Traversal – use external structured ontologies (e.g., Large-Scale Concept Ontology, ESA graphs) to obtain pre-curated hierarchical orderings.

2. Objectives
* Generate concept chains of length 5–20 with clear topical coherence.
* Provide sequence-based training data for LVMs (concept → next-concept prediction).
* Balance diversity (GraphRAG, WikiSearch) with logical rigor (Ontology).
* Maintain auditability: every sequence must be traceable back to sources and rationale.

3. Functional Requirements
3.1 GraphRAG Walks
* Input: CPESH graph nodes + weighted edges (confidence ≥0.6).
* Process: weighted random walk, max depth = 10, pruning low-confidence edges.
* Output: ordered concept sequences [C1 → C2 → … → Cn].
* Configurable: min/max chain length, edge confidence threshold.
3.2 WikiSearch Anchoring
* Input: CPESH concept_text.
* Process:
    1. Map to Wikipedia page (exact or nearest).
    2. Extract related concepts (links, subsections).
    3. Validate candidates via cosine similarity (≥0.82) against CPESH vectors.
* Output: topic-aligned ordered lists of related concepts.
* Configurable: max related terms per concept; cosine threshold.
3.3 Ontology Traversal
* Input: external ontology nodes (domain-specific or general).
* Process: breadth-first or depth-first traversal within domain.
* Output: canonical sequences ordered by ontology edges.
* Configurable: traversal depth, relation types (e.g., causal, temporal).

4. Data Flow

Sources → CPESH Builder → GraphRAG/Neo4j → (GWOM Pipeline)
   ├─ GraphRAG Walks → Sequences A
   ├─ WikiSearch Anchoring → Sequences B
   └─ Ontology Traversal → Sequences C
            ▼
       Sequence Sampler
            ▼
      Training Batches (concept chains)

5. Outputs
* Ordered Sequence Dataset (JSONL):

{
  "seq_id": "uuid",
  "method": "graphrag|wikisearch|ontology",
  "concept_chain": ["C1", "C2", "C3", ...],
  "source_refs": [{"cpe_id": "...", "wiki_url": "...", "ontology_id": "..."}],
  "quality_score": 0.0–1.0
}
* Metrics:
    * Mean chain length
    * Topical coherence (avg cosine sim between neighbors)
    * Coverage across lanes/domains
    * % validated vs discarded

6. Non-Functional Requirements
* Scalability: handle 10M+ concepts across 32,768 lanes.
* Auditability: sequences must link back to source graph/wiki/ontology.
* Extensibility: pluggable modules for new knowledge sources.
* Performance: sequence generation ≥1000 chains/min on commodity GPU+CPU.

7. Risks & Mitigations
* Noise from Wiki – apply cosine validation and lane filters.
* Ontology incompleteness – fallback to GraphRAG/Wiki.
* Over-fragmented chains – enforce min coherence threshold (≥0.7).
* Scaling bottlenecks – use batch traversal, vector pre-caching.

8. Open Questions
* Should we assign different weights to each method (e.g., 50% GraphRAG, 30% Wiki, 20% Ontology)?
* How to best merge overlapping chains (deduplication, consensus ranking)?
* Do we maintain separate datasets per method or unify into one blended corpus?

9. Acceptance Criteria
* End-to-end pipeline generates ≥10k validated sequences/day.
* Each sequence traceable to sources with ≥0.82 mean cosine coherence.
* JSONL output validated against schema.
* Dashboard shows method contribution, chain stats, and quality metrics.


EXPANDED:

GWOM PRD (v0.1)
GraphRAG + WikiSearch + Ontology Model for Ordered PrepositionsDate: 2025-09-27Authors: Trent Carter + ChatGPT-5

1) Objectives & Non-Goals
Objectives
* Generate ordered concept sequences for training vector-native models (Mamba/LVM).
* Integrate three complementary sequence builders:
    1. GraphRAG weighted walks (local CPESH graph).
    2. Wikipedia anchoring + validation (external wiki).
    3. Ontology traversal (structured domain ontologies).
* Persist results in a tiered sequence lake with full provenance.
* Provide operator tooling (dashboards, snapshots) for validation and monitoring.
* Ensure outputs feed into training loops (concept→next prediction, RLVecRAG, curriculum design).
Non-Goals
* No token-based alignment layers.
* No auto-deletion of sequences (append-only lake).
* No reliance on managed external APIs; Wikipedia fetches cached or mirrored.

2) Terminology
* GWOM: GraphRAG + WikiSearch + Ontology Model.
* Concept Chain: Ordered list of related concepts, length 5–20.
* Provenance: Links to CPESH IDs, Wikipedia URLs, Ontology IDs.
* Quality Score: Composite metric (cosine coherence + edge confidence + wiki validation).

3) Primary Use Cases
* LVM Training: Provide ordered chains for sequential next-concept prediction.
* RLVecRAG: Reward shaping via chain validation (echo checks on predicted vs actual).
* Curriculum Sampling: Order training samples by chain quality to optimize learning.
* Debugging: Visualize sequences across methods to inspect semantic continuity.

4) Data Model
4.1 Sequence Record

{
  "seq_id": "uuid",
  "method": "graphrag|wikisearch|ontology",
  "concept_chain": ["C1", "C2", "C3", "C4"],
  "source_refs": [
    {"cpe_id": "uuid", "wiki_url": "url", "ontology_id": "id"}
  ],
  "quality_score": 0.0,
  "created_at": "ISO8601",
  "validation_status": "pending|passed|failed"
}
4.2 Attachments
* Vectors: [768D fused vectors per concept] stored alongside.
* Relations: adjacency list for GraphRAG sequences, wiki link context, ontology relation type.

5) Tiered Storage

artifacts/
  gwom_active.jsonl        # hot append
  gwom_segments/           # rotated Parquet segments
    seq_YYYYMMDD.parquet
  gwom_manifest.jsonl      # lineage
  gwom_index.db            # SQLite/DuckDB: seq_id → segment
* Rotation threshold: 1M lines or 100MB.
* Never delete; only compress + segment.

6) Sequence Generation Methods
6.1 GraphRAG Walks
* Input: CPESH Neo4j subgraph.
* Process: weighted random walk; depth ≤10; edge confidence ≥0.6.
* Output: chains of 5–15 nodes.
6.2 WikiSearch Anchoring
* Input: concept_text → Wikipedia.
* Extract links/subsections, filter with cosine ≥0.82.
* Output: wiki-anchored chains (topic aligned).
6.3 Ontology Traversal
* Input: ontology node.
* Traverse depth 2–4; relation types filtered (causal, temporal).
* Output: canonical curated chains.

7) APIs & CLIs
HTTP
* POST /sequence/gen?method=graphrag|wikisearch|ontology → generates chains.
* GET /sequence/manifest → list segments.
* GET /sequence/stats → counts, avg length, quality distribution.
CLI
* make gwom-status → ASCII dashboard (counts, avg length, quality).
* make gwom-rotate → rotate active → Parquet.
* make gwom-snapshot → save validation stats.

8) Observability & SLOs
* Artifacts: gwom_manifest.jsonl, gwom_index.db, quality_scores.jsonl.
* Dashboards: chain length histograms, coherence scores, method mix %.
* SLOs:
    * Mean chain length = 7–12.
    * Coherence (cosine sim avg) ≥0.78.
    * ≥80% of chains validated “passed.”

9) Runbook
* Ingest: Graph updates, Wiki dump, Ontology updates.
* Nightly rotation: append active → Parquet, refresh manifest.
* Weekly: recompute quality scores, retrain Faiss IVF centroids if vectors drift.
* Monthly: ontology refresh, wiki mirror sync.

10) Security & Governance
* No PII; redaction hooks optional.
* Wikipedia fetches cached; no outbound reliance in production.
* Full provenance stored in source_refs.

11) Acceptance Criteria (DoD)
* gwom_active.jsonl exists and rotates cleanly.
* Dashboard shows 3 sections: chain counts, method distribution, quality stats.
* /sequence/stats returns JSON with coherence ≥0.78 average.
* Parquet segments validated in CI.

12) Open Questions
* Do we need blended training batches (mixed GraphRAG/Wiki/Ontology) or per-method isolation?
* Should ontology traversal be prioritized (weighting) over noisy methods?
* Do we store negatives (anti-chains) to train discriminators?

High-Level Flow (ASCII)
                        ┌──────────────────────────────┐
                        │          Sources             │
                        │ CPESH Graph / Wikipedia /    │
                        │ Ontology (domain corpora)    │
                        └─────────────┬────────────────┘
                                      │
                                      ▼
                        ┌──────────────────────────────┐
                        │     GWOM Sequence Builder    │
                        │  (GraphRAG | Wiki | Ontology)│
                        └─────────────┬────────────────┘
                                      │ append
                                      ▼
                        ┌──────────────────────────────┐
                        │   Active Log (JSONL)         │
                        │  artifacts/gwom_active.jsonl │
                        └─────────────┬────────────────┘
                                      │ rotate threshold
                                      ▼
       ┌──────────────────────────────────────────────────────────┐
       │                     GWOM Data Lake                       │
       │  ┌────────────────────────────┐  ┌────────────────────┐  │
       │  │  seg_20250927.parquet      │  │  seg_20251001.parq │  │
       │  └───────────┬────────────────┘  └───────────┬────────┘  │
       │              │ manifest append                │           │
       │              ▼                                ▼           │
       │   gwom_manifest.jsonl + gwom_index.db (seq_id→segment)    │
       └──────────────────────────────────────────────────────────┘
                                      │
                                      ▼
                         ┌──────────────────────────────┐
                         │   Training / Serving         │
                         │  - LVM next-concept pred     │
                         │  - RLVecRAG reward shaping   │
                         │  - Curriculum sampling       │
                         └──────────────────────────────┘


Sequence Generation Methods
GraphRAG Walk


   [C1]──(0.9 causes)──►[C2]──(0.8 enables)──►[C3]
     │
   (0.7 requires)
     ▼
    [C4]


WikiSearch Anchoring

 Concept: "Photosynthesis"
    │ Wikipedia page → links/subsections
    ▼
   Extracted: [Chlorophyll] → [Light reactions] → [Oxygen release]
    │
   Cosine filter (≥0.82) vs CPESH vectors
    ▼
   Ordered wiki-anchored chain


Ontology Traversal

   Ontology Node: "Cell Division"
     ├─is_a→ "Mitosis"
     ├─is_a→ "Meiosis"
     └─part_of→ "Cell Cycle"
   Ordered traversal (depth=3) → sequence

Operator Tooling

$ make gwom-status
──────────────────────────────────────────────────────────
 GWOM STATUS DASHBOARD
──────────────────────────────────────────────────────────
 Active JSONL:       520k chains
 Parquet Segments:   12 (Sep–Oct)
 Avg Chain Length:   8.4
 Method Mix:         42% GraphRAG | 38% Wiki | 20% Ontology
 Mean Coherence:     0.81
──────────────────────────────────────────────────────────


Observability & SLOs

Metrics: 
  - Chain Length Histogram
  - Coherence Distribution
  - Method Contribution %

SLOs:
  - Mean length 7–12
  - Coherence ≥0.78
  - ≥80% passed validation


Runbook (ASCII)

 Ingest → Build Chains → Append Active JSONL
    │                     │
    │ rotate nightly      ▼
    │             ┌───────────────────────────┐
    │             │  Parquet Segments + Index │
    │             └───────────────────────────┘
    │                     │
    ▼                     ▼
 Validation         Training/Serving (LVM, RL)
   │
   ▼
 gwom-snapshot → quality_scores.jsonl


GWOM White Paper:
GWOM — GraphRAG + WikiSearch + Ontology Model for Ordered Concept SequencesDate: 2025-09-27Authors: Trent Carter + ChatGPT-5

Abstract
Traditional Retrieval-Augmented Generation (RAG) pipelines capture static concept representations but often lack the ordered sequences of concepts required to train generative latent vector models (LVMs). Without sequence order, models risk degenerating into sophisticated retrieval systems, unable to predict or generate new knowledge chains.
We propose GWOM (GraphRAG + WikiSearch + Ontology Model), a hybrid framework that constructs ordered concept chains by combining three complementary approaches: GraphRAG weighted walks, WikiSearch anchoring, and Ontology traversal. GWOM converts disconnected fact repositories into narrative concept sequences suitable for training vector-native generative models (e.g., Mamba, VMMoE) that can predict “next concepts” in a chain.

1. Introduction
Vector-native architectures (Mamba, LNSP, VMMoE) require training data that reflects not only semantic proximity but also sequential continuity. Prior work (LNSPRAG PRD, Semantic GPS) established strong foundations for vector retrieval and concept clustering, but lacked mechanisms to form ordered knowledge paths.
GWOM addresses this gap by leveraging three complementary data sources:
1. GraphRAG Walks – exploiting local CPESH graphs for weighted edge traversals.
2. WikiSearch Anchoring – using Wikipedia link structure validated against CPESH vectors.
3. Ontology Traversal – structured, hierarchical progressions through curated ontologies.
Together, these methods generate concept sequences that serve as training curriculum for generative latent models.

2. Motivation
Static concept embeddings (CPE/CPESH) provide excellent clustering but fail to provide directionality. For models to generate, they must learn temporal or causal flows between concepts. Without such flows, retrieval saturates but prediction collapses.
GWOM reframes concept storage as sequence generation:
* Instead of asking “which concept is similar?”, we ask “which concept follows next?”
* This transforms the model objective into vector-to-vector sequence prediction.

3. Architecture
3.1 High-Level Flow

Sources (CPESH Graph / Wikipedia / Ontology)
        │
        ▼
   GWOM Sequence Builder
 (GraphRAG | WikiSearch | Ontology)
        │ append
        ▼
Active Log (gwom_active.jsonl)
        │ rotate threshold
        ▼
GWOM Data Lake (Parquet Segments + Index)
        │
        ▼
Training / Serving
  - LVM next-concept prediction
  - RLVecRAG continual learning
  - Curriculum sampling

3.2 Sequence Methods
GraphRAG Walks

[C1] --(0.9 causes)--> [C2] --(0.8 enables)--> [C3]
  │
(0.7 requires)
  ▼
 [C4]
* Random walk with edge weights ≥0.6.
* Produces local narrative chains.
WikiSearch Anchoring

 Concept: "Photosynthesis"
   │  Wikipedia page → links/subsections
   ▼
   [Chlorophyll] → [Light reactions] → [Oxygen release]
   │
   Cosine filter (≥0.82 vs CPESH vectors)
* Anchors sequences in topical wiki pages.
Ontology Traversal

Ontology Node: "Cell Division"
 ├─ is_a → "Mitosis"
 ├─ is_a → "Meiosis"
 └─ part_of → "Cell Cycle"
* Uses curated relation edges (causal, temporal).

4. Data Model
Each sequence is persisted as a CPESH-linked record:

{
  "seq_id": "uuid",
  "method": "graphrag|wikisearch|ontology",
  "concept_chain": ["C1","C2","C3"],
  "source_refs": [{"cpe_id":"...","wiki_url":"...","ontology_id":"..."}],
  "quality_score": 0.0–1.0,
  "created_at": "ISO8601"
}
Vectors (768D fused) accompany each chain for training.

5. Training Applications
1. Generative LVM Training
    * Predict the next concept vector given a chain prefix.
    * Turns retrieval into generative progression.
2. RLVecRAG Feedback
    * Echo loop compares predicted next concept vs actual.
    * Reinforcement signals improve retrieval + generation.
3. Curriculum Learning
    * Order chains by quality_score.
    * Present cleaner sequences first; add noisy GraphRAG/Wiki later.

6. Observability & SLOs
* Metrics:
    * Avg chain length = 7–12
    * Mean cosine coherence ≥0.78
    * ≥80% chains validated “passed”
* Dashboards (ASCII/JSON):

GWOM STATUS
──────────────────────────────
 Active JSONL:   500k chains
 Segments:       12
 Method Mix:     42% GraphRAG
                 38% Wiki
                 20% Ontology
 Mean Coherence: 0.81
──────────────────────────────

7. Advantages
* Hybrid robustness: GraphRAG captures local nuance, Wiki anchors topicality, Ontology enforces logical structure.
* Permanent, auditable lake: no data loss; all sequences traceable.
* Extensible: new ontologies or knowledge sources can plug in.
* Efficiency: Lean storage (~6–9 KB per sequence with fused vectors).

8. Risks & Mitigations
* Wiki noise → cosine filtering + lane routing.
* Ontology incompleteness → fallback to GraphRAG/Wiki.
* Over-fragmented chains → enforce min coherence thresholds.

9. Future Work
* Chain merging: blend overlapping paths into unified narratives.
* Negative sequences: train discriminators with anti-chains.
* Cross-lane walks: test transitions across domains for analogical reasoning.

10. Conclusion
GWOM represents a step beyond static RAG and embedding databases. By converting disconnected facts into ordered, validated concept chains, it enables vector-native generative models to predict, not just retrieve. The hybrid design (GraphRAG, Wiki, Ontology) provides balance between flexibility, scale, and rigor — ensuring both coverage and coherence.
