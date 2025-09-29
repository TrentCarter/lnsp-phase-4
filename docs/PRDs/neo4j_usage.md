PRD — Neo4j for LNSP (vecRAG + GraphRAG)
0. Purpose & Scope
Goal: Specify how Neo4j is used to persist and serve the knowledge graph for LNSP: capturing CPESH-derived entities and relations with audit metadata, enabling GraphRAG traversal and hybrid retrieval alongside FAISS (VecRAG), and supporting RLvecRAG (feedback-driven updates).
In scope
Graph data model (nodes/relationships/properties).
Ingestion from CPESH outputs (Active JSONL + Parquet segments).
APIs to search/hop the graph and blend with vector retrieval.
Indexes/constraints, security, ops, scale, testing, SLOs.
Out of scope
Storing/serving dense vectors inside Neo4j (handled by FAISS/NPZ).
Long-term data lake storage (Parquet/SQLite index remain source-of-truth).
1. Primary Use Cases
GraphRAG retrieval
Expand candidate documents by traversing from CPESH-anchored concepts and relation triples (semantic neighbors, typed edges).
VecRAG → GraphRAG fusion
Start from vector hits (top-k doc_ids), map to cpe_id / concepts, then graph-expand to re-rank or broaden recall.
RLvecRAG feedback
Write positive/negative feedback as edge weights & counters; decay old evidence; promote new relations.
Audit & Explainability
Show why a result was retrieved (paths, predicates, sources, timestamps, quality scores).
2. Data Model
2.1 Node Labels
:Concept
Keys: cpe_id (string, unique), concept_text (string short),
tmd_bits (int), lane_index (int),
quality (float?), echo_score (float?),
created_at (string ISO), last_accessed (string ISO), access_count (int),
dataset_source (string), content_type (string),
doc_id (string), chunk_start (int?), chunk_end (int?)
:Source (document/article/source provenance)
Keys: source_id (string, unique), title (string?), uri (string?), checksum (string?)
:Mission (optional: campaign/run context)
Keys: mission_id (string, unique), name (string), started_at (string ISO)
:Predicate (optional lookup for canonical relation names)
Keys: name (string unique), description (string?)
Note: We do not create separate :Probe/:Answer nodes by default; those remain properties in CPESH. We keep the graph focused on concepts and their relationships.
2.2 Relationships
:RELATES_TO (typed edge with pred)
Directed edge between :Concept → :Concept
Properties:
pred (string; e.g., "is_a", "part_of", "causes", freeform allowed),
weight (float; default 1.0),
quality (float?), echo_score (float?),
insufficient_evidence (bool; default false),
created_at (string ISO), last_accessed (string ISO), access_count (int),
source_id (string?), doc_id (string?), lane_index (int),
curator_score (float?), recency_decay (float?).
:MENTIONS (concept appears in source)
(:Source)-[:MENTIONS {doc_id, created_at, ...}]->(:Concept)
:IN_MISSION
(:Concept)-[:IN_MISSION]->(:Mission) (optional if missions are used)
Relationship labels are intentionally few. The predicate is a property so we can keep the type system compact but expressive. If you prefer many rel types, we can convert pred to relationship labels later.
2.3 Required Constraints & Indexes
// Uniqueness
CREATE CONSTRAINT concept_cpe_id IF NOT EXISTS
FOR (c:Concept) REQUIRE c.cpe_id IS UNIQUE;

CREATE CONSTRAINT source_id IF NOT EXISTS
FOR (s:Source) REQUIRE s.source_id IS UNIQUE;

CREATE CONSTRAINT mission_id IF NOT EXISTS
FOR (m:Mission) REQUIRE m.mission_id IS UNIQUE;

// Fast lookups
CREATE INDEX concept_lane IF NOT EXISTS
FOR (c:Concept) ON (c.lane_index);

CREATE FULLTEXT INDEX concept_text_fts IF NOT EXISTS
FOR (c:Concept) ON EACH [c.concept_text];

// Optional: predicate lookup
CREATE CONSTRAINT pred_name IF NOT EXISTS
FOR (p:Predicate) REQUIRE p.name IS UNIQUE;
3. Ingestion & Upsert
3.1 Inputs
Active JSONL: artifacts/cpesh_active.jsonl
Parquet Segments: enumerated by artifacts/cpesh_manifest.jsonl
Each CPESH record (normalized) must include:
cpe_id, concept_text, probe_question, expected_answer
tmd_bits, lane_index
created_at, last_accessed, access_count
doc_id, dataset_source, content_type
Optional: relations_text (list of {subj, pred, obj})
3.2 Upsert Semantics
MERGE on :Concept{cpe_id}
Update non-null fields (text/meta/audit), bump access_count, update last_accessed.
For relations_text triples, normalize:
Resolve subj & obj to :Concept nodes (by cpe_id if present; else by text hash deterministic ID).
MERGE (s)-[r:RELATES_TO {pred}]->(o); set weights/quality; increment access_count.
For provenance, optionally MERGE :Source{source_id} and connect with :MENTIONS.
3.3 Cypher Upserts (reference)
// Concept upsert
MERGE (c:Concept {cpe_id: $cpe_id})
SET c.concept_text = $concept_text,
    c.tmd_bits = $tmd_bits,
    c.lane_index = $lane_index,
    c.quality = coalesce($quality, c.quality),
    c.echo_score = coalesce($echo_score, c.echo_score),
    c.created_at = coalesce(c.created_at, $created_at),
    c.last_accessed = $last_accessed,
    c.access_count = coalesce(c.access_count, 0) + 1,
    c.dataset_source = $dataset_source,
    c.content_type = $content_type,
    c.doc_id = $doc_id,
    c.chunk_start = coalesce($chunk_start, c.chunk_start),
    c.chunk_end = coalesce($chunk_end, c.chunk_end);

// Relation upsert (subject/object are cpe_ids; fallback path = text hash)
MATCH (s:Concept {cpe_id: $subj_id})
MATCH (o:Concept {cpe_id: $obj_id})
MERGE (s)-[r:RELATES_TO {pred: $pred}]->(o)
ON CREATE SET r.weight = $weight, r.created_at = $created_at, r.access_count = 1,
              r.quality = $quality, r.echo_score = $echo_score,
              r.insufficient_evidence = $insufficient_evidence,
              r.source_id = $source_id, r.doc_id = $doc_id, r.lane_index = $lane_index
ON MATCH SET  r.weight = coalesce(r.weight,1.0) + coalesce($weight,0.0),
              r.last_accessed = $last_accessed, r.access_count = r.access_count + 1;
4. Serving (API)
4.1 Feature Flag
Env: LNSP_GRAPHRAG_ENABLED=1 enables graph endpoints.
If not set → return 501 (not enabled), never crash.
4.2 Endpoints
POST /graph/search
Input: q (query text), top_k (int), optional lane, optional seed_ids (cpe_id list)
Behavior:
If seed_ids provided: start from those nodes.
Else: do full-text on :Concept.concept_text OR map from VecRAG results.
Return top-k concepts with scores and minimal neighbor context.
Output: [{cpe_id, concept_text, score, neighbors:[{cpe_id, pred, score}]}]
POST /graph/hop
Input: node_id (cpe_id), max_hops (default 2), top_k
Behavior: BFS or weighted expansion on :RELATES_TO, filtered by lane, optional predicates, return reachable nodes with path summaries.
4.3 Query Patterns (Cypher)
Full-text search (ranked):
CALL db.index.fulltext.queryNodes('concept_text_fts', $q) YIELD node, score
WHERE ($lane IS NULL OR node.lane_index = $lane)
RETURN node.cpe_id AS cpe_id, node.concept_text AS concept_text, score
ORDER BY score DESC
LIMIT $top_k;
K-hop expansion with weights & freshness:
MATCH (s:Concept {cpe_id: $seed})
CALL apoc.path.expandConfig(s, {
  relationshipFilter: 'RELATES_TO>',
  minLevel: 1, maxLevel: $max_hops, uniqueness: 'NODE_GLOBAL'
}) YIELD path
WITH relationships(path) AS rels, last(nodes(path)) AS n
WITH n, reduce(total=0.0, r IN rels |
  total + coalesce(r.weight,1.0) * (1.0 - coalesce(r.recency_decay,0.0))
) AS score
WHERE ($lane IS NULL OR n.lane_index = $lane)
RETURN n.cpe_id AS cpe_id, n.concept_text AS concept_text, score
ORDER BY score DESC LIMIT $top_k;
Personalized PageRank (optional re-rank):
Seed with seed_ids or top-k VecRAG hits to get graph-aware importance.
5. Scoring & Fusion
5.1 Relation Score (per-edge)
Default:
edge_score = w_base
           + w_quality * quality
           + w_echo    * echo_score
           - w_decay   * recency_decay
5.2 Node Score (path aggregation)
Sum or max over path edges (see Cypher above).
Optional RRF fusion with VecRAG ranks:
RRF_score(doc) = Σ_i 1/(k + rank_i)
Where ranks come from (a) vector search, (b) graph expansion.
6. Ops & Security
6.1 Envs
NEO4J_URI (e.g., bolt://localhost:7687)
NEO4J_USER, NEO4J_PASSWORD
Optional: NEO4J_DATABASE (multi-db)
LNSP_GRAPHRAG_ENABLED=1 to serve endpoints
API uses read-only credentials; ingestion uses writer creds.
6.2 Roles
Writer (ingestion jobs, rotation/index refresh): WRITE only to target DB.
Reader (API service): READ + CALL on APOC/path functions if used.
Secrets stored in local .env, not committed; production via secret manager.
6.3 Backups
Single-node local: neo4j-admin database dump neo4j --to-path backups/ daily; retain 7–14 days.
Incremental/cluster (future): enable online backups or use Neo4j Aura/cluster snapshots.
7. Performance & Scale
7.1 Local Targets (developer)
Up to 10M nodes/edges feasible with tuned page cache and selective indexing.
Keep short text in graph; do not store full documents; store doc_id and small preview only.
7.2 Production Considerations (10B upper bound roadmap)
Partition by lane_index and/or mission; multiple DBs or sharded clusters.
Pre-compute neighbor caches (e.g., top relations per concept) for hot lanes.
Use APOC path expansion judiciously; bound hops, use uniqueness ‘NODE_GLOBAL’.
For massive graphs, consider sampling or PPR on subgraphs seeded by VecRAG.
8. Observability & SLOs
8.1 Metrics
Ingestion counters: nodes merged, rels merged, per-run sums.
API: request counts, p50/p95 latency, error rates, top predicates used.
Quality: edge acceptance rate (insufficient_evidence=false), average edge weight by lane.
8.2 Alerts
Graph endpoints returning 500s > 1% over 5 minutes.
Ingestion failing to MERGE > N times per minute.
Page cache hit rate < target (e.g., 90%).
SLOs (local dev):
/graph/hop p95 ≤ 200 ms on 10k–100k node subsets, 2 hops.
9. Testing
9.1 Unit
db_neo4j.insert_relation_triple() idempotency (MERGE does not duplicate).
Constraint violations fail fast.
9.2 Integration (flagged)
With LNSP_GRAPHRAG_ENABLED=1, test /graph/search and /graph/hop happy paths and error behavior (501 when disabled).
9.3 Data Validation
Periodic job validates distribution of pred values; flags unknowns.
Spot-check: a sample of RELATES_TO edges must have non-empty source_id or doc_id.
10. Migration & Rollout
Create constraints/indexes (section 2.3).
Backfill from Active JSONL + Parquet segments (batch job reads CPESH, emits concepts + triples).
Enable writer credentials in ingestion.
Feature-flag GraphRAG in API (LNSP_GRAPHRAG_ENABLED=1), verify endpoints.
Add nightly refresh: CPESH rotation → SQLite index → graph backfill → compact/maintain indexes.
11. Risks & Mitigations
Graph bloat: exploding nodes from raw strings → Mitigation: deterministically derive IDs; avoid duplicating long text; merge on cpe_id.
Predicate drift: many freeform predicates → Mitigation: predicate whitelist w/ fallback, weekly normalization job.
Hotspot queries: unbounded traversal → Mitigation: hop limits, lane filters, predicate filters, precomputed neighbor caches.
Security leakage: writer creds in API → Mitigation: separate read-only user for serving; write-only creds in ingestion pipeline.
12. Non-Goals
Serving dense vector search from Neo4j.
Using Neo4j as the primary data lake (that remains Parquet/JSONL + SQLite index).
Long-term retention policies: graph mirrors the lake (no TTL by default).
13. Acceptance Criteria (DoD)
Schema online: constraints + indexes exist.
Upsert path works: insert_relation_triple() called by ingestion; concepts + edges visible in Neo4j Browser.
Graph endpoints:
Disabled returns 501 when feature flag off.
Enabled returns 200 with realistic results on the dev sample.
Hybrid flow: Basic fusion demo—start from VecRAG hits, expand via graph, return fused list.
Observability: Minimal counters emitted and visible in logs; /graph/* p95 under target on dev set.
14. Developer Cheatsheet
ENV
export NEO4J_URI=bolt://localhost:7687
export NEO4J_USER=neo4j
export NEO4J_PASSWORD=secret
export LNSP_GRAPHRAG_ENABLED=1
Create constraints & indexes (once)
// paste the statements from §2.3
Programmatic insert (Python)
db.insert_relation_triple(
  cpe_id="cpe:abc",
  triple={"subj":"cpe:abc","pred":"is_a","obj":"cpe:def"},
  meta={"doc_id":"fw:42","lane_index":0,"quality":0.9}
)
API checks
curl -s -X POST 'http://127.0.0.1:8094/graph/search?q=photosynthesis&top_k=5' | jq
curl -s -X POST 'http://127.0.0.1:8094/graph/hop' -d 'node_id=cpe:abc&max_hops=2&top_k=10' | jq
15. Future Enhancements
 Predicate normalization table + UI (curation).
 Precomputed PPR vectors per lane/mission for instant graph re-rank.
 Feedback loops: auto-boost edges from accepted answers; decay stale ones.
 Graph sampling for self-distillation tasks in LVM training.

16. Implementation Summary & Use Cases
Architecture Snapshot
- Neo4j stores CPESH-derived concepts and relations while FAISS maintains dense vectors; ingestion jobs keep both in sync.
- Core node labels (:Concept, :Source, :Mission, optional :Predicate) capture minimal but high-value metadata needed for traversal and auditing.
- Relationships (:RELATES_TO, :MENTIONS, :IN_MISSION) encode provenance, weights, and access counters so GraphRAG can reason over evidence strength.

Operational Flow
- Batch ingest: CPESH Active JSONL + Parquet segments power deterministic MERGE cycles that upsert concepts, sources, and relation triples.
- Realtime/feedback ingest: RLvecRAG writes updated edge weights and counters, enabling continuous learning without schema churn.
- Constraints (unique cpe_id/source_id) and indexes (lane index, concept_text full-text) guard consistency and keep hop queries performant.

Query & Retrieval Patterns
- GraphRAG: start from a concept seed, expand across predicate-filtered relations, return audited paths with doc_ids for downstream ranking.
- VecRAG fusion: take top-k FAISS hits, map to concepts, traverse in Neo4j, and re-order results via reciprocal rank fusion or custom scoring.
- Audit & explainability: expose stored provenance (source_id, timestamps, quality metrics) to show why a concept or relation was surfaced.

Testing & Reliability
- Unit coverage focuses on idempotent MERGE behavior, constraint enforcement, and deterministic subject/object resolution.
- Integration tests validate /graph/search and /graph/hop endpoints behind the LNSP_GRAPHRAG_ENABLED flag, ensuring service toggles behave.
- Observability hinges on ingestion counters, API latency/error metrics, and periodic data validation jobs flagging unknown predicates or orphaned nodes.

Primary Use Cases
- Analyst workflows needing explainable graph hops over CPESH concepts with traceable provenance.
- Retrieval augmentation that blends vector similarity with graph semantics to broaden recall while maintaining precision.
- Feedback loops where accepted/ rejected answers directly adjust graph edge weights, enabling adaptive RLvecRAG refinement.
