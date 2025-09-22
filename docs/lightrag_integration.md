# LightRAG Integration (Adapter-First)

**Version:** 0.1 (09/22/2025)

## Summary

We vendor the minimal LightRAG utilities required for relation normalization and hybrid retrieval into `third_party/lightrag/` and expose adapter layers in `src/integrations/lightrag/`. This keeps the pipeline free from heavyweight dependencies while matching the LightRAG data model.

## Triples Schema

All relations emitted by the adapters must conform to the following structure before writing to Neo4j:

| Field          | Type        | Description                                                  |
|----------------|-------------|--------------------------------------------------------------|
| `src_cpe_id`   | `UUID`      | Source concept (CPE) identifier                              |
| `dst_cpe_id`   | `UUID`      | Target concept (CPE) identifier                              |
| `type`         | `str`       | Relation predicate (default `related_to` unless specified)   |
| `confidence`   | `float32`   | Adapter-assigned confidence (floor = `LIGHTRAG_REL_CONF_FLOOR`) |
| `properties`   | `dict`      | Arbitrary key/value metadata (e.g., `{ "source": "lightrag" }`) |

Adapters should enrich `properties` with:

- `source`: static tag (`"lightrag"` by default)
- `text`: human-readable triple string
- `lane_index`: originating lane for downstream filtering

## Lane Prefilter Strategy

- Compute `lane_index` from TMD bits (`lane_index_from_bits`).
- Neo4j writes and LightRAG graph lookups remain lane-aware by default.
- Graph expansion is limited to neighbors where `abs(neighbor_lane - query_lane) ≤ 4` to avoid unrelated hops. This window is configurable via `LIGHTRAG_LANE_WINDOW` (default 4).
- Faiss-to-LightRAG hybrid reranking only considers candidates already in the query’s lane; cross-lane expansion is a secondary step and must respect the same window.

## Upstream Pin & Licensing

- **Source:** PyPI `lightrag-hku==1.4.8.2` (Hong Kong University) downloaded 09/23/2025.
- **Commit/SHA:** Not provided upstream; we retain the package version identifier in `THIRD_PARTY_NOTICES.md`.
- **Phase 4 Note:** Pinned to 1.4.8.2 for production stability; version >=1.3.9 required for CVE fix (critical path-traversal vulnerability in ≤1.3.8).
- **Library Usage:** Direct import, no vendoring required. Uses HYBRID mode with our embeddings + KG edges.
- **Embedding Model:** Compatible with both `bge-small-en-v1.5` (384D) and `gtr-t5-base` (768D) to match our `vectorizer.py`.
- **Store:** Maintains our FAISS IVF for dense retrieval; LightRAG builds NetworkX KG in-process for graph expansion.
- License: MIT. Full text is preserved in `THIRD_PARTY_NOTICES.md`.

## Runtime Feature Flags

| Env Var                    | Default | Purpose                                   |
|----------------------------|---------|-------------------------------------------|
| `LIGHTRAG_GRAPH`           | `0`     | Enable relation normalization              |
| `LIGHTRAG_QUERY`           | `0`     | Enable hybrid retriever reranking          |
| `LIGHTRAG_REL_CONF_FLOOR`  | `0.5`   | Minimum relation confidence                |
| `LIGHTRAG_REL_SELF`        | `0`     | Emit self-relation when none extracted     |
| `LIGHTRAG_QUERY_TOPK`      | `10`    | Override default reranker top-k            |
| `LIGHTRAG_QUERY_WEIGHT`    | `0.6`   | Blend ratio between LightRAG and Faiss     |
| `LIGHTRAG_LANE_WINDOW`     | `4`     | Max lane distance for graph expansion      |

## Testing Checklist

- `tests/integrations/test_lightrag_adapters.py` validates adapter confidence injection and reranking behavior.
- End-to-end ingest tests must assert that Neo4j receives ≥ 50 LightRAG-tagged relations on the 1k run.
- Retrieval smoke tests should confirm fallback behavior when LightRAG is disabled.

## Open Work

- Extend adapters to stream provenance metadata (LLM prompt ID, extraction timestamp).
- Layer in caching for hybrid reranking to avoid recomputing full matrix during batch queries.
- Document operational runbooks once the agent-facing APIs are wired.

