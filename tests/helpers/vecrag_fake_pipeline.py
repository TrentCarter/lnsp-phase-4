"""Synthetic VecRAG pipeline primitives for deterministic testing.

The real system relies on Postgres, FAISS, and Neo4j. These helpers provide a
repeatable, in-memory stand-in so the regression tests can exercise the vecRAG
flow (query → FAISS shortlist → graph expansion) without external services.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

import numpy as np

try:  # Optional dependency in CI
    import faiss  # type: ignore
except Exception:  # pragma: no cover - fall back to pure NumPy search
    faiss = None  # type: ignore

_DIM = 784  # Fused concept + TMD dimension used in production

# ---------------------------------------------------------------------------
# Synthetic dataset specification
# ---------------------------------------------------------------------------
_CLUSTER_DEFS: Dict[str, Dict[str, Any]] = {
    "alignment": {
        "lane": 12345,
        "center_seed": 31337,
        "concepts": [
            {"cpe_id": "CPE:BLAST", "concept_text": "BLAST sequence alignment tool", "representative": True},
            {"cpe_id": "CPE:SMITH", "concept_text": "Smith-Waterman local alignment", "noise_scale": 0.01},
            {"cpe_id": "CPE:NEEDLEMAN", "concept_text": "Needleman-Wunsch dynamic programming", "noise_scale": 0.01},
            {"cpe_id": "CPE:FASTA", "concept_text": "FASTA pairwise alignments"},
            {"cpe_id": "CPE:MAFFT", "concept_text": "MAFFT multiple alignment"},
            {"cpe_id": "CPE:CLUSTAL", "concept_text": "Clustal Omega aligner"},
            {"cpe_id": "CPE:HMMER", "concept_text": "HMMER profile HMM search"},
            {"cpe_id": "CPE:BOWTIE", "concept_text": "Bowtie short-read mapper"},
            {"cpe_id": "CPE:BWA", "concept_text": "BWA MEM aligner"},
            {"cpe_id": "CPE:DPALIGN", "concept_text": "Dynamic programming alignment primer"},
            {"cpe_id": "CPE:KMERALIGN", "concept_text": "k-mer seed alignment"},
            {"cpe_id": "CPE:GAPPEDALIGN", "concept_text": "Gapped alignment refinement"},
        ],
    },
    "bioinfo": {
        "lane": 12345,
        "center_seed": 42424,
        "concepts": [
            {"cpe_id": "CPE:BIOSUITE", "concept_text": "Bioinformatics software toolkit", "representative": True},
            {"cpe_id": "CPE:VARCALL", "concept_text": "Variant calling workflow"},
            {"cpe_id": "CPE:GENEONTO", "concept_text": "Gene ontology annotation"},
            {"cpe_id": "CPE:PATHWAY", "concept_text": "Pathway enrichment analyzer"},
            {"cpe_id": "CPE:METAGEN", "concept_text": "Metagenomics pipeline"},
            {"cpe_id": "CPE:VISUAL", "concept_text": "Omics visualization suite"},
            {"cpe_id": "CPE:QCPIPE", "concept_text": "Sequencing QC automation"},
            {"cpe_id": "CPE:WORKFLOW", "concept_text": "Workflow orchestration engine"},
            {"cpe_id": "CPE:PIPEKIT", "concept_text": "Pipeline kit prebuilt tasks"},
            {"cpe_id": "CPE:NOTEBOOK", "concept_text": "Analysis notebook templates"},
            {"cpe_id": "CPE:REPORTGEN", "concept_text": "Auto-report generation"},
            {"cpe_id": "CPE:SHADOW", "concept_text": "Shadow bioinformatics mirror", "lane_override": 67890, "noise_scale": 0.001},
        ],
    },
}

_GRAPH_EXPANSION_DATA: Dict[str, List[Dict[str, Any]]] = {
    "alignment": [
        {"target": "CPE:SMITH", "baseline_hops": 6, "shortcut_hops": 2},
        {"target": "CPE:NEEDLEMAN", "baseline_hops": 6, "shortcut_hops": 2},
        {"target": "CPE:MAFFT", "baseline_hops": 7, "shortcut_hops": 3},
        {"target": "CPE:CLUSTAL", "baseline_hops": 6, "shortcut_hops": 2},
        {"target": "CPE:HMMER", "baseline_hops": 5, "shortcut_hops": 2},
        {"target": "CPE:BOWTIE", "baseline_hops": 7, "shortcut_hops": 3},
    ],
    "bioinfo": [
        {"target": "CPE:VARCALL", "baseline_hops": 6, "shortcut_hops": 2},
        {"target": "CPE:GENEONTO", "baseline_hops": 5, "shortcut_hops": 2},
        {"target": "CPE:WORKFLOW", "baseline_hops": 7, "shortcut_hops": 3},
        {"target": "CPE:PIPEKIT", "baseline_hops": 6, "shortcut_hops": 2},
        {"target": "CPE:REPORTGEN", "baseline_hops": 5, "shortcut_hops": 2},
        {"target": "CPE:QCPIPE", "baseline_hops": 6, "shortcut_hops": 2},
    ],
}

_QUERY_TO_CLUSTER: Dict[str, str] = {
    "BLAST is analysis software": "alignment",
    "sequence alignment algorithm": "alignment",
    "bioinformatics software tool": "bioinfo",
}

_CONCEPTS: List[Dict[str, Any]] = []
_CONCEPT_BY_ID: Dict[str, Dict[str, Any]] = {}
_CLUSTER_QUERY_VECTORS: Dict[str, np.ndarray] = {}
_VECTORS: np.ndarray
_INDEX: Optional[Any] = None


def _unit(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    if norm == 0.0:
        return np.zeros_like(vec)
    return vec / norm


def _generate_vector(center: np.ndarray, *, seed: int, noise_scale: float) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if noise_scale == 0.0:
        noise = np.zeros_like(center)
    else:
        noise = rng.normal(size=center.shape).astype(np.float32)
    return _unit(center + noise_scale * noise)


def _build_concepts() -> None:
    global _CONCEPTS, _CONCEPT_BY_ID, _CLUSTER_QUERY_VECTORS, _VECTORS, _INDEX

    concepts: List[Dict[str, Any]] = []
    cluster_vectors: Dict[str, np.ndarray] = {}

    for cluster_name, cfg in _CLUSTER_DEFS.items():
        lane = int(cfg["lane"])
        center_seed = int(cfg["center_seed"])
        center_rng = np.random.default_rng(center_seed)
        center_vec = _unit(center_rng.normal(size=_DIM).astype(np.float32))
        cluster_vectors[cluster_name] = center_vec

        for offset, concept in enumerate(cfg["concepts"]):
            noise_scale = float(concept.get("noise_scale", 0.05))
            if concept.get("representative"):
                vec = center_vec.copy()
            else:
                vec = _generate_vector(center_vec, seed=center_seed + 100 + offset, noise_scale=noise_scale)

            lane_override = concept.get("lane_override")
            record = {
                "cpe_id": concept["cpe_id"],
                "concept_text": concept["concept_text"],
                "tmd_lane": int(lane_override) if lane_override is not None else lane,
                "cluster": cluster_name,
                "vector": vec.astype(np.float32),
                "is_representative": bool(concept.get("representative", False)),
            }
            concepts.append(record)

    _CONCEPTS = concepts
    _CONCEPT_BY_ID = {item["cpe_id"]: item for item in concepts}
    _CLUSTER_QUERY_VECTORS = {name: vec.astype(np.float32) for name, vec in cluster_vectors.items()}
    _VECTORS = np.vstack([item["vector"] for item in concepts]).astype(np.float32)

    if faiss is not None:
        index = faiss.IndexFlatIP(_DIM)
        index.add(_VECTORS)
        _INDEX = index
    else:
        _INDEX = None


_build_concepts()


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def vecrag_search(query: str, top_k: int = 10, tmd_lane: Optional[int] = None) -> List[Dict[str, Any]]:
    """Return top-k synthetic retrieval results for a query string."""
    cluster = _QUERY_TO_CLUSTER.get(query)
    if cluster is None:
        return []

    query_vec = _CLUSTER_QUERY_VECTORS[cluster].reshape(1, -1)

    if _INDEX is not None:
        distances, indices = _INDEX.search(query_vec, len(_CONCEPTS))
        ordered: Iterable[tuple[int, float]] = zip(indices[0].tolist(), distances[0].tolist())
    else:
        scores = (_VECTORS @ query_vec.T).reshape(-1)
        ranked = np.argsort(scores)[::-1]
        ordered = [(int(idx), float(scores[idx])) for idx in ranked]

    results: List[Dict[str, Any]] = []
    for rank, (idx, similarity) in enumerate(ordered, start=1):
        concept = _CONCEPTS[idx]
        if tmd_lane is not None and concept["tmd_lane"] != tmd_lane:
            continue
        payload = {
            "cpe_id": concept["cpe_id"],
            "concept_text": concept["concept_text"],
            "similarity": float(similarity),
            "tmd_lane": concept["tmd_lane"],
            "rank": rank,
        }
        results.append(payload)
        if len(results) >= top_k:
            break

    return results


def neo4j_expand(
    seed_concepts: Iterable[Dict[str, Any]],
    *,
    max_hops: int = 3,
    use_shortcuts: bool = False,
) -> List[Dict[str, Any]]:
    """Return synthetic graph expansions with or without shortcuts."""
    expansions: List[Dict[str, Any]] = []

    for seed in seed_concepts:
        seed_record = _CONCEPT_BY_ID.get(seed.get("cpe_id", ""))
        if seed_record is None:
            continue
        cluster = seed_record["cluster"]
        lane = seed_record["tmd_lane"]
        entries = _GRAPH_EXPANSION_DATA.get(cluster, [])

        count = 0
        for entry in entries:
            target = _CONCEPT_BY_ID.get(entry["target"])
            if target is None or target["tmd_lane"] != lane:
                continue

            hops = int(entry["shortcut_hops"] if use_shortcuts else entry["baseline_hops"])
            expansions.append(
                {
                    "seed_cpe_id": seed_record["cpe_id"],
                    "cpe_id": target["cpe_id"],
                    "concept_text": target["concept_text"],
                    "tmd_lane": target["tmd_lane"],
                    "hops": hops,
                    "hops_from_seed": hops,
                }
            )
            count += 1
            if count >= max_hops:
                break

    return expansions


def sample_queries(n: int) -> List[str]:
    """Provide a deterministic list of synthetic queries for benchmarking."""
    base = list(_QUERY_TO_CLUSTER.keys())
    if not base:
        return []
    return [base[i % len(base)] for i in range(n)]


__all__ = ["vecrag_search", "neo4j_expand", "sample_queries"]
