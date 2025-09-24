"""Entry point for building a LightRAG knowledge graph over FactoidWiki."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict

import numpy as np
import yaml

from .embedder_gtr import get_embedder
from .vectorstore_faiss import get_vector_store

logger = logging.getLogger(__name__)


def _load_config(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing LightRAG config at {path}")
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError("LightRAG config must be a mapping")
    return data


def _ensure_chunks(chunks_path: Path) -> int:
    if not chunks_path.exists():
        raise FileNotFoundError(f"Corpus chunks not found: {chunks_path}")

    line_count = 0
    with chunks_path.open("r", encoding="utf-8") as fh:
        for line_count, _ in enumerate(fh, start=1):
            pass

    if line_count == 0:
        raise RuntimeError(f"Corpus chunks file {chunks_path} is empty")
    logger.info("Validated corpus with %d chunks", line_count)
    return line_count


def _enforce_zero_vector_guard(meta_npz: Path, expected_dim: int) -> None:
    npz = np.load(meta_npz, allow_pickle=True)
    vectors = None
    for key in ("vectors", "fused"):
        if key in npz:
            vectors = np.asarray(npz[key], dtype=np.float32)
            break
    if vectors is None:
        raise ValueError(f"{meta_npz} missing 'vectors' or 'fused' array")
    if vectors.ndim != 2 or vectors.shape[1] != expected_dim:
        raise ValueError(
            f"Expected vectors with shape (*,{expected_dim}), got {vectors.shape}"
        )
    norms = np.linalg.norm(vectors, axis=1)
    if np.any(norms < 1e-6):
        raise RuntimeError("Zero or near-zero vectors detected in metadata NPZ")


def _load_lightrag() -> Any:
    try:
        from lightrag import LightRAG  # type: ignore
    except ImportError as exc:  # pragma: no cover - import guard
        raise RuntimeError(
            "LightRAG is not installed. Run scripts/vendor_lightrag.sh inside your virtualenv"
        ) from exc
    return LightRAG


def _build_graph_with_lightrag(
    config: Dict[str, Any],
    chunks_path: Path,
    out_nodes: Path,
    out_edges: Path,
    stats_path: Path,
    load_neo4j: bool,
) -> None:
    """Build LightRAG knowledge graph from corpus chunks."""
    LightRAG = _load_lightrag()

    # Load chunks
    chunks = []
    with chunks_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                chunk = json.loads(line)
                chunks.append(chunk)

    logger.info("Loaded %d chunks from %s", len(chunks), chunks_path)

    # Extract texts for LightRAG
    texts = []
    for chunk in chunks:
        text = chunk.get("concept", chunk.get("text", chunk.get("contents", "")))
        if text.strip():
            texts.append(text)

    logger.info("Extracted %d texts for graph construction", len(texts))

    # Get embedder and vector store
    embedder = get_embedder()

    vector_store_cfg = config.get("vector_store", {})
    index_path = vector_store_cfg.get("index_path")
    meta_npz = vector_store_cfg.get("meta_npz")
    dim = vector_store_cfg.get("dim", 768)

    vector_store = get_vector_store(str(index_path), str(meta_npz), dim)

    # Initialize LightRAG
    working_dir = str(out_nodes.parent)
    rag = LightRAG(
        working_dir=working_dir,
        llm_model_func=None,  # Not needed for graph construction
        embedding_func=embedder.embed_batch,
        vector_store=vector_store,
    )

    # Insert documents to build graph
    logger.info("Inserting documents into LightRAG...")
    for i, text in enumerate(texts):
        if (i + 1) % 100 == 0:
            logger.info("Inserted %d/%d documents", i + 1, len(texts))
        rag.insert(text)

    logger.info("Graph construction complete")

    # Extract graph data
    logger.info("Extracting graph artifacts...")

    # Get entities and relations (these methods may vary by LightRAG version)
    try:
        entities = rag.get_entities()
        relations = rag.get_relations()

        logger.info("Extracted %d entities and %d relations", len(entities), len(relations))

        # Write nodes
        with out_nodes.open("w", encoding="utf-8") as fh:
            for entity in entities:
                fh.write(json.dumps(entity) + "\n")

        # Write edges
        with out_edges.open("w", encoding="utf-8") as fh:
            for relation in relations:
                fh.write(json.dumps(relation) + "\n")

        # Calculate stats
        stats = {
            "nodes": len(entities),
            "edges": len(relations),
            "chunks_processed": len(chunks),
            "texts_processed": len(texts),
            "config": config,
        }

        # Coverage analysis
        entity_texts = set()
        for entity in entities:
            if isinstance(entity, dict):
                name = entity.get("entity_name", entity.get("name", ""))
                if name:
                    entity_texts.add(name.lower())

        # Count chunks that produced at least one entity
        chunks_with_entities = 0
        for chunk in chunks:
            chunk_text = chunk.get("concept", chunk.get("text", chunk.get("contents", ""))).lower()
            # Simple heuristic: check if any entity name appears in chunk
            for entity_text in entity_texts:
                if entity_text in chunk_text:
                    chunks_with_entities += 1
                    break

        stats["chunks_with_entities"] = chunks_with_entities
        stats["entity_coverage"] = chunks_with_entities / len(chunks) if chunks else 0

        # Write stats
        with stats_path.open("w", encoding="utf-8") as fh:
            json.dump(stats, fh, indent=2, ensure_ascii=False)

        logger.info("Graph stats: %d nodes, %d edges, %.1f%% coverage",
                   len(entities), len(relations), stats["entity_coverage"] * 100)

        # Validate gates
        if len(entities) == 0:
            raise RuntimeError("No entities extracted - check chunk quality or LightRAG configuration")
        if len(relations) == 0:
            raise RuntimeError("No relations extracted - check extraction parameters")

        coverage_threshold = 0.6  # 60% coverage required
        if stats["entity_coverage"] < coverage_threshold:
            logger.warning("Low entity coverage: %.1f%% (threshold: %.1f%%)",
                          stats["entity_coverage"] * 100, coverage_threshold * 100)

        # Optional Neo4j loading
        if load_neo4j:
            logger.info("Loading edges to Neo4j...")
            _load_edges_to_neo4j(config, relations)
            logger.info("Neo4j loading complete")

    except AttributeError as e:
        logger.error("LightRAG API methods not available: %s", e)
        logger.error("This may indicate a version mismatch or incomplete LightRAG installation")
        raise RuntimeError("LightRAG graph extraction methods unavailable") from e


def _load_edges_to_neo4j(config: Dict[str, Any], relations: list) -> None:
    """Load relations into Neo4j if configured."""
    neo4j_cfg = config.get("graph", {}).get("storage", {}).get("neo4j", {})
    uri = neo4j_cfg.get("uri")
    user = neo4j_cfg.get("user")
    password = neo4j_cfg.get("pass")

    if not all([uri, user, password]):
        logger.warning("Neo4j config incomplete, skipping load")
        return

    try:
        from neo4j import GraphDatabase

        driver = GraphDatabase.driver(uri, auth=(user, password))

        with driver.session() as session:
            # Create constraints/indexes if needed
            session.run("CREATE CONSTRAINT entity_name IF NOT EXISTS FOR (e:Entity) REQUIRE e.name IS UNIQUE")

            # Load relations
            for relation in relations:
                if isinstance(relation, dict):
                    src = relation.get("source_entity")
                    rel_type = relation.get("relation_type", "RELATED_TO")
                    dst = relation.get("target_entity")

                    if src and dst:
                        session.run("""
                            MERGE (a:Entity {name: $src})
                            MERGE (b:Entity {name: $dst})
                            MERGE (a)-[r:`%s`]->(b)
                            SET r.confidence = $confidence
                            SET r.source = 'lightrag'
                        """ % rel_type.replace("-", "_").upper(),
                        src=str(src), dst=str(dst), confidence=relation.get("confidence", 0.8))

        driver.close()
        logger.info("Loaded %d relations to Neo4j", len(relations))

    except ImportError:
        logger.warning("Neo4j driver not available, skipping load")
    except Exception as e:
        logger.error("Neo4j loading failed: %s", e)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description="Build LightRAG graph artifacts")
    parser.add_argument("--config", default="configs/lightrag.yml", help="Path to LightRAG YAML config")
    parser.add_argument("--out-nodes", required=True, help="Output path for nodes.jsonl")
    parser.add_argument("--out-edges", required=True, help="Output path for edges.jsonl")
    parser.add_argument("--stats", required=True, help="Output path for stats.json")
    parser.add_argument("--load-neo4j", action="store_true", help="Push edges into Neo4j after build")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    config_path = Path(args.config)
    out_nodes = Path(args.out_nodes)
    out_edges = Path(args.out_edges)
    stats_path = Path(args.stats)

    cfg = _load_config(config_path)

    vector_store_cfg = cfg.get("vector_store", {})
    meta_npz = Path(vector_store_cfg.get("meta_npz", ""))
    index_path = Path(vector_store_cfg.get("index_path", ""))
    dim = int(vector_store_cfg.get("dim", 768))

    if not meta_npz:
        raise ValueError("vector_store.meta_npz must be configured")
    if not index_path:
        raise ValueError("vector_store.index_path must be configured")

    _enforce_zero_vector_guard(meta_npz, dim)

    corpus_cfg = cfg.get("corpus", {})
    raw_chunks = corpus_cfg.get("chunks_path")
    if not raw_chunks:
        raise ValueError("corpus.chunks_path must be configured")
    chunks_path = Path(raw_chunks)
    _ensure_chunks(chunks_path)

    # Instantiate adapters to validate runtime prerequisites
    get_embedder()
    get_vector_store(str(index_path), str(meta_npz), dim)

    out_nodes.parent.mkdir(parents=True, exist_ok=True)
    out_edges.parent.mkdir(parents=True, exist_ok=True)
    stats_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        _build_graph_with_lightrag(cfg, chunks_path, out_nodes, out_edges, stats_path, args.load_neo4j)
    except NotImplementedError as exc:
        logger.error("Graph build halted: %s", exc)
        return 2

    logger.info("Graph build complete: %s", stats_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
