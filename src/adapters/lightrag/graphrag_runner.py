"""Execute LightRAG GraphRAG queries and persist telemetry."""

from __future__ import annotations

import argparse
import json
import logging
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml

from .embedder_gtr import get_embedder
from .vectorstore_faiss import get_vector_store
from ...db.rag_session_store import RAGSessionStore

logger = logging.getLogger(__name__)


@dataclass
class QueryItem:
    lane: str
    text: str


def _load_config(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing LightRAG config at {path}")
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError("LightRAG config must be a mapping")
    return data


def _load_queries(path: Path, default_lane: str) -> List[QueryItem]:
    if not path.exists():
        raise FileNotFoundError(f"Query file not found: {path}")
    queries: List[QueryItem] = []
    with path.open("r", encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if "\t" in line:
                lane, text = line.split("\t", 1)
            elif "|" in line:
                lane, text = line.split("|", 1)
            else:
                lane, text = default_lane, line
            queries.append(QueryItem(lane=lane.strip(), text=text.strip()))
    if not queries:
        raise RuntimeError(f"No queries found in {path}")
    return queries


def _load_lightrag() -> Any:
    try:
        from lightrag import LightRAG  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "LightRAG is not installed. Run scripts/vendor_lightrag.sh inside your virtualenv"
        ) from exc
    return LightRAG


def _ensure_runtime(cfg: Dict[str, Any]) -> Tuple[Any, Any, Any]:
    vector_cfg = cfg.get("vector_store", {})
    index_path = vector_cfg.get("index_path")
    meta_npz = vector_cfg.get("meta_npz")
    dim = int(vector_cfg.get("dim", 768))

    if not index_path or not meta_npz:
        raise ValueError("vector_store.index_path and meta_npz must be configured")

    embedder = get_embedder()
    vector_store = get_vector_store(index_path=index_path, meta_npz_path=meta_npz, dim=dim)
    LightRAG = _load_lightrag()
    return embedder, vector_store, LightRAG


def _run_query(
    LightRAG: Any,
    cfg: Dict[str, Any],
    query: QueryItem,
) -> Dict[str, Any]:
    """Execute a single GraphRAG query with full instrumentation."""
    # Get LLM function
    llm_provider = cfg.get("llm", {}).get("provider", "unknown")
    llm_model = cfg.get("llm", {}).get("model", "unknown")

    # Create LLM function (use real local Llama client)
    def llm_func(prompt: str, **kwargs) -> Dict[str, Any]:
        """Real LLM function using local Llama client."""
        from ...llm.local_llama_client import call_local_llama

        # Use system prompt if provided
        system_prompt = kwargs.get("system_prompt")
        response = call_local_llama(prompt, system_prompt)

        return {
            "text": response.text,
            "usage": {
                "prompt_tokens": response.bytes_in // 4,  # Rough token estimate
                "completion_tokens": response.bytes_out // 4,
                "total_tokens": (response.bytes_in + response.bytes_out) // 4
            },
            "latency_ms": response.latency_ms
        }

    # Get vector store config
    vector_store_cfg = cfg.get("vector_store", {})
    index_path = vector_store_cfg.get("index_path")
    meta_npz = vector_store_cfg.get("meta_npz")
    dim = vector_store_cfg.get("dim", 768)

    # Get embedder and vector store
    embedder = get_embedder()
    vector_store = get_vector_store(index_path=index_path, meta_npz_path=meta_npz, dim=dim)

    # Initialize LightRAG instance for this query
    working_dir = str(Path("artifacts/kg").absolute())
    rag = LightRAG(
        working_dir=working_dir,
        llm_model_func=llm_func,
        embedding_func=embedder.embed_batch,
        vector_store=vector_store,
    )

    # Configure retrieval parameters
    retrieval_cfg = cfg.get("retrieval", {})
    topk_vectors = retrieval_cfg.get("topk_vectors", 12)
    graph_depth = retrieval_cfg.get("graph_depth", 2)
    graph_method = retrieval_cfg.get("graph_method", "ppr")

    # Run the query
    # LightRAG's query method with mode="hybrid" should do graph-enhanced retrieval
    try:
        response = rag.query(
            query.text,
            param=rag.QueryParam(
                mode="hybrid",
                top_k=topk_vectors,
                graph_depth=graph_depth,
                graph_method=graph_method
            )
        )

        # Extract results
        result = {
            "answer": getattr(response, 'response', str(response)),
            "model": llm_model,
            "provider": llm_provider,
        }

        # Extract usage if available
        if hasattr(response, 'usage'):
            usage = response.usage
            result.update({
                "usage_prompt": getattr(usage, 'prompt_tokens', 0),
                "usage_completion": getattr(usage, 'completion_tokens', 0),
                "usage_total": getattr(usage, 'total_tokens', 0),
            })

        # Extract graph information if available
        if hasattr(response, 'graph_info'):
            graph_info = response.graph_info
            result.update({
                "graph_nodes_used": len(graph_info.get('nodes', [])),
                "graph_edges_used": len(graph_info.get('edges', [])),
                "graph_entities": [n.get('name', '') for n in graph_info.get('nodes', [])]
            })

        # Extract context information if available
        if hasattr(response, 'context'):
            context = response.context
            result["context"] = []
            result["doc_ids"] = []

            for i, chunk in enumerate(context):
                chunk_info = {
                    "rank": i + 1,
                    "doc_id": getattr(chunk, 'doc_id', ''),
                    "score": getattr(chunk, 'score', 0.0),
                    "text": getattr(chunk, 'text', ''),
                }
                result["context"].append(chunk_info)
                if chunk_info["doc_id"]:
                    result["doc_ids"].append(chunk_info["doc_id"])

        # Extract graph edges used
        if hasattr(response, 'graph_edges'):
            result["graph_edges"] = []
            for edge in response.graph_edges:
                edge_info = {
                    "src": getattr(edge, 'src', ''),
                    "rel": getattr(edge, 'rel', ''),
                    "dst": getattr(edge, 'dst', ''),
                    "weight": getattr(edge, 'weight', 0.0),
                    "doc_id": getattr(edge, 'doc_id', ''),
                }
                result["graph_edges"].append(edge_info)

        return result

    except Exception as e:
        logger.error("GraphRAG query execution failed: %s", e)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description="Run LightRAG GraphRAG evaluation")
    parser.add_argument("--config", default="configs/lightrag.yml", help="LightRAG YAML config")
    parser.add_argument("--lane", default="L1_FACTOID", help="Default lane for queries without explicit lane tag")
    parser.add_argument("--query-file", required=True, help="Text file containing queries (lane\tquery)")
    parser.add_argument("--out", required=True, help="Output JSONL path")
    parser.add_argument("--persist-postgres", action="store_true", help="Persist runs to PostgreSQL")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    config_path = Path(args.config)
    output_path = Path(args.out)
    queries = _load_queries(Path(args.query_file), args.lane)
    cfg = _load_config(config_path)

    try:
        embedder, vector_store, LightRAG = _ensure_runtime(cfg)
    except Exception as exc:
        logger.error("Runtime check failed: %s", exc)
        return 2

    store = RAGSessionStore(enabled=args.persist_postgres)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(tz=timezone.utc).isoformat()
    completed = 0

    with output_path.open("w", encoding="utf-8") as fh:
        for idx, query in enumerate(queries, start=1):
            session_id = str(uuid.uuid4())
            logger.info("[%s/%s] %s", idx, len(queries), query.text)
            started = time.perf_counter()
            try:
                result = _run_query(LightRAG, cfg, query)
                status = "ok"
            except NotImplementedError as exc:
                logger.error("Query execution blocked: %s", exc)
                result = {"error": str(exc)}
                status = "blocked"
            except Exception as exc:  # pragma: no cover - runtime guard
                logger.exception("Query failed: %s", exc)
                result = {"error": str(exc)}
                status = "error"

            latency_ms = int((time.perf_counter() - started) * 1000)
            payload = {
                "id": session_id,
                "timestamp": timestamp,
                "query": query.text,
                "lane": query.lane,
                "status": status,
                "latency_ms": latency_ms,
                "result": result,
            }
            fh.write(json.dumps(payload) + "\n")
            completed += 1

            if status == "ok" and args.persist_postgres:
                try:
                    store.insert_session(
                        session_id=session_id,
                        query=query.text,
                        lane=query.lane,
                        model=str(result.get("model", "unknown")),
                        provider=str(cfg.get("llm", {}).get("provider", "unknown")),
                        usage_prompt=int(result.get("usage_prompt", 0)),
                        usage_completion=int(result.get("usage_completion", 0)),
                        latency_ms=latency_ms,
                        answer=str(result.get("answer", "")),
                        hit_k=result.get("hit_k"),
                        faiss_top_ids=result.get("faiss_top_ids", []),
                        graph_node_ct=result.get("graph_nodes_used"),
                        graph_edge_ct=result.get("graph_edges_used"),
                        doc_ids=result.get("doc_ids", []),
                    )

                    for chunk in result.get("context", []):
                        store.insert_context_chunk(
                            session_id=session_id,
                            rank=int(chunk.get("rank", 0)),
                            doc_id=chunk.get("doc_id"),
                            score=chunk.get("score"),
                            text=chunk.get("text"),
                        )

                    for edge in result.get("graph_edges", []):
                        store.insert_graph_edge(
                            session_id=session_id,
                            src=edge.get("src", ""),
                            rel=edge.get("rel", ""),
                            dst=edge.get("dst", ""),
                            weight=edge.get("weight"),
                            doc_id=edge.get("doc_id"),
                        )
                except Exception as exc:  # pragma: no cover
                    logger.error("Failed to persist session %s: %s", session_id, exc)

    store.close()
    logger.info("Completed %d queries", completed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
