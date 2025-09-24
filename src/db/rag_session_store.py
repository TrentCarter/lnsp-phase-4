"""PostgreSQL writer for GraphRAG session telemetry."""

from __future__ import annotations

import logging
import os
from typing import Iterable, Optional

try:  # Optional dependency
    import psycopg2
    import psycopg2.extras
except ImportError:  # pragma: no cover
    psycopg2 = None  # type: ignore

from .db_postgres import PG_DSN

logger = logging.getLogger(__name__)


class RAGSessionStore:
    """Persist GraphRAG sessions, chunks, and graph edges to PostgreSQL."""

    def __init__(self, enabled: bool = True, dsn: Optional[str] = None) -> None:
        self.enabled = enabled and psycopg2 is not None
        self._dsn = dsn or os.getenv("PG_DSN", PG_DSN)
        self._conn = None

        if not self.enabled:
            if enabled:
                logger.warning("psycopg2 not available; rag session store running in stub mode")
            return

        try:
            self._conn = psycopg2.connect(self._dsn)
            self._conn.autocommit = True
            logger.info("Connected to PostgreSQL for GraphRAG session logging")
        except Exception as exc:  # pragma: no cover - runtime guard
            logger.error("Failed to connect to PostgreSQL: %s", exc)
            self.enabled = False
            self._conn = None

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            logger.info("Closed PostgreSQL session connection")

    # ------------------------------------------------------------------
    # Insert helpers
    # ------------------------------------------------------------------
    def insert_session(
        self,
        *,
        session_id: str,
        query: str,
        lane: str,
        model: str,
        provider: str,
        usage_prompt: int,
        usage_completion: int,
        latency_ms: int,
        answer: str,
        hit_k: Optional[int],
        faiss_top_ids: Iterable[int],
        graph_node_ct: Optional[int],
        graph_edge_ct: Optional[int],
        doc_ids: Iterable[str],
    ) -> None:
        if not self.enabled or self._conn is None:
            logger.debug("[stub] GraphRAG session %s", session_id)
            return

        sql = """
        INSERT INTO rag_sessions (
            id, query, lane, model, provider,
            usage_prompt, usage_completion, latency_ms,
            answer, hit_k, faiss_top_ids, graph_node_ct, graph_edge_ct, doc_ids
        ) VALUES (
            %(id)s, %(query)s, %(lane)s, %(model)s, %(provider)s,
            %(usage_prompt)s, %(usage_completion)s, %(latency_ms)s,
            %(answer)s, %(hit_k)s, %(faiss_top_ids)s, %(graph_node_ct)s, %(graph_edge_ct)s, %(doc_ids)s
        )
        ON CONFLICT (id) DO UPDATE SET
            query = EXCLUDED.query,
            lane = EXCLUDED.lane,
            model = EXCLUDED.model,
            provider = EXCLUDED.provider,
            usage_prompt = EXCLUDED.usage_prompt,
            usage_completion = EXCLUDED.usage_completion,
            latency_ms = EXCLUDED.latency_ms,
            answer = EXCLUDED.answer,
            hit_k = EXCLUDED.hit_k,
            faiss_top_ids = EXCLUDED.faiss_top_ids,
            graph_node_ct = EXCLUDED.graph_node_ct,
            graph_edge_ct = EXCLUDED.graph_edge_ct,
            doc_ids = EXCLUDED.doc_ids;
        """

        payload = {
            "id": session_id,
            "query": query,
            "lane": lane,
            "model": model,
            "provider": provider,
            "usage_prompt": usage_prompt,
            "usage_completion": usage_completion,
            "latency_ms": latency_ms,
            "answer": answer,
            "hit_k": hit_k,
            "faiss_top_ids": list(faiss_top_ids),
            "graph_node_ct": graph_node_ct,
            "graph_edge_ct": graph_edge_ct,
            "doc_ids": list(doc_ids),
        }

        with self._conn.cursor() as cur:
            cur.execute(sql, payload)

    def insert_context_chunk(
        self,
        *,
        session_id: str,
        rank: int,
        doc_id: Optional[str],
        score: Optional[float],
        text: Optional[str],
    ) -> None:
        if not self.enabled or self._conn is None:
            logger.debug("[stub] GraphRAG chunk %s r=%s", session_id, rank)
            return

        sql = """
        INSERT INTO rag_context_chunks (session_id, rank, doc_id, score, text)
        VALUES (%(session_id)s, %(rank)s, %(doc_id)s, %(score)s, %(text)s)
        ON CONFLICT (session_id, rank) DO UPDATE SET
            doc_id = EXCLUDED.doc_id,
            score = EXCLUDED.score,
            text = EXCLUDED.text;
        """

        payload = {
            "session_id": session_id,
            "rank": rank,
            "doc_id": doc_id,
            "score": score,
            "text": text,
        }

        with self._conn.cursor() as cur:
            cur.execute(sql, payload)

    def insert_graph_edge(
        self,
        *,
        session_id: str,
        src: str,
        rel: str,
        dst: str,
        weight: Optional[float],
        doc_id: Optional[str],
    ) -> None:
        if not self.enabled or self._conn is None:
            logger.debug("[stub] GraphRAG edge %s %s-%s-%s", session_id, src, rel, dst)
            return

        sql = """
        INSERT INTO rag_graph_edges_used (session_id, src, rel, dst, weight, doc_id)
        VALUES (%(session_id)s, %(src)s, %(rel)s, %(dst)s, %(weight)s, %(doc_id)s);
        """

        payload = {
            "session_id": session_id,
            "src": src,
            "rel": rel,
            "dst": dst,
            "weight": weight,
            "doc_id": doc_id,
        }

        with self._conn.cursor() as cur:
            cur.execute(sql, payload)


__all__ = ["RAGSessionStore"]
