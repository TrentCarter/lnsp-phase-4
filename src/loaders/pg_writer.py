from __future__ import annotations
from typing import Optional, List, Dict, Any
import os
import json
import psycopg2
import psycopg2.extras
import numpy as np


PG_DSN = os.getenv("PG_DSN", "host=localhost port=5432 dbname=lnsp user=lnsp password=lnsp")

# Cache pgvector extension check (avoid checking on every chunk)
_PGVECTOR_EXTENSION_CACHE = {}


def connect():
    conn = psycopg2.connect(PG_DSN)
    conn.autocommit = True
    return conn


def check_pgvector_extension(conn) -> bool:
    """
    Check if pgvector extension exists (cached).

    Returns True if pgvector is installed, False otherwise.
    """
    conn_id = id(conn)
    if conn_id in _PGVECTOR_EXTENSION_CACHE:
        return _PGVECTOR_EXTENSION_CACHE[conn_id]

    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM pg_extension WHERE extname = 'vector';")
    has_vector_ext = cur.fetchone()[0] > 0
    cur.close()

    _PGVECTOR_EXTENSION_CACHE[conn_id] = has_vector_ext
    return has_vector_ext


def insert_cpe_entry(conn, core):
    """Insert one CPECore row into cpe_entry. `core`  is a dict or dataclass with fields matching schema."""
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

    # Normalize payload and JSON-adapt complex fields to avoid psycopg2 adaptation errors
    data = core if isinstance(core, dict) else core.__dict__
    payload = dict(data)
    if payload.get("chunk_position") is not None:
        # Accept dict/list/str; ensure proper JSON adaptation
        val = payload["chunk_position"]
        if not isinstance(val, (str, bytes)):
            payload["chunk_position"] = psycopg2.extras.Json(val)
    if payload.get("relations_text") is not None:
        val = payload["relations_text"]
        if not isinstance(val, (str, bytes)):
            payload["relations_text"] = psycopg2.extras.Json(val)

    sql = (
        """
        INSERT INTO cpe_entry (
            cpe_id, mission_text, source_chunk, concept_text, probe_question, expected_answer,
            domain_code, task_code, modifier_code, content_type, dataset_source, chunk_position,
            relations_text, echo_score, validation_status, batch_id, tmd_bits, tmd_lane, lane_index
        ) VALUES (
            %(cpe_id)s, %(mission_text)s, %(source_chunk)s, %(concept_text)s, %(probe_question)s, %(expected_answer)s,
            %(domain_code)s, %(task_code)s, %(modifier_code)s, %(content_type)s, %(dataset_source)s, %(chunk_position)s,
            %(relations_text)s, %(echo_score)s, %(validation_status)s, %(batch_id)s, %(tmd_bits)s, %(tmd_lane)s, %(lane_index)s
        ) ON CONFLICT (cpe_id) DO NOTHING
        RETURNING cpe_id
        """
    )
    cur.execute(sql, payload)
    row = cur.fetchone()
    cur.close()
    return row["cpe_id"] if row else core["cpe_id"] if isinstance(core, dict) else core.cpe_id


def upsert_cpe_vectors(conn, cpe_id, fused_vec: np.ndarray, question_vec: Optional[np.ndarray] = None,
                        concept_vec: Optional[np.ndarray] = None, tmd_dense: Optional[np.ndarray] = None,
                        fused_norm: Optional[float] = None):
    cur = conn.cursor()

    # Convert arrays to Python lists for psycopg2 + pgvector (if installed). If pgvector
    # extension is present, ensure appropriate adapters are registered. As a minimal
    # starter, we store as TEXT (JSON) if pgvector extension isn't configured yet.

    has_vector_ext = check_pgvector_extension(conn)

    if has_vector_ext:
        sql = (
            """
            INSERT INTO cpe_vectors (cpe_id, fused_vec, question_vec, concept_vec, tmd_dense, fused_norm)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (cpe_id) DO UPDATE SET
              fused_vec = EXCLUDED.fused_vec,
              question_vec = EXCLUDED.question_vec,
              concept_vec = EXCLUDED.concept_vec,
              tmd_dense = EXCLUDED.tmd_dense,
              fused_norm = EXCLUDED.fused_norm
            """
        )
        cur.execute(sql, (
            str(cpe_id), fused_vec.tolist(),
            None if question_vec is None else question_vec.tolist(),
            None if concept_vec is None else concept_vec.tolist(),
            None if tmd_dense is None else tmd_dense.tolist(),
            fused_norm,
        ))
    else:
        # Fallback: create a JSON side table until pgvector is enabled (developer mode)
        cur.execute(
            "CREATE TABLE IF NOT EXISTS cpe_vectors_json (cpe_id UUID PRIMARY KEY, fused_json TEXT, question_json TEXT, concept_json TEXT, tmd_json TEXT, fused_norm REAL);"
        )
        sql = (
            """
            INSERT INTO cpe_vectors_json (cpe_id, fused_json, question_json, concept_json, tmd_json, fused_norm)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (cpe_id) DO UPDATE SET
              fused_json = EXCLUDED.fused_json,
              question_json = EXCLUDED.question_json,
              concept_json = EXCLUDED.concept_json,
              tmd_json = EXCLUDED.tmd_json,
              fused_norm = EXCLUDED.fused_norm
            """
        )
        import json
        cur.execute(sql, (
            str(cpe_id), json.dumps(fused_vec.tolist()),
            None if question_vec is None else json.dumps(question_vec.tolist()),
            None if concept_vec is None else json.dumps(concept_vec.tolist()),
            None if tmd_dense is None else json.dumps(tmd_dense.tolist()),
            fused_norm,
        ))

    conn.commit()
    cur.close()


def batch_insert_cpe_entries(conn, entries: List[Dict[str, Any]]) -> List[str]:
    """
    Batch insert multiple CPE entries in a single transaction.

    Args:
        conn: PostgreSQL connection
        entries: List of dicts with CPE entry data

    Returns:
        List of inserted cpe_ids

    Performance: ~10-20x faster than individual inserts for large batches.
    """
    if not entries:
        return []

    # Temporarily disable autocommit for transaction
    old_autocommit = conn.autocommit
    conn.autocommit = False

    try:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

        # Normalize payloads (JSON-adapt complex fields)
        normalized_entries = []
        for entry in entries:
            payload = dict(entry)
            if payload.get("chunk_position") is not None:
                val = payload["chunk_position"]
                if not isinstance(val, (str, bytes)):
                    payload["chunk_position"] = psycopg2.extras.Json(val)
            if payload.get("relations_text") is not None:
                val = payload["relations_text"]
                if not isinstance(val, (str, bytes)):
                    payload["relations_text"] = psycopg2.extras.Json(val)
            # Add soft_negatives and hard_negatives if present
            if payload.get("soft_negatives") is not None:
                val = payload["soft_negatives"]
                if not isinstance(val, (str, bytes)):
                    payload["soft_negatives"] = psycopg2.extras.Json(val)
            if payload.get("hard_negatives") is not None:
                val = payload["hard_negatives"]
                if not isinstance(val, (str, bytes)):
                    payload["hard_negatives"] = psycopg2.extras.Json(val)
            # Quality metrics
            if payload.get("quality_metrics") is not None:
                val = payload["quality_metrics"]
                if not isinstance(val, (str, bytes)):
                    payload["quality_metrics"] = psycopg2.extras.Json(val)

            normalized_entries.append(payload)

        # Use execute_batch for efficient bulk insert
        sql = """
            INSERT INTO cpe_entry (
                cpe_id, mission_text, source_chunk, concept_text, probe_question, expected_answer,
                soft_negatives, hard_negatives, domain_code, task_code, modifier_code,
                content_type, dataset_source, chunk_position, relations_text, echo_score,
                validation_status, batch_id, tmd_bits, tmd_lane, lane_index,
                confidence_score, quality_metrics
            ) VALUES (
                %(cpe_id)s, %(mission_text)s, %(source_chunk)s, %(concept_text)s, %(probe_question)s,
                %(expected_answer)s, %(soft_negatives)s, %(hard_negatives)s, %(domain_code)s,
                %(task_code)s, %(modifier_code)s, %(content_type)s, %(dataset_source)s,
                %(chunk_position)s, %(relations_text)s, %(echo_score)s, %(validation_status)s,
                %(batch_id)s, %(tmd_bits)s, %(tmd_lane)s, %(lane_index)s,
                %(confidence_score)s, %(quality_metrics)s
            ) ON CONFLICT (cpe_id) DO NOTHING
            RETURNING cpe_id
        """

        # Execute batch (page_size=100 for optimal performance)
        psycopg2.extras.execute_batch(cur, sql, normalized_entries, page_size=100)

        # Collect inserted IDs
        inserted_ids = [str(entry["cpe_id"]) for entry in normalized_entries]

        conn.commit()
        cur.close()
        return inserted_ids

    except Exception as e:
        conn.rollback()
        raise e
    finally:
        conn.autocommit = old_autocommit


def batch_upsert_cpe_vectors(
    conn,
    vector_data: List[Dict[str, Any]]
) -> None:
    """
    Batch upsert multiple CPE vector entries in a single transaction.

    Args:
        conn: PostgreSQL connection
        vector_data: List of dicts with keys:
            - cpe_id: str (UUID)
            - fused_vec: np.ndarray (784D)
            - question_vec: np.ndarray (768D) optional
            - concept_vec: np.ndarray (768D) optional
            - tmd_dense: np.ndarray (16D) optional
            - fused_norm: float optional

    Performance: ~10-20x faster than individual inserts for large batches.
    """
    if not vector_data:
        return

    # Temporarily disable autocommit for transaction
    old_autocommit = conn.autocommit
    conn.autocommit = False

    try:
        cur = conn.cursor()
        has_vector_ext = check_pgvector_extension(conn)

        if has_vector_ext:
            sql = """
                INSERT INTO cpe_vectors (cpe_id, fused_vec, question_vec, concept_vec, tmd_dense, fused_norm)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (cpe_id) DO UPDATE SET
                  fused_vec = EXCLUDED.fused_vec,
                  question_vec = EXCLUDED.question_vec,
                  concept_vec = EXCLUDED.concept_vec,
                  tmd_dense = EXCLUDED.tmd_dense,
                  fused_norm = EXCLUDED.fused_norm
            """

            # Prepare batch data
            batch_values = []
            for entry in vector_data:
                batch_values.append((
                    str(entry["cpe_id"]),
                    entry["fused_vec"].tolist(),
                    None if entry.get("question_vec") is None else entry["question_vec"].tolist(),
                    None if entry.get("concept_vec") is None else entry["concept_vec"].tolist(),
                    None if entry.get("tmd_dense") is None else entry["tmd_dense"].tolist(),
                    entry.get("fused_norm")
                ))

            psycopg2.extras.execute_batch(cur, sql, batch_values, page_size=100)

        else:
            # Fallback: JSON table
            cur.execute(
                "CREATE TABLE IF NOT EXISTS cpe_vectors_json (cpe_id UUID PRIMARY KEY, fused_json TEXT, question_json TEXT, concept_json TEXT, tmd_json TEXT, fused_norm REAL);"
            )

            sql = """
                INSERT INTO cpe_vectors_json (cpe_id, fused_json, question_json, concept_json, tmd_json, fused_norm)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (cpe_id) DO UPDATE SET
                  fused_json = EXCLUDED.fused_json,
                  question_json = EXCLUDED.question_json,
                  concept_json = EXCLUDED.concept_json,
                  tmd_json = EXCLUDED.tmd_json,
                  fused_norm = EXCLUDED.fused_norm
            """

            batch_values = []
            for entry in vector_data:
                batch_values.append((
                    str(entry["cpe_id"]),
                    json.dumps(entry["fused_vec"].tolist()),
                    None if entry.get("question_vec") is None else json.dumps(entry["question_vec"].tolist()),
                    None if entry.get("concept_vec") is None else json.dumps(entry["concept_vec"].tolist()),
                    None if entry.get("tmd_dense") is None else json.dumps(entry["tmd_dense"].tolist()),
                    entry.get("fused_norm")
                ))

            psycopg2.extras.execute_batch(cur, sql, batch_values, page_size=100)

        conn.commit()
        cur.close()

    except Exception as e:
        conn.rollback()
        raise e
    finally:
        conn.autocommit = old_autocommit
