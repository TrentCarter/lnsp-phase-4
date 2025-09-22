from __future__ import annotations
from typing import Optional
import os
import psycopg2
import psycopg2.extras
import numpy as np


PG_DSN = os.getenv("PG_DSN", "host=localhost port=5432 dbname=lnsp user=lnsp password=lnsp")


def connect():
    conn = psycopg2.connect(PG_DSN)
    conn.autocommit = True
    return conn


def insert_cpe_entry(conn, core):
    """Insert one CPECore row into cpe_entry. `core`  is a dict or dataclass with fields matching schema."""
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
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
    cur.execute(sql, core if isinstance(core, dict) else core.__dict__)
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

    has_vector_ext = False
    cur.execute("SELECT COUNT(*) FROM pg_extension WHERE extname = 'vector';")
    if cur.fetchone()[0] > 0:
        has_vector_ext = True

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
