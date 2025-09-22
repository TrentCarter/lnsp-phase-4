import os
import psycopg2
import psycopg2.extras

PG_DSN = os.getenv("PG_DSN", "host=localhost dbname=lnsp user=lnsp password=lnsp")


def connect():
    conn = psycopg2.connect(PG_DSN)
    conn.autocommit = True
    return conn


def insert_entry(conn, core: dict):
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    sql = """
    INSERT INTO cpe_entry (cpe_id, mission_text, source_chunk, concept_text,
        probe_question, expected_answer, domain_code, task_code, modifier_code,
        content_type, dataset_source, chunk_position, relations_text, echo_score,
        validation_status, batch_id, tmd_bits, tmd_lane, lane_index)
    VALUES (%(cpe_id)s, %(mission_text)s, %(source_chunk)s, %(concept_text)s,
        %(probe_question)s, %(expected_answer)s, %(domain_code)s, %(task_code)s,
        %(modifier_code)s, %(content_type)s, %(dataset_source)s, %(chunk_position)s,
        %(relations_text)s, %(echo_score)s, %(validation_status)s, %(batch_id)s,
        %(tmd_bits)s, %(tmd_lane)s, %(lane_index)s)
    ON CONFLICT (cpe_id) DO NOTHING RETURNING cpe_id
    """
    cur.execute(sql, core)
    row = cur.fetchone()
    cur.close()
    return row["cpe_id"] if row else core["cpe_id"]


def upsert_vectors(conn, cpe_id: str, fused_vec, question_vec, concept_vec):
    cur = conn.cursor()
    sql = """
    INSERT INTO cpe_vectors (cpe_id, fused_vec, question_vec, concept_vec)
    VALUES (%s, %s, %s, %s)
    ON CONFLICT (cpe_id) DO UPDATE SET
      fused_vec = EXCLUDED.fused_vec,
      question_vec = EXCLUDED.question_vec,
      concept_vec = EXCLUDED.concept_vec
    """
    cur.execute(sql, (cpe_id, fused_vec.tolist(), question_vec.tolist(), concept_vec.tolist()))
    conn.commit()
    cur.close()


class PostgresDB:
    def __init__(self, enabled: bool = True, dsn: str = PG_DSN):
        self.enabled = enabled
        self.dsn = dsn
        self.conn = None
        if self.enabled:
            try:
                self.conn = connect()
                print("[PostgresDB] Connected")
            except Exception as exc:
                print(f"[PostgresDB] Connection failed: {exc} — using stub mode")
                self.enabled = False
                self.conn = None
        else:
            print("[PostgresDB] Running in stub mode")

    def insert_cpe(self, cpe_record: dict) -> bool:
        if not self.enabled or self.conn is None:
            print(f"[PostgresDB STUB] Would insert CPE {cpe_record['cpe_id']}")
            return True
        payload = dict(cpe_record)
        payload["chunk_position"] = json.dumps(payload.get("chunk_position", {}))
        payload["relations_text"] = json.dumps(payload.get("relations_text", []))
        return bool(insert_entry(self.conn, payload))

    def close(self) -> None:
        if self.conn:
            self.conn.close()
            print("[PostgresDB] Connection closed")
