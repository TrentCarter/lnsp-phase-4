"""
Outbox Pattern Implementation for LNSP Phase 4

Ensures eventual consistency across PostgreSQL, Neo4j, and FAISS.
See: docs/PRDs/PRD_Inference_LVM_v2_PRODUCTION.md (lines 209-294)
"""

import json
import time
import logging
from uuid import UUID, uuid4
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import psycopg2
from psycopg2.extras import RealDictCursor

logger = logging.getLogger(__name__)


@dataclass
class OutboxEvent:
    """Outbox event record."""
    id: int
    aggregate_id: UUID
    event_type: str
    payload: Dict[str, Any]
    status: str
    retry_count: int
    created_at: Any


class OutboxWriter:
    """
    Writes concepts with outbox events atomically.

    Usage:
        writer = OutboxWriter(db_conn)
        concept_id = writer.create_concept_with_outbox(
            concept_text="machine learning",
            tmd_bits=b"\\x01\\x02",
            tmd_dense=[0.1, 0.2, ...],
            vector_784d=[0.5, 0.3, ...],
            parent_hint=parent_uuid
        )
    """

    def __init__(self, db_conn):
        self.conn = db_conn

    def create_concept_with_outbox(
        self,
        concept_text: str,
        tmd_bits: bytes,
        tmd_dense: List[float],
        vector_784d: List[float],
        parent_hint: Optional[UUID] = None,
        child_hint: Optional[UUID] = None,
        concept_id: Optional[UUID] = None
    ) -> UUID:
        """
        Atomically create concept and outbox event.

        Args:
            concept_text: Human-readable concept text
            tmd_bits: Binary TMD representation
            tmd_dense: 16D TMD vector
            vector_784d: Fused 784D vector (768D GTR + 16D TMD)
            parent_hint: Optional parent concept ID from retrieval
            child_hint: Optional child concept ID from retrieval
            concept_id: Optional explicit UUID (generates if None)

        Returns:
            UUID of created concept (status='staged')
        """
        if concept_id is None:
            concept_id = uuid4()

        with self.conn.cursor() as cur:
            # Use PostgreSQL function for atomicity
            cur.execute("""
                SELECT create_concept_with_outbox(
                    %s::UUID, %s, %s, %s::JSONB, %s::JSONB, %s::UUID, %s::UUID
                )
            """, (
                str(concept_id),
                concept_text,
                tmd_bits,
                json.dumps(tmd_dense),
                json.dumps(vector_784d),
                str(parent_hint) if parent_hint else None,
                str(child_hint) if child_hint else None
            ))
            self.conn.commit()

        logger.info(f"Created concept {concept_id} with outbox event (status=staged)")
        return concept_id


class OutboxWorker:
    """
    Background worker to process outbox events.

    Polls outbox_events table, applies to Neo4j/FAISS, marks processed.
    """

    def __init__(
        self,
        db_conn,
        faiss_index,
        neo4j_driver,
        batch_size: int = 100,
        poll_interval_sec: float = 0.1
    ):
        self.conn = db_conn
        self.faiss = faiss_index
        self.neo4j = neo4j_driver
        self.batch_size = batch_size
        self.poll_interval = poll_interval_sec
        self._running = False

    def start(self):
        """Start worker loop (blocks)."""
        self._running = True
        logger.info("Outbox worker started")

        while self._running:
            try:
                processed = self._process_batch()
                if processed == 0:
                    time.sleep(self.poll_interval)
            except Exception as e:
                logger.error(f"Worker error: {e}", exc_info=True)
                time.sleep(1.0)  # Back off on error

    def stop(self):
        """Stop worker loop."""
        self._running = False
        logger.info("Outbox worker stopped")

    def _process_batch(self) -> int:
        """Process one batch of pending events."""
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            # Fetch pending events
            cur.execute("""
                SELECT id, aggregate_id, event_type, payload, status, retry_count, created_at
                FROM outbox_events
                WHERE status = 'pending'
                ORDER BY created_at ASC
                LIMIT %s
                FOR UPDATE SKIP LOCKED
            """, (self.batch_size,))

            events = [OutboxEvent(**row) for row in cur.fetchall()]

        if not events:
            return 0

        # Process each event
        for event in events:
            try:
                self._process_event(event)
            except Exception as e:
                self._mark_failed(event.id, str(e))

        return len(events)

    def _process_event(self, event: OutboxEvent):
        """Process single outbox event (idempotent)."""
        payload = event.payload

        # 1. Upsert FAISS (idempotent: same ID → replace)
        vector_784d = payload["vector_784d"]
        self.faiss.add_with_ids(
            vectors=[vector_784d],
            ids=[int(event.aggregate_id.int)]  # Convert UUID to int for FAISS
        )

        # 2. Upsert Neo4j (idempotent: MERGE by ID)
        with self.neo4j.session() as session:
            session.run("""
                MERGE (c:Concept {id: $id})
                SET c.text = $text,
                    c.tmd_bits = $tmd_bits,
                    c.status = 'ready'
            """, {
                "id": str(event.aggregate_id),
                "text": payload["text"],
                "tmd_bits": payload["tmd_bits"]
            })

            # 3. Create provisional edges (if hints provided)
            if payload.get("parent_hint"):
                session.run("""
                    MATCH (parent:Concept {id: $parent_id}), (child:Concept {id: $child_id})
                    MERGE (parent)-[r:BROADER {provisional: true, confidence: 0.5}]->(child)
                """, {
                    "parent_id": payload["parent_hint"],
                    "child_id": str(event.aggregate_id)
                })

        # 4. Mark processed
        self._mark_processed(event.id, event.aggregate_id)

        logger.debug(f"Processed outbox event {event.id} for concept {event.aggregate_id}")

    def _mark_processed(self, event_id: int, aggregate_id: UUID):
        """Mark event as processed."""
        with self.conn.cursor() as cur:
            cur.execute("SELECT mark_outbox_processed(%s, %s::UUID)", (event_id, str(aggregate_id)))
            self.conn.commit()

    def _mark_failed(self, event_id: int, error_msg: str):
        """Mark event as failed."""
        with self.conn.cursor() as cur:
            cur.execute("SELECT mark_outbox_failed(%s, %s)", (event_id, error_msg))
            self.conn.commit()
        logger.warning(f"Marked outbox event {event_id} as failed: {error_msg}")


# ============================================================================
# Monitoring utilities
# ============================================================================

def get_outbox_lag_metrics(db_conn) -> Dict[str, Any]:
    """Get current outbox lag metrics."""
    with db_conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute("SELECT * FROM outbox_lag_metrics")
        result = cur.fetchone()
        return dict(result) if result else {}


def get_failed_events_summary(db_conn) -> List[Dict[str, Any]]:
    """Get summary of failed events."""
    with db_conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute("SELECT * FROM outbox_failed_summary")
        return [dict(row) for row in cur.fetchall()]


if __name__ == "__main__":
    print("Outbox pattern implementation")
    print("See docs/PRDs/PRD_Inference_LVM_v2_PRODUCTION.md for details")
    print("\nUsage:")
    print("  1. Apply schema: psql lnsp < src/lvm/outbox_schema.sql")
    print("  2. Use OutboxWriter for writes")
    print("  3. Run OutboxWorker in background daemon")
