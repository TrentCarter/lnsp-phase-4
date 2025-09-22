from __future__ import annotations
from typing import Dict, Any, Optional
import os
import uuid
from datetime import datetime

try:
    from ..loaders.pg_writer import connect, insert_cpe_entry
except ImportError:
    # Fallback if pg not available
    def insert_cpe_entry(conn, core):
        print(f"[TEXT_DB STUB] Would insert CPE: {core.get('cpe_id', 'unknown')}")
    def connect():
        return None


class TextDB:
    """Text database writer for CPE entries."""

    def __init__(self):
        self.use_pg = os.getenv("USE_POSTGRES", "false").lower() == "true"

    def insert_cpe_entry(self, cpe_id: str, sample: Dict[str, Any], extraction: Dict[str, Any],
                        batch_id: Optional[str] = None) -> str:
        """Insert CPE entry into text database."""

        core_data = {
            "cpe_id": cpe_id,
            "mission_text": extraction["mission"],
            "source_chunk": sample["contents"],
            "concept_text": extraction["concept"],
            "probe_question": extraction["probe"],
            "expected_answer": extraction["expected"],
            "domain_code": extraction["domain_code"],
            "task_code": extraction["task_code"],
            "modifier_code": extraction["modifier_code"],
            "content_type": "factual",  # hardcoded for now
            "dataset_source": "factoid-wiki-large",
            "chunk_position": {
                "doc_id": sample["id"],
                "start": 0,
                "end": len(sample["contents"])
            },
            "relations_text": extraction["relations"],
            "echo_score": extraction["echo_score"],
            "validation_status": extraction["validation_status"],
            "batch_id": batch_id or str(uuid.uuid4()),
            "created_at": datetime.utcnow().isoformat(),
            "tmd_bits": extraction["tmd_bits"],
            "tmd_lane": extraction["tmd_lane"],
            "lane_index": extraction["lane_index"]
        }

        if self.use_pg:
            conn = connect()
            if conn:
                result_id = insert_cpe_entry(conn, core_data)
                conn.close()
                print(f"[TEXT_DB] Inserted CPE {cpe_id}")
                return result_id

        # Fallback: print to console
        print(f"[TEXT_DB STUB] CPE {cpe_id}: {extraction['concept'][:50]}...")
        return cpe_id
