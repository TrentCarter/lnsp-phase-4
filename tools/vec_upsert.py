#!/usr/bin/env python
"""
VecRAG Upsert (single item): creates a FULL entry everywhere:
- CPESH text + metadata to Active JSONL (artifacts/cpesh_active.jsonl)
- Optional Postgres text row (PG_DSN)
- Optional Neo4j relations (LNSP_WRITE_NEO4J=1)
- Embeddings (768D) + TMD16 -> fused 784D; append NPZ store
- Optional FAISS rebuild (if requested)
- Updates Parquet manifest/index when rotation runs (not here)

Usage examples:
  PYTHONPATH=src python -m tools.vec_upsert --text "Photosynthesis is ..." --doc-id doc_abc
  PYTHONPATH=src python -m tools.vec_upsert --tsv data/factoid.tsv --row 0
Env:
  LNSP_LLM_ENDPOINT, LNSP_LLM_MODEL, LNSP_CPESH_TIMEOUT_S
  LNSP_EMBEDDER_PATH (for local GTR-T5), HF_HUB_OFFLINE=1, TRANSFORMERS_OFFLINE=1
  PG_DSN (optional), LNSP_WRITE_NEO4J=1 (optional)
  LNSP_NPZ_PATH (default artifacts/fw10k_vectors_768.npz)
  LNSP_FAISS_REBUILD=1 triggers index rebuild via src/faiss_index.py
"""

import os, sys, json, uuid, time, argparse
from pathlib import Path

# Ensure src is in path
ROOT_DIR = Path(__file__).parent.parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

ACTIVE = Path("artifacts/cpesh_active.jsonl")
NPZ_PATH = Path(os.getenv("LNSP_NPZ_PATH", "artifacts/fw10k_vectors_768.npz"))

def _uuid5(ns: str, name: str) -> str:
    return str(uuid.uuid5(uuid.UUID("6ba7b811-9dad-11d1-80b4-00c04fd430c8"), f"{ns}:{name}"))

def _iso_now():
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()) + "Z"

def _load_tsv_row(path: str, row: int) -> dict:
    import csv
    with open(path, newline="", encoding="utf-8") as f:
        r = list(csv.DictReader(f, delimiter="\t"))
    assert 0 <= row < len(r), f"row {row} out of range (0..{len(r)-1})"
    return r[row]

def _extract_cpesh(text: str, source_id: str):
    # 1) LLM path (preferred)
    try:
        from src.prompt_extractor import extract_cpe_from_text
        cpesh = extract_cpe_from_text(text)
    except Exception:
        # 2) Heuristic fallback
        cpesh = {
            "concept_text": text.split(".")[0][:128],
            "probe_question": "What is the main concept?",
            "expected_answer": text[:256],
            "soft_negative": None, "hard_negative": None
        }
    # TMD & routing
    try:
        from src.utils.tmd import pack_tmd, lane_index_from_bits, tmd16_deterministic
        # If the extractor already returned domain/task/modifier, use them
        d = cpesh.get("domain_code"); t = cpesh.get("task_code"); m = cpesh.get("modifier_code")
        bits = cpesh.get("tmd_bits") or (pack_tmd(d, t, m) if all(x is not None for x in (d,t,m)) else None)
        lane_idx = cpesh.get("lane_index") or (lane_index_from_bits(bits) if bits is not None else 0)
        tmd16 = tmd16_deterministic(d, t, m) if all(x is not None for x in (d,t,m)) else None
    except Exception:
        bits, lane_idx, tmd16 = None, 0, None
    return cpesh, bits, lane_idx, tmd16

def _embed(concept: str, probe: str, tmd16):
    try:
        from src.vectorizer import EmbeddingBackend
        import numpy as np
        eb = EmbeddingBackend()
        c = eb.encode([concept])[0]  # 768D
        q = eb.encode([probe])[0]    # 768D
        # fuse: [tmd16 || concept768] -> 784D (normalize)
        if tmd16 is None:
            tmd16 = np.zeros(16, dtype="float32")
        fused = np.concatenate([np.array(tmd16, dtype="float32"), c]).astype("float32")
        norm = np.linalg.norm(fused) + 1e-12
        fused /= norm
        return c, q, tmd16, fused
    except Exception:
        # deterministic stub
        import numpy as np, hashlib
        def hvec(s, d):
            h = hashlib.sha256(s.encode("utf-8")).digest()
            num_repeats = (d * 4 + len(h) - 1) // len(h)
            full_buffer = (h * num_repeats)[:d*4]
            v = np.frombuffer(full_buffer, dtype="float32")
            v = v / (np.linalg.norm(v) + 1e-12)
            return v
        c = hvec(concept or "concept", 768)
        q = hvec(probe or "probe", 768)
        t = tmd16 if tmd16 is not None else np.zeros(16, dtype="float32")
        fused = np.concatenate([np.array(t,dtype="float32"), c])
        fused /= (np.linalg.norm(fused) + 1e-12)
        return c, q, t, fused

def _append_active_jsonl(rec: dict):
    ACTIVE.parent.mkdir(parents=True, exist_ok=True)
    with ACTIVE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

def _append_npz(cpe_id: str, doc_id: str, lane_idx: int, concept_text: str, tmd16, cvec, qvec, fused):
    import numpy as np
    NPZ_PATH.parent.mkdir(parents=True, exist_ok=True)
    if NPZ_PATH.exists():
        z = np.load(NPZ_PATH, allow_pickle=True)
        V = z["vectors"]; C = z["concept_vecs"]; Q = z["question_vecs"]
        T = z["tmd_dense"]; L = z["lane_indices"]
        dids = list(z["doc_ids"]); cids = list(z["cpe_ids"]); ctxt = list(z["concept_texts"])
        # append
        V = np.vstack([V, fused[None,:]])
        C = np.vstack([C, cvec[None,:]])
        Q = np.vstack([Q, qvec[None,:]])
        T = np.vstack([T, np.array(tmd16, dtype="float32")[None,:]])
        L = np.concatenate([L, np.array([lane_idx], dtype=L.dtype)])
        dids.append(doc_id); cids.append(cpe_id); ctxt.append(concept_text or "")
    else:
        V = fused[None,:].astype("float32")
        C = cvec[None,:].astype("float32")
        Q = qvec[None,:].astype("float32")
        T = np.array([tmd16], dtype="float32")
        L = np.array([lane_idx], dtype="int16")
        dids = [doc_id]; cids = [cpe_id]; ctxt = [concept_text or ""]
    np.savez_compressed(NPZ_PATH,
        vectors=V, concept_vecs=C, question_vecs=Q, tmd_dense=T, lane_indices=L,
        doc_ids=np.array(dids, dtype=object), cpe_ids=np.array(cids, dtype=object),
        concept_texts=np.array(ctxt, dtype=object)
    )

def _insert_postgres(cpesh_full: dict):
    dsn = os.getenv("PG_DSN")
    if not dsn:
        print("[vec_upsert] PG_DSN not set; skipping Postgres")
        return
    try:
        from src.db_postgres import PostgresDB
        db = PostgresDB(enabled=True)
        db.insert_cpe(cpesh_full)
    except Exception as e:
        print(f"[vec_upsert] Postgres insert failed: {e}")

def _insert_neo4j(cpesh_full: dict):
    if os.getenv("LNSP_WRITE_NEO4J","0") != "1":
        return
    try:
        from src.db_neo4j import Neo4jDB
        rels = cpesh_full.get("relations_text") or []
        if not rels: return
        db = Neo4jDB()
        db.insert_concept(cpesh_full)
        for r in rels:
            dst_id = r.get("obj_id") or r.get("obj")
            if not dst_id: continue
            rel_type = r.get("pred", "related_to")
            db.insert_relation_triple(cpesh_full["cpe_id"], dst_id, rel_type)
    except Exception as e:
        print(f"[vec_upsert] Neo4j insert failed: {e}")

def _maybe_rebuild_faiss():
    if os.getenv("LNSP_FAISS_REBUILD","0") != "1":
        return
    try:
        import subprocess, sys
        cmd = [sys.executable, "-m", "src.faiss_index",
               "--vectors", str(NPZ_PATH), "--type", "ivf_flat", "--metric", "ip",
               "--nlist", os.getenv("LNSP_NLIST","512"), "--nprobe", os.getenv("LNSP_NPROBE","16")]
        print("[faiss] rebuilding:", " ".join(cmd))
        subprocess.check_call(cmd)
    except Exception as e:
        print(f"[faiss] rebuild skipped: {e}")

def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--text", type=str, help="Raw text to extract CPESH/TMD from")
    g.add_argument("--tsv", type=str, help="TSV file path")
    ap.add_argument("--row", type=int, default=0, help="Row index when using --tsv")
    ap.add_argument("--doc-id", type=str, default="", help="Optional doc id (else derived)")
    args = ap.parse_args()

    if args.tsv:
        row = _load_tsv_row(args.tsv, args.row)
        text = row.get("text") or row.get("body") or row.get("content") or ""
        doc_id = args.doc_id or row.get("id") or f"tsv:{Path(args.tsv).name}:{args.row}"
        dataset_source = f"tsv:{args.tsv}"
    else:
        text = args.text
        doc_id = args.doc_id or _uuid5("doc", text[:120])
        dataset_source = "adhoc"

    cpe_id = _uuid5("cpe", doc_id)
    cpesh, tmd_bits, lane_idx, tmd16 = _extract_cpesh(text, doc_id)
    created = _iso_now()

    cpesh_full = {
        "cpe_id": cpe_id,
        "concept_text": cpesh.get("concept_text",""),
        "probe_question": cpesh.get("probe_question",""),
        "expected_answer": cpesh.get("expected_answer",""),
        "soft_negative": cpesh.get("soft_negative"),
        "hard_negative": cpesh.get("hard_negative"),
        "mission_text": cpesh.get("mission_text"),
        "dataset_source": dataset_source,
        "content_type": cpesh.get("content_type") or "text/plain",
        "chunk_position": cpesh.get("chunk_position") or {"doc_id": doc_id, "start": 0, "end": len(text)},
        "relations_text": cpesh.get("relations") or [],
        "tmd_bits": tmd_bits,
        "tmd_lane": cpesh.get("tmd_lane"),
        "lane_index": lane_idx,
        "quality": cpesh.get("quality"),
        "echo_score": cpesh.get("echo_score"),
        "insufficient_evidence": bool(cpesh.get("insufficient_evidence", False)),
        "created_at": created,
        "last_accessed": created,
        "access_count": 0,
        "doc_id": doc_id,
        "raw_text_preview": text[:320]
    }

    # Append to active lake
    _append_active_jsonl(cpesh_full)

    # Embeddings + NPZ
    cvec, qvec, t16, fused = _embed(cpesh_full["concept_text"], cpesh_full["probe_question"], tmd16)
    _append_npz(cpe_id, doc_id, lane_idx, cpesh_full["concept_text"], t16, cvec, qvec, fused)

    # Text DB
    _insert_postgres(cpesh_full)

    # Graph DB
    _insert_neo4j(cpesh_full)

    # Optional FAISS rebuild
    _maybe_rebuild_faiss()

    print(json.dumps({"ok": True, "cpe_id": cpe_id, "doc_id": doc_id, "lane_index": lane_idx, "npz": str(NPZ_PATH)}))

if __name__ == "__main__":
    main()
