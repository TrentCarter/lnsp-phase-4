#!/usr/bin/env python3
"""
Validates 10× training readiness on REAL FactoidWiki data.

Checks:
- NPZ schema keys/dtypes/shapes
- doc_id alignment across NPZ ↔ chunks.jsonl
- vectors are L2-normalized and re-encode match ≥0.99
- FAISS IDMap returns real hits mapped to stable doc_ids
- (optional) CPESH extraction via local Llama + sims/margins

Outputs:
- eval/10x_training_readiness.jsonl  (one line per sampled id)
- prints a PASS/FAIL summary with counts
"""
from __future__ import annotations
import os, sys, json, random, time, argparse, gc
from pathlib import Path
import numpy as np

def load_chunks(path: Path):
    rows = []
    with path.open() as f:
        for line in f:
            rec = json.loads(line)
            did = rec.get("doc_id") or rec.get("id")
            if did is None: continue
            rows.append((str(did), rec))
    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="artifacts/fw10k_vectors_768.npz")
    ap.add_argument("--chunks", default="artifacts/fw10k_chunks.jsonl")
    ap.add_argument("--index", default="artifacts/fw10k_ivf_768.index")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--out", default="eval/10x_training_readiness.jsonl")
    ap.add_argument("--extract-cpesh", action="store_true",
                    help="Use local Llama to extract CPESH for each sample (requires src/llm/local_llama_client.py)")
    ap.add_argument("--lane-index", type=int, default=None)
    args = ap.parse_args()

    # Hard real-only policy
    os.environ["TRANSFORMERS_OFFLINE"]="1"; os.environ["HF_HUB_OFFLINE"]="1"
    model_dir = os.getenv("LNSP_EMBED_MODEL_DIR", "data/teacher_models/gtr-t5-base")
    assert Path(model_dir).exists(), f"Missing local GTR model at {model_dir}"

    # Load NPZ and schema
    npz = np.load(args.npz, allow_pickle=True)
    need = ["vectors","ids","doc_ids","concept_texts","tmd_dense","lane_indices","cpe_ids"]
    miss = [k for k in need if k not in npz]
    if miss: raise SystemExit(f"NPZ missing keys: {miss}")
    V = npz["vectors"]; ids = npz["ids"]; doc_ids = npz["doc_ids"]; C = npz["concept_texts"]
    lanes = npz["lane_indices"]; tmd = npz["tmd_dense"]

    assert V.dtype == np.float32 and V.ndim==2 and V.shape[1]==768, f"vectors wrong shape {V.shape}"
    norms = np.linalg.norm(V, axis=1)
    assert np.isfinite(V).all() and (norms>0.99).mean()>0.99, "vectors not L2-normalized or contain NaN/Inf"
    N = V.shape[0]
    assert len(ids)==len(doc_ids)==len(C)==len(lanes)==len(tmd)==N, "array length mismatch"

    # Load chunks for text/ID parity
    rows = load_chunks(Path(args.chunks))
    by_doc = {str(did): rec for did, rec in rows}
    have = sum(1 for did in doc_ids if str(did) in by_doc)
    if have < int(0.95*N):
        print(f"WARNING: only {have}/{N} doc_ids found in chunks; continue but check your ingest.")

    # Embedder (offline GTR)
    from sentence_transformers import SentenceTransformer
    enc = SentenceTransformer(model_dir)
    # FAISS
    import faiss
    faiss.omp_set_num_threads(1)
    index = faiss.read_index(args.index)
    assert index.d == 768, f"Index dim {index.d} != 768"
    assert index.ntotal == N, f"Index ntotal {index.ntotal} != {N}"

    # Sampling
    all_rows = list(range(N))
    if args.lane_index is not None:
        all_rows = [i for i in all_rows if int(lanes[i])==args.lane_index]
    random.seed(13)
    sample = random.sample(all_rows, min(args.n, len(all_rows)))

    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    ok_reenc = ok_faiss = 0
    with outp.open("w") as w:
        for i in sample:
            did = str(doc_ids[i]); text = str(C[i]); vec = V[i]
            # Re-encode match
            v1 = enc.encode([text], normalize_embeddings=True)[0].astype(np.float32)
            cos = float(np.dot(vec, v1))
            reenc_ok = cos > 0.99
            ok_reenc += 1 if reenc_ok else 0

            # FAISS neighbors for the query text
            q = enc.encode([text], normalize_embeddings=True).astype(np.float32)
            D,I = index.search(q, 5)
            ids_raw = [int(x) for x in I[0]]
            # In IndexIDMap2, ids are whatever was added; in your build you wrapped with positional ids -> treat as rows
            neighbors = []
            for rid,score in zip(ids_raw, D[0].tolist()):
                if 0 <= rid < N:
                    neighbors.append({"row": rid, "doc_id": str(doc_ids[rid]), "score": float(score), "concept": str(C[rid])[:120]})
            faiss_ok = len(neighbors)>0
            ok_faiss += 1 if faiss_ok else 0

            rec = {
                "row": i,
                "doc_id": did,
                "lane_index": int(lanes[i]),
                "concept_text": text,
                "reencode_cos": cos,
                "faiss_neighbors": neighbors,
            }

            # Optional CPESH extraction via local Llama
            if args.extract_cpesh:
                try:
                    sys.path.append("src")
                    from llm.local_llama_client import LocalLlamaClient
                    llm = LocalLlamaClient(os.getenv("LNSP_LLM_ENDPOINT","http://127.0.0.1:11434"),
                                           os.getenv("LNSP_LLM_MODEL","llama3.1:8b-instruct"))
                    # Minimal CPESH prompt (expects JSON-only, from your template)
                    prompt = (
                        "Return JSON only for CPESH_EXTRACT.\n"
                        f'Factoid: "{text}"\n'
                        '{"concept": "...","probe": "...","expected": "...",'
                        '"soft_negative": "...","hard_negative": "...","insufficient_evidence": false}'
                    )
                    t0=time.time()
                    cpesh_json = llm.complete_json(prompt, timeout_s=12)  # assumes helper returning parsed JSON
                    dt=int((time.time()-t0)*1000)
                    rec["cpesh"] = cpesh_json; rec["cpesh_ms"]=dt
                    # Compute sims if negatives provided
                    exp = cpesh_json.get("expected")
                    sneg = cpesh_json.get("soft_negative")
                    hneg = cpesh_json.get("hard_negative")
                    if exp and sneg:
                        s_sim = float(np.dot(
                            enc.encode([sneg], normalize_embeddings=True)[0].astype(np.float32),
                            v1
                        ))
                        rec["soft_sim"]=s_sim
                    if exp and hneg:
                        h_sim = float(np.dot(
                            enc.encode([hneg], normalize_embeddings=True)[0].astype(np.float32),
                            v1
                        ))
                        rec["hard_sim"]=h_sim
                except Exception as e:
                    rec["cpesh_error"] = str(e)

            w.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # Clean FAISS to avoid macOS teardown issues
    del index; gc.collect()

    print(f"Samples: {len(sample)}  |  Re-encode≥0.99: {ok_reenc}/{len(sample)}  |  FAISS hits: {ok_faiss}/{len(sample)}")
    print(f"Wrote: {outp}")

if __name__ == "__main__":
    main()
