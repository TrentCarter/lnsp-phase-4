#!/usr/bin/env python3
from __future__ import annotations
import os, re, json, gc, argparse
from pathlib import Path
import numpy as np

TEXT_MIN, TEXT_MAX = 60, 500
ALNUM_RE = re.compile(r"[A-Za-z0-9]")
CTRL_RE  = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F]")

def text_health(s: str) -> float:
    if not s: return 0.0
    L = len(s)
    len_score = max(0.0, min(1.0, (L - TEXT_MIN) / (TEXT_MAX - TEXT_MIN))) if L < TEXT_MAX else max(0.0, 1.0 - (L - TEXT_MAX)/500)
    alnum_ratio = sum(ch.isalnum() for ch in s) / max(1, L)
    ctrl_pen = 1.0 if CTRL_RE.search(s) else 0.0
    return max(0.0, min(1.0, 0.7*len_score + 0.3*alnum_ratio - 0.5*ctrl_pen))

def load_graph_degree(edges_path: Path, doc_ids) -> np.ndarray:
    if not edges_path.exists(): return np.zeros(len(doc_ids), np.float32)
    deg = {str(d):0 for d in doc_ids}
    with edges_path.open() as f:
        for line in f:
            e = json.loads(line)
            a = str(e.get("src_doc_id") or e.get("src") or "")
            b = str(e.get("dst_doc_id") or e.get("dst") or "")
            if a in deg: deg[a]+=1
            if b in deg: deg[b]+=1
    # degree → [0..1] with soft saturation
    out = np.zeros(len(doc_ids), np.float32)
    for i,d in enumerate(doc_ids):
        k = deg.get(str(d), 0)
        out[i] = k/(k+5)  # 0, ~0.17@1, 0.5@5, ->1 as k grows
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="artifacts/fw10k_vectors_768.npz")
    ap.add_argument("--index", default="artifacts/fw10k_ivf_768.index")
    ap.add_argument("--edges", default="artifacts/kg/edges.jsonl")
    ap.add_argument("--out-jsonl", default="artifacts/id_quality.jsonl")
    ap.add_argument("--out-npz", default="artifacts/id_quality.npz")
    ap.add_argument("--kdup", type=int, default=5, help="neighbors to inspect for duplicate penalty")
    ap.add_argument("--w_text", type=float, default=0.4)
    ap.add_argument("--w_graph", type=float, default=0.3)
    ap.add_argument("--w_dup", type=float, default=0.2)
    ap.add_argument("--w_cpesh", type=float, default=0.1)
    ap.add_argument("--use-cpesh", action="store_true", help="compute CPESH margin via local Llama (real only)")
    args = ap.parse_args()

    os.environ["TRANSFORMERS_OFFLINE"]="1"; os.environ["HF_HUB_OFFLINE"]="1"
    D = np.load(args.npz, allow_pickle=True)
    V = D["vectors"].astype(np.float32, copy=False)
    doc_ids = D["doc_ids"]; texts = D["concept_texts"]

    # text feature
    text_feat = np.array([text_health(str(t)) for t in texts], np.float32)

    # graph feature
    graph_feat = load_graph_degree(Path(args.edges), doc_ids)

    # duplicate penalty from FAISS neighbors
    import faiss
    faiss.omp_set_num_threads(1)
    index = faiss.read_index(args.index)
    assert index.d == V.shape[1] == 768
    # Use each stored vector to query top-k dup candidates
    dup_pen = np.zeros(len(V), np.float32)
    for i in range(len(V)):
        Dv, Iv = index.search(V[i:i+1], args.kdup)
        # count neighbors (excluding self) with cosine >= 0.999
        c = 0
        for j,(rid,sim) in enumerate(zip(Iv[0], Dv[0])):
            if int(rid)==i: continue
            if sim >= 0.999: c += 1
        # convert to [0..1] penalty (cap at 3 exact dups)
        dup_pen[i] = min(1.0, c/3.0)
    del index; gc.collect()

    # optional CPESH margin
    cpesh_margin = np.zeros(len(V), np.float32)
    if args.use_cpesh:
        # lightweight local Llama extraction + embed expected/negatives once per item
        from sentence_transformers import SentenceTransformer
        enc = SentenceTransformer(os.getenv("LNSP_EMBED_MODEL_DIR","data/teacher_models/gtr-t5-base"))
        import sys, pathlib
        sys.path.insert(0, str(pathlib.Path("src").resolve()))
        from llm.local_llama_client import LocalLlamaClient
        llm = LocalLlamaClient(os.getenv("LNSP_LLM_ENDPOINT","http://localhost:11434"),
                               os.getenv("LNSP_LLM_MODEL","llama3.1:8b"))
        def margin(txt: str, base_vec: np.ndarray)->float:
            prompt = ("Return JSON only for CPESH_EXTRACT.\n"
                      f'Factoid: "{txt}"\n'
                      '{"concept":"...","probe":"...","expected":"...",'
                      '"soft_negative":"...","hard_negative":"...",'
                      '"insufficient_evidence":false}')
            try:
                j = llm.complete_json(prompt, timeout_s=10)
                exp, s, h = j.get("expected"), j.get("soft_negative"), j.get("hard_negative")
                if not exp: return 0.0
                vexp = enc.encode([exp], normalize_embeddings=True)[0].astype(np.float32)
                vs = enc.encode([s], normalize_embeddings=True)[0].astype(np.float32) if s else None
                vh = enc.encode([h], normalize_embeddings=True)[0].astype(np.float32) if h else None
                ssoft = float(vexp@vs) if vs is not None else -1.0
                shard = float(vexp@vh) if vh is not None else -1.0
                return max(0.0, 1.0*1.0 - max(ssoft, shard)) if max(ssoft, shard)>=0 else 0.3 # modest default
            except Exception:
                return 0.0
        for i in range(len(V)):
            cpesh_margin[i] = margin(str(texts[i]), V[i])

    # aggregate quality
    wT,wG,wD,wC = args.w_text, args.w_graph, args.w_dup, args.w_cpesh
    quality = (wT*text_feat + wG*graph_feat + wD*(1.0-dup_pen) + wC*cpesh_margin).astype(np.float32)

    # write artifacts
    Path(args.out_jsonl).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_jsonl, "w") as w:
        for i,did in enumerate(doc_ids):
            w.write(json.dumps({
                "doc_id": str(did),
                "quality": float(quality[i]),
                "features": {
                    "text": float(text_feat[i]),
                    "graph": float(graph_feat[i]),
                    "dup_penalty": float(dup_pen[i]),
                    "cpesh_margin": float(cpesh_margin[i])
                }
            }, ensure_ascii=False) + "\n")
    np.savez(args.out_npz, doc_ids=doc_ids, quality=quality,
             text=text_feat, graph=graph_feat, dup_penalty=dup_pen, cpesh_margin=cpesh_margin)
    print(f"OK: wrote {args.out_jsonl} and {args.out_npz}")

if __name__ == "__main__":
    main()
