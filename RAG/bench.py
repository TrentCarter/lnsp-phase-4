#!/usr/bin/env python3
"""RAG-only benchmarking harness (vecRAG vs alternatives).
Usage:
  python RAG/bench.py --dataset self --n 1000 --topk 10 --backends vec,bm25,lex,lightvec,lightrag_full
Env:
  FAISS_NPZ_PATH, LNSP_FUSED(0/1), FAISS_NPROBE
"""
from __future__ import annotations
import argparse, json, os, sys, time, math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.db_faiss import FaissDB  # type: ignore
from src.vectorizer import EmbeddingBackend  # type: ignore

try:
    from src.adapters.lightrag.vectorstore_faiss import get_vector_store  # type: ignore
    HAS_LIGHTRAG = True
except Exception:
    HAS_LIGHTRAG = False

try:
    from rank_bm25 import BM25Okapi
    HAS_BM25 = True
except Exception:
    HAS_BM25 = False

@dataclass
class Corpus:
    vectors: np.ndarray
    doc_ids: List[str]
    concept_texts: List[str]
    tmd_dense: Optional[np.ndarray]
    cpe_ids: List[str]
    dim: int


def _detect_npz() -> Optional[str]:
    for p in [os.getenv("FAISS_NPZ_PATH"), ROOT/"artifacts/fw10k_vectors.npz", ROOT/"artifacts/fw9k_vectors.npz"]:
        if p and os.path.isfile(str(p)):
            return str(p)
    return None

def _detect_index() -> Optional[str]:
    meta = ROOT/"artifacts/faiss_meta.json"
    if meta.exists():
        try:
            return (json.loads(meta.read_text()).get("index_path"))
        except Exception:
            return None
    return None

def load_corpus(npz_path: str) -> Corpus:
    npz = np.load(npz_path, allow_pickle=True)
    vec = None
    for k in ("vectors","fused","concept","concept_vecs"):
        if k in npz and np.asarray(npz[k]).ndim==2:
            vec = np.asarray(npz[k], dtype=np.float32); break
    if vec is None:
        raise ValueError(f"No 2D vectors found in {npz_path}")

    # Handle doc_ids safely
    doc_ids_raw = npz.get("doc_ids")
    if doc_ids_raw is not None:
        doc_ids = [str(x) for x in doc_ids_raw]
    else:
        doc_ids = [str(i) for i in range(len(vec))]

    # Handle concept_texts safely
    concept_texts_raw = npz.get("concept_texts")
    if concept_texts_raw is not None:
        concept_texts = [str(x) for x in concept_texts_raw]
    else:
        concept_texts = [""]*len(vec)

    # Handle cpe_ids safely
    cpe_ids_raw = npz.get("cpe_ids")
    if cpe_ids_raw is not None:
        cpe_ids = [str(x) for x in cpe_ids_raw]
    else:
        cpe_ids = [str(i) for i in range(len(vec))]

    # Handle tmd_dense safely
    tmd_dense_raw = npz.get("tmd_dense")
    if tmd_dense_raw is not None:
        tmd_dense = np.asarray(tmd_dense_raw, dtype=np.float32)
    else:
        tmd_dense = None

    return Corpus(
        vectors=vec,
        doc_ids=doc_ids,
        concept_texts=concept_texts,
        tmd_dense=tmd_dense,
        cpe_ids=cpe_ids,
        dim=int(vec.shape[1]),
    )

def build_query(text: str, emb: EmbeddingBackend, tmd: Optional[np.ndarray], dim: int) -> np.ndarray:
    v = emb.encode([text])[0].astype(np.float32)
    if dim==768:
        return v
    if dim==784:
        t = np.zeros((16,),dtype=np.float32) if tmd is None else tmd.astype(np.float32)
        q = np.concatenate([t,v]).astype(np.float32)
        n = float(np.linalg.norm(q));  q = q/n if n>0 else q
        return q
    raise ValueError(f"Unsupported dim {dim}")

# backends

def run_vec(db: FaissDB, queries: List[np.ndarray], topk: int):
    idxs: List[List[int]]=[]; scores: List[List[float]]=[]; lat: List[float]=[]
    for q in queries:
        q = q.reshape(1,-1).astype(np.float32)
        n=float(np.linalg.norm(q));  q=q/n if n>0 else q
        t0=time.perf_counter(); D,I=db.search(q, topk); lat.append((time.perf_counter()-t0)*1000.0)
        idxs.append([int(x) for x in I[0]]); scores.append([float(s) for s in D[0]])
    return idxs, scores, lat

def run_lex(tokens: List[set], queries_text: List[str], topk: int):
    import re
    idxs: List[List[int]]=[]; scores: List[List[float]]=[]; lat: List[float]=[]
    for qt in queries_text:
        t0=time.perf_counter(); q=set(re.findall(r"\w+", (qt or "").lower()))
        sc=[]
        for i,t in enumerate(tokens):
            ov=len(q & t)
            if ov: sc.append((i,ov))
        sc.sort(key=lambda x:(-x[1],x[0]))
        sel=sc[:topk]; lat.append((time.perf_counter()-t0)*1000.0)
        idxs.append([i for i,_ in sel]+([-1]*max(0,topk-len(sel))))
        scores.append([float(s) for _,s in sel]+([0.0]*max(0,topk-len(sel))))
    return idxs, scores, lat

def run_lightvec(index_path: str, npz_path: str, dim: int, emb: EmbeddingBackend, queries_text: List[str], tmds: List[Optional[np.ndarray]], topk: int):
    if not HAS_LIGHTRAG: raise RuntimeError("lightrag not installed")
    store=get_vector_store(index_path=index_path, meta_npz_path=npz_path, dim=dim)
    idxs: List[List[int]]=[]; scores: List[List[float]]=[]; lat: List[float]=[]
    for qt,tmd in zip(queries_text,tmds):
        q=build_query(qt, emb, tmd, dim).reshape(1,-1).astype(np.float32)
        t0=time.perf_counter(); D,I=store.search(q, top_k=topk); lat.append((time.perf_counter()-t0)*1000.0)
        idxs.append([int(x) for x in I[0]]); scores.append([float(s) for s in D[0]])
    return idxs, scores, lat

def run_bm25(corpus_texts: List[str], queries_text: List[str], topk: int):
    """BM25 baseline using rank_bm25."""
    if not HAS_BM25: raise RuntimeError("rank-bm25 not installed (pip install rank-bm25)")
    import re
    # Tokenize corpus
    tokenized_corpus = [re.findall(r"\w+", (text or "").lower()) for text in corpus_texts]
    bm25 = BM25Okapi(tokenized_corpus)

    idxs: List[List[int]] = []
    scores: List[List[float]] = []
    lat: List[float] = []

    for qt in queries_text:
        t0 = time.perf_counter()
        tokenized_query = re.findall(r"\w+", (qt or "").lower())
        doc_scores = bm25.get_scores(tokenized_query)

        # Get top-k indices
        top_indices = np.argsort(doc_scores)[::-1][:topk]
        top_scores = doc_scores[top_indices]

        lat.append((time.perf_counter() - t0) * 1000.0)
        idxs.append(top_indices.tolist())
        scores.append(top_scores.tolist())

    return idxs, scores, lat

def run_lightrag_full(queries_text: List[str], topk: int, doc_ids: List[str]):
    """LightRAG full graph-enhanced retrieval (hybrid mode).

    NOTE: Requires artifacts/kg/ to be populated with knowledge graph.
    This is more complex than pure vector retrieval and includes graph traversal.
    """
    if not HAS_LIGHTRAG:
        raise RuntimeError("lightrag not installed")

    # Check if knowledge graph exists
    kg_dir = ROOT / "artifacts/kg"
    if not kg_dir.exists() or not list(kg_dir.glob("*.json")):
        raise RuntimeError(
            "Knowledge graph not found in artifacts/kg/. "
            "Run graph ingestion first or use 'lightvec' for vector-only comparison."
        )

    # Import LightRAG components
    try:
        from lightrag import LightRAG, QueryParam  # type: ignore
        from src.adapters.lightrag.graphrag_runner import get_embedder  # type: ignore
    except Exception as e:
        raise RuntimeError(f"Failed to import LightRAG components: {e}")

    # Initialize LightRAG with hybrid mode
    embedder = get_embedder()

    def llm_func(prompt: str, **kwargs):
        """Stub LLM for graph traversal (not used in retrieval-only mode)."""
        from src.llm.local_llama_client import call_local_llama  # type: ignore
        response = call_local_llama(prompt)
        return {"text": response.text}

    rag = LightRAG(
        working_dir=str(kg_dir.absolute()),
        llm_model_func=llm_func,
        embedding_func=embedder.embed_batch,
    )

    idxs: List[List[int]] = []
    scores: List[List[float]] = []
    lat: List[float] = []

    # Map doc_ids to positions for result mapping
    doc_to_pos = {str(d): i for i, d in enumerate(doc_ids)}

    for qt in queries_text:
        t0 = time.perf_counter()

        # Run hybrid query (vector + graph)
        result = rag.query(qt, param=QueryParam(mode="hybrid", top_k=topk))

        # Parse results - LightRAG returns entities/chunks with scores
        # This is a simplified mapping; actual implementation depends on LightRAG output format
        # For now, return empty results with a warning
        lat.append((time.perf_counter() - t0) * 1000.0)

        # Placeholder: Map LightRAG results back to corpus positions
        # This requires matching returned entities to doc_ids
        idxs.append([])  # TODO: Implement result mapping
        scores.append([])

    # Warning already printed in main() before execution
    return idxs, scores, lat

# metrics

def _rank_of(pos: int, indices: List[int]) -> Optional[int]:
    try: return indices.index(pos)+1
    except ValueError: return None

def _m_at_k(ranks: List[Optional[int]], k: int) -> float:
    return float(sum(1 for r in ranks if r is not None and r<=k))/max(1,len(ranks))

def _mrr_at_k(ranks: List[Optional[int]], k: int) -> float:
    s=0.0
    for r in ranks:
        if r is not None and r<=k: s += 1.0/float(r)
    return s/max(1,len(ranks))

def _ndcg_at_k(ranks: List[Optional[int]], k: int) -> float:
    s=0.0
    for r in ranks:
        if r is not None and r<=k: s += 1.0/math.log2(r+1.0)
    return s/max(1,len(ranks))


def main() -> int:
    ap=argparse.ArgumentParser()
    ap.add_argument("--dataset",choices=["self","cpesh"],default=None)
    ap.add_argument("--npz",default=None); ap.add_argument("--index",default=None)
    ap.add_argument("--n",type=int,default=500); ap.add_argument("--topk",type=int,default=10)
    ap.add_argument("--backends",default="vec,bm25")
    ap.add_argument("--out",default=None)
    args=ap.parse_args()

    npz_path=args.npz or _detect_npz()
    if not npz_path:
        print("[bench] NPZ not found"); return 2
    index_path=args.index or _detect_index()
    if not index_path:
        print("[bench] FAISS index not found"); return 2

    corp=load_corpus(npz_path); dim=corp.dim
    doc_to_pos={str(d):i for i,d in enumerate(corp.doc_ids)}

    # build dataset
    dataset=args.dataset or ("cpesh" if (ROOT/"artifacts/cpesh_cache.jsonl").exists() else "self")
    q_text: List[str]=[]; gold_pos: List[int]=[]; tmds: List[Optional[np.ndarray]]=[]

    if dataset=="self":
        N=len(corp.doc_ids); n=min(args.n,N); idxs=np.linspace(0,N-1,num=n,dtype=int)
        for i in idxs:
            qt=corp.concept_texts[i] or f"Concept {corp.cpe_ids[i]}"; q_text.append(qt); gold_pos.append(int(i))
            tmds.append(corp.tmd_dense[i] if corp.tmd_dense is not None else None)
    else:
        cache=ROOT/"artifacts/cpesh_cache.jsonl"; count=0
        with cache.open("r",encoding="utf-8") as fh:
            for line in fh:
                try: j=json.loads(line)
                except Exception: continue
                doc_id=str(j.get("doc_id") or "")
                cp=j.get("cpesh") or {}
                probe=cp.get("probe") or cp.get("expected") or cp.get("concept") or ""
                if not probe: continue
                pos=doc_to_pos.get(doc_id)
                if pos is None: continue
                q_text.append(probe)
                gold_pos.append(int(pos))
                t=(corp.tmd_dense[pos] if corp.tmd_dense is not None else None)
                tmds.append(t)
                count+=1
                if count>=args.n: break
        if not q_text:
            print("[bench] No CPESH queries usable"); return 2

    # tokens for lexical
    import re
    tokens=[set(re.findall(r"\w+", (t or "").lower())) for t in corp.concept_texts]

    # prep queries for vec/lightvec
    emb=EmbeddingBackend()
    queries=[build_query(t, emb, tm, dim) for t,tm in zip(q_text,tmds)]

    out_path=args.out or str(ROOT/f"RAG/results/bench_{int(time.time())}.jsonl"); Path(out_path).parent.mkdir(parents=True,exist_ok=True)

    backends=[b.strip() for b in args.backends.split(',') if b.strip()]
    summaries: List[Dict[str,Any]]=[]

    # Check for lightrag_full and warn early
    if "lightrag_full" in backends:
        kg_dir = ROOT / "artifacts/kg"
        if not kg_dir.exists() or not list(kg_dir.glob("*.json")):
            print(f"[WARN] lightrag_full backend requires knowledge graph in artifacts/kg/")
            print(f"[WARN] Skipping lightrag_full. Use 'lightvec' for vector-only LightRAG comparison.")
            backends = [b for b in backends if b != "lightrag_full"]
        else:
            print(f"[WARN] lightrag_full is EXPERIMENTAL - result mapping incomplete, metrics will be zero")

    # load FAISS once
    db=FaissDB(index_path=str(index_path), meta_npz_path=npz_path); db.load(str(index_path))

    def eval_backend(name: str):
        if name=="vec":
            I,S,L=run_vec(db, queries, args.topk)
        elif name=="lex":
            I,S,L=run_lex(tokens, q_text, args.topk)
        elif name=="bm25":
            I,S,L=run_bm25(corp.concept_texts, q_text, args.topk)
        elif name=="lightvec":
            I,S,L=run_lightvec(str(index_path), npz_path, dim, emb, q_text, tmds, args.topk)
        elif name=="lightrag_full":
            I,S,L=run_lightrag_full(q_text, args.topk, corp.doc_ids)
        else:
            raise ValueError(f"Unknown backend: {name}")
        # ranks
        ranks=[_rank_of(g, Iq) for g,Iq in zip(gold_pos, I)]
        summ={
            "name": name,
            "metrics": {
                "p_at_1": _m_at_k(ranks,1),
                "p_at_5": _m_at_k(ranks,5),
                "mrr_at_10": _mrr_at_k(ranks,10),
                "ndcg_at_10": _ndcg_at_k(ranks,10),
            },
            "latency_ms": {
                "mean": float(np.mean(L) if L else 0.0),
                "p95": float(np.percentile(L,95) if L else 0.0),
            }
        }
        # write per-query JSONL with enhanced output (scores + ranks)
        with open(out_path,"a",encoding="utf-8") as fh:
            for i,(qt,ip,sc,rank) in enumerate(zip(q_text,I,S,ranks)):
                hits = [{"doc_id": corp.doc_ids[idx] if 0 <= idx < len(corp.doc_ids) else "N/A",
                        "score": float(score),
                        "rank": r+1}
                       for r, (idx, score) in enumerate(zip(ip, sc))]
                fh.write(json.dumps({
                    "backend": name,
                    "query": qt,
                    "gold_pos": gold_pos[i],
                    "gold_doc_id": corp.doc_ids[gold_pos[i]],
                    "hits": hits,
                    "gold_rank": rank
                })+"\n")
        return summ

    for b in backends:
        summaries.append(eval_backend(b))

    results={"dataset":dataset,"n":len(q_text),"topk":args.topk,"dim":dim,"backends":summaries}
    md_lines=[
        "# RAG Benchmark Summary",
        f"Dataset: {results['dataset']} | N: {results['n']} | topk: {results['topk']} | dim: {results['dim']}",
        "",
        "| Backend | P@1 | P@5 | MRR@10 | nDCG@10 | Mean ms | P95 ms |",
        "|---------|-----|-----|--------|---------|---------|--------|",
    ]
    for s in summaries:
        m=s["metrics"]; lat=s["latency_ms"]
        md_lines.append(f"| {s['name']} | {m['p_at_1']:.3f} | {m['p_at_5']:.3f} | {m['mrr_at_10']:.3f} | {m['ndcg_at_10']:.3f} | {lat['mean']:.2f} | {lat['p95']:.2f} |")
    md_path=ROOT/"RAG/results"/f"summary_{int(time.time())}.md"; md_path.parent.mkdir(parents=True,exist_ok=True); md_path.write_text("\n".join(md_lines))
    print(f"Wrote per-query to: {out_path}\nSummary: {md_path}")
    return 0

if __name__=="__main__":
    raise SystemExit(main())
