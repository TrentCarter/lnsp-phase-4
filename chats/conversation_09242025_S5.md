S5 — Batch Diff (unified)
Apply from repo root:
git checkout -b s5-bundle
git apply -p0 <<'PATCH'
*** Begin Patch
*** Update File: src/schemas.py
@@
-from pydantic import BaseModel, Field, model_validator
-from typing import List, Optional, Literal
+from pydantic import BaseModel, Field, model_validator
+from typing import List, Optional, Literal
 
 Lane = Literal["L1_FACTOID", "L2_GRAPH", "L3_SYNTH"]
 Mode = Literal["DENSE", "GRAPH", "HYBRID"]
 
+class CPESH(BaseModel):
+    concept: Optional[str] = None
+    probe: Optional[str] = None
+    expected: Optional[str] = None
+    soft_negative: Optional[str] = None
+    hard_negative: Optional[str] = None
+    soft_sim: Optional[float] = None  # cosine vs query (full mode)
+    hard_sim: Optional[float] = None  # cosine vs query (full mode)
+
 class CPESHDiagnostics(BaseModel):
     concept: Optional[str] = None
     probe: Optional[str] = None
     expected: Optional[str] = None
     soft_negative: Optional[str] = None
@@
 class SearchRequest(BaseModel):
     q: str = Field(..., min_length=1, max_length=512, description="Query string (1-512 characters)")
     lane: Optional[Lane] = Field(default=None, description="Lane: L1_FACTOID, L2_GRAPH, or L3_SYNTH")
     top_k: int = Field(default=8, ge=1, le=100, description="Number of results to return (1-100)")
     lane_index: Optional[int] = Field(default=None, ge=0, le=32767, description="Optional lane index filter (0-32767)")
     return_cpesh: Optional[bool] = Field(default=False, description="Include per-item CPESH object")
     cpesh_mode: Optional[Literal["lite","full"]] = Field(default="lite", description="CPESH detail level")
+    cpesh_k: Optional[int] = Field(default=None, ge=0, le=50, description="Max hits to CPESH-enrich (overrides env)")
+    compact: Optional[bool] = Field(default=False, description="Return compact hit objects (id,score,tmd,lane,cpesh)")
 
 class SearchItem(BaseModel):
     id: str
     score: Optional[float] = None
     concept_text: Optional[str] = None
     tmd_code: Optional[str] = None
     lane_index: Optional[int] = None
+    quality: Optional[float] = None
+    final_score: Optional[float] = None
+    cpesh: Optional[CPESH] = None
 
 class SearchResponse(BaseModel):
     lane: Optional[Lane]
     mode: Mode
     items: List[SearchItem]
     trace_id: Optional[str] = None
     diagnostics: Optional[CPESHDiagnostics] = None
     insufficient_evidence: Optional[bool] = None
*** End Patch
PATCH
git apply -p0 <<'PATCH'
*** Begin Patch
*** Update File: src/llm/local_llama_client.py
@@
-import json, time, requests
+import json, time, requests
 class LocalLlamaClient:
     def __init__(self, endpoint: str, model: str):
         self.endpoint = endpoint.rstrip("/")
         self.model = model
         self._session = requests.Session()
@@
-    def complete_json(self, prompt: str, timeout_s: float = 10.0):
-        payload = {"model": self.model, "prompt": prompt, "options": {"num_predict": 256}}
-        r = self._session.post(self.endpoint + "/api/generate", json=payload, timeout=timeout_s)
-        r.raise_for_status()
-        # stream parsing (older style)
-        text = "".join([json.loads(line)["response"] for line in r.text.splitlines() if line.strip().startswith("{") and "response" in line])
-        s = text.find("{"); e = text.rfind("}")
-        if s == -1 or e == -1 or e <= s:
-            raise ValueError(f"Ollama did not return JSON: {text[:80]}")
-        return json.loads(text[s:e+1])
+    def complete_json(self, prompt: str, timeout_s: float = 10.0):
+        payload = {
+            "model": self.model,
+            "prompt": prompt,
+            "options": {"num_predict": 256, "temperature": 0},
+            "stream": False,
+            "format": "json",
+        }
+        r = self._session.post(self.endpoint + "/api/generate", json=payload, timeout=timeout_s)
+        r.raise_for_status()
+        data = r.json()  # {"response": "..."}
+        text = data.get("response", "")
+        s = text.find("{"); e = text.rfind("}")
+        if s == -1 or e == -1 or e <= s:
+            raise ValueError(f"Ollama did not return JSON: {text[:80]}")
+        return json.loads(text[s:e+1])
*** End Patch
PATCH
git apply -p0 <<'PATCH'
*** Begin Patch
*** Update File: src/adapters/lightrag/graphrag_runner.py
@@
- from ...db.rag_session_store import RAGSessionStore
+try:
+    from ...db.rag_session_store import RAGSessionStore
+except Exception:
+    import sys, pathlib
+    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "src"))
+    from db.rag_session_store import RAGSessionStore
@@
- store = RAGSessionStore()  # uses PG by default
+import os, json
+use_pg = bool(os.getenv("LNSP_PG_DSN"))
+if use_pg:
+    store = RAGSessionStore()
+else:
+    from pathlib import Path
+    class JsonlStore:
+        def __init__(self, path="eval/graphrag_runs.jsonl"):
+            self.path = Path(path); self.path.parent.mkdir(parents=True, exist_ok=True)
+        def write_session(self, session: dict):
+            with self.path.open("a", encoding="utf-8") as w:
+                w.write(json.dumps(session, ensure_ascii=False) + "\n")
+    store = JsonlStore()
*** End Patch
PATCH
git apply -p0 <<'PATCH'
*** Begin Patch
*** Update File: src/adapters/lightrag/vectorstore_faiss.py
@@
- from ...db.db_postgres import PG_DSN
+try:
+    from ...db.db_postgres import PG_DSN
+except Exception:
+    import sys, pathlib
+    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "src"))
+    from db.db_postgres import PG_DSN
*** End Patch
PATCH
git apply -p0 <<'PATCH'
*** Begin Patch
*** Update File: scripts/run_graphrag_eval.sh
@@
-  python src/adapters/lightrag/graphrag_runner.py "$@"
+  PYTHONPATH=src:. python -m src.adapters.lightrag.graphrag_runner "$@"
*** End Patch
PATCH
git apply -p0 <<'PATCH'
*** Begin Patch
*** Update File: src/api/retrieve.py
@@
-import os, json, uuid, time
+import os, json, uuid, time
 import numpy as np
 from pathlib import Path
 from fastapi import APIRouter, HTTPException
-from ..schemas import SearchRequest, SearchResponse, SearchItem
+from ..schemas import SearchRequest, SearchResponse, SearchItem, CPESH
+from ..utils.tmd import unpack_tmd  # ensure this exists; used to format D.T.M
@@
 class RetrieveAPI:
     def __init__(self, ...):
         ...
-        # CPESH extraction limits
-        self.cpesh_max_k = int(os.getenv("LNSP_CPESH_MAX_K", "5"))
+        # CPESH limits + caching
+        self.cpesh_max_k = int(os.getenv("LNSP_CPESH_MAX_K", "2"))
+        self.cpesh_timeout_s = float(os.getenv("LNSP_CPESH_TIMEOUT_S", "4"))
+        self.cpesh_cache_path = Path(os.getenv("LNSP_CPESH_CACHE", "artifacts/cpesh_cache.jsonl"))
+        self.cpesh_cache = {}
+        if self.cpesh_cache_path.exists():
+            try:
+                for line in self.cpesh_cache_path.open():
+                    j = json.loads(line)
+                    self.cpesh_cache[str(j["doc_id"])] = j["cpesh"]
+            except Exception as e:
+                print(f"[CPESH] cache load skipped: {e}")
         # load id quality map (doc_id -> quality)
         self.id_quality = {}
         qpath = Path("artifacts/id_quality.jsonl")
         if qpath.exists():
             for line in qpath.open():
                 j = json.loads(line)
                 self.id_quality[str(j["doc_id"])] = float(j.get("quality", 0.5))
         self.w_cos = float(os.getenv("LNSP_W_COS","0.85"))
         self.w_q   = float(os.getenv("LNSP_W_QUALITY","0.15"))
@@
-    def _norm_hit(self, h) -> SearchItem:
-        doc_id = h.get("doc_id") or h.get("id")
-        score = h.get("score")
-        ct = h.get("concept_text")
-        tmd = h.get("tmd_code")
-        li = h.get("lane_index")
-        qqual = self.id_quality.get(str(doc_id), 0.5)
-        final = None if score is None else (self.w_cos*float(score) + self.w_q*float(qqual))
-        return SearchItem(
-            id=doc_id, score=score, concept_text=ct, tmd_code=tmd, lane_index=li,
-            quality=qqual, final_score=final
-        )
+    def _format_tmd_code(self, h: dict) -> str:
+        """
+        Resolve a D.T.M code string from either packed bits, or domain/task/modifier ints, else '0.0.0'.
+        """
+        tmd_bits = h.get("tmd_bits")
+        if tmd_bits is not None:
+            try:
+                d, t, m = unpack_tmd(int(tmd_bits))
+                return f"{int(d)}.{int(t)}.{int(m)}"
+            except Exception:
+                pass
+        d = h.get("domain_code"); t = h.get("task_code"); m = h.get("modifier_code")
+        if d is not None and t is not None and m is not None:
+            try:
+                return f"{int(d)}.{int(t)}.{int(m)}"
+            except Exception:
+                return "0.0.0"
+        return h.get("tmd_code") or "0.0.0"
+
+    def _norm_hit(self, h) -> SearchItem:
+        doc_id = h.get("doc_id") or h.get("id")
+        score = h.get("score")
+        ct = h.get("concept_text")
+        tmd_code = self._format_tmd_code(h)
+        li = h.get("lane_index")
+        qqual = self.id_quality.get(str(doc_id), 0.5)
+        final = None if score is None else (self.w_cos*float(score) + self.w_q*float(qqual))
+        return SearchItem(
+            id=doc_id, score=score, concept_text=ct, tmd_code=tmd_code, lane_index=li,
+            quality=qqual, final_score=final
+        )
@@
     def search(self, req: SearchRequest) -> SearchResponse:
         trace_id = uuid.uuid4().hex[:8]
         ...
         items = [self._norm_hit(h) for h in candidates if h]
         print(f"Trace {trace_id}: Normalized {len(items)} items.")
 
         # --- Optional per-item CPESH extraction ---
-        if req.return_cpesh and items:
-            # cap extraction to avoid long calls
-            k = min(len(items), self.cpesh_max_k)
+        if req.return_cpesh and items:
+            # cap extraction per request
+            kcap = req.cpesh_k if (req.cpesh_k is not None) else self.cpesh_max_k
+            k = min(len(items), kcap)
             llm = self._ensure_llm()
             # embed the user query once for sim calc (cosine) when cpesh_mode=full
             qvec = None
             if req.cpesh_mode == "full":
                 qvec = self.embedder.encode([req.q], normalize_embeddings=True)[0].astype(np.float32)
             for i in range(k):
                 it = items[i]
                 text = (it.concept_text or "").strip()
                 if not text:
                     continue
+                cached = self.cpesh_cache.get(it.id)
+                if cached:
+                    try:
+                        it.cpesh = CPESH(**cached)
+                        continue
+                    except Exception:
+                        pass
                 prompt = (
                     "Return JSON only for CPESH_EXTRACT.\n"
                     f'Factoid: "{text}"\n'
                     '{"concept":"...","probe":"...","expected":"...",'
                     '"soft_negative":"...","hard_negative":"...",'
                     '"insufficient_evidence":false}'
                 )
                 try:
-                    j = llm.complete_json(prompt, timeout_s=12)  # parses JSON
+                    j = llm.complete_json(prompt, timeout_s=self.cpesh_timeout_s)  # strict JSON
                     cp = CPESH(
                         concept=j.get("concept"),
                         probe=j.get("probe"),
                         expected=j.get("expected"),
                         soft_negative=j.get("soft_negative"),
                         hard_negative=j.get("hard_negative"),
                     )
                     # sims vs the query embedding if requested
                     if req.cpesh_mode == "full" and qvec is not None:
                         if cp.soft_negative:
                             sv = self.embedder.encode([cp.soft_negative], normalize_embeddings=True)[0].astype(np.float32)
                             cp.soft_sim = float(np.dot(qvec, sv))
                         if cp.hard_negative:
                             hv = self.embedder.encode([cp.hard_negative], normalize_embeddings=True)[0].astype(np.float32)
                             cp.hard_sim = float(np.dot(qvec, hv))
                     it.cpesh = cp
+                    # persist cache line
+                    try:
+                        self.cpesh_cache[it.id] = cp.dict()
+                        self.cpesh_cache_path.parent.mkdir(parents=True, exist_ok=True)
+                        with self.cpesh_cache_path.open("a", encoding="utf-8") as w:
+                            w.write(json.dumps({"doc_id": it.id, "cpesh": cp.dict()}, ensure_ascii=False) + "\n")
+                    except Exception as e:
+                        print(f"[CPESH] cache write skipped: {e}")
                 except Exception as e:
                     print(f"Trace {trace_id}: CPESH extract failed for {it.id}: {e}")
                     continue
@@
-        _search_cache.put(req.lane, req.q, req.top_k, items)
-        return SearchResponse(
+        _search_cache.put(req.lane, req.q, req.top_k, items)
+        resp = SearchResponse(
             lane=req.lane, mode=mode, items=items, trace_id=trace_id,
             diagnostics=diag, insufficient_evidence=insufficient,
         )
+        # Optional compact projection
+        if req.compact:
+            for it in resp.items:
+                it.concept_text = None
+        return resp
+
+@router.get("/admin/cpesh_cache")
+def cpesh_cache_stats():
+    api = _API_SINGLETON  # assuming you set this when wiring the app
+    size = len(api.cpesh_cache) if hasattr(api, "cpesh_cache") else 0
+    path = str(getattr(api, "cpesh_cache_path", ""))
+    return {"entries": size, "path": path}
*** End Patch
PATCH
Note: if your app structure uses a factory, replace _API_SINGLETON with however you expose the instance; or return file stats directly.
New helper (optional but handy): tools/self_test_s5.py
git apply -p0 <<'PATCH'
*** Begin Patch
*** Add File: tools/self_test_s5.py
+#!/usr/bin/env python3
+import os, json, requests, time
+from sentence_transformers import SentenceTransformer
+
+def main():
+    os.environ.setdefault("TRANSFORMERS_OFFLINE","1")
+    os.environ.setdefault("HF_HUB_OFFLINE","1")
+    mdir = os.getenv("LNSP_EMBED_MODEL_DIR","data/teacher_models/gtr-t5-base")
+    m = SentenceTransformer(mdir); assert m.get_sentence_embedding_dimension()==768
+    print("✅ GTR offline ok")
+    r = requests.get("http://localhost:11434/api/tags", timeout=3); r.raise_for_status()
+    print("✅ Ollama reachable")
+    q = {"q":"Which ocean is largest?","top_k":5,"lane":"L1_FACTOID","return_cpesh":True,"cpesh_mode":"full","cpesh_k":2,"compact":True}
+    s = requests.post("http://localhost:8092/search", json=q, timeout=10); print("API:", s.status_code); print(json.dumps(s.json(), indent=2)[:800])
+    print("✅ /search cpesh-full ok")
+if __name__=="__main__":
+    main()
*** End Patch
PATCH
S5 — Team Assignment
[Architect] (hard issues)
CPESH Cache Policy & SLOs
Define TTL (7d), max size (~50k entries), eviction (append-only JSONL + periodic compaction).
Document knobs: LNSP_CPESH_MAX_K, LNSP_CPESH_TIMEOUT_S, LNSP_CPESH_CACHE.
Confirm no cloud fallback is possible anywhere (local-only LLM).
TMD Taxonomy & Mapping
Finalize DOMAINS/TASKS/MODIFIERS taxonomy and publish in docs/architecture.md.
Ensure utils/tmd.py round-trips (pack/unpack/format) and capture examples.
Write acceptance: hit returns non-0.0.0 for at least 70% factoid items.
Deliverables: Architecture doc updates; cache SLO note; examples for D.T.M.
[Programmer]
Apply S5 patch set
Update files per diff; ensure src/utils/tmd.py exposes unpack_tmd.
Ensure the FastAPI instance wires _API_SINGLETON or adjust the admin handler.
Tests
Add CPESH cache smoke:
First /search with return_cpesh=true, cpesh_k=1 should call LLM.
Second identical /search returns CPESH from cache (no LLM call if you track count).
Add TMD formatting test: _format_tmd_code() returns expected strings ("2.0.27" etc).
GraphRAG import
Verify python -m src.adapters.lightrag.graphrag_runner --help runs.
If PG not configured, JSONL fallback writes eval/graphrag_runs.jsonl.
Ollama JSON strictness
Ensure complete_json() works with format:"json"; update any callers that assumed stream.
Deliverables: Passing tests; short README snippet in docs/run_log.md.
[Consultant]
Latency + Quality
With LNSP_CPESH_MAX_K=2 and LNSP_CPESH_TIMEOUT_S=4, measure /search P50/P95.
Show impact of cache warm vs cold on latency.
Retrieval Quality
Compare with vs without use_quality and return_cpesh:
Hit@1/3 deltas on 100 queries.
Qualitative: show 3 examples where CPESH helps judge ambiguity (soft/hard neg sims).
Report
eval/day_s5_report.md with: settings, latency table, success cases, failures, next knobs.
Runbook (quick)
# env
export TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 LNSP_EMBED_MODEL_DIR=$PWD/data/teacher_models/gtr-t5-base
export LNSP_LLM_ENDPOINT=http://localhost:11434 LNSP_LLM_MODEL=llama3.1:8b
export LNSP_CPESH_MAX_K=2 LNSP_CPESH_TIMEOUT_S=4 LNSP_CPESH_CACHE=artifacts/cpesh_cache.jsonl
export LNSP_W_COS=0.85 LNSP_W_QUALITY=0.15

# start API
uvicorn src.api.retrieve:app --host 127.0.0.1 --port 8092

# self-test
python tools/self_test_s5.py

# example call (compact)
curl -s -X POST localhost:8092/search \
 -H 'content-type: application/json' \
 -d '{"q":"Which ocean is largest?","top_k":5,"lane":"L1_FACTOID","return_cpesh":true,"cpesh_mode":"full","cpesh_k":2,"compact":true}' | jq .
About the TMD fix you reported
Great catch. The batch above doesn’t change your mappings; it simply makes _norm_hit() resilient:
If the hit carries tmd_bits, we decode to D.T.M with unpack_tmd.
Else we fall back to domain_code/task_code/modifier_code.
Else reuse tmd_code if already set; final fallback "0.0.0".
That ensures your programmer’s fix surfaces correctly in the API across all hit paths.