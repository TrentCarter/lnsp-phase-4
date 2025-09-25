[Architect] we want each /search hit to return a compact, per-item bundle:
CPESH + TMD + ID + Score (optionally with CPESH sims), not just a big payload with diagnostics.
Below is a tight, drop-in patch that adds this without breaking anything you already have.
✅ What this adds
Request knobs
return_cpesh: bool (default false)
cpesh_mode: "lite" | "full" (default "lite"; "full" also returns soft_sim/hard_sim)
Per-item fields
id (stable doc_id)
score (cosine from FAISS)
tmd_code and lane_index (you already output these)
cpesh: {concept, probe, expected, soft_negative, hard_negative, soft_sim?, hard_sim?}
Guards
Top-K CPESH extraction capped via LNSP_CPESH_MAX_K (default 5)
Timeouts via your local Llama client; if extraction fails, item simply omits cpesh
No synthetic data; CPESH is generated live by local Llama per result’s concept_text, and sims are computed with your real 768D embedder.
1) schemas.py — add per-item CPESH + request flags
*** a/src/schemas.py
--- b/src/schemas.py
@@
 from pydantic import BaseModel, Field, model_validator
 from typing import List, Optional, Literal
 
 Lane = Literal["L1_FACTOID", "L2_GRAPH", "L3_SYNTH"]
 Mode = Literal["DENSE", "GRAPH", "HYBRID"]
 
+class CPESH(BaseModel):
+    concept: Optional[str] = None
+    probe: Optional[str] = None
+    expected: Optional[str] = None
+    soft_negative: Optional[str] = None
+    hard_negative: Optional[str] = None
+    soft_sim: Optional[float] = None  # cosine vs query (only if cpesh_mode=full)
+    hard_sim: Optional[float] = None  # cosine vs query (only if cpesh_mode=full)
+
 class SearchRequest(BaseModel):
     q: str = Field(..., min_length=1, max_length=512, description="Query string (1-512 characters)")
     lane: Optional[Lane] = Field(default=None, description="Lane: L1_FACTOID, L2_GRAPH, or L3_SYNTH")
     top_k: int = Field(default=8, ge=1, le=100, description="Number of results to return (1-100)")
     lane_index: Optional[int] = Field(default=None, ge=0, le=32767, description="Optional lane index filter (0-32767)")
+    return_cpesh: Optional[bool] = Field(default=False, description="Include per-item CPESH object")
+    cpesh_mode: Optional[Literal["lite","full"]] = Field(default="lite", description="CPESH detail level")
@@
 class SearchItem(BaseModel):
     id: str
     score: Optional[float] = None
     concept_text: Optional[str] = None
     tmd_code: Optional[str] = None
     lane_index: Optional[int] = None
+    cpesh: Optional[CPESH] = None
 
 class SearchResponse(BaseModel):
     lane: Optional[Lane]
     mode: Mode
     items: List[SearchItem]
     trace_id: Optional[str] = None
-    diagnostics: Optional[CPESHDiagnostics] = None
-    insufficient_evidence: Optional[bool] = None
+    diagnostics: Optional[CPESHDiagnostics] = None
+    insufficient_evidence: Optional[bool] = None
(Keeps your existing diagnostics & insufficient_evidence on the response. New per-item cpesh is optional.)
2) retrieve.py — fill cpesh on each item (opt-in)
*** a/src/api/retrieve.py
--- b/src/api/retrieve.py
@@
 import os, json, uuid, time
 import numpy as np
 from pathlib import Path
 from fastapi import APIRouter, HTTPException
 from ..schemas import SearchRequest, SearchResponse, SearchItem
+from ..schemas import CPESH  # new
@@
 class RetrieveAPI:
     def __init__(self, ...):
         ...
+        # CPESH extraction limits
+        self.cpesh_max_k = int(os.getenv("LNSP_CPESH_MAX_K", "5"))
+        # local llama client (lazy)
+        self._llm = None
+
+    def _ensure_llm(self):
+        if self._llm is None:
+            # Local-only client, no cloud fallback
+            from ..llm.local_llama_client import LocalLlamaClient
+            endpoint = os.getenv("LNSP_LLM_ENDPOINT","http://localhost:11434")
+            model = os.getenv("LNSP_LLM_MODEL","llama3.1:8b")
+            self._llm = LocalLlamaClient(endpoint, model)
+        return self._llm
@@
     def _norm_hit(self, h) -> SearchItem:
         doc_id = h.get("doc_id") or h.get("id")
         score = h.get("score")
         ct = h.get("concept_text")
         tmd = h.get("tmd_code")
         li = h.get("lane_index")
-        q = self.id_quality.get(str(doc_id), 0.5)
-        final = None if score is None else (self.w_cos*float(score) + self.w_q*float(q))
-        return SearchItem(id=doc_id, score=score, concept_text=ct, tmd_code=tmd, lane_index=li,
-                          quality=q, final_score=final)
+        qqual = self.id_quality.get(str(doc_id), 0.5)
+        final = None if score is None else (self.w_cos*float(score) + self.w_q*float(qqual))
+        return SearchItem(
+            id=doc_id, score=score, concept_text=ct, tmd_code=tmd, lane_index=li,
+            quality=qqual, final_score=final
+        )
@@
     def search(self, req: SearchRequest) -> SearchResponse:
         trace_id = uuid.uuid4().hex[:8]
         ...
         items = [self._norm_hit(h) for h in candidates if h]
         print(f"Trace {trace_id}: Normalized {len(items)} items.")
 
+        # --- Optional per-item CPESH extraction ---
+        if req.return_cpesh and items:
+            # cap extraction to avoid long calls
+            k = min(len(items), self.cpesh_max_k)
+            llm = self._ensure_llm()
+            # embed the user query once for sim calc (cosine) when cpesh_mode=full
+            qvec = None
+            if req.cpesh_mode == "full":
+                qvec = self.embedder.encode([req.q], normalize_embeddings=True)[0].astype(np.float32)
+            for i in range(k):
+                it = items[i]
+                text = (it.concept_text or "").strip()
+                if not text:
+                    continue
+                prompt = (
+                    "Return JSON only for CPESH_EXTRACT.\n"
+                    f'Factoid: "{text}"\n'
+                    '{"concept":"...","probe":"...","expected":"...",'
+                    '"soft_negative":"...","hard_negative":"...",'
+                    '"insufficient_evidence":false}'
+                )
+                try:
+                    j = llm.complete_json(prompt, timeout_s=12)  # parses JSON
+                    cp = CPESH(
+                        concept=j.get("concept"),
+                        probe=j.get("probe"),
+                        expected=j.get("expected"),
+                        soft_negative=j.get("soft_negative"),
+                        hard_negative=j.get("hard_negative"),
+                    )
+                    # sims vs the query embedding if requested
+                    if req.cpesh_mode == "full" and qvec is not None:
+                        if cp.soft_negative:
+                            sv = self.embedder.encode([cp.soft_negative], normalize_embeddings=True)[0].astype(np.float32)
+                            cp.soft_sim = float(np.dot(qvec, sv))
+                        if cp.hard_negative:
+                            hv = self.embedder.encode([cp.hard_negative], normalize_embeddings=True)[0].astype(np.float32)
+                            cp.hard_sim = float(np.dot(qvec, hv))
+                    it.cpesh = cp
+                except Exception as e:
+                    print(f"Trace {trace_id}: CPESH extract failed for {it.id}: {e}")
+                    continue
+
         # Cache and return
         _search_cache.put(req.lane, req.q, req.top_k, items)
         return SearchResponse(
             lane=req.lane, mode=mode, items=items, trace_id=trace_id,
             diagnostics=diag, insufficient_evidence=insufficient,
         )
This only touches the top-K items (default 5) to keep latency tight. If you want all hits to include CPESH, set LNSP_CPESH_MAX_K higher.
3) Call it the way you described
You asked for: CPESH + TMD + ID + Score. Here’s the exact call:
curl -s -X POST http://localhost:8092/search \
  -H 'content-type: application/json' \
  -d '{
        "q":"Which ocean is largest?",
        "top_k":5,
        "lane":"L1_FACTOID",
        "use_quality": true,
        "return_cpesh": true,
        "cpesh_mode": "full"
      }' | jq '.items[] | {id, score, tmd_code, lane_index, cpesh}'
You’ll now see, per item:
id (same as doc_id)
score (FAISS cosine)
tmd_code, lane_index
cpesh (with concept/probe/expected/soft/hard and, in "full" mode, soft_sim/hard_sim)