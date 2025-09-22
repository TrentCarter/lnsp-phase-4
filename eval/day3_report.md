# Day 3 Retrieval Report (P4) - 2025-09-22

## Summary
- Echo pass: **NN.N%** (passes / 20)
- Modes: DENSE vs GRAPH vs HYBRID distribution
- IVF Config: ivf=8192, pq=64, metric=COSINE

## Lane Distribution
| Lane       | Queries | Pass | P@1 | P@5 | Notes |
|------------|---------|------|-----|-----|-------|
| L1_FACTOID |         |      |     |     |       |
| L2_GRAPH   |         |      |     |     |       |
| L3_SYNTH   |         |      |     |     |       |

## Samples
```json
GET /search?q=Define%20FactoidWiki&lane=L1_FACTOID&top_k=8
{ "items":[{"id":"doc_000123","score":0.91,"why":"..."}], "lane":"L1_FACTOID","mode":"DENSE" }
```
Findings
Strengths: ...
Misses: ...
Next Actions:
Tune IVF (try ivf=16384, pq=64)
Add ALIASES extraction to KG
Expand paraphrase positives for Q6

---

## Exact P4 Command Script (copy/paste)

```bash
# Status log
mkdir -p chats && echo "[P4 start] $(date -Iseconds)" >> chats/conversation_09222025.md

# Architect: drop files
git add docs/enums.md src/enums.py

# Programmer: init NO_DOCKER and vendor
./scripts/init_pg.sh
./scripts/vendor_lightrag.sh

# Ingest 1k and build IVF
./scripts/ingest_1k.sh
./scripts/build_faiss_1k.sh

# API up
uvicorn src.api.retrieve:app --host 0.0.0.0 --port 8080 --reload &
sleep 2

# Consultant: quick eval smoke
mkdir -p eval/day3_samples
./eval_echo.sh   # or python -m src.eval_runner ... as above

echo "[P4 done] $(date -Iseconds)" >> chats/conversation_09222025.md
```
Quick notes & receipts
Pin LightRAG at 1.4.8.2 (released Sep 16, 2025) to dodge a critical path-traversal vuln reported for <=1.3.8; multiple advisories confirm fix beyond that range. 
PyPI
+1
LightRAG's hybrid + graph retrieval aligns with our lane-aware plan; paper/site back this pattern.
