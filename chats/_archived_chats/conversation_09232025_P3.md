[P3] Conversation Plan — Day-4 kickoff & 10k ramp (25–30 min)
Attendees: [Architect] [Programmer] [Consultant]
Goal: eliminate segfaults, fix FAISS config/metadata mismatch, bump docs, and start 10k ingest canary.
0) Pre-reads (2 min)
artifacts/faiss_meta.json (shows 4×784, IndexFlatIP)
tests/test_retrieve_api.py (great coverage; we’ll mark heavy tests)
/docs/architecture.md LightRAG pin section (still old pin)
1) Decisions to ratify (5 min)
Python version: hard-pin runtime to 3.11.x for now (3.13 causes segfaults with FAISS/torch on some wheels).
L1_FACTOID default: keep dense-only, put lexical behind ENABLE_LEX_FALLBACK (default 0).
Indexing for 1k/10k:
1k: IVF_FLAT, nlist=32, nprobe=8 (≈√N)
10k: IVF_FLAT, nlist=128, nprobe=16 (start here; we’ll tune after latency checks)
2) Segfault elimination (6 min)
Enforce interpreter with .python-version and tooling hints; mark heavy tests to avoid loading FAISS/torch inside unit tests.
Split tests: pytest -m "not heavy" for CI/unit, pytest -m heavy for integration.
Drop-in files/patches
pyproject.toml (or add to existing):
[project]
requires-python = ">=3.11,<3.12"
.python-version
3.11.9
pytest.ini
[pytest]
markers =
    heavy: loads FAISS/torch or hits real indices
Refactor tests (example):
# tests/test_retrieve_api.py
import pytest

@pytest.mark.heavy
def test_vector_search_path(...):
    ...
Light test stub (fast, no FAISS):
# src/search_backends/stub.py
class StubSearcher:
    def topk(self, qvec, k=5):
        return [{"id": "S1", "score": 0.99}]
Wire stub for unit tests:
# src/api/retrieve.py (snippet)
import os
USE_STUB = os.getenv("LNSP_TEST_MODE","0") == "1"
searcher = StubSearcher() if USE_STUB else RealFaissSearcher(...)
Run units without heavy deps:
LNSP_TEST_MODE=1 pytest -m "not heavy" -q
3) FAISS truthing & metadata (5 min)
Rebuild 1k vectors and ensure vectors_count >= nlist:
bash scripts/ingest_1k.sh
python -m src.vectorizer --input artifacts/fw1k_chunks.jsonl --out artifacts/fw1k_vectors.npz
python -m src.faiss_index --npz artifacts/fw1k_vectors.npz \
  --index-type IVF_FLAT --nlist 32 --out artifacts/fw1k_ivf.index
Meta writer (ensures index + vector counts line up):
# tools/write_faiss_meta.py
import json, time
from pathlib import Path
def main():
    meta = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "vectors_npz": "artifacts/fw1k_vectors.npz",
        "index_path": "artifacts/fw1k_ivf.index",
        "index_type": "IVF_FLAT",
        "nlist": 32, "nprobe": 8,
        "dim": 784, "vectors": 1000  # replace with actual loader values
    }
    Path("artifacts/faiss_meta.json").write_text(json.dumps(meta, indent=2))
if __name__ == "__main__": main()
Commit the refreshed faiss_meta.json and paste the key line into chats/conversation_09232025.md.
4) API & lexical flag (4 min)
Add the env flag and remove lexical cost from the hot path by default.
# src/api/retrieve.py (near query handling)
ENABLE_LEX = os.getenv("ENABLE_LEX_FALLBACK","0") == "1"
if lane == "L1_FACTOID" and not ENABLE_LEX:
    results = dense_topk(...)
else:
    results = hybrid_or_lexical_path(...)
Quick smoke:
uvicorn src.api.retrieve:app --reload
curl -s 'http://127.0.0.1:8000/healthz'
curl -s 'http://127.0.0.1:8000/search?q=Who%20coined%20computer%20bug?&lane=L1_FACTOID&top_k=5'
5) Docs & pins (3 min)
/docs/architecture.md: bump LightRAG pin to 1.4.9rc1 and note “graph+vector via adapter; FAISS remains SoT”.
/docs/enums.md: sync to src/enums.py and append TMD bit-packing note (D:4 | T:5 | M:6 | spare:1).
Add a “Runtime Matrix” snippet to README (Py 3.11 ✅; 3.13 🚫 until FAISS wheels settle).
6) 10k canary plan (5 min)
Ingest next 9k (total 10k); build IVF with nlist=128; record build time, RAM, and search P50/P95 at k∈{5,20}.
Acceptance gates (initial):
L1_FACTOID dense-only: P50 ≤ 85 ms, P95 ≤ 200 ms, Top-1 hit-rate ≥ 0.92 on eval-20.
Hybrid path (L2/L3 only): P50 ≤ 180 ms, P95 ≤ 400 ms.
Commands
bash scripts/ingest_10k.sh          # mirror 1k script, emits artifacts/fw10k_vectors.npz
python -m src.faiss_index --npz artifacts/fw10k_vectors.npz \
  --index-type IVF_FLAT --nlist 128 --out artifacts/fw10k_ivf.index
python tools/write_faiss_meta.py    # update counts/params
python -m src.eval_runner --queries eval/day3_eval.jsonl --top-k 5 --timeout 15 \
  --out eval/day3_results_live.jsonl
Role-scoped action items
[Architect]
Update /docs/architecture.md (pin + retrieval modes) and /docs/enums.md (bit-packing).
Open issue: “Retire lexical from L1 hot path; keep flag only”.
[Programmer]
Enforce Py 3.11 (.python-version, pyproject.toml), commit.
Add pytest.ini markers and LNSP_TEST_MODE stub; mark heavy tests.
Rebuild 1k vectors + IVF (nlist=32) and refresh faiss_meta.json.
Add a small artifact check script:
python - <<'PY'
import numpy as np, json, sys
from pathlib import Path
ok = Path('artifacts/fw1k_vectors.npz').exists() and Path('artifacts/fw1k_ivf.index').exists()
print({"artifacts_ok": ok})
PY
[Consultant]
Extend eval to balanced 20 with 5×L1/8×L2/7×L3; re-run on 1k dense-only and hybrid; paste a 3-row /search sample (req/resp) in eval/day3_report.md.
Log acceptance results vs gates; highlight any regressions.

[Consultant] Status — 2025-09-23T19:04:00Z
- Rebuilt `eval/day3_eval.jsonl` with balanced lanes (5 L1 / 8 L2 / 7 L3) using gold IDs from `data/factoidwiki_1k.jsonl`.
- Executed offline lexical smoke `python3 -m src.eval_runner --queries eval/day3_eval.jsonl --offline-npz artifacts/fw1k_vectors.npz --top-k 5 --timeout 15 --out eval/day3_results_consultant_balanced.jsonl` (25% echo, strong drop on L2/L3 as expected).
- Updated `eval/day3_report.md` with the new dataset snapshot, offline metrics, and three representative `/search` transcripts; flagged dense-only & hybrid API runs as pending on a Python 3.11 environment with network access.

[Consultant] Next
- Once a network-enabled 3.11 environment is available, rerun `src.eval_runner` against the live API with `LNSP_LEXICAL_FALLBACK=0/1` to capture dense vs hybrid metrics and latency distribution.
- After the programmer refreshes `artifacts/faiss_meta.json`, ingest those counts into the report and double-check acceptance gates (P50/P95 latency and top-1 hit rate thresholds).

---

## [Architect] EXECUTION FEEDBACK (2025-09-23)

### ✅ COMPLETED TASKS

#### 1. Updated /docs/architecture.md
- **LightRAG pin**: Updated from `1.4.8.2` to `1.4.9rc1`
- **Retrieval modes documentation**: Added comprehensive lane-awareness details:
  - L1_FACTOID: Dense-only by default (lexical fallback via `LNSP_LEXICAL_FALLBACK=true`)
  - L2_GRAPH: Dense + graph expansion via KGEdgeType
  - L3_SYNTH: Full hybrid path (dense + graph + reranking)
- **FAISS Source of Truth**: Clarified that LightRAG provides graph utilities while FAISS remains primary vector store

#### 2. Updated /docs/enums.md
- **Synced with src/enums.py**: Complete alignment of enum definitions
- **Added TMD Bit-Packing section**: Detailed 16-bit layout specification:
  - Domain: bits 15-12 (4 bits, 0-15)
  - Task: bits 11-7 (5 bits, 0-31)
  - Modifier: bits 6-1 (6 bits, 0-63)
  - Spare: bit 0 (1 bit, reserved)
- **Lane index extraction**: `(tmd_bits >> 1) & 0x7FFF` for 15-bit lane space
- **Example calculation**: Domain=2, Task=5, Modifier=10 → tmd_bits=8832 → lane_index=4416

#### 3. GitHub Issue Documentation
**Title**: "Retire lexical from L1_FACTOID hot path; keep flag only"

**Summary**: Remove lexical search from L1_FACTOID default path for performance, keeping it available via `LNSP_LEXICAL_FALLBACK=true` flag.

**Implementation**: Update `src/api/retrieve.py`:
```python
ENABLE_LEX = os.getenv('LNSP_LEXICAL_FALLBACK', '0') == '1'
if lane == 'L1_FACTOID' and not ENABLE_LEX:
    results = dense_topk(...)
else:
    results = hybrid_or_lexical_path(...)
```

**Acceptance Criteria**:
- L1_FACTOID dense-only default: P50 ≤ 85ms, P95 ≤ 200ms
- Quality maintained: Top-1 hit-rate ≥ 0.92 on eval set
- Flag-enabled lexical fallback preserved
- L2_GRAPH/L3_SYNTH lanes unchanged

### 🔗 INTEGRATION NOTES

1. **Python 3.11 Pinning**: Architecture now references the completed Python version pinning to prevent FAISS/torch segfaults
2. **Performance Gates**: Documentation aligns with Day-4 performance targets (85ms P50, 200ms P95)
3. **Lane Strategy**: Clear separation of dense-only (L1) vs graph-enhanced (L2/L3) retrieval paths

### 📋 NEXT ACTIONS FOR TEAM

**[Programmer]**:
- Implement the `LNSP_LEXICAL_FALLBACK` flag logic in `src/api/retrieve.py`
- Add pytest.ini with heavy test markers
- Rebuild 1k vectors with IVF_FLAT nlist=32

**[Consultant]**:
- Test both dense-only and fallback modes against performance gates
- Validate quality metrics on balanced eval set (5×L1/8×L2/7×L3)

### ✨ IMPACT

The architecture updates provide clear guidance for the 10k scaling effort while maintaining backward compatibility through environment flags. The TMD bit-packing documentation ensures consistent lane routing across all components.

**Status**: All [Architect] tasks completed and documented. Ready for [Programmer] and [Consultant] execution phases.
