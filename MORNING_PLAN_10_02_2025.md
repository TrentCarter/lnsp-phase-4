# Morning Action Plan - October 2, 2025

**Priority:** Fix critical data issues BEFORE LVM training
**Time Estimate:** 4 hours
**Goal:** High-quality ontology data for LVM training

---

## 🚨 Critical Issues to Fix

### Issue 1: TMD Misclassification (HIGH)
- **1,562/4,484 (34.8%)** have zero TMD vectors
- Remaining TMDs incorrectly classify domains
- **Root Cause:** TMD classifier trained on FactoidWiki, not ontologies

### Issue 2: Missing Negatives (MEDIUM)
- **171/4,484 (3.8%)** have empty soft/hard negatives
- Need to generate from chain structure

---

## ⚡ Quick Start (Recommended Path)

### Option A: Simple Rule-Based Fix (FASTEST - 30 min)
```bash
# Use source-based TMD assignment
./.venv/bin/python tools/fix_ontology_tmd_simple.py
```

**Logic:**
- SWO (software) → engineering/code_generation/neutral
- GO (biology) → medicine/fact_retrieval/neutral
- DBpedia (people) → art/fact_retrieval/neutral

**Pros:** Immediate, accurate for known sources
**Cons:** Not learning from concept text

### Option B: LLM-Based Classification (BEST - 2 hours)
```bash
# Use Ollama Llama 3.1 for accurate classification
./.venv/bin/python tools/fix_ontology_tmd.py --mode all --limit 100  # test first
./.venv/bin/python tools/fix_ontology_tmd.py --mode all  # full run
```

**Pros:** High accuracy, learns from concept semantics
**Cons:** Takes ~4 minutes for 4,484 concepts

---

## 📋 Step-by-Step Workflow

### Step 1: Create Simple TMD Fixer (15 min)
```bash
cat > tools/fix_ontology_tmd_simple.py << 'EOF'
#!/usr/bin/env python3
"""Quick fix: Rule-based TMD assignment for ontology sources."""
import sys
from pathlib import Path
import psycopg2
import json

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.tmd_encoder import pack_tmd

# Source-based TMD mapping
TMD_MAP = {
    'swo': (2, 1, 0),      # engineering, code_generation, neutral
    'go': (4, 0, 0),       # medicine, fact_retrieval, neutral
    'dbpedia': (9, 0, 0),  # art, fact_retrieval, neutral
    None: (0, 0, 0)        # default: science, fact_retrieval, neutral
}

conn = psycopg2.connect('dbname=lnsp')
cur = conn.cursor()

# Get all concepts
cur.execute("""
    SELECT v.cpe_id, e.source
    FROM cpe_vectors v
    JOIN cpe_entry e ON v.cpe_id = e.cpe_id
    WHERE v.tmd_dense IS NOT NULL
""")

rows = cur.fetchall()
print(f'Fixing TMD for {len(rows)} concepts...')

for i, (cpe_id, source) in enumerate(rows):
    if i % 500 == 0:
        print(f'Progress: {i}/{len(rows)}')

    domain, task, modifier = TMD_MAP.get(source, TMD_MAP[None])
    tmd_dense = pack_tmd(domain, task, modifier)

    cur.execute("""
        UPDATE cpe_vectors
        SET tmd_dense = %s
        WHERE cpe_id = %s
    """, (json.dumps(tmd_dense.tolist()), cpe_id))

conn.commit()
print('✅ All TMD vectors updated!')

# Verify
cur.execute("""
    SELECT COUNT(*)
    FROM cpe_vectors
    WHERE tmd_dense::jsonb = '[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]'::jsonb
""")
zero_count = cur.fetchone()[0]
print(f'Zero TMD vectors remaining: {zero_count}')

cur.close()
conn.close()
EOF

chmod +x tools/fix_ontology_tmd_simple.py
```

### Step 2: Run TMD Fix (1 min)
```bash
./.venv/bin/python tools/fix_ontology_tmd_simple.py
```

### Step 3: Generate Soft/Hard Negatives (CREATE SCRIPT - 30 min)
```bash
cat > tools/generate_ontology_negatives.py << 'EOF'
#!/usr/bin/env python3
"""Generate soft/hard negatives from ontology chain structure."""
import sys
from pathlib import Path
import psycopg2
import json
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[1]

conn = psycopg2.connect('dbname=lnsp')
cur = conn.cursor()

# Load all chains from JSONL files
chains = []
for jsonl_path in (ROOT / "data/ontology_chains").glob("*.jsonl"):
    with open(jsonl_path) as f:
        for line in f:
            chain = json.loads(line)
            chains.append(chain["chain"])  # list of concepts

print(f'Loaded {len(chains)} chains')

# Build parent → children index
parent_to_children = defaultdict(set)
for chain in chains:
    for i in range(len(chain) - 1):
        parent = chain[i]
        child = chain[i + 1]
        parent_to_children[parent].add(child)

# Generate negatives for each concept
updates = []
for chain in chains:
    concept = chain[-1]  # target concept

    # Soft negatives: siblings (share parent)
    soft_negatives = set()
    if len(chain) > 1:
        parent = chain[-2]
        siblings = parent_to_children.get(parent, set())
        soft_negatives = {s for s in siblings if s != concept}

    # Hard negatives: random from other chains (not in current chain)
    hard_negatives = set()
    for other_chain in chains:
        if other_chain == chain:
            continue
        # Take last concept from other chains
        candidate = other_chain[-1]
        if candidate not in chain:
            hard_negatives.add(candidate)
            if len(hard_negatives) >= 5:
                break

    updates.append((
        concept,
        list(soft_negatives)[:5],
        list(hard_negatives)[:5]
    ))

print(f'Generated negatives for {len(updates)} concepts')

# Update database
for i, (concept, soft_neg, hard_neg) in enumerate(updates):
    if i % 500 == 0:
        print(f'Update progress: {i}/{len(updates)}')

    cur.execute("""
        UPDATE cpe_entry
        SET soft_negatives = %s, hard_negatives = %s
        WHERE concept_text = %s
    """, (json.dumps(soft_neg), json.dumps(hard_neg), concept))

conn.commit()
print('✅ Soft/hard negatives generated!')

# Verify
cur.execute("""
    SELECT COUNT(*)
    FROM cpe_entry
    WHERE jsonb_array_length(soft_negatives) = 0
       OR jsonb_array_length(hard_negatives) = 0
""")
empty_count = cur.fetchone()[0]
print(f'Concepts with empty negatives: {empty_count}')

cur.close()
conn.close()
EOF

chmod +x tools/generate_ontology_negatives.py
```

### Step 4: Run Negative Generation (2 min)
```bash
./.venv/bin/python tools/generate_ontology_negatives.py
```

### Step 5: Export Corrected NPZ (2 min)
```bash
./.venv/bin/python -c "
import numpy as np
import psycopg2
import json

conn = psycopg2.connect('dbname=lnsp')
cur = conn.cursor()

cur.execute('''
    SELECT v.cpe_id, v.concept_vec, e.concept_text, v.tmd_dense
    FROM cpe_vectors v
    JOIN cpe_entry e ON v.cpe_id = e.cpe_id
    WHERE v.concept_vec IS NOT NULL
    ORDER BY v.cpe_id
''')

rows = cur.fetchall()
print(f'Exporting {len(rows)} vectors...')

cpe_ids = [r[0] for r in rows]
concept_vecs = np.array([json.loads(r[1]) for r in rows], dtype=np.float32)
concept_texts = [r[2] for r in rows]
tmd_dense = np.array([json.loads(r[3]) for r in rows], dtype=np.float32)

np.savez_compressed(
    'artifacts/ontology_4k_corrected.npz',
    vectors=concept_vecs,
    concept_vecs=concept_vecs,
    doc_ids=np.array(cpe_ids),
    cpe_ids=np.array(cpe_ids),
    concept_texts=np.array(concept_texts),
    tmd_dense=tmd_dense
)

print('✅ Saved to artifacts/ontology_4k_corrected.npz')
cur.close()
conn.close()
"
```

### Step 6: Rebuild FAISS Index (1 min)
```bash
./.venv/bin/python -c "
import faiss
import numpy as np
import json
from pathlib import Path

npz = np.load('artifacts/ontology_4k_corrected.npz')
vectors = npz['vectors'].astype('float32')

# Normalize for IP
faiss.normalize_L2(vectors)

# Build index
index = faiss.IndexFlatIP(768)
index.add(vectors)
faiss.write_index(index, 'artifacts/ontology_4k_corrected.index')

# Update metadata
meta = {
    'num_vectors': int(index.ntotal),
    'index_type': 'IndexFlatIP',
    'dimension': 768,
    'npz_path': 'artifacts/ontology_4k_corrected.npz',
    'index_path': 'artifacts/ontology_4k_corrected.index',
    'last_updated': '2025-10-02T08:00:00'
}
Path('artifacts/faiss_meta.json').write_text(json.dumps(meta, indent=2))
print('✅ FAISS index rebuilt')
"
```

### Step 7: Validate with RAG Benchmark (3 min)
```bash
export FAISS_NPZ_PATH=artifacts/ontology_4k_corrected.npz
./.venv/bin/python RAG/bench.py \
  --dataset self \
  --n 100 \
  --topk 10 \
  --backends vec,bm25

# Check results
cat RAG/results/summary_*.md | tail -10
```

**Expected:**
- P@1 > 0.95 ✅
- No more zero TMD vectors ✅
- No more empty negatives ✅

---

## ⏱️ Time Breakdown

| Task | Time | Critical |
|------|------|----------|
| Create simple TMD fixer | 15 min | ✅ YES |
| Run TMD fix | 1 min | ✅ YES |
| Create negative generator | 30 min | ⚠️ MEDIUM |
| Run negative generation | 2 min | ⚠️ MEDIUM |
| Export corrected NPZ | 2 min | ✅ YES |
| Rebuild FAISS index | 1 min | ✅ YES |
| Validate with RAG | 3 min | ✅ YES |
| **TOTAL** | **54 min** | - |

**With Option A (Simple):** ~1 hour
**With Option B (LLM):** ~2 hours

---

## ✅ Success Criteria

Before LVM training:
- [ ] Zero TMD vectors: 0 (was 1,562)
- [ ] TMD domains match sources (SWO→engineering, GO→medicine, DBpedia→art)
- [ ] Empty soft negatives: 0 (was 171)
- [ ] Empty hard negatives: 0 (was 171)
- [ ] RAG P@1 > 0.95
- [ ] Artifacts exported with correct metadata

---

## 🚀 Then Start LVM Training!

Once fixes are complete:
```bash
# Extract OCP sequences from corrected data
python src/extract_ocp_sequences.py \
  --input artifacts/ontology_4k_corrected.npz \
  --output artifacts/ocp_sequences.npz

# Train Mamba LVM
python src/train_lvm.py \
  --sequences artifacts/ocp_sequences.npz \
  --epochs 50 \
  --batch-size 32

# Evaluate
python src/eval_lvm.py \
  --model artifacts/lvm_checkpoint.pt \
  --test artifacts/ocp_sequences_test.npz
```

---

**RECOMMENDATION:** Use Option A (Simple TMD Fix) to get started quickly (1 hour), then optionally run LLM-based classification in parallel during LVM training if you want even better quality.

**Status:** READY TO EXECUTE
**Start Time:** 8:00 AM
**Finish Time:** 9:00 AM (Option A) or 10:00 AM (Option B)
**LVM Training:** 10:00 AM ✅

