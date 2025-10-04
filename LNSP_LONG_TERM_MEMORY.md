# 🧠 LNSP Long-Term Memory: Core Principles & Invariants

**Purpose**: Permanent record of fundamental design decisions that must NEVER be violated
**Priority**: CRITICAL - These are not preferences, they are REQUIREMENTS
**Last Updated**: October 3, 2025

---

## 🚨 CARDINAL RULES (NEVER VIOLATE)

### 1. Data Synchronization is Sacred

**PRINCIPLE**: PostgreSQL + Neo4j + FAISS must ALWAYS be synchronized atomically

```
PostgreSQL ↔ Neo4j ↔ FAISS
   ▲           ▲        ▲
   └───────────┴────────┘
   MUST ALL MATCH EXACTLY!
```

**WHY**: GraphRAG depends on concept[i] in vectors matching Concept[i] in graph. If desynchronized, GraphRAG fails completely (0% accuracy).

**ENFORCEMENT**:
- ✅ ALWAYS use `./scripts/ingest_10k.sh` (writes to all three)
- ❌ NEVER run `tools/regenerate_*_vectors.py` (only updates PostgreSQL)
- ✅ ALWAYS verify sync with `./scripts/verify_data_sync.sh` before GraphRAG operations
- ✅ ALWAYS ingest with `--write-pg --write-neo4j --faiss-out` flags together

**CONSEQUENCE OF VIOLATION**: Total system failure, GraphRAG returns 0 neighbors, 0% improvement

**REFERENCE**: `docs/CRITICAL_GraphRAG_Data_Synchronization.md`

---

### 2. NO FactoidWiki Data - ONTOLOGIES ONLY

**PRINCIPLE**: LNSP is an ontology-based system. FactoidWiki is NOT ontological and MUST NOT be used for training.

**FORBIDDEN DATASETS**:
- ❌ FactoidWiki (album names, people, places - NOT ontological)
- ❌ Wikipedia articles (too noisy, not structured)
- ❌ General web scrapes (no semantic structure)

**REQUIRED DATASETS**:
- ✅ **SWO** (Software Ontology) - Bioinformatics software concepts
- ✅ **GO** (Gene Ontology) - Biological processes, functions
- ✅ **ConceptNet** - Structured concept relationships
- ✅ **DBpedia** - Structured knowledge base
- ✅ Domain-specific ontologies (medical, scientific, technical)

**WHY**:
1. **Training LVM**: FactoidWiki has no semantic structure → LVM learns noise, not concepts
2. **vecRAG**: Ontology concepts have clear boundaries → better retrieval
3. **GraphRAG**: Ontology relationships are semantic → meaningful graph walks
4. **Inference**: LVM outputs must map to ontology concepts → structured decoding

**ENFORCEMENT**:
- ✅ Ingestion scripts MUST reject `dataset_source` containing "factoid" or "wiki"
- ✅ Add validation check: `if 'factoid' in dataset_source.lower(): raise ValueError(...)`
- ✅ Document in ALL README files: "NO FACTOIDWIKI EVER"

**CONSEQUENCE OF VIOLATION**: Wasted training compute, poor LVM performance, unusable for scientific/technical domains

**REFERENCE**: This document, section 2

---

### 3. Complete Data Pipeline: CPESH + TMD + Graph

**PRINCIPLE**: Every ingestion run must create ALL data artifacts for both vecRAG+GraphRAG AND LVM training

```
Source Data (Ontology)
        │
        ▼
   ┌────────────────────────┐
   │  CPESH Extraction      │  ← LLM generates Concept/Probe/Expected/Soft/Hard
   │  (LightRAG)            │
   └───────────┬────────────┘
               │
        ┌──────┴───────┐
        │              │
        ▼              ▼
    ┌───────┐      ┌──────────┐
    │  TMD  │      │  Graph   │
    │ (16D) │      │ (Neo4j)  │
    └───┬───┘      └────┬─────┘
        │               │
        └───────┬───────┘
                ▼
        ┌───────────────┐
        │  PostgreSQL   │  ← cpe_entry (CPESH + TMD)
        │  Neo4j        │  ← Concept nodes + RELATES_TO edges
        │  FAISS        │  ← 784D vectors (16D TMD + 768D semantic)
        └───────────────┘
```

**REQUIRED ARTIFACTS PER INGESTION**:
1. **CPESH** (Concept-Probe-Expected-Soft-Hard negatives)
   - Stored in PostgreSQL `cpe_entry.soft_negatives`, `hard_negatives`
   - Used for contrastive training of LVM

2. **TMD** (Task-Method-Domain 16D encoding)
   - Stored in PostgreSQL `cpe_vectors.tmd_dense`
   - Concatenated with 768D semantic → 784D fused vectors
   - Used for semantic routing and retrieval

3. **Graph** (Neo4j concept relationships)
   - Concept nodes with `text`, `cpe_id`, `tmd_bits`
   - RELATES_TO edges with confidence scores
   - SHORTCUT_6DEG edges for graph walks
   - Used for GraphRAG neighbor expansion

4. **Vectors** (FAISS 784D embeddings)
   - NPZ file: `[16D TMD | 768D GTR-T5]` per concept
   - FAISS index for fast similarity search
   - Used for vecRAG dense retrieval

**WHY ALL FOUR**:
- **vecRAG**: Needs vectors (FAISS) + TMD routing
- **GraphRAG**: Needs vectors (FAISS) + graph (Neo4j) synchronized
- **LVM Training**: Needs CPESH for contrastive learning + graph for sequence prediction
- **LVM Inference**: Needs FAISS index as decoder (vector → nearest concept → text)

**ENFORCEMENT**:
- ✅ Default to `--write-pg --write-neo4j --faiss-out` in ALL ingestion scripts
- ✅ Validate CPESH has non-empty soft_negatives, hard_negatives arrays
- ✅ Verify Neo4j receives graph relationships (not just concept nodes)
- ✅ Check FAISS NPZ has both `tmd_dense` and `vectors` arrays

**REFERENCE**: `docs/PRDs/PRD_KnownGood_vecRAG_Data_Ingestion.md`

---

### 4. LVM Architecture: Tokenless Vector-Native

**PRINCIPLE**: The LVM processes 768D/784D vectors directly, NO tokens involved

```
Input Text → GTR-T5 (768D) → [Optional TMD 16D] → LVM (Mamba-2) → 768D Output → Decoder
             Frozen                                 12 layers                    ↓
                                                                          FAISS + Vec2Text
```

**DECODER IS vecRAG+GraphRAG**:
1. LVM outputs 768D vector
2. FAISS finds nearest neighbors (threshold 0.85)
3. GraphRAG expands via Neo4j relationships
4. Vec2Text handles out-of-distribution vectors
5. LLM smoother optional for fluency

**WHY**:
- **24x faster** than token-based LLMs (no tokenization overhead)
- **Infinite vocabulary** (any 768D vector = valid concept)
- **Perfect RAG alignment** (same vector space as retrieval)
- **No hallucination** (always retrieve or decode from graph)

**TRAINING DATA**:
- **CPESH** (4,500 contrastive pairs) → learn concept boundaries
- **GWOM** (10K graph walks) → learn concept transitions
- **Objective**: Predict next concept vector in sequence

**INFERENCE**:
- Input: Query text → 768D vector
- LVM: Processes sequence, outputs next concept vector
- Decoder: vecRAG finds nearest concept + GraphRAG expands context
- Output: Grounded concept text (not hallucinated)

**REFERENCE**: `docs/TOKENLESS_MAMBA_ARCHITECTURE.md`, `docs/PRDs/PRD_P15_Latent_LVM_Implementation_Plan.md`

---

### 5. Six Degrees of Separation + Shortcuts

**PRINCIPLE**: Use 6-degrees theory with 0.5-3% shortcut edges to achieve ≤6-hop convergence from any question to answer

```
Without Shortcuts:                With Shortcuts (1% of edges):
A → B → C → D → E → F            A ─────────────→ F  (1 hop via shortcut)
    (6 hops)                      A → B → C → F       (3 hops via shortcut at C)
```

**MATHEMATICAL BASIS**:
- Small-world networks: Average path length L ~ log(N) / log(k)
- Adding just 1% random shortcuts reduces L from O(N) to O(log N)
- For N=10,000 concepts, L drops from ~5,000 to ~4 hops

**IMPLEMENTATION**:
- **SHORTCUT_6DEG** edges in Neo4j
- Created via graph analysis: identify concepts 4-6 hops apart with semantic similarity >0.7
- Target: 0.5-3% of total edges (~50-300 shortcuts for 10K concepts)
- Edge properties: `{type: "SHORTCUT_6DEG", distance: 5, similarity: 0.82}`

**WHY**:
1. **Fast convergence**: Question → Answer in ≤6 hops (proven by Milgram/Watts-Strogatz)
2. **Efficient graph walks**: GWOM sequences stay focused, not wandering
3. **Analogical reasoning**: Shortcuts connect distant but related concepts
4. **Scalability**: Works for 1M+ concepts without explosion

**ENFORCEMENT**:
- ✅ Run `./scripts/generate_6deg_shortcuts.sh` after every ingestion
- ✅ Target: 1% of RELATES_TO edges should be SHORTCUT_6DEG
- ✅ Validate shortcuts: `cypher-shell "MATCH ()-[r:SHORTCUT_6DEG]->() RETURN count(r)"`
- ✅ Check average path length: Should be ≤6 between random concept pairs

**REFERENCE**: `src/graph/add_6deg_shortcuts.py`, Watts-Strogatz small-world model

---

## 📋 Mandatory Checklist for Every Data Operation

Before ANY ingestion/training/inference run, verify:

### Pre-Ingestion
- [ ] Source data is ONTOLOGY (not FactoidWiki!)
- [ ] LLM is running (Ollama + Llama 3.1:8b)
- [ ] PostgreSQL + Neo4j services are up
- [ ] Previous data is cleared if doing fresh ingest

### During Ingestion
- [ ] Using `--write-pg --write-neo4j --faiss-out` flags
- [ ] CPESH extraction generates non-empty negatives arrays
- [ ] TMD codes are deterministic (same concept → same 16D vector)
- [ ] Neo4j receives both Concept nodes AND relationships
- [ ] FAISS NPZ has 784D vectors (16D TMD + 768D semantic)

### Post-Ingestion
- [ ] Run `./scripts/verify_data_sync.sh` → ✅ ALL CHECKS PASSED
- [ ] PostgreSQL count = Neo4j count = FAISS count
- [ ] Sample concept exists in all three stores
- [ ] Neo4j has >100 relationships (not just nodes)
- [ ] Generate SHORTCUT_6DEG edges (target 1% of edges)
- [ ] Build FAISS index: `make build-faiss`

### Before GraphRAG/Training
- [ ] Verify sync again: `./scripts/verify_data_sync.sh`
- [ ] Check shortcuts exist: `cypher-shell "MATCH ()-[r:SHORTCUT_6DEG]->() RETURN count(r)"`
- [ ] Validate CPESH completeness: `psql lnsp -c "SELECT count(*) FROM cpe_entry WHERE jsonb_array_length(soft_negatives) > 0"`

---

## 🎯 Success Metrics (Know Your Numbers)

### vecRAG Baseline
- **P@1**: 54.4% (current)
- **P@5**: 77.8%
- **Latency**: 0.04ms mean

### GraphRAG Target
- **P@1**: 60-70% (+10-15% over vecRAG)
- **Graph neighbors**: 1-10 per query (NOT 0!)
- **Latency**: <5ms (100x slower than vecRAG but still fast)

### LVM Training Target
- **CPESH loss**: <0.1 (contrastive triplet loss)
- **GWOM MSE**: <0.05 (next vector prediction)
- **Echo score**: >0.85 (predicted ≈ expected)

### LVM Inference Target
- **FAISS recall@1**: >0.80 (vector → concept accuracy)
- **GraphRAG boost**: +10% over pure FAISS
- **Vec2Text fallback**: <20% of queries (most should hit FAISS)

---

## 🚫 Common Mistakes (DO NOT DO THESE)

### ❌ Mistake 1: Using FactoidWiki
```bash
# WRONG - FactoidWiki is not ontological!
python -m src.ingest_factoid --file-path data/factoidwiki_10k.jsonl
```

### ❌ Mistake 2: Updating Stores Independently
```bash
# WRONG - Only updates PostgreSQL!
python tools/regenerate_all_tmd_vectors.py

# WRONG - Only updates Neo4j!
cypher-shell "CREATE (c:Concept {text: 'foo'})"
```

### ❌ Mistake 3: Skipping Sync Verification
```bash
# WRONG - No verification before GraphRAG!
./scripts/run_graphrag_benchmark.sh

# CORRECT - Always verify first!
./scripts/verify_data_sync.sh && ./scripts/run_graphrag_benchmark.sh
```

### ❌ Mistake 4: No Shortcuts in Graph
```bash
# WRONG - Graph has no long-range connections!
# Average path length will be O(N) instead of O(log N)

# CORRECT - Generate shortcuts after ingestion
./scripts/generate_6deg_shortcuts.sh
```

### ❌ Mistake 5: Empty CPESH Negatives
```bash
# Check for this bug:
psql lnsp -c "SELECT count(*) FROM cpe_entry WHERE soft_negatives = '[]'::jsonb;"
# If count > 0, CPESH extraction failed - re-ingest with real LLM!
```

### ❌ Mistake 6: Hardcoded dataset_source Labels (FIXED Oct 4, 2025)
```python
# WRONG - Hardcoded label in process_sample()!
cpe_record = {
    "dataset_source": "factoid-wiki-large",  # ❌ Wrong for ontology data!
}

# CORRECT - Use parameter
def process_sample(..., dataset_source: str = "factoid-wiki-large"):
    cpe_record = {
        "dataset_source": dataset_source,  # ✅ Pass "ontology-swo" etc.
    }
```

### ❌ Mistake 7: Forgetting to Save FAISS Vectors (FIXED Oct 4, 2025)
```python
# WRONG - No save() call after processing!
for sample in samples:
    faiss_db.add_vector(cpe_record)
# Files never written to disk!

# CORRECT - Call save() after loop
for sample in samples:
    faiss_db.add_vector(cpe_record)
faiss_db.save()  # ✅ Persist to NPZ file
```

---

## 📚 Documentation Hierarchy

### Level 1: THIS FILE (Long-Term Memory)
Read first. Contains immutable principles.

### Level 2: Critical Operations
- `docs/CRITICAL_GraphRAG_Data_Synchronization.md` - Data sync rules
- `docs/FUNDAMENTAL_PRINCIPLES.md` - Design decisions
- `CLAUDE.md` - Instructions for Claude Code

### Level 3: Implementation Guides
- `docs/PRDs/PRD_KnownGood_vecRAG_Data_Ingestion.md` - Ingestion procedure
- `docs/TOKENLESS_MAMBA_ARCHITECTURE.md` - LVM architecture
- `docs/GraphRAG_Implementation.md` - GraphRAG technical details

### Level 4: Quick References
- `docs/GraphRAG_QuickStart.md` - 30-second test
- `QUICKSTART_ONTOLOGY_PIPELINE.md` - Fast setup
- `README.md` - Project overview

---

## 🔄 Change Log

### October 4, 2025 - Critical Fixes Applied
- **Fixed**: `dataset_source` labeling bug in `ingest_factoid.py`
  - Hardcoded `"factoid-wiki-large"` → Parameterized with `dataset_source` argument
  - Ontology ingestions now use `ontology-{source}` labels (e.g., `ontology-swo`)
- **Fixed**: Missing `faiss_db.save()` call in `ingest_ontology_simple.py`
  - FAISS NPZ files now created automatically with `--write-faiss` flag
- **Fixed**: Validation script false positives in `validate_no_factoidwiki.sh`
  - Now checks concept content patterns, not just labels
- **Created**: Comprehensive fix documentation in `docs/FIXES_Oct4_2025_FactoidWiki_Labeling.md`

**Why**: 6K overnight ingestion completed but had mislabeled data. All 4,484 concepts were ontological but labeled `factoid-wiki-large`. Fixes prevent recurrence.

### October 3, 2025 - Initial Creation
- Documented Cardinal Rules 1-5
- Established NO FACTOIDWIKI policy
- Defined complete data pipeline requirements
- Documented 6-degrees shortcuts theory
- Created mandatory checklist
- Defined success metrics

**Why**: Oct 2-3 incident showed that critical principles weren't documented. GraphRAG failed due to data desynchronization. This file ensures it never happens again.

---

## 🎓 For Claude Code / Future Developers

**When in doubt, re-read this file.**

These are not suggestions. These are REQUIREMENTS that the entire system depends on.

If you violate any Cardinal Rule, the system will break in subtle ways that are hard to debug. Follow the checklist. Verify synchronization. Use ontologies only. Generate shortcuts. Your future self will thank you.

**Remember**: Fast re-ingestion is better than slow debugging of desynchronized data.

---

**END OF LONG-TERM MEMORY**

_This document must be updated whenever new fundamental principles are discovered._
_Last review: October 4, 2025_
_Last update: October 4, 2025 - Added fixes for dataset_source labeling and FAISS save issues_
