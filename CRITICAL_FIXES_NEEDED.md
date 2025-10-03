# CRITICAL FIXES NEEDED - Consultant Review

**Date:** October 2, 2025 (Morning)
**Priority:** HIGH - Must fix before LVM training
**Source:** Consultant feedback on ontology ingestion

---

## 🚨 Issues Identified

### 1. TMD Classification Drift (HIGH PRIORITY)
**Problem:**
- **1,562/4,484 (34.8%)** have all-zero TMD vectors
- Remaining TMDs misclassify domains (BioConductor→engineering, legal→geography)
- Hard-coded domain/task/modifier don't match ontology semantics

**Evidence:**
```
metaArray (BioConductor) → engineering/fact_retrieval/ethical ❌
Philosoph (DBpedia)      → geography/entailment/robust ❌
```

**Root Cause:**
- TMD classifier trained on FactoidWiki (general knowledge)
- Doesn't understand domain-specific ontology concepts (software, biology, people)
- Applies engineering/fact_retrieval as default fallback

**Impact:**
- LVM will learn incorrect domain associations
- Retrieval will mix incompatible domains
- Training efficiency degraded by misaligned metadata

---

### 2. Missing Soft/Hard Negatives (MEDIUM PRIORITY)
**Problem:**
- **171/4,484 (3.8%)** have empty soft/hard negative arrays
- Code hard-codes empty arrays for ontology chains (src/ingest_ontology.py:83-88)

**Evidence:**
```python
# Current code (ingest_ontology.py)
soft_negatives = []  # ← Hard-coded empty!
hard_negatives = []  # ← Hard-coded empty!
```

**Impact:**
- Contrastive learning degraded (no negative examples)
- Can't distinguish similar concepts
- Reduced retrieval precision

---

### 3. Ontology Order (VERIFIED CORRECT ✅)
**Status:** NO ISSUE - Consultant verified all chains are root→leaf ordered
- SWO: Checked sample at line 1310 ✅
- GO: Checked sample at line 115 ✅
- DBpedia: Checked sample at line 13 ✅

---

## 🔧 Fix Plan

### Fix 1: Retrain TMD Classifier for Ontology Domains

#### Option A: Domain-Specific Mappings (FAST - 1 hour)
**Approach:** Rule-based domain assignment
```python
def classify_ontology_domain(concept_text, source):
    if source == 'swo':
        return Domain.SCIENCE, Task.CODE_GENERATION, Modifier.ROBUST
    elif source == 'go':
        return Domain.BIOLOGY, Task.FACT_RETRIEVAL, Modifier.MULTILINGUAL
    elif source == 'dbpedia':
        return Domain.ARTS, Task.FACT_RETRIEVAL, Modifier.NEUTRAL
```

**Pros:**
- Immediate fix
- Accurate for known sources
- No retraining needed

**Cons:**
- Not scalable
- Doesn't learn from concept text
- Requires manual mapping

#### Option B: Fine-tune TMD Classifier (BETTER - 3 hours)
**Approach:** Few-shot learning on ontology examples
```python
# Create training set from ontology concepts
train_data = [
    ("BioConductor package metaArray", Domain.SCIENCE, Task.CODE_GENERATION),
    ("Gene Ontology: protein binding", Domain.BIOLOGY, Task.FACT_RETRIEVAL),
    ("DBpedia: Philosoph", Domain.ARTS, Task.FACT_RETRIEVAL),
    ...
]

# Fine-tune existing TMD classifier
model.fine_tune(train_data, epochs=10)
```

**Pros:**
- Learns concept semantics
- Generalizes to new concepts
- Improves over time

**Cons:**
- Requires labeled examples
- Longer implementation
- Need validation set

#### Option C: LLM-Based Classification (BEST - 2 hours)
**Approach:** Use local Llama 3.1 for classification
```python
def classify_with_llm(concept_text):
    prompt = f"""
    Classify this concept into domain, task, modifier:
    Concept: {concept_text}

    Domains: science, engineering, arts, biology, geography, law
    Tasks: fact_retrieval, code_generation, entailment, qa
    Modifiers: robust, multilingual, ethical, neutral

    Output JSON: {{"domain": "...", "task": "...", "modifier": "..."}}
    """
    response = llama_client.call(prompt)
    return parse_json(response)
```

**Pros:**
- High accuracy (Llama 3.1:8b is smart)
- No training needed
- Handles edge cases
- Already have Ollama running

**Cons:**
- Slower (50ms per concept)
- Need batch processing
- Total time: 4,484 × 50ms = 4 minutes (acceptable!)

**RECOMMENDATION: Option C (LLM-based) - Best accuracy/effort tradeoff**

---

### Fix 2: Generate Soft/Hard Negatives

#### Approach: Sibling Concept Sampling
```python
def generate_negatives(chain, all_chains):
    concept = chain[-1]  # Target concept

    # Soft negatives: Same parent, different children (siblings)
    if len(chain) > 1:
        parent = chain[-2]
        siblings = find_concepts_with_parent(parent, all_chains)
        soft_negatives = [s for s in siblings if s != concept][:5]

    # Hard negatives: Same domain, different subtree
    domain = extract_domain(concept)
    hard_negatives = find_concepts_in_domain_excluding_chain(
        domain, chain, all_chains
    )[:5]

    return soft_negatives, hard_negatives
```

**Implementation:**
1. Build parent→children index from chains
2. For each concept, find siblings (share parent)
3. Find cousins (same domain, different parent)
4. Store in soft/hard negative arrays

**Time:** 1-2 hours

---

### Fix 3: Regenerate Artifacts

**Steps:**
1. Re-run TMD classification with LLM (4 minutes)
2. Generate soft/hard negatives (10 minutes)
3. Update PostgreSQL cpe_entry table
4. Export new NPZ with corrected TMD vectors
5. Rebuild FAISS index
6. Update metadata

**Time:** 30 minutes

---

## 📊 Implementation Plan

### Phase 1: TMD Fix (Morning - 2 hours)
```bash
# 1. Create LLM-based TMD classifier
python tools/fix_ontology_tmd.py --mode classify --llm ollama

# 2. Update database
python tools/fix_ontology_tmd.py --mode update-db

# 3. Verify classifications
python tools/fix_ontology_tmd.py --mode verify
```

**Expected Output:**
- 0 zero TMD vectors (was 1,562)
- Correct domain alignment (SWO→science, GO→biology, DBpedia→arts)

### Phase 2: Negatives Fix (Morning - 1.5 hours)
```bash
# 1. Generate negatives from chain structure
python tools/generate_ontology_negatives.py --input data/ontology_chains/*.jsonl

# 2. Update database
python tools/generate_ontology_negatives.py --mode update-db

# 3. Verify coverage
python tools/generate_ontology_negatives.py --mode verify
```

**Expected Output:**
- 0 empty soft/hard negatives (was 171)
- 5 soft negatives per concept (siblings)
- 5 hard negatives per concept (cousins)

### Phase 3: Regenerate Artifacts (Morning - 30 min)
```bash
# 1. Export corrected NPZ
python tools/export_ontology_npz.py --output artifacts/ontology_4k_corrected.npz

# 2. Rebuild FAISS index
python src/faiss_index.py \
  --npz artifacts/ontology_4k_corrected.npz \
  --index artifacts/ontology_4k_corrected.index \
  --type flat \
  --metric ip

# 3. Update metadata
python tools/update_faiss_meta.py --index artifacts/ontology_4k_corrected.index
```

### Phase 4: Validate (Morning - 15 min)
```bash
# Run RAG benchmark on corrected data
export FAISS_NPZ_PATH=artifacts/ontology_4k_corrected.npz
./test_rag_tomorrow.sh
```

**Expected Results:**
- P@1 > 0.95 (unchanged, retrieval quality maintained)
- TMD diversity increased (no more engineering-only)
- Soft/hard negatives populated

---

## 🎯 Success Criteria

### Before Fixes
- ❌ 1,562/4,484 (34.8%) zero TMDs
- ❌ TMD domains misaligned (BioConductor→engineering)
- ❌ 171/4,484 (3.8%) empty negatives
- ⚠️ Risk of training on incorrect metadata

### After Fixes
- ✅ 0 zero TMDs
- ✅ TMD domains aligned (SWO→science, GO→biology, DBpedia→arts)
- ✅ 0 empty negatives (5 soft + 5 hard per concept)
- ✅ Ready for high-quality LVM training

---

## ⏱️ Time Estimate

| Task | Time | Priority |
|------|------|----------|
| Fix 1: LLM-based TMD classifier | 2 hours | HIGH |
| Fix 2: Generate negatives | 1.5 hours | MEDIUM |
| Fix 3: Regenerate artifacts | 30 min | HIGH |
| Fix 4: Validate with RAG test | 15 min | HIGH |
| **TOTAL** | **4 hours 15 min** | - |

**Timeline:**
- Start: 8:00 AM
- Finish: 12:15 PM
- **LVM training start: 1:00 PM** ✅

---

## 📝 Implementation Scripts Needed

### 1. `tools/fix_ontology_tmd.py`
```python
# LLM-based TMD classification for ontology concepts
# - Batch process 4,484 concepts
# - Use Ollama Llama 3.1:8b
# - Update PostgreSQL cpe_vectors table
```

### 2. `tools/generate_ontology_negatives.py`
```python
# Generate soft/hard negatives from chain structure
# - Build parent→children index
# - Sample siblings (soft) and cousins (hard)
# - Update PostgreSQL cpe_entry table
```

### 3. `tools/export_ontology_npz.py`
```python
# Export corrected data to NPZ
# - Include fixed TMD vectors
# - Include all metadata
# - Compatible with RAG/bench.py
```

---

## 🚀 Next Steps

1. **Acknowledge consultant feedback** ✅
2. **Implement Fix 1 (TMD)** - Start immediately
3. **Implement Fix 2 (Negatives)** - Parallel if possible
4. **Regenerate artifacts** - After fixes complete
5. **Validate with RAG** - Before LVM training
6. **Proceed to LVM training** - With confidence!

---

## ✅ Consultant Sign-Off Checklist

Before proceeding to LVM training, verify:
- [ ] Zero TMD vectors: 0 (was 1,562)
- [ ] TMD domain accuracy: Spot-check 20 examples
- [ ] Empty soft negatives: 0 (was 171)
- [ ] Empty hard negatives: 0 (was 171)
- [ ] RAG P@1 score: > 0.95
- [ ] Artifacts regenerated with correct metadata

---

**Status:** READY TO IMPLEMENT
**Priority:** HIGH - Block LVM training until fixed
**Owner:** Claude Code
**ETA:** 4 hours 15 minutes

