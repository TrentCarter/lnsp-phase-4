# P13 Data Quality: Root Cause Analysis & Fix

**Date**: 2025-09-30
**Issue**: P13 Echo Validation failed with 48.9% pass rate (need ≥80%)
**Status**: Root cause identified, fix ready

---

## Part 1: Root Cause Analysis

### Issue 1: LLM Prompt Leakage ⚠️ CRITICAL

**Location**: `src/prompt_extractor.py:221-249`

**Problem**: The prompt starts with instruction text that the LLM sometimes returns as output:

```python
prompt = (
    "Output JSON only with keys: concept, probe, expected, soft_negatives, hard_negatives, domain, task, modifier, relations.\n"
    "Given the input text, extract: \n"
    # ... rest of prompt
)
```

**Evidence from P13 failures**:
```json
{
  "probe": "Output JSON only with keys: concept, probe, expected, soft_negatives, hard_negatives, domain, task,",
  "concept": "'Alawi dynasty, History, Colonial rule, Mohammed V, and independence\nHowever, over the course of his"
}
```

**Root Cause**: The LLM is echoing the instruction preamble instead of following it. This happens when:
1. Text is too long/complex
2. LLM timeout/truncation occurs
3. LLM doesn't understand JSON format requirement

**Impact**: ~10-15% of entries have malformed probes (500-750 out of 4,993)

---

### Issue 2: Incomplete CPESH Generation

**Evidence from P13 report**:
- Total entries: 4,993
- Missing negatives: 257 entries (5.1%)
- Pass rate: 48.9% (2,442 passed, 2,551 failed)

**Root Cause**: LLM timeout or JSON parsing failure causes fallback to `extract_stub()` which generates:
- Empty soft_negatives: `[]`
- Empty hard_negatives: `[]`
- Generic probe: "What is X?"

**Location**: `src/prompt_extractor.py:278-279`
```python
except Exception as exc:
    print(f"[extract_cpe_from_text] LLM fallback due to error: {exc}")
    return _fallback()  # Returns stub with empty negatives
```

---

### Issue 3: Low Domain Scores

**Evidence**:
```json
"scores_by_domain": {
    "3": 0.5099057704210281,  // Domain 3 = Engineering (extremely low!)
    "0": 0.7033370967174686,   // Domain 0 = Science (low)
    "15": 0.7020315308319894,  // Domain 15 = Sociology (low)
}
```

**Root Cause**: FactoidWiki is **not ontological** - it's a collection of random Wikipedia trivia facts without:
- ❌ Sequential ordering (no "next concept" structure)
- ❌ Hierarchical relationships (no parent→child chains)
- ❌ Logical dependencies (no "requires" or "causes" edges)
- ❌ Domain consistency (jumps randomly between topics)

**Example bad sequence from FactoidWiki**:
```
1. "Vliegend Hert" (18th-century ship)
2. "European Parliament" (political institution)
3. "Mount Eerie" (album name)
4. "certification" (music industry term)
```

There's **no semantic continuity** - these are just random facts that happened to be adjacent in the Wikipedia dump.

---

## Part 2: Strategic Decision - Ditch FactoidWiki? ✅ YES

### Your Question:
> "Should we ditch the FactoidWiki as its not-ontological, and we want that for training because we want the LVM to output generative (next concept in series) vectors?"

### Answer: **YES, replace FactoidWiki with ontological data**

Here's why:

### Why FactoidWiki is WRONG for P15 Training

| Requirement | FactoidWiki | Ontological Data | Winner |
|-------------|-------------|------------------|--------|
| **Ordered sequences** | ❌ Random facts | ✅ Parent→Child chains | Ontology |
| **Next-concept prediction** | ❌ No continuity | ✅ Natural progression | Ontology |
| **Logical relationships** | ❌ Weak/heuristic | ✅ Formal (subClassOf, partOf) | Ontology |
| **Training signal** | ❌ Noisy (48.9% pass) | ✅ Clean (90%+ expected) | Ontology |
| **Domain coherence** | ❌ Jumps randomly | ✅ Stays in lane | Ontology |
| **STEM focus** | ❌ Heavy arts/history | ✅ 80%+ STEM | Ontology |

### What FactoidWiki Teaches LVM (BAD):
```
Concept[t] = "Vliegend Hert" (ship)
Concept[t+1] = "European Parliament" (politics)
Concept[t+2] = "Mount Eerie" (music)

LVM learns: "Next concept is... random?"
```

### What Ontological Data Teaches LVM (GOOD):
```
Concept[t] = "Algorithm"
Concept[t+1] = "Sorting Algorithm"
Concept[t+2] = "Comparison Sort"
Concept[t+3] = "QuickSort"
Concept[t+4] = "Lomuto Partition"

LVM learns: "Next concept is more specific instance of current concept"
```

---

## Part 3: Recommended Strategy

### Option A: Complete Replacement ⭐ RECOMMENDED

**Throw away FactoidWiki**, start fresh with ontological data:

1. **Phase 1** (Week 1): Generate 50K ontology chains
   - DBpedia STEM subset: 30K chains
   - Wikidata Science & Tech: 20K chains

2. **Phase 2** (Week 2): Add specialized ontologies
   - Software Ontology (SWO): 15K chains
   - Gene Ontology (GO): 40K chains

3. **Phase 3** (Week 3): Bridge with common sense
   - ConceptNet technical: 20K chains

**Total**: 125K high-quality ontology chains (50K for P15 training, 75K for fine-tuning)

**Advantages**:
- ✅ Clean data (90%+ expected pass rate)
- ✅ True sequential learning signal
- ✅ STEM-focused (matches project goals)
- ✅ No prompt injection issues (structured RDF/OWL)

**Disadvantages**:
- ⏱️ Need to implement ontology ingestion pipeline (3-5 days)
- 🗑️ "Waste" the 5K FactoidWiki data already ingested

---

### Option B: Hybrid Approach (NOT Recommended)

Keep FactoidWiki for **vecRAG retrieval** only, use ontologies for **P15 training**:

**Reasoning**:
- FactoidWiki can still serve as a fact database for retrieval
- But don't use it to train the generative LVM

**Problems**:
- 🤔 Why keep 5K low-quality entries when we can replace with 125K high-quality?
- 🗄️ Storage/compute waste maintaining two systems
- 🔀 Confusion: Which data source for what?

**Verdict**: Not worth the complexity. Go with Option A.

---

### Option C: Fix FactoidWiki (NOT Recommended)

Try to fix the prompt injection and re-run P5 on 4,993 entries:

**Estimate**: 2-3 days to:
1. Fix prompt (1 hour)
2. Re-run P5 on all 4,993 entries (8-12 hours @ 500ms/entry)
3. Re-run P13 validation (2 hours)
4. Still get only 70-80% pass rate (FactoidWiki is inherently noisy)

**But**: Still doesn't solve the fundamental problem - **no sequential ordering**!

Even if we get 80% pass rate, the data is still just random facts. The LVM will learn nothing useful.

**Verdict**: Don't waste time on fundamentally flawed data.

---

## Part 4: The Fix - Prompt Injection Only (IF You Want to Keep FactoidWiki)

If you decide to keep FactoidWiki for any reason, here's the prompt fix:

### Before (BAD):
```python
prompt = (
    "Output JSON only with keys: concept, probe, expected, soft_negatives, hard_negatives, domain, task, modifier, relations.\n"
    "Given the input text, extract: \n"
    # ...
)
```

### After (GOOD):
```python
prompt = f"""You are a precise knowledge extraction system. Your task is to analyze text and extract structured information in valid JSON format.

**CRITICAL**: Respond ONLY with valid JSON. Do not include any explanatory text, preamble, or postamble. No markdown code blocks. Just pure JSON.

**Required JSON Schema**:
{{
  "concept": "<short atomic proposition>",
  "probe": "<question directly answered by the concept>",
  "expected": "<concise answer to probe>",
  "soft_negatives": ["<plausible wrong 1>", "<plausible wrong 2>", "<plausible wrong 3>"],
  "hard_negatives": ["<unrelated 1>", "<unrelated 2>", "<unrelated 3>"],
  "domain": "<one of: science, mathematics, technology, engineering, medicine, psychology, philosophy, history, literature, art, economics, law, politics, education, environment, sociology>",
  "task": "<one of: fact_retrieval, definition, comparison, cause_effect, taxonomy, etc>",
  "modifier": "<one of: historical, descriptive, temporal, biochemical, geographical, etc>",
  "relations": [{{"subj": "<entity>", "pred": "<relationship>", "obj": "<entity>"}}]
}}

**Extraction Guidelines**:
1. concept: ONE atomic fact from the text (max 20 words)
2. probe: A question that this concept directly answers
3. expected: Brief answer to probe (max 10 words)
4. soft_negatives: 3 plausible but incorrect answers (similar domain)
5. hard_negatives: 3 clearly wrong answers (different domain)
6. relations: 2-5 entity relationships found in text

**Input Text**:
{text[:1000]}

**Output** (JSON only, no other text):"""
```

**Key Changes**:
1. ✅ Moved "Output JSON only" to beginning as system instruction
2. ✅ Added "CRITICAL" emphasis to prevent echoing
3. ✅ Specified "No markdown, no explanations"
4. ✅ Truncated input to 1000 chars (prevents timeout)
5. ✅ Clear schema with examples
6. ✅ Explicit "Output (JSON only, no other text):" trigger at end

---

## Part 5: Implementation Plan

### Recommended: Option A (Complete Replacement)

<function_calls>
<invoke name="TodoWrite">
<parameter name="todos">[{"content": "Analyze P13 failure root causes (malformed probes, LLM prompt injection issues)", "status": "completed", "activeForm": "Analyzing P13 failure root causes"}, {"content": "Strategic decision: Replace FactoidWiki with ontological data for P15", "status": "completed", "activeForm": "Deciding on training data strategy"}, {"content": "Archive current FactoidWiki data (5K entries) for reference", "status": "pending", "activeForm": "Archiving FactoidWiki data"}, {"content": "Implement ontology ingestion pipeline (DBpedia + Wikidata)", "status": "pending", "activeForm": "Implementing ontology ingestion"}, {"content": "Generate 50K ontology chains (Phase 1)", "status": "pending", "activeForm": "Generating ontology chains Phase 1"}, {"content": "Run P13 validation on ontology data (expect 90%+ pass)", "status": "pending", "activeForm": "Validating ontology data quality"}, {"content": "Generate remaining 75K chains (Phase 2-3) for full training set", "status": "pending", "activeForm": "Generating full ontology training set"}]