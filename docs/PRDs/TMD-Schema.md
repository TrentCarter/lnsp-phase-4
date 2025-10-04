# TMD Schema (Domain-Task-Modifier)

**IMPORTANT NAMING CLARIFICATION**:
- **Historically called**: "TMD" (Task-Modifier-Domain)
- **Actual implementation order**: `(domain, task, modifier)` - should be "DTM"!
- **Convention**: We use "TMD" as the name but parameters are ALWAYS `(domain, task, modifier)`
- **Code signature**: `pack_tmd(domain: int, task: int, modifier: int)`
- **Bit layout**: `[Domain 4b | Task 5b | Modifier 6b]`

---

## Ranges (0-indexed)

- **Domains**: 0-15 (16 total)
- **Tasks**: 0-31 (32 total)
- **Modifiers**: 0-63 (64 total)
- **Total combinations**: 16 × 32 × 64 = **32,768 unique TMD codes**

---

## 🌍 Domains (0-15, 16 total)

These represent broad semantic territories—ideal for clustering and routing.

```
0: Science
1: Mathematics
2: Technology
3: Engineering
4: Medicine
5: Psychology
6: Philosophy
7: History
8: Literature
9: Art
10: Economics
11: Law
12: Politics
13: Education
14: Environment
15: Software
```

---

## 🧠 Tasks (0-31, 32 total)

These reflect cognitive or linguistic operations—perfect for expert specialization.

```
0: Fact Retrieval
1: Definition Matching
2: Analogical Reasoning
3: Causal Inference
4: Classification
5: Entity Recognition
6: Relationship Extraction
7: Schema Adherence
8: Summarization
9: Paraphrasing
10: Translation
11: Sentiment Analysis
12: Argument Evaluation
13: Hypothesis Testing
14: Code Generation
15: Function Calling
16: Mathematical Proof
17: Diagram Interpretation
18: Temporal Reasoning
19: Spatial Reasoning
20: Ethical Evaluation
21: Policy Recommendation
22: Roleplay Simulation
23: Creative Writing
24: Instruction Following
25: Error Detection
26: Output Repair
27: Question Generation
28: Conceptual Mapping
29: Knowledge Distillation
30: Tool Use
31: Prompt Completion
```

---

## 🎨 Modifiers (0-63, 64 total)

These act as semantic adjectives—great for embedding nuance and routing precision.

```
0: Biochemical
1: Evolutionary
2: Computational
3: Logical
4: Ethical
5: Historical
6: Legal
7: Philosophical
8: Emotional
9: Technical
10: Creative
11: Abstract
12: Concrete
13: Visual
14: Auditory
15: Spatial
16: Temporal
17: Quantitative
18: Qualitative
19: Procedural
20: Declarative
21: Comparative
22: Analogical
23: Causal
24: Hypothetical
25: Experimental
26: Narrative
27: Descriptive
28: Prescriptive
29: Diagnostic
30: Predictive
31: Reflective
32: Strategic
33: Tactical
34: Symbolic
35: Functional
36: Structural
37: Semantic
38: Syntactic
39: Pragmatic
40: Normative
41: Statistical
42: Probabilistic
43: Deterministic
44: Stochastic
45: Modular
46: Hierarchical
47: Distributed
48: Localized
49: Global
50: Contextual
51: Generalized
52: Specialized
53: Interdisciplinary
54: Multimodal
55: Ontological
56: Epistemic
57: Analog-sensitive
58: Schema-bound
59: Role-based
60: Feedback-driven
61: Entailment-aware
62: Alignment-focused
63: Compression-optimized
```

---

## LLM-Based TMD Extraction Examples

### Test Results (Llama 3.1:8b, 2025-10-04)

**Format**: `(domain, task, modifier)` = (Domain Name, Task Name, Modifier Name)

| Concept | TMD Code | Interpretation | Quality |
|---------|----------|----------------|---------|
| "software" | (2, 5, 9) | Technology / Entity Recognition / Technical | ✅ Excellent |
| "Gene Ontology" | (4, 5, 55) | Medicine / Entity Recognition / **Ontological** | ✅ Perfect! |
| "Python programming language" | (2, 14, 9) | Technology / **Code Generation** / Technical | ✅ Excellent |
| "World War II" | (7, 6, 45) | History / Relationship Extraction / Modular | ✅ Great |
| "cardiac arrest" | (4, 5, 9) | Medicine / Entity Recognition / Technical | ✅ Good |

### Pattern-Based Extraction Problems (OLD SYSTEM)

**Before LLM extraction** (using `src/tmd_extractor_v2.py`):

| Concept | TMD Code | Interpretation | Quality |
|---------|----------|----------------|---------|
| "oxidoreductase activity" | (2, 1, 1) | Science / Default / Biochemical | ❌ Generic |
| "material entity" | (2, 1, 1) | Science / Default / Biochemical | ❌ Generic |
| "software" | (1, 1, 1) | Technology / Default / **Biochemical** | ❌ WRONG! |
| "Gene Ontology" | (2, 1, 1) | Science / Default / Biochemical | ❌ Generic |
| "biochemical process" | (2, 4, 1) | Science / Classification / Biochemical | ⚠️ Partial |

**Problems with pattern matching**:
- ❌ Assigns "Biochemical" modifier to **everything** (including "software"!)
- ❌ Uses only ~5 out of 32,768 possible TMD codes (<0.02% utilization)
- ❌ No discriminative power for retrieval/re-ranking

---

## LLM Prompt for TMD Extraction

See: `src/llm_tmd_extractor.py`

**Key requirements**:
1. Analyze concept semantically (not just keyword matching)
2. Choose BEST matching codes (not defaults!)
3. Return ONLY three numbers: `domain,task,modifier`
4. Example valid output: `"15,14,9"` (Software, Code Generation, Technical)
5. Example INVALID: `"Software, 14, Technical"` or explanations

**Model**: Llama 3.1:8b (local, via Ollama)
**Latency**: ~500ms-1s per concept
**Accuracy**: High (see test results above)

---

## Encoding Details

### Bit-Packed Format (uint16)

```
Bits [15..12]: Domain  (4 bits, 0-15)
Bits [11..7]:  Task    (5 bits, 0-31)
Bits [6..1]:   Modifier (6 bits, 0-63)
Bit  [0]:      Spare   (unused)
```

### Dense Vector Encoding (16D)

TMD is encoded as a **16-dimensional float vector** using:

1. **Convert to 15D binary**:
   - Domain: 4 bits → 4D binary vector
   - Task: 5 bits → 5D binary vector
   - Modifier: 6 bits → 6D binary vector
   - Total: 15D binary vector

2. **Random projection** (seed=1337):
   - Fixed 16×15 projection matrix
   - Projects 15D binary → 16D dense

3. **L2 normalization**:
   - Normalize to unit length

**Result**: 16D dense vector that preserves all 32,768 TMD combinations via stable random projection

**Implementation**: `src/utils/tmd.py:encode_tmd16(domain, task, modifier)`

---

## Usage in LNSP

### At Ingestion Time

```python
from src.llm_tmd_extractor import extract_tmd_with_llm
from src.utils.tmd import encode_tmd16

# Extract TMD codes from concept text
tmd_dict = extract_tmd_with_llm("oxidoreductase activity")
# Returns: {'domain_code': 4, 'task_code': 5, 'modifier_code': 0}

# Encode to 16D vector
tmd_vector = encode_tmd16(
    domain=tmd_dict['domain_code'],
    task=tmd_dict['task_code'],
    modifier=tmd_dict['modifier_code']
)
# Returns: np.array([...], dtype=float32, shape=(16,))

# Prepend to GTR-T5 embedding (768D) → 784D total
full_vector = np.concatenate([tmd_vector, gtr_embedding])
```

### At Query Time (TMD-ReRank)

```python
from RAG.vecrag_tmd_rerank import run_vecrag_tmd_rerank

# 1. Generate TMD for query text
query_tmd = generate_tmd_for_query("software engineering")

# 2. Get top-20 from vecRAG (FAISS)
vec_indices, vec_scores = faiss_search(query_vector, k=20)

# 3. Extract TMD from retrieved vectors (first 16 dims)
result_tmds = corpus_vectors[vec_indices, :16]

# 4. Compute TMD similarity
tmd_similarities = cosine_similarity(query_tmd, result_tmds)

# 5. Combine scores
final_scores = 0.7 * vec_scores + 0.3 * tmd_similarities

# 6. Re-rank and return top-10
reranked_indices = argsort(-final_scores)[:10]
```

---

## Migration Notes

### Renaming TMD → DTM?

**Current state**:
- Functions/files named "TMD" but parameter order is `(domain, task, modifier)`
- Potentially confusing!

**Options**:
1. **Keep "TMD" name** (historical convention, widely used in codebase)
2. **Rename to "DTM"** (matches actual parameter order)

**Recommendation**: Keep "TMD" as the name but document clearly that parameters are ALWAYS `(domain, task, modifier)`. Less breaking change.

**Files to update if renaming**:
- `src/utils/tmd.py` → `src/utils/dtm.py`
- `src/tmd_extractor_v2.py` → `src/dtm_extractor_v2.py`
- `src/llm_tmd_extractor.py` → `src/llm_dtm_extractor.py`
- `RAG/vecrag_tmd_rerank.py` → `RAG/vecrag_dtm_rerank.py`
- All references in docs, tests, etc.

---

**Document Version**: 2025-10-04
**Status**: LLM-based extraction working, ready for re-ingestion
**Next Steps**: Re-ingest corpus with LLM-generated TMD codes
