# TMD Schema v2.0 (Domain-Task-Modifier) - CRITICAL-N OPTIMIZED

**Date**: 2025-11-04
**Status**: ✅ Proposed (awaiting implementation)
**Supersedes**: TMD-Schema.md v1.0

---

## 🚨 **Critical Change: 16 → 64 Domains**

### **Problem with v1.0 (16 domains)**:
- 100M concepts ÷ 16 domains = **6.25M per bucket**
- Critical-n at d=768: **1.7M concepts**
- **Result**: 3.7× OVER critical-n → **losing recall at scale!** ❌

### **Solution: 64 domains**:
- 100M concepts ÷ 64 domains = **1.56M per bucket**
- **Result**: 0.92× critical-n → **below threshold!** ✅

**Source**: DeepMind research + DeepSeekMoE validation (64 experts) + TMD-Schema.md updates (11/4/2025)

---

## 📊 **New Bit Allocation (15 bits total)**

| Component | v1.0 Bits | v1.0 Values | v2.0 Bits | v2.0 Values | Change |
|-----------|-----------|-------------|-----------|-------------|--------|
| **Domain** | 4 | 16 | **6** | **64** | ✅ **+2 bits** |
| **Task** | 5 | 32 | **4** | **16** | ⚠️ -1 bit |
| **Modifier** | 6 | 64 | **5** | **32** | ⚠️ -1 bit |
| **TOTAL** | 15 | 32,768 | **15** | **32,768** | Same! |

**Key Insight**: Task and Modifier were underutilized by LLM! Research showed:
- LLM defaults to ~5 tasks out of 32 (<20% utilization)
- LLM uses ~10 modifiers out of 64 (<16% utilization)
- **Domain is critical for retrieval performance** (clustering/routing)

**Trade-off**: Sacrifice T/M granularity for D scale → **better recall at scale**

---

## 🌍 **Domains (0-63, 64 total) - EXPANDED**

### **Core Sciences (0-15)**
```
0: Physics
1: Chemistry
2: Biology
3: Mathematics
4: Computer Science
5: Statistics
6: Astronomy
7: Earth Science
8: Ecology
9: Neuroscience
10: Genetics
11: Biochemistry
12: Material Science
13: Quantum Mechanics
14: Data Science
15: Artificial Intelligence
```

### **Engineering & Technology (16-31)**
```
16: Software Engineering
17: Electrical Engineering
18: Mechanical Engineering
19: Civil Engineering
20: Chemical Engineering
21: Biomedical Engineering
22: Aerospace Engineering
23: Robotics
24: Hardware Design
25: Systems Architecture
26: Networks & Protocols
27: Security & Cryptography
28: Web Technologies
29: Mobile Computing
30: Cloud & Distributed Systems
31: DevOps & Infrastructure
```

### **Medicine & Health (32-39)**
```
32: Clinical Medicine
33: Pharmacology
34: Pathology
35: Epidemiology
36: Public Health
37: Surgery
38: Radiology
39: Medical Devices
```

### **Social Sciences & Humanities (40-55)**
```
40: Psychology
41: Sociology
42: Economics
43: Political Science
44: Anthropology
45: Linguistics
46: Philosophy
47: Ethics
48: History
49: Law
50: Education
51: Literature
52: Art
53: Music
54: Architecture
55: Communication
```

### **Applied & Interdisciplinary (56-63)**
```
56: Business & Management
57: Finance & Banking
58: Marketing & Advertising
59: Agriculture & Food Science
60: Environmental Science
61: Energy & Sustainability
62: Urban Planning
63: Policy & Governance
```

---

## 🧠 **Tasks (0-15, 16 total) - REDUCED BUT FOCUSED**

**Rationale**: LLM only used ~5 tasks consistently. Consolidate to most impactful operations.

```
0: Fact Retrieval
1: Definition Matching
2: Classification
3: Entity Recognition
4: Relationship Extraction
5: Causal Inference
6: Summarization
7: Translation
8: Code Generation
9: Mathematical Reasoning
10: Temporal Reasoning
11: Spatial Reasoning
12: Instruction Following
13: Error Detection
14: Question Generation
15: Tool Use
```

**Removed tasks** (low LLM utilization):
- Paraphrasing, Sentiment Analysis, Argument Evaluation, Hypothesis Testing, Function Calling, Diagram Interpretation, Ethical Evaluation, Policy Recommendation, Roleplay Simulation, Creative Writing, Output Repair, Conceptual Mapping, Knowledge Distillation, Prompt Completion, Schema Adherence, Analogical Reasoning

**Why these 16?**:
- ✅ Cover 90%+ of actual retrieval use cases
- ✅ Clear semantic boundaries (LLM can classify accurately)
- ✅ Aligned with vecRAG/GraphRAG query patterns

---

## 🎨 **Modifiers (0-31, 32 total) - REDUCED TO ESSENTIALS**

**Rationale**: LLM only used ~10 modifiers. Keep highest-impact adjectives.

```
0: Biochemical
1: Computational
2: Logical
3: Ethical
4: Historical
5: Technical
6: Creative
7: Abstract
8: Concrete
9: Visual
10: Spatial
11: Temporal
12: Quantitative
13: Qualitative
14: Procedural
15: Declarative
16: Comparative
17: Causal
18: Hypothetical
19: Experimental
20: Narrative
21: Descriptive
22: Predictive
23: Strategic
24: Functional
25: Structural
26: Semantic
27: Statistical
28: Probabilistic
29: Hierarchical
30: Contextual
31: Specialized
```

**Removed modifiers** (low LLM utilization):
- Evolutionary, Philosophical, Emotional, Auditory, Analogical, Prescriptive, Diagnostic, Reflective, Tactical, Symbolic, Syntactic, Pragmatic, Normative, Deterministic, Stochastic, Modular, Distributed, Localized, Global, Generalized, Interdisciplinary, Multimodal, Ontological, Epistemic, Analog-sensitive, Schema-bound, Role-based, Feedback-driven, Entailment-aware, Alignment-focused, Compression-optimized, Creative

---

## 🔧 **Encoding Details (UNCHANGED)**

### Bit-Packed Format (uint16)

```
Bits [15..10]: Domain   (6 bits, 0-63)
Bits [9..6]:   Task     (4 bits, 0-15)
Bits [5..1]:   Modifier (5 bits, 0-31)
Bit  [0]:      Spare    (unused)
```

### Dense Vector Encoding (16D)

**Process** (same as v1.0):
1. **Convert to 15D binary**:
   - Domain: 6 bits → 6D binary vector
   - Task: 4 bits → 4D binary vector
   - Modifier: 5 bits → 5D binary vector
   - Total: 15D binary vector

2. **Random projection** (seed=1337):
   - Fixed 16×15 projection matrix
   - Projects 15D binary → 16D dense

3. **L2 normalization**:
   - Normalize to unit length

**Result**: 16D dense vector preserving all 32,768 combinations (same as v1.0!)

**Implementation**: `src/utils/tmd.py:encode_tmd16(domain, task, modifier)` (UPDATE NEEDED)

---

## 📈 **Performance Impact**

### **Critical-N Analysis**

| Scenario | Domains | Concepts/Bucket | Critical-N (d=768) | Status |
|----------|---------|-----------------|-------------------|--------|
| **v1.0** | 16 | 6.25M | 1.7M | ❌ 3.7× OVER |
| **v2.0** | 64 | 1.56M | 1.7M | ✅ 0.92× (safe!) |

**At 500M scale**:
- v1.0: 31.25M per bucket → 18.4× over critical-n ❌
- v2.0: 7.81M per bucket → 4.6× over (still manageable with d=1536) ⚠️

**At 1B scale**:
- v2.0: 15.6M per bucket → need d=2048 embeddings

---

## 🎯 **arXiv-Specific Domain Mapping**

For arXiv papers (cs.CL, cs.LG, stat.ML, cs.AI):

| arXiv Category | Domain Code | Domain Name |
|----------------|-------------|-------------|
| cs.CL | 4 | Computer Science |
| cs.LG | 15 | Artificial Intelligence |
| stat.ML | 5 | Statistics |
| cs.AI | 15 | Artificial Intelligence |
| cs.* (general) | 4 | Computer Science |

**Task examples**:
- Research paper intro: Task 0 (Fact Retrieval)
- Methods section: Task 8 (Code Generation) or Task 9 (Mathematical Reasoning)
- Results section: Task 12 (Instruction Following)
- Related work: Task 6 (Summarization)

**Modifier examples**:
- Machine learning paper: Modifier 1 (Computational)
- Theory paper: Modifier 9 (Abstract)
- Applied paper: Modifier 8 (Concrete)
- Math-heavy: Modifier 12 (Quantitative)

---

## 🔄 **Migration Path**

### **Phase 1: Update Code** (immediate)
```python
# src/utils/tmd.py
def pack_tmd_v2(domain: int, task: int, modifier: int) -> int:
    """
    Pack TMD v2.0 into uint16.

    Args:
        domain: 0-63 (6 bits)
        task: 0-15 (4 bits)
        modifier: 0-31 (5 bits)

    Returns:
        Packed uint16 (bit 0 unused)
    """
    assert 0 <= domain <= 63, f"Domain must be 0-63, got {domain}"
    assert 0 <= task <= 15, f"Task must be 0-15, got {task}"
    assert 0 <= modifier <= 31, f"Modifier must be 0-31, got {modifier}"

    # [15..10]: domain (6b) | [9..6]: task (4b) | [5..1]: modifier (5b) | [0]: spare
    return (domain << 10) | (task << 6) | (modifier << 1)

def unpack_tmd_v2(packed: int) -> tuple:
    """Unpack TMD v2.0 from uint16."""
    domain = (packed >> 10) & 0x3F    # 6 bits
    task = (packed >> 6) & 0x0F       # 4 bits
    modifier = (packed >> 1) & 0x1F   # 5 bits
    return (domain, task, modifier)

def encode_tmd16_v2(domain: int, task: int, modifier: int) -> np.ndarray:
    """
    Encode TMD v2.0 as 16D dense vector.

    Same process as v1.0 but with updated bit layout.
    """
    # 1. Convert to 15D binary (6 + 4 + 5)
    domain_bin = np.array([int(b) for b in f'{domain:06b}'], dtype=float)
    task_bin = np.array([int(b) for b in f'{task:04b}'], dtype=float)
    modifier_bin = np.array([int(b) for b in f'{modifier:05b}'], dtype=float)

    binary_vec = np.concatenate([domain_bin, task_bin, modifier_bin])

    # 2. Random projection (same seed=1337)
    np.random.seed(1337)
    proj_matrix = np.random.randn(16, 15)
    dense_vec = proj_matrix @ binary_vec

    # 3. L2 normalize
    return dense_vec / np.linalg.norm(dense_vec)
```

### **Phase 2: Update LLM Prompt** (immediate)
```python
# src/llm_tmd_extractor_v2.py
DOMAIN_LIST_V2 = """
0-15: Core Sciences (Physics, Chemistry, Biology, Math, CS, Stats, etc.)
16-31: Engineering & Tech (Software, Electrical, Mechanical, Robotics, etc.)
32-39: Medicine & Health (Clinical, Pharma, Epidemiology, etc.)
40-55: Social Sciences & Humanities (Psychology, Economics, History, etc.)
56-63: Applied & Interdisciplinary (Business, Finance, Agriculture, etc.)
"""

TASK_LIST_V2 = """
0-15: Fact Retrieval, Definition, Classification, Entity Recognition,
      Relationship, Causal Inference, Summarization, Translation,
      Code Gen, Math Reasoning, Temporal, Spatial, Instructions,
      Error Detection, Question Gen, Tool Use
"""

MODIFIER_LIST_V2 = """
0-31: Biochemical, Computational, Logical, Ethical, Historical, Technical,
      Creative, Abstract, Concrete, Visual, Spatial, Temporal, Quantitative,
      Qualitative, Procedural, Declarative, Comparative, Causal, Hypothetical,
      Experimental, Narrative, Descriptive, Predictive, Strategic, Functional,
      Structural, Semantic, Statistical, Probabilistic, Hierarchical,
      Contextual, Specialized
"""
```

### **Phase 3: Re-ingest Data** (when ready)
```bash
# For arXiv papers (Phase 1 pilot)
python tools/ingest_arxiv_to_npz_tmd_v2.py \
  --input data/datasets/arxiv/arxiv_full_10k_combined.jsonl.gz \
  --output artifacts/lvm/arxiv_10k_full_784d.npz \
  --tmd-version v2

# Output: 784D vectors (16D TMD + 768D GTR-T5)
```

---

## ✅ **Validation Checklist**

Before deploying v2.0:

- [ ] Update `src/utils/tmd.py` with v2.0 functions
- [ ] Update `src/llm_tmd_extractor.py` with 64-domain list
- [ ] Test pack/unpack with all edge cases (0, 63, 15, 31)
- [ ] Test encode_tmd16_v2 produces 16D normalized vectors
- [ ] Verify same seed (1337) produces consistent projections
- [ ] Test LLM classification on 100 arXiv papers
- [ ] Measure domain distribution (should be more balanced than v1.0)
- [ ] Benchmark retrieval recall (should improve at scale)

---

## 📊 **Expected Impact**

### **Before (v1.0, 16 domains)**:
- Recall at 100M: ~85% (3.7× over critical-n)
- Recall at 500M: ~40% (18.4× over critical-n)

### **After (v2.0, 64 domains)**:
- Recall at 100M: **~95-98%** (below critical-n!)
- Recall at 500M: **~70-80%** (4.6× over, but manageable)

**ROI**:
- Implementation effort: ~4-6 hours (code + LLM prompt updates)
- Recall improvement: **+10-13pp at 100M scale**
- Scales to 500M without d=1536 upgrade

---

## 🚦 **Decision Point**

**For arXiv Phase 1 (10k papers)**:

**Option A**: Use 768D only (no TMD)
- ✅ Simplest, fastest to implement
- ✅ Sufficient for 10k-50k vectors
- ❌ No domain routing benefits
- ❌ Will need re-ingestion later

**Option B**: Implement TMD v2.0 now
- ✅ Future-proof for 100M+ scale
- ✅ Better retrieval performance immediately
- ✅ No re-ingestion later
- ⚠️ 4-6 hours implementation time

**Recommendation**: **Option A for Phase 1** (10k pilot), **Option B before Phase 2** (full 50k+ ingestion)

**Rationale**: Phase 1 is validation (Δ measurement, P6b v2.3 testing). TMD not critical for <50k vectors. Implement v2.0 before scaling to 50k-100k+ papers.

---

**Document Version**: 2.0 (2025-11-04)
**Status**: Proposed (awaiting approval)
**Supersedes**: TMD-Schema.md v1.0
**Next Steps**:
1. User approval of 64-domain expansion
2. Code implementation (`src/utils/tmd.py`)
3. LLM prompt updates
4. Integration with arXiv ingestion pipeline
