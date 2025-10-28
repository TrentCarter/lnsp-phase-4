# Ontological Training Data - DEPRECATED

**Date Archived**: October 11, 2025
**Reason**: Ontological data is unsuitable for autoregressive LVM training

---

## Why These Files Were Archived

### Problem Identified

Ontological datasets (WordNet, SWO, GO, DBpedia) contain **taxonomic hierarchies**, not **sequential narratives**. They teach classification relationships ("dog → mammal → animal") rather than temporal/causal flow needed for autoregressive generation.

### Files Archived

1. **wordnet_training_sequences.npz** (33MB)
   - Training sequences derived from WordNet taxonomies
   - Contains hierarchical IS-A relationships, not narrative progressions

2. **ontology_chains/** directory
   - Source chains from SWO, GO, WordNet, DBpedia
   - All taxonomic/ontological relationships

3. **Models trained on this data:**
   - `lvm_wordnet.pt` (19MB)
   - `lvm_mamba_new.pt` (35MB)
   - `lvm_lstm_retrained.pt` (19MB)

### Why This Data Doesn't Work

**Example from WordNet chains:**
```
"ormer" → "abalone" → "gastropod" → "mollusk" → "invertebrate"
```

This teaches the model: "Given 'gastropod', predict 'mollusk'"

**Problem**: This is a taxonomic IS-A relationship, not a narrative continuation.

**What we need instead:**
```
"Photosynthesis begins..." → "Light is absorbed..." → "Energy is converted..."
```

This teaches: "Given a process description, predict the next step in that process"

### Correct Data for LVM Training

✅ **Sequential document data:**
- Wikipedia articles (narrative progression)
- Textbooks (instructional sequences)
- Scientific papers (methods → results → conclusions)
- Programming tutorials (step-by-step procedures)

❌ **Ontological data (what's archived here):**
- WordNet (lexical hierarchies)
- SWO/GO (ontological classifications)
- DBpedia ontology chains (taxonomic structures)

### Impact on LVM

Models trained on this data learned:
- Classification relationships
- Hierarchical categorization
- Taxonomic structure

Models DID NOT learn:
- Narrative continuation
- Causal/temporal relationships
- Sequential reasoning

### Validation Results

Sequential coherence testing (see `docs/LVM_TRAINING_CRITICAL_FACTS.md`) showed:
- Ontology chains: Low cosine similarity between consecutive concepts
- Document chunks: High cosine similarity (coherent narrative flow)

### Recovery Plan

See sprint plan: `sprints/sprint_10112025_S1.md`

**New approach:**
1. Validate existing document datasets for sequential coherence
2. Ingest Wikipedia/textbook data with semantic chunking
3. Train LVM on sequential document data only
4. Use ontologies for GraphRAG retrieval (their correct use case)

---

## Files Preserved For Reference Only

**DO NOT use these files for LVM training.**

They remain archived for:
- Historical reference
- Understanding what didn't work
- Comparison with proper training data

For GraphRAG and Neo4j knowledge graphs, ontologies are still valuable. But for LVM training, use sequential document data only.
