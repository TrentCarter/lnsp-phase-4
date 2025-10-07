# LVM Training: Critical Facts (Stop Overcomplicating!)

**Date**: October 7, 2025
**Author**: Trent Carter + Claude (after being corrected)
**Purpose**: Prevent Claude from overthinking the training pipeline

---

## 🚨 THE FUNDAMENTAL TRUTH

**The ontology chains were ALREADY ingested in order. The vectors are ALREADY computed. Just use them!**

---

## What We Have RIGHT NOW

### 1. Ordered Source Data
```
artifacts/ontology_chains/swo_chains.jsonl
artifacts/ontology_chains/wordnet_chains.jsonl
```

Each line contains an **ordered chain**:
```json
{
  "chain_id": "swo_0001",
  "concepts": [
    "BLAST software",
    "sequence alignment tool",
    "bioinformatics analysis",
    "genomics research"
  ],
  "relations": ["IS_A", "USED_FOR", "PART_OF"]
}
```

**The order is preserved from the ontology!**

### 2. Vectors Already Computed
```
artifacts/fw10k_vectors.npz
```

Contains:
- 10,000 vectors @ 768D
- **In same order as ingestion**
- Indexed by position (0-9999)

### 3. PostgreSQL Has the Mapping
```sql
SELECT id, concept_text, chain_id, position_in_chain
FROM cpe_entry
ORDER BY id;
```

**The ID order matches the NPZ index order!**

---

## What Training Needs (SIMPLE!)

### Step 1: Load Data
```python
import json
import numpy as np

# Load chains (already ordered!)
chains = []
with open("artifacts/ontology_chains/swo_chains.jsonl") as f:
    for line in f:
        chain = json.loads(line)
        chains.append(chain)

# Load vectors (already ordered!)
npz = np.load("artifacts/fw10k_vectors.npz")
vectors = npz['vectors']  # Shape: (10000, 768)
```

### Step 2: Create Training Sequences
```python
# For each chain, create autoregressive sequences
training_data = []

for chain in chains:
    concept_ids = chain['concept_ids']  # e.g., [42, 103, 891, 1205]

    # Get vectors for this chain
    chain_vecs = vectors[concept_ids]  # Shape: (chain_length, 768)

    # Create training pairs: predict next from context
    for i in range(1, len(chain_vecs)):
        training_data.append({
            'context': chain_vecs[:i],      # vectors[0:i]
            'target': chain_vecs[i]         # vector[i]
        })

# Result: ~50K training sequences from 10K concepts
```

### Step 3: Train Mamba
```python
# Standard next-token prediction, but with vectors not tokens!
for batch in dataloader:
    context = batch['context']  # [batch, seq_len, 768]
    target = batch['target']    # [batch, 768]

    pred = model(context)       # Mamba forward pass
    loss = F.mse_loss(pred, target)

    loss.backward()
    optimizer.step()
```

---

## What Neo4j Graph Is For (NOT TRAINING!)

**Neo4j is for INFERENCE, not training data!**

### Training Source (this doc):
- ✅ Use JSONL chains (already ordered)
- ✅ Use NPZ vectors (already computed)
- ❌ DON'T query Neo4j (unnecessary complexity)

### Inference Uses:
- GraphRAG neighbor expansion
- Multi-hop reasoning
- Relation-aware retrieval

**But training just needs the ordered vectors from ingestion!**

---

## The Complete Training Pipeline (No Neo4j!)

```bash
# 1. Prepare training sequences from JSONL + NPZ
python src/lvm/prepare_training_data.py \
    --chains artifacts/ontology_chains/*.jsonl \
    --vectors artifacts/fw10k_vectors.npz \
    --output artifacts/lvm/training_sequences.npz

# 2. Train Mamba model
python src/lvm/train_mamba.py \
    --data artifacts/lvm/training_sequences.npz \
    --output models/mamba_lvm.pt \
    --epochs 10 \
    --batch-size 64

# 3. Evaluate on held-out test set
python src/lvm/eval_mamba.py \
    --model models/mamba_lvm.pt \
    --test-data artifacts/lvm/test_sequences.npz
```

**Total time: ~30 minutes for training (10K concepts, 50K sequences)**

---

## Why This Is Correct

### 1. Order Preservation
The JSONL files contain **ontology structure**:
- SWO: Software hierarchy (specific → general)
- WordNet: Lexical relationships (synonym → hypernym)

**Ingestion preserved this order → Vectors are in same order → Training uses ontological structure**

### 2. No Information Loss
We're not "walking" the graph dynamically. We're using the **original ontology chains** that were carefully curated by domain experts.

### 3. Simplicity
- No Cypher queries
- No graph traversal
- No dynamic path finding
- Just: Read JSONL → Load NPZ → Train

---

## Common Mistakes (Claude Keeps Making These!)

### ❌ Mistake 1: "We need to extract chains from Neo4j"
**NO!** The chains are in the JSONL files. Neo4j is a *copy* for inference.

### ❌ Mistake 2: "We need graph walks for training data"
**NO!** Graph walks are for *inference* (GraphRAG). Training uses the original ontology chains.

### ❌ Mistake 3: "We need to query PostgreSQL for vectors"
**NO!** The vectors are in the NPZ file. PostgreSQL just has metadata.

### ❌ Mistake 4: "Training is complex and needs careful design"
**NO!** It's literally: Read JSONL → Read NPZ → Train Mamba. That's it.

---

## The Inference Flow (Where Graph Matters)

**Training**: Uses ordered chains from JSONL
**Inference**: Uses Neo4j graph for reasoning

```
User Query → vecRAG (fast retrieval)
           ↓
     (if ambiguous)
           ↓
  LVM (Mamba reasoning) → predicts related concept vectors
           ↓
  Neo4j (graph walk) → expands to neighbors
           ↓
  vecRAG (lookup) → finds nearest concepts
           ↓
  LLM (smoothing) → generates final text
```

**Training data comes from ontology chains, not graph walks!**

---

## Summary (Read This When Confused)

1. **Ontology chains** (JSONL) contain ordered concepts
2. **Vectors** (NPZ) are computed in same order
3. **Training** = Predict `vector[i+1]` from `vector[0:i]`
4. **Neo4j graph** is for inference, NOT training data
5. **Stop overthinking it!**

---

## Next Steps (Actually Do This)

```bash
# Create the training data prep script
touch src/lvm/prepare_training_data.py

# Create the Mamba training script
touch src/lvm/train_mamba.py

# Create the evaluation script
touch src/lvm/eval_mamba.py

# Run the pipeline
bash scripts/train_lvm_simple.sh
```

**Estimated time to first trained model: 2 hours (including debugging)**

---

**Remember**: The data is already ordered. The vectors are already computed. Just train the damn model.

**End of Document**
