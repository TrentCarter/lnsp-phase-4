# Epoch 4 Go/No-Go Checklist

**Quality Gates for Two-Tower Retrieval (Epoch 4)**

## 1. Check Gate Metrics (PRIMARY GATES)

```bash
jq . artifacts/lvm/eval_epoch4/metrics.json
```

**Pass Criteria (EITHER condition):**
- ✅ `R@5 >= 0.30` (30% recall at rank 5)
- ✅ `MRR >= 0.20` (mean reciprocal rank)

**If PASS**: Skip to Step 4 (Tag & Ship)

---

## 2. If JUST SHORT → Train/Apply Reranker

**When to use**: Metrics close but not quite passing (R@5 = 0.27-0.29 or MRR = 0.18-0.19)

### 2a. Train Reranker
```bash
# Train vector-only MLP reranker on top-50 hits
./.venv/bin/python tools/reranker_train.py \
  --hits artifacts/lvm/eval_epoch4/hits50_ep4.jsonl \
  --scores artifacts/lvm/eval_epoch4/scores_ep4.json \
  --epochs 10 \
  --hidden 128 \
  --out artifacts/lvm/reranker/mlp_v0.pt
```

### 2b. Apply Reranker
```bash
# Rerank top-50 candidates
./.venv/bin/python tools/reranker_apply.py \
  --hits artifacts/lvm/eval_epoch4/hits50_ep4.jsonl \
  --model artifacts/lvm/reranker/mlp_v0.pt \
  --out artifacts/lvm/eval_epoch4/reranked_ep4.jsonl
```

### 2c. Recompute Metrics
```bash
# Check if reranking pushes us over the threshold
./.venv/bin/python tools/compute_metrics.py \
  --hits artifacts/lvm/eval_epoch4/reranked_ep4.jsonl \
  --truth artifacts/lvm/eval_clean_disjoint.npz \
  --out artifacts/lvm/eval_epoch4/metrics_reranked_ep4.json

jq . artifacts/lvm/eval_epoch4/metrics_reranked_ep4.json
```

**If PASS after reranking**: Go to Step 4 (Tag & Ship with reranker)
**If still SHORT**: Continue to Step 3

---

## 3. If Still Short → Near-Miss Bank + Resume Training

**When to use**: Metrics significantly short (R@5 < 0.27 or MRR < 0.18)

### 3a. Mine Near-Miss Bank (Large-Scale Hard Negatives)
```bash
# Mine 4096 near-miss negatives per query (stratified by lane)
./.venv/bin/python tools/mine_nearmiss_bank.py \
  --pvec artifacts/eval/p_train_ep3.npy \
  --qvec artifacts/eval/q_train_ep3.npy \
  --index artifacts/lvm/train_index_ep3.faiss \
  --topk 4096 \
  --out artifacts/corpus/near_miss_bank_ep4.npy

# Output: [N_queries, 4096] array of negative indices
```

### 3b. Resume Training (Epoch 5)
```bash
# Resume from Epoch 4 with near-miss bank
./.venv/bin/python app/lvm/train_twotower_fast.py \
  --resume artifacts/lvm/models/twotower_fast/epoch4.pt \
  --train-npz artifacts/lvm/train_clean_disjoint.npz \
  --same-article-k 3 \
  --nearmiss-bank artifacts/corpus/near_miss_bank_ep4.npy \
  --p-cache-npy artifacts/eval/p_train_ep3.npy \
  --epochs 1 \
  --batch-size 256 \
  --lr 5e-5 \
  --device cpu \
  --save-dir artifacts/lvm/models/twotower_ep5
```

### 3c. Re-evaluate Epoch 5
```bash
# Run full evaluation pipeline again
./scripts/eval_epoch5_pipeline.sh
```

---

## 4. Tag & Ship (PASS)

### 4a. Tag Retriever
```bash
# Tag the passing model
git tag -a retriever-v0-ep4 -m "Two-tower retriever (R@5=${R5}%, MRR=${MRR})"
```

### 4b. Tag Reranker (if used)
```bash
# If reranker was applied
git tag -a reranker-v0 -m "MLP reranker (boost R@5 by ${BOOST}pp)"
```

### 4c. Wire into vecRAG
```bash
# Update vecRAG config to use new retriever
echo "LNSP_RETRIEVER_MODEL=artifacts/lvm/models/twotower_fast/epoch4.pt" >> .env
echo "LNSP_RERANKER_MODEL=artifacts/lvm/reranker/mlp_v0.pt" >> .env  # if used

# Restart vecRAG service
./scripts/restart_vecrag.sh
```

---

## Nitpicks (FROZEN - Do Not Change)

### Storage Math
- **Lean path**: 6.6-9.7 KB/entry (concept_vec + metadata)
- **Full path**: ~12.9 KB/entry (with CPESH + graph edges)
- **Plan around**: ~10 KB/entry average

### Type Clarity
- **16 bits (routing)**: `uint16` lane index (0-65535 lanes)
- **16D (learned dense)**: 16-dimensional routing vector (NOT same as 16 bits!)
- **Don't conflate these two**

### Lane Index Type
- **Current**: `int2` with range check (0-3 for 4 lanes)
- **Future**: Can switch to `uint16` if scaling to 65k lanes
- **Keep validation**: Always check lane_index is in valid range

---

## Decision Tree

```
Start
  |
  v
Check metrics.json
  |
  ├─> R@5 >= 0.30 OR MRR >= 0.20?
  |     YES --> PASS --> Tag & Ship (Step 4)
  |     NO  --> Continue
  |
  v
JUST SHORT? (R@5 = 0.27-0.29)
  |
  ├─> YES --> Train Reranker (Step 2)
  |     |
  |     v
  |     Check reranked metrics
  |       |
  |       ├─> PASS? --> Tag & Ship (Step 4 with reranker)
  |       └─> NO --> Continue to Step 3
  |
  └─> NO (R@5 < 0.27) --> Mine Near-Miss Bank + Epoch 5 (Step 3)
        |
        v
        Re-evaluate Epoch 5
        |
        v
        Check metrics --> If PASS, Tag & Ship
```

---

## Expected Outcomes

**Epoch 3 Baseline**: R@5 = 29.9%
**Epoch 4 Target**: R@5 = 33-36% (+3-6pp from near-miss negatives)

**Reranker boost**: +1-3pp (if needed)
**Near-Miss Bank boost**: +5-10pp (if needed)

**Final Target**: R@5 >= 30% for production deployment
