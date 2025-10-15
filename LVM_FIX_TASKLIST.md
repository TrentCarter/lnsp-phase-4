# LVM Training Fix - Systematic Task List

**Date**: October 13, 2025
**Root Cause**: Distribution mismatch between model output (L2-normalized) and vec2text decoder expectations (raw encoder distribution)

## Critical Issues Identified

1. ❌ **Forcing L2 norm=1.0 collapses variance** → vec2text produces gibberish
2. ❌ **Plain MSE/cosine loss** → encourages mode collapse to mean vector
3. ❌ **Single output head** → need separate heads for cosine retrieval vs decoder input
4. ❌ **Weak projection head** → single linear layer too easy to collapse
5. ❌ **Missing hard negatives** → InfoNCE needs within-doc negatives
6. ❌ **Batch construction** → near-duplicates make average vector look good

---

## Phase 1: Diagnostic Tests (DO FIRST)

### Task 1.1: Moment Matching Test
**File**: `tools/test_moment_matching.py`

```python
# Compare per-dimension μ/σ of encoder(x) vs model(x) pre-norm
# If σ is flat/low, decoder will produce gibberish
```

**Expected Output**:
- GTR-T5 encoder: μ ≈ 0, σ ≈ varies per dimension
- Model pre-norm: should match encoder distribution
- Model post-norm: σ = constant (this is the problem!)

**Action**:
- [ ] Create test script
- [ ] Run on current GRU model
- [ ] Document μ/σ distributions

---

### Task 1.2: Mean Vector Baseline
**File**: `tools/test_mean_vector_baseline.py`

```python
# Compute cosine between targets and global mean of target set
# If model cosine ≈ baseline cosine, you have average-vector collapse
```

**Expected Output**:
- Global mean baseline: ~75-80% cosine
- Current model: 78% cosine (MATCHES baseline → mode collapse confirmed)

**Action**:
- [ ] Create baseline script
- [ ] Compare model vs baseline
- [ ] Document findings

---

### Task 1.3: Decoder A/B/C Test
**File**: `tools/test_decoder_distributions.py`

```python
# Test A: decoder(encoder(x)) → should be fine
# Test B: decoder(L2(encoder(x))) → if degraded, decoder not trained on unit-norm
# Test C: decoder(model_out_pre_norm) → this is what you should ship
```

**Expected Output**:
- Test A: Good reconstruction
- Test B: Gibberish (smoking gun!)
- Test C: Should match Test A after fixes

**Action**:
- [ ] Create A/B/C test
- [ ] Document which fails
- [ ] Confirm vec2text decoder expectations

---

## Phase 2: Architecture Fixes

### Task 2.1: Split Output Heads
**File**: `app/lvm/train_mamba2.py` (and LSTM/Transformer)

**Current (BROKEN)**:
```python
self.fc = nn.Linear(hidden_dim, input_dim)

def forward(self, x):
    output = self.fc(last_hidden)  # [batch, 768]
    output = nn.functional.normalize(output, p=2, dim=-1)  # ❌ BREAKS DECODER
    return output
```

**Fixed**:
```python
# Two-layer projection head
self.fc1 = nn.Linear(hidden_dim, input_dim)
self.gelu = nn.GELU()
self.ln = nn.LayerNorm(input_dim)
self.fc2 = nn.Linear(input_dim, input_dim)

def forward(self, x, return_both=False):
    # Shared projection
    h = self.fc1(last_hidden)
    h = self.gelu(h)
    h = self.ln(h)
    y_dec = self.fc2(h)  # [batch, 768] - RAW for decoder

    # L2-normalized head for cosine/retrieval
    y_cos = nn.functional.normalize(y_dec, p=2, dim=-1)

    if return_both:
        return y_dec, y_cos  # decoder, cosine
    return y_cos  # backward compat for training loop
```

**Action**:
- [ ] Update Mamba2VectorPredictor
- [ ] Update LSTMVectorPredictor
- [ ] Update TransformerVectorPredictor
- [ ] Add return_both parameter

---

### Task 2.2: Moment-Matching Loss
**File**: `app/lvm/train_mamba2.py`

**Add to training loop**:
```python
# Compute statistics from GTR-T5 encoder vectors
# (compute once, cache as model attributes)
target_mean = targets.mean(0)  # [768]
target_std = targets.std(0)    # [768]

# Moment matching on y_dec (pre-norm)
y_dec, y_cos = model(contexts, return_both=True)
pred_mean = y_dec.mean(0)
pred_std = y_dec.std(0)

moment_loss = (
    ((pred_mean - target_mean) ** 2).mean() +
    ((pred_std - target_std) ** 2).mean()
)

loss = mse_loss + 1e-3 * moment_loss
```

**Action**:
- [ ] Add moment loss to all 3 trainers
- [ ] Compute target statistics from data
- [ ] Weight appropriately (1e-3 baseline)

---

### Task 2.3: InfoNCE Contrastive Loss
**File**: `app/lvm/train_mamba2.py`

**Replace MSE loss**:
```python
def symmetric_infonce_loss(y_cos, t_cos, temperature=0.07):
    """
    Symmetric InfoNCE with in-batch negatives
    y_cos, t_cos: L2-normalized [batch, 768]
    """
    # Compute similarity matrices
    logits_y2t = (y_cos @ t_cos.T) / temperature  # [B, B]
    logits_t2y = (t_cos @ y_cos.T) / temperature  # [B, B]

    # Diagonal is positive pairs
    labels = torch.arange(len(y_cos), device=y_cos.device)

    loss = (
        F.cross_entropy(logits_y2t, labels) +
        F.cross_entropy(logits_t2y, labels)
    )

    return loss

# In training loop:
y_dec, y_cos = model(contexts, return_both=True)
t_cos = nn.functional.normalize(targets, p=2, dim=-1)

contrastive_loss = symmetric_infonce_loss(y_cos, t_cos)
variance_loss = torch.relu(1.0 - y_cos.std(0)).mean()  # prevent collapse

loss = contrastive_loss + 1e-3 * variance_loss + 1e-3 * moment_loss
```

**Action**:
- [ ] Implement symmetric_infonce_loss
- [ ] Add variance regularizer
- [ ] Balance loss weights
- [ ] Update all 3 trainers

---

### Task 2.4: Hard Negatives in Batch
**File**: `app/lvm/train_mamba2.py`

**Update dataset to include hard negatives**:
```python
class VectorSequenceDataset(Dataset):
    def __init__(self, npz_path):
        data = np.load(npz_path)
        self.contexts = data['context_sequences']
        self.targets = data['target_vectors']

        # Group by document/source for hard negatives
        # (requires metadata in NPZ)
        self.doc_ids = data.get('doc_ids', None)

    def __getitem__(self, idx):
        context = self.contexts[idx]
        target = self.targets[idx]

        # Sample hard negative from same document
        if self.doc_ids is not None:
            doc_id = self.doc_ids[idx]
            same_doc_indices = np.where(self.doc_ids == doc_id)[0]
            if len(same_doc_indices) > 1:
                neg_idx = np.random.choice(
                    same_doc_indices[same_doc_indices != idx]
                )
                hard_negative = self.targets[neg_idx]
            else:
                hard_negative = target  # fallback
        else:
            hard_negative = target

        return context, target, hard_negative
```

**Action**:
- [ ] Add doc_ids to training data NPZ
- [ ] Update dataset to return hard negatives
- [ ] Integrate into InfoNCE loss
- [ ] Ensure batch diversity (cap samples per doc)

---

## Phase 3: Training Data Fixes

### Task 3.1: Add Document IDs to NPZ
**File**: `tools/extract_ordered_training_data.py`

**Update to include**:
```python
np.savez_compressed(
    output_file,
    context_sequences=contexts,
    target_vectors=targets,
    doc_ids=doc_ids,  # NEW: document source IDs
    concept_texts=concept_texts  # optional for debugging
)
```

**Action**:
- [ ] Update extraction script
- [ ] Regenerate training_sequences_ctx5.npz
- [ ] Verify doc_ids are correct

---

### Task 3.2: Batch Sampler with Diversity
**File**: `app/lvm/train_mamba2.py`

**Custom batch sampler**:
```python
class DiverseBatchSampler:
    def __init__(self, doc_ids, batch_size, max_per_doc=8):
        self.doc_ids = doc_ids
        self.batch_size = batch_size
        self.max_per_doc = max_per_doc

    def __iter__(self):
        # Group indices by doc
        doc_to_indices = {}
        for idx, doc_id in enumerate(self.doc_ids):
            doc_to_indices.setdefault(doc_id, []).append(idx)

        # Sample batches with diversity
        # ...implementation...
```

**Action**:
- [ ] Implement DiverseBatchSampler
- [ ] Integrate into DataLoader
- [ ] Verify batch diversity

---

## Phase 4: Inference Fixes

### Task 4.1: Update Inference to Use y_dec
**File**: `app/lvm/evaluate_models.py`, `tools/test_gru_inference_samples.py`

**Current (BROKEN)**:
```python
predictions = model(contexts)  # Returns L2-normalized y_cos
# Send to vec2text decoder → GIBBERISH
```

**Fixed**:
```python
y_dec, y_cos = model(contexts, return_both=True)

# For vec2text decoding: use y_dec (raw distribution)
decoded_text = vec2text_decoder(y_dec)

# For cosine similarity/retrieval: use y_cos
cosine = compute_cosine(y_cos, target_normalized)
```

**Action**:
- [ ] Update all inference scripts
- [ ] Use y_dec for vec2text
- [ ] Use y_cos for metrics
- [ ] Document in comments

---

### Task 4.2: Update LVM Server API
**File**: `app/api/lvm_server.py`

**Add endpoint parameter**:
```python
@app.post("/predict")
async def predict(
    context_vectors: List[List[float]],
    output_type: str = "decoder"  # "decoder" or "cosine"
):
    context_tensor = torch.FloatTensor(context_vectors)
    y_dec, y_cos = model(context_tensor, return_both=True)

    if output_type == "decoder":
        return y_dec.tolist()  # For port 8766
    else:
        return y_cos.tolist()  # For retrieval
```

**Action**:
- [ ] Add output_type parameter
- [ ] Update API docs
- [ ] Test both endpoints

---

## Phase 5: Validation

### Task 5.1: Full Pipeline Test
**File**: `tools/test_lvm_fixed_pipeline.py`

```python
# Test sequence:
# 1. Input text → GTR-T5 encoder → vec_in
# 2. Context [vec_in[0:5]] → LVM → y_dec, y_cos
# 3. y_dec → vec2text decoder → text_out
# 4. Compare text_in[5] vs text_out
```

**Expected Results**:
- Cosine (y_cos vs target): 70-80%
- Vec2text decode quality: GOOD (not gibberish)
- Semantic similarity: HIGH

**Action**:
- [ ] Create end-to-end test
- [ ] Run on 100 samples
- [ ] Document metrics

---

### Task 5.2: Training Curve Validation
**File**: Monitor during retraining

**Watch for**:
- ✅ Contrastive loss decreasing
- ✅ Variance loss stable (not → 0)
- ✅ Moment loss → 0
- ✅ Per-dimension σ matches target distribution
- ✅ Output vectors NOT all identical (check variance)

**Action**:
- [ ] Add logging for all losses
- [ ] Plot training curves
- [ ] Validate no mode collapse

---

## Phase 6: Retraining

### Task 6.1: Retrain All 3 Models
**Commands**:
```bash
# With fixed architecture and losses
python app/lvm/train_lstm_baseline.py --epochs 20 --device mps
python app/lvm/train_mamba2.py --epochs 20 --device mps
python app/lvm/train_transformer.py --epochs 20 --device mps
```

**Action**:
- [ ] Clear old checkpoints
- [ ] Run training with monitoring
- [ ] Save best models
- [ ] Document final metrics

---

### Task 6.2: Validate Fixed Models
**File**: `tools/test_fixed_models.py`

**Tests**:
1. Moment matching: μ/σ match GTR-T5
2. Output diversity: variance > 0.01
3. Vec2text decode: coherent text
4. Cosine metrics: 70-80%
5. A/B/C test: Test C works

**Action**:
- [ ] Run all validation tests
- [ ] Compare to broken models
- [ ] Document improvements

---

## Summary Checklist

### Prerequisites (DO FIRST)
- [ ] Task 1.1: Moment matching test
- [ ] Task 1.2: Mean vector baseline
- [ ] Task 1.3: Decoder A/B/C test

### Architecture Fixes
- [ ] Task 2.1: Split output heads (y_dec, y_cos)
- [ ] Task 2.2: Add moment-matching loss
- [ ] Task 2.3: Replace MSE with InfoNCE
- [ ] Task 2.4: Add hard negatives

### Data Fixes
- [ ] Task 3.1: Add doc_ids to NPZ
- [ ] Task 3.2: Diverse batch sampler

### Inference Fixes
- [ ] Task 4.1: Use y_dec for vec2text
- [ ] Task 4.2: Update API endpoints

### Validation
- [ ] Task 5.1: Full pipeline test
- [ ] Task 5.2: Training curve validation

### Retraining
- [ ] Task 6.1: Retrain all 3 models
- [ ] Task 6.2: Validate fixed models

---

## Expected Timeline

| Phase | Time Estimate | Priority |
|-------|--------------|----------|
| Phase 1: Diagnostics | 1 hour | CRITICAL |
| Phase 2: Architecture | 2 hours | CRITICAL |
| Phase 3: Data | 1 hour | HIGH |
| Phase 4: Inference | 1 hour | HIGH |
| Phase 5: Validation | 30 min | MEDIUM |
| Phase 6: Retraining | 4 hours | FINAL |

**Total**: ~8-10 hours (surgical fixes + retraining)

---

## Key Insights

1. **The model WAS learning** - 75-78% cosine during training was real
2. **I broke it at inference** - forced L2 norm collapsed the variance vec2text needs
3. **Need TWO heads**: y_cos for retrieval, y_dec for decoding
4. **InfoNCE > MSE** - prevents mode collapse to mean vector
5. **Distribution matching matters** - decoder expects raw encoder statistics

---

## Success Criteria

✅ **Fixed Model Should**:
- Maintain 70-80% cosine similarity (y_cos)
- Produce varied outputs (σ > 0.01 per dimension)
- Match GTR-T5 encoder μ/σ distribution (y_dec)
- Decode to coherent text via vec2text
- Not collapse to mean vector (diversity maintained)
