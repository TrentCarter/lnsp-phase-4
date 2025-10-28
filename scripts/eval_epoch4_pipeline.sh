#!/bin/bash
# Automated Evaluation Pipeline for Epoch 4
# Runs immediately when checkpoint appears

set -e

CHECKPOINT="artifacts/lvm/models/twotower_fast/epoch4.pt"
EVAL_NPZ="artifacts/lvm/eval_clean_disjoint.npz"
INDEX_NPZ="artifacts/lvm/train_clean_disjoint.npz"
OUTPUT_DIR="artifacts/lvm/eval_epoch4"

echo "========================================"
echo "EPOCH 4 EVALUATION PIPELINE"
echo "========================================"
echo "Waiting for checkpoint: $CHECKPOINT"
echo ""

# Wait for checkpoint to appear
while [ ! -f "$CHECKPOINT" ]; do
    echo -ne "\r[$(date +%H:%M:%S)] Waiting for checkpoint..."
    sleep 10
done

echo ""
echo "✅ CHECKPOINT FOUND!"
ls -lh "$CHECKPOINT"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# ============================================================
# Step 1: Emit Q-vectors for eval split
# ============================================================
echo "========================================"
echo "STEP 1: Emitting Q-vectors for eval"
echo "========================================"

./.venv/bin/python -u << 'EOF'
import torch
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))
from app.lvm.train_twotower import QueryTower

# Load checkpoint
print("Loading checkpoint...")
ckpt = torch.load("artifacts/lvm/models/twotower_fast/epoch4.pt",
                  map_location='cpu', weights_only=False)
args = ckpt['args']

# Load eval data
print("Loading eval data...")
data = np.load("artifacts/lvm/eval_clean_disjoint.npz", allow_pickle=True)
contexts = data['context_sequences']
print(f"  Eval samples: {len(contexts)}")

# Create Q-tower
print("Creating Q-tower...")
q_tower = QueryTower(
    backbone_type=args.get('arch_q', 'mamba_s'),
    d_model=args.get('d_model', 768),
    n_layers=args.get('n_layers', 8),
    d_state=args.get('d_state', 128),
    conv_sz=args.get('conv_sz', 4),
    expand=args.get('expand', 2),
    dropout=0.0  # No dropout for eval
)
q_tower.load_state_dict(ckpt['q_tower_state_dict'])
q_tower.eval()

# Emit Q-vectors
print("Emitting Q-vectors...")
q_vectors = []
batch_size = 512

with torch.no_grad():
    for i in range(0, len(contexts), batch_size):
        batch = contexts[i:i+batch_size]
        ctx = torch.from_numpy(batch).float()
        q = q_tower(ctx)
        q_vectors.append(q.cpu().numpy())

        if (i // batch_size) % 10 == 0:
            print(f"  Progress: {i}/{len(contexts)}")

q_vectors = np.vstack(q_vectors)
print(f"  Shape: {q_vectors.shape}")

# Normalize
norms = np.linalg.norm(q_vectors, axis=1, keepdims=True) + 1e-12
q_vectors = q_vectors / norms

# Save
output_path = "artifacts/lvm/eval_epoch4/q_eval_ep4.npy"
np.save(output_path, q_vectors)
print(f"✅ Saved: {output_path}")
print(f"   Size: {q_vectors.nbytes / 1024**2:.1f} MB")
EOF

echo ""

# ============================================================
# Step 2: Build FAISS index from training vectors
# ============================================================
echo "========================================"
echo "STEP 2: Building FAISS index"
echo "========================================"

./.venv/bin/python -u << 'EOF'
import numpy as np
import faiss

# Load training P-vectors (already normalized from mining step)
print("Loading training P-vectors...")
P_train = np.load("artifacts/eval/p_train_ep3.npy").astype('float32')
print(f"  Training vectors: {P_train.shape}")

# Build index
print("Building FAISS Flat IP index...")
dim = P_train.shape[1]
index = faiss.IndexFlatIP(dim)
index.add(P_train)
print(f"  Index size: {index.ntotal} vectors")

# Save index
faiss.write_index(index, "artifacts/lvm/eval_epoch4/train_index_ep3.faiss")
print("✅ Index saved: artifacts/lvm/eval_epoch4/train_index_ep3.faiss")
EOF

echo ""

# ============================================================
# Step 3: Retrieve top-50 and compute metrics
# ============================================================
echo "========================================"
echo "STEP 3: Retrieval and Scoring"
echo "========================================"

./.venv/bin/python -u << 'EOF'
import numpy as np
import faiss
import json
from collections import defaultdict

# Load data
print("Loading data...")
q_eval = np.load("artifacts/lvm/eval_epoch4/q_eval_ep4.npy").astype('float32')
index = faiss.read_index("artifacts/lvm/eval_epoch4/train_index_ep3.faiss")

# Load ground truth
eval_data = np.load("artifacts/lvm/eval_clean_disjoint.npz", allow_pickle=True)
train_data = np.load("artifacts/lvm/train_clean_disjoint.npz", allow_pickle=True)

eval_truth = eval_data['truth_keys']
train_truth = train_data['truth_keys']

print(f"  Eval queries: {len(q_eval)}")
print(f"  Index size: {index.ntotal}")
print("")

# Retrieve top-50
print("Retrieving top-50...")
K = 50
scores, indices = index.search(q_eval, K)

# Save hits and scores for reranker
print("Saving hits and scores...")
hits_data = []
for i in range(len(q_eval)):
    hits_data.append({
        "query_idx": int(i),
        "retrieved_indices": indices[i].tolist(),
        "scores": scores[i].tolist(),
        "gold_article": int(eval_truth[i][0]),
        "gold_chunk": int(eval_truth[i][1])
    })

with open("artifacts/lvm/eval_epoch4/hits50_ep4.jsonl", "w") as f:
    for hit in hits_data:
        f.write(json.dumps(hit) + "\n")

scores_summary = {
    "mean_score": float(scores.mean()),
    "std_score": float(scores.std()),
    "min_score": float(scores.min()),
    "max_score": float(scores.max()),
}
with open("artifacts/lvm/eval_epoch4/scores_ep4.json", "w") as f:
    json.dump(scores_summary, f, indent=2)

print("✅ Saved hits50_ep4.jsonl and scores_ep4.json")

# Compute metrics
print("Computing metrics...")
hits_at_k = defaultdict(int)
mrr_sum = 0.0
contain_count = 0

for i in range(len(q_eval)):
    gold_article, gold_chunk = eval_truth[i]
    retrieved = indices[i]

    # Check containment (gold article in top-50)
    retrieved_articles = train_truth[retrieved][:, 0]
    if gold_article in retrieved_articles:
        contain_count += 1

        # Find rank of first gold article chunk
        for rank, ret_idx in enumerate(retrieved):
            ret_article, ret_chunk = train_truth[ret_idx]
            if ret_article == gold_article:
                # Hit at this rank
                for k in [1, 3, 5, 10, 20, 50]:
                    if rank < k:
                        hits_at_k[k] += 1

                # MRR
                mrr_sum += 1.0 / (rank + 1)
                break

# Calculate metrics
N = len(q_eval)
metrics = {
    "epoch": 4,
    "eval_samples": int(N),
    "contain@50": float(contain_count / N),
    "R@1": float(hits_at_k[1] / N),
    "R@3": float(hits_at_k[3] / N),
    "R@5": float(hits_at_k[5] / N),
    "R@10": float(hits_at_k[10] / N),
    "R@20": float(hits_at_k[20] / N),
    "R@50": float(hits_at_k[50] / N),
    "MRR": float(mrr_sum / N),
}

# Save metrics
with open("artifacts/lvm/eval_epoch4/metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

# Print results
print("")
print("=" * 60)
print("EPOCH 4 RESULTS")
print("=" * 60)
print(f"Eval samples:  {N:,}")
print(f"Contain@50:    {metrics['contain@50']:.1%}")
print(f"R@1:           {metrics['R@1']:.1%}")
print(f"R@3:           {metrics['R@3']:.1%}")
print(f"R@5:           {metrics['R@5']:.1%}")
print(f"R@10:          {metrics['R@10']:.1%}")
print(f"R@20:          {metrics['R@20']:.1%}")
print(f"MRR:           {metrics['MRR']:.4f}")
print("=" * 60)
print("")

# Quality gates
print("QUALITY GATES:")
if metrics['R@5'] >= 0.30:
    print(f"✅ R@5 = {metrics['R@5']:.1%} >= 30% (PASS)")
elif metrics['MRR'] >= 0.20:
    print(f"✅ MRR = {metrics['MRR']:.4f} >= 0.20 (PASS)")
else:
    print(f"⚠️  R@5 = {metrics['R@5']:.1%} < 30%")
    print(f"⚠️  MRR = {metrics['MRR']:.4f} < 0.20")
    print("Consider vector-only MLP reranker to boost R@5")

print("")
print(f"✅ Results saved: artifacts/lvm/eval_epoch4/metrics.json")
EOF

echo ""
echo "========================================"
echo "EVALUATION COMPLETE"
echo "========================================"
echo "Results: artifacts/lvm/eval_epoch4/metrics.json"
echo ""
