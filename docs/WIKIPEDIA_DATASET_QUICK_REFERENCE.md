# Wikipedia 790K Dataset - Quick Reference

## 🗂️ File Locations at a Glance

| What | Where | Size | Count |
|------|-------|------|-------|
| **Raw Text** | `data/datasets/wikipedia/wikipedia_500k.jsonl` | 2.1 GB | 500K articles |
| **PostgreSQL** | `lnsp` database | - | 790,391 chunks |
| **Vectors** | `artifacts/wikipedia_500k_corrected_vectors.npz` | 2.1 GB | 771,115 vectors |
| **Training Data** | `artifacts/wikipedia_584k_fresh.npz` | - | 584K+ sequences |
| **Test Sample** | `data/datasets/wikipedia/full_wikipedia.jsonl` | - | Full dataset |
| **Small Sample** | `test_data/100_test_chunks.jsonl` | - | 100 chunks |

## 🚀 Common Commands

```bash
# Count Wikipedia chunks in JSONL
wc -l data/datasets/wikipedia/wikipedia_500k.jsonl

# View first chunk
head -n 1 data/datasets/wikipedia/wikipedia_500k.jsonl | jq '.'

# Check vector dimensions
python -c "import numpy as np; d=np.load('artifacts/wikipedia_500k_corrected_vectors.npz'); print(d['embeddings'].shape)"

# Test DIRECT pipeline on dashboard
curl http://localhost:8999/evaluate -X POST -H "Content-Type: application/json" \
  -d '{"models":["DIRECT"],"test_mode":"both","num_test_cases":5}'
```

## 🔌 Service Ports

- **GTR-T5 Encoder:** Port 7001 (text → 768D vectors)
- **Vec2Text Decoder:** Port 7002 (768D vectors → text)
- **LVM Dashboard:** Port 8999 (evaluation interface)
- **PostgreSQL:** Port 5432 (database)

## 📊 Key Numbers

- **Total Wikipedia Chunks:** 790,391
- **Available Vectors:** 771,115
- **Vector Dimensions:** 768
- **Raw Articles:** 500,000
- **Training Sequences:** 584,000+
- **File Size:** ~2.1 GB each (JSONL & NPZ)

## 🎯 Primary Uses

1. **LVM Training:** P6 sequences from `wikipedia_584k_fresh.npz`
2. **Evaluation:** Dashboard uses `wikipedia_500k.jsonl` chunks
3. **Vector Search:** FAISS index from `wikipedia_500k_corrected_vectors.npz`
4. **API Backend:** PostgreSQL with 790K chunks
5. **Pipeline Testing:** DIRECT model (text→GTR-T5→vec2text)

## ⚡ Quick Python Access

```python
# Load Wikipedia chunks
import json
with open('data/datasets/wikipedia/wikipedia_500k.jsonl') as f:
    chunk = json.loads(f.readline())
    print(f"Title: {chunk['title']}")
    print(f"Text: {chunk['text'][:200]}...")

# Load vectors
import numpy as np
vectors = np.load('artifacts/wikipedia_500k_corrected_vectors.npz')['embeddings']
print(f"Loaded {vectors.shape[0]} vectors of dimension {vectors.shape[1]}")
```

---
*For detailed information, see [WIKIPEDIA_790K_DATASET_GUIDE.md](./WIKIPEDIA_790K_DATASET_GUIDE.md)*
