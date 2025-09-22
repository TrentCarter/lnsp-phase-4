#!/usr/bin/env bash
set -euo pipefail
API="${1:-http://localhost:8080/search}"
QUERIES="${2:-eval/day3_eval.jsonl}"
NPZ="${3:-artifacts/fw1k_vectors.npz}"
TOPK="${4:-8}"

./venv/bin/python - <<'PY' "$API" "$QUERIES" "$NPZ" "$TOPK"
import json, sys, time, requests, numpy as np
from pathlib import Path

api = sys.argv[1]; qpath = Path(sys.argv[2]); npz = sys.argv[3]; topk = int(sys.argv[4])
data = np.load(npz, allow_pickle=True)
doc_ids = data["doc_ids"].tolist()
cpe_ids = data["cpe_ids"].tolist()
texts   = data["concept_texts"].tolist()

def offline_ids_for(q, k=8):
    # tiny lexical scorer using word overlap
    import re
    q_words = set(re.findall(r'\w+', q.lower()))
    scores = []
    for i, t in enumerate(texts):
        t_words = set(re.findall(r'\w+', t.lower()))
        overlap = len(q_words & t_words)
        if overlap > 0:
            scores.append((i, overlap))
    scores.sort(key=lambda x: x[1], reverse=True)
    ids = [cpe_ids[i] for i,s in scores[:k]]
    return ids

def api_ids_for(q, lane, k=8):
    r = requests.post(api, json={"q": q, "lane": lane, "top_k": k}, timeout=15)
    r.raise_for_status()
    js = r.json()
    return [it.get("id") for it in js.get("items", [])]

Q = [json.loads(l) for l in qpath.read_text().splitlines() if l.strip()]
ok, total = 0, 0
for row in Q:
    total += 1
    q, lane = row["query"], row["lane"]
    ids_off = set(offline_ids_for(q, topk))
    ids_api = set(api_ids_for(q, lane, topk))
    inter = len(ids_off & ids_api)
    jacc = inter / max(1, len(ids_off | ids_api))
    print(f'{row["id"]}: jacc={jacc:.2f} | api={len(ids_api)} off={len(ids_off)} inter={inter}')
    if jacc > 0: ok += 1

print(f"\nNon-empty intersection on {ok}/{total} queries.")
PY