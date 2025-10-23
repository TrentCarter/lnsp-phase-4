#!/usr/bin/env bash
set -euo pipefail
CFG=${1:-configs/dual_path.yaml}
CKPT=${2:-runs/twotower_v4_cpu_test/checkpoints/epoch_001_pre_validation.pt}
PYTHON=${PYTHON:-python3}

$PYTHON - <<'PY'
import yaml, numpy as np
from pathlib import Path
from src.training.train_twotower_sync import train_sync
import faiss

cfg_path = "configs/dual_path.yaml"
with open(cfg_path) as f:
    cfg = yaml.safe_load(f)

# Load FAISS index and bank vectors
index = faiss.index_factory(768, "IVF1024,Flat", faiss.METRIC_INNER_PRODUCT)
# NOTE: Replace with your persisted index loader

bank = np.memmap("data/bank_vectors.fp32", dtype="float32", mode="r", shape=(771000, 768))

train_sync(cfg|{"train": {"npz_list": [], "batch_size": 8, "epochs": 30}}, index, bank, resume_ckpt="runs/twotower_v4_cpu_test/checkpoints/epoch_001_pre_validation.pt")
PY
