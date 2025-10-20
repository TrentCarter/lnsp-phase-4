#!/bin/bash
cd /Users/trentcarter/Artificial_Intelligence/AI_Projects/lnsp-phase-4
./.venv/bin/python -m app.lvm.train_improved --model-type memory_gru --data artifacts/lvm/data_extended/training_sequences_ctx100.npz --epochs 2 --batch-size 16 --device mps
