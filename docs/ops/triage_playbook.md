# TwoTower v4 Crash Triage Playbook

## Symptoms
- "Output queue empty" warnings
- it/s decay then crash during training (not validation)

## Likely Causes
1. Multiprocessing miner + DataLoader deadlock
2. Memory creep in queues (IDs, D arrays)

## Fix‑Order (must follow)
1) Switch to **synchronous miner** (no MP). `num_workers=0`, `pin_memory=False`.
2) Shrink scope: batch=8, bank=5k (subset index) for a clean epoch 0.
3) Add `memprof.log_mem()` per 500 steps; alert if +500 MB.
4) If stable, scale K→500, bank→771k.
5) Only then try `ThreadedMiner` (no MP). Never run FAISS in forked workers.

## Validation Gates
- Complete 1 full epoch without stalls.
- Peak RSS ≤ baseline + 1.0 GB.
- Miner latency p95 ≤ 2.5 ms (771k, IVF1024,nprobe=8) on CPU.
