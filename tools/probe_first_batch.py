#!/usr/bin/env python3
# tools/probe_first_batch.py
from __future__ import annotations
import os, sys, time, argparse
import threading

# --- Stability env (must be set before importing numpy/torch/faiss) ---
os.environ.setdefault("PYTHONUNBUFFERED", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_MAX_THREADS", "1")

import faulthandler; faulthandler.enable(all_threads=True)

# Optional external stackdump trigger via SIGUSR1
try:
    import signal
    faulthandler.register(signal.SIGUSR1, file=sys.stderr, all_threads=True)
except Exception:
    pass

# repo imports (adjust if your package root differs)
import numpy as np
import torch
from torch.utils.data import DataLoader

# Local modules
try:
    from src.retrieval.query_tower import QueryTower
    from src.retrieval.miner_sync import SyncFaissMiner
except Exception as e:
    print(f"[probe] Failed to import modules: {e}", file=sys.stderr)
    sys.exit(2)

# Minimal dataset stub; replace if you have a packaged dataset class
class ContextDataset(torch.utils.data.Dataset):
    def __init__(self, npz_list):
        self.paths = npz_list
    def __len__(self):
        return len(self.paths)
    def __getitem__(self, i):
        arr = np.load(self.paths[i])["context"].astype(np.float32)  # (T,768) unit‑norm
        return torch.from_numpy(arr), torch.tensor(0, dtype=torch.long)


def watchdog(last_tick_ref, timeout_s=120, poll=5):
    while True:
        time.sleep(poll)
        if time.time() - last_tick_ref[0] > timeout_s:
            sys.stderr.write("\n[WATCHDOG] No progress — dumping stacks\n"); sys.stderr.flush()
            faulthandler.dump_traceback(file=sys.stderr, all_threads=True)
            last_tick_ref[0] = time.time()  # avoid spam


def main():
    p = argparse.ArgumentParser(description="Probe a single training step for hangs.")
    p.add_argument("--bank", required=True, help="Path to bank vectors .fp32 memmap (shape: N,768)")
    p.add_argument("--index", required=True, help="Path to FAISS index (IVF/HNSW/Flat)")
    p.add_argument("--npz_list", required=True, help="Text file listing .npz context files (one per line)")
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--K", type=int, default=200)
    p.add_argument("--nprobe", type=int, default=8)
    p.add_argument("--timeout", type=int, default=120)
    args = p.parse_args()

    # Device: force CPU for maximal determinism
    device = torch.device("cpu")

    # Load bank memmap
    # Expect a header line with rows if needed, otherwise infer via file size
    bank = np.memmap(args.bank, dtype="float32", mode="r")
    if bank.size % 768 != 0:
        print(f"[probe] Bank file size not divisible by 768 floats: {bank.size}", file=sys.stderr)
        sys.exit(3)
    bank = bank.reshape((-1, 768))

    # Load FAISS index
    import faiss
    index = faiss.read_index(args.index)
    if index.ntotal != bank.shape[0]:
        print(f"[probe] Index/bank mismatch: {index.ntotal} vs {bank.shape[0]}", file=sys.stderr)
        sys.exit(4)
    try:
        faiss.ParameterSpace().set_index_parameter(index, "nprobe", args.nprobe)
    except Exception:
        pass

    # Build dataset & loader
    with open(args.npz_list) as f:
        npz_paths = [ln.strip() for ln in f if ln.strip()]
    if not npz_paths:
        print("[probe] npz_list is empty", file=sys.stderr); sys.exit(5)

    ds = ContextDataset(npz_paths)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=0, pin_memory=False)

    model = QueryTower().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)

    miner = SyncFaissMiner(index, nprobe=args.nprobe)

    last_tick = [time.time()]
    threading.Thread(target=watchdog, args=(last_tick, args.timeout), daemon=True).start()

    # ---- Timed probe ----
    t0 = time.time()
    batch = next(iter(dl))
    t_load = time.time()

    ctx, _ = batch
    ctx = ctx.to(device)
    q = model(ctx)  # (B,768)
    t_fwd = time.time()

    q_np = q.detach().cpu().numpy().astype(np.float32)
    I, D = miner.search(q_np, k=args.K)
    t_retr = time.time()

    # Simple InfoNCE‑ish loss with pseudo positives
    docs = bank[I]  # (B,K,768)
    sims = (q_np[:, None, :] * docs).sum(-1)
    sims_t = torch.from_numpy(sims).to(device)
    logits = sims_t / 0.07
    labels = torch.zeros(logits.size(0), dtype=torch.long, device=device)
    sorted_logits, _ = torch.sort(logits, dim=1, descending=True)
    loss = torch.nn.functional.cross_entropy(sorted_logits, labels)
    t_loss = time.time()

    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    t_bwd = time.time()
    last_tick[0] = time.time()

    print(
        (
            f"[probe] OK | load={t_load-t0:.3f}s fwd={t_fwd-t_load:.3f}s "
            f"retr={t_retr-t_fwd:.3f}s loss={t_loss-t_retr:.3f}s bwd={t_bwd-t_loss:.3f}s | "
            f"loss={loss.item():.4f} B={args.batch} K={args.K} nprobe={args.nprobe}"
        ), flush=True
    )

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("[probe] Interrupted", file=sys.stderr)
        sys.exit(130)
