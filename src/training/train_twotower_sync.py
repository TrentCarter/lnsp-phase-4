"""Stabilized trainer: synchronous FAISS, no multiprocessing anywhere.
- DataLoader: num_workers=0, pin_memory=False (MPS friendly)
- Optional resume from pre‑validation checkpoint
- Periodic miner refresh without multiprocessing
"""
from __future__ import annotations
import os, time
import torch
from torch.utils.data import DataLoader
import numpy as np
from src.retrieval.query_tower import QueryTower
from src.retrieval.miner_sync import SyncFaissMiner

# --- Replace with real dataset ---
class ContextDataset(torch.utils.data.Dataset):
    def __init__(self, npz_paths):
        self.paths = npz_paths
    def __len__(self):
        return len(self.paths)
    def __getitem__(self, i):
        # returns (T,768) fp32 unit‑norm, and positive id
        arr = np.load(self.paths[i])["context"].astype(np.float32)
        return torch.from_numpy(arr), torch.tensor(0, dtype=torch.long)


def l2norm(t: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.normalize(t, p=2, dim=-1)


def train_sync(cfg, faiss_index, bank_vectors_cpu: np.ndarray, resume_ckpt: str | None = None):
    device = torch.device("cpu")  # stabilize; enable MPS later

    model = QueryTower().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)

    if resume_ckpt and os.path.exists(resume_ckpt):
        ckpt = torch.load(resume_ckpt, map_location=device)
        model.load_state_dict(ckpt.get("model_q", ckpt))
        if "opt" in ckpt:
            opt.load_state_dict(ckpt["opt"])

    miner = SyncFaissMiner(faiss_index, nprobe=cfg.get("faiss", {}).get("nprobe", 8))

    ds = ContextDataset(cfg["train"].get("npz_list", []))
    dl = DataLoader(ds, batch_size=cfg["train"].get("batch_size", 8), shuffle=True, num_workers=0, pin_memory=False)

    tau = cfg["train"].get("temperature", 0.07)

    for epoch in range(cfg["train"].get("epochs", 30)):
        model.train()
        t0 = time.time()
        for step, (ctx, pos_id) in enumerate(dl):
            ctx = ctx.to(device)
            q = model(ctx)  # (B,768)
            q_np = q.detach().cpu().numpy().astype(np.float32)

            I, D = miner.search(q_np, k=cfg["faiss"].get("K", 500))
            # Gather doc vectors (identity tower)
            docs = bank_vectors_cpu[I]  # (B,K,768)
            # In‑batch softmax with single positive chosen as best match in Top‑K
            sims = (q_np[:, None, :] * docs).sum(-1)  # cosine (unit‑norm)
            sims_t = torch.from_numpy(sims).to(device)
            # Take top‑1 as pseudo positive
            pos_sim, _ = sims_t.max(dim=1)  # (B,)
            logits = sims_t / tau
            labels = torch.zeros(logits.size(0), dtype=torch.long, device=device)  # position 0 after sort
            # sort per row so that max appears at position 0
            sorted_logits, _ = torch.sort(logits, dim=1, descending=True)
            loss = torch.nn.functional.cross_entropy(sorted_logits, labels)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            if (step + 1) % 200 == 0:
                print(f"epoch {epoch} step {step+1} loss {loss.item():.4f} it/s ~{(step+1)/(time.time()-t0):.2f}")

        # Save checkpoint each epoch
        os.makedirs(cfg["train"].get("out_dir", "runs/two_tower_sync"), exist_ok=True)
        torch.save({"model_q": model.state_dict(), "opt": opt.state_dict()},
                   os.path.join(cfg["train"].get("out_dir", "runs/two_tower_sync"), f"epoch_{epoch:03d}.pt"))
