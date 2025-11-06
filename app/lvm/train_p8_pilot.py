#!/usr/bin/env python3
import argparse, os, time, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ---- Try to import your P8 implementations; fall back to minimal local versions
ModelImported = False
LossImported  = False
try:
    from app.lvm.models_p8_constrained import TransformerP8Constrained  # your impl
    ModelImported = True
except Exception:
    pass
try:
    from app.lvm.losses_p8_listwise import listwise_loss_with_prev_margin  # your impl
    LossImported = True
except Exception:
    pass

# ---- Minimal stand-ins (only used if your modules aren't importable)
class TinyP8Mixture(nn.Module):
    """Constrained mixture head: q = norm(sum_i alpha_i * c_i). Context encoder is a tiny MLP."""
    def __init__(self, dim=768, ctx_len=5, hidden=512):
        super().__init__()
        self.ctx_len = ctx_len
        self.dim = dim
        self.enc = nn.Sequential(
            nn.Linear(ctx_len*dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU())
        self.attn = nn.Linear(hidden, ctx_len)

    def forward(self, C):  # C: (B, L=5, D)
        B, L, D = C.shape
        x = C.reshape(B, L*D)
        h = self.enc(x)
        alpha = torch.softmax(self.attn(h), dim=-1)              # (B, L)
        q = (alpha.unsqueeze(-1) * F.normalize(C, dim=-1)).sum(dim=1)
        return F.normalize(q, dim=-1), alpha

def tiny_listwise_with_prev(q, candidates, temperature=0.07, w_prev=1.0, margin=0.07):
    """
    candidates: (B, Lcand, D) where index 0 = true next, index 1 = prev, others = negatives.
    """
    q = F.normalize(q, dim=-1)
    cand = F.normalize(candidates, dim=-1)
    scores = (q.unsqueeze(1) * cand).sum(dim=-1)            # cosine
    loss_listwise = F.cross_entropy(scores/temperature, torch.zeros(q.size(0), dtype=torch.long, device=q.device))
    sp = scores[:, 0]                                       # next
    sn_prev = scores[:, 1]                                  # prev
    loss_prev = torch.clamp(margin - sp + sn_prev, min=0).mean()
    return loss_listwise + w_prev*loss_prev, scores

# ---- Dataset
class SequencesNPZDataset(Dataset):
    def __init__(self, npz_path):
        self.data = np.load(npz_path, allow_pickle=True)
        # Be tolerant to schema names
        # contexts: (N, L, D), targets: (N, D)
        for ck in ["contexts", "context", "C"]:
            if ck in self.data: self.contexts = self.data[ck]; break
        else: raise ValueError("No contexts key found in NPZ (expected one of: contexts/context/C)")
        for tk in ["targets", "target", "Y", "t"]:
            if tk in self.data: self.targets = self.data[tk]; break
        else: raise ValueError("No targets key found in NPZ (expected one of: targets/target/Y/t)")

        self.article_ids = self.data["article_ids"] if "article_ids" in self.data else None
        self.N, self.L, self.D = self.contexts.shape
        # previous chunk is the most recent context element
        self.prevs = self.contexts[:, -1, :]

    def __len__(self): return self.N
    def __getitem__(self, i):
        return (self.contexts[i].astype(np.float32),
                self.targets[i].astype(np.float32),
                self.prevs[i].astype(np.float32),
                (-1 if self.article_ids is None else int(self.article_ids[i])))

# ---- Utilities
def build_inbatch_candidates(t_pos, t_prev, inbatch_targets, n_extra=30):
    """
    Build listwise candidates: [t_next, t_prev, random in-batch negatives...]
    """
    B, D = t_pos.shape
    # random negatives from in-batch (avoid self)
    idx = torch.randint(0, inbatch_targets.size(0), (B, n_extra), device=t_pos.device)
    row = torch.arange(B, device=t_pos.device).unsqueeze(1)
    idx[row==idx] = (idx[row==idx] + 1) % inbatch_targets.size(0)  # avoid self
    negs = inbatch_targets[idx]  # (B, n_extra, D)
    cand = torch.cat([t_pos.unsqueeze(1), t_prev.unsqueeze(1), negs], dim=1)  # (B, 2+n_extra, D)
    return cand

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    m_cos_next = []; m_cos_prev = []; m_margin = []; m_anchor = []
    # Collect all q and all targets for R@5
    all_q = []; all_t = []
    for C, t, p, _ in loader:
        C = C.to(device); t = t.to(device); p = p.to(device)
        if ModelImported: q, _ = model(C)  # your model returns (q, alpha) or q
        else: q, _ = model(C)
        q = F.normalize(q, dim=-1); t = F.normalize(t, dim=-1); p = F.normalize(p, dim=-1)
        cos_next = (q * t).sum(-1)
        cos_prev = (q * p).sum(-1)
        margin = cos_next - cos_prev
        # proxy "anchor" = cos(q, mean(C)) since q∈span(C)
        anchor = (q * F.normalize(C.mean(dim=1), dim=-1)).sum(-1)
        m_cos_next.append(cos_next.mean().item())
        m_cos_prev.append(cos_prev.mean().item())
        m_margin.append(margin.mean().item())
        m_anchor.append(anchor.mean().item())
        all_q.append(q.cpu()); all_t.append(t.cpu())
    Q = torch.cat(all_q, dim=0)     # (N,D)
    T = torch.cat(all_t, dim=0)     # (N,D)
    sims = Q @ T.T                  # (N,N)
    # R@5
    top5 = sims.topk(5, dim=1).indices
    correct = torch.arange(sims.size(0)).unsqueeze(1)
    r5 = (top5 == correct).any(dim=1).float().mean().item()
    return {
        "cos_next": float(np.mean(m_cos_next)),
        "cos_prev": float(np.mean(m_cos_prev)),
        "margin":   float(np.mean(m_margin)),
        "cos_anchor": float(np.mean(m_anchor)),
        "R_at_5": r5
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--bsz", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--temperature", type=float, default=0.07)
    ap.add_argument("--w_prev", type=float, default=1.0)
    ap.add_argument("--margin", type=float, default=0.07)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    train_ds = SequencesNPZDataset(args.train)
    val_ds   = SequencesNPZDataset(args.val)
    train_dl = DataLoader(train_ds, batch_size=args.bsz, shuffle=True, num_workers=0, drop_last=True)
    val_dl   = DataLoader(val_ds, batch_size=args.bsz, shuffle=False, num_workers=0)

    dim, L = train_ds.D, train_ds.L
    if ModelImported:
        model = TransformerP8Constrained(dim=dim, ctx_len=L)  # your ctor
    else:
        model = TinyP8Mixture(dim=dim, ctx_len=L)
    model.to(args.device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    print(f"Pilot start | device={args.device} | epochs={args.epochs} | dim={dim} | L={L}")

    for epoch in range(1, args.epochs+1):
        model.train()
        losses = []
        for C, t, p, _ in train_dl:
            C = C.to(args.device); t = t.to(args.device); p = p.to(args.device)
            if ModelImported:
                q, _ = model(C)
            else:
                q, _ = model(C)
            candidates = build_inbatch_candidates(t, p, t, n_extra=30)
            if LossImported:
                loss, _ = listwise_loss_with_prev_margin(q, candidates,
                                                         temperature=args.temperature,
                                                         w_prev=args.w_prev, margin=args.margin)
            else:
                loss, _ = tiny_listwise_with_prev(q, candidates,
                                                  temperature=args.temperature,
                                                  w_prev=args.w_prev, margin=args.margin)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            losses.append(loss.item())
        eval_metrics = evaluate(model, val_dl, args.device)
        print(f"[E{epoch}] loss={np.mean(losses):.4f} | "
              f"margin={eval_metrics['margin']:.4f} | "
              f"cos_next={eval_metrics['cos_next']:.4f} | "
              f"cos_prev={eval_metrics['cos_prev']:.4f} | "
              f"cos_anchor={eval_metrics['cos_anchor']:.4f} | "
              f"R@5={eval_metrics['R_at_5']:.3f}")
        # Kill criteria (fast feedback)
        if epoch >= 2 and eval_metrics["margin"] < 0:
            print("Margin negative after epoch 2 → aborting pilot.")
            break

if __name__ == "__main__":
    main()
