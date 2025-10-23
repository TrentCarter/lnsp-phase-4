# ---- async_miner.py (drop-in snippet) ---------------------------------------
import threading, queue, numpy as np, torch, faiss, time

class AsyncMiner:
    """Batched FAISS miner with TTL cache + producer/consumer queues."""
    def __init__(self, faiss_index, k=128, qbatch=1024, prefetch=2, ttl=3):
        self.index = faiss_index; self.k = k; self.qb = qbatch; self.ttl = ttl
        self.in_q  = queue.Queue(maxsize=prefetch)
        self.out_q = queue.Queue(maxsize=prefetch)
        self.cache = {}; self._stop = False
        self.th = threading.Thread(target=self._worker, daemon=True)

    def start(self): self.th.start()
    def stop(self):
        """Stop worker thread with proper cleanup (critical for MPS)"""
        self._stop = True
        # Drain input queue to unblock worker
        try:
            while not self.in_q.empty():
                self.in_q.get_nowait()
        except: pass
        # Wait longer for FAISS operations to complete (MPS needs this)
        self.th.join(timeout=5.0)
        if self.th.is_alive():
            print("⚠️  AsyncMiner thread still alive after 5s shutdown")

    def submit(self, step, Q):  # Q: [N,768] torch or np.float32, L2-normalized
        if isinstance(Q, torch.Tensor): Q = Q.detach().cpu().float().numpy()
        self.in_q.put((step, Q), block=True)

    def try_get(self):
        try: return self.out_q.get_nowait()  # (step, I: np.int64 [N,k])
        except queue.Empty: return None

    def _search(self, Q):
        out = []
        for i in range(0, len(Q), self.qb):
            _, I = self.index.search(Q[i:i+self.qb], self.k)
            out.append(I)
        return np.concatenate(out, 0)

    def _worker(self):
        while not self._stop:
            try: step, Q = self.in_q.get(timeout=0.1)
            except queue.Empty: continue
            key = hash(Q.tobytes())
            entry = self.cache.get(key)
            if entry and entry["ttl"] > 0:
                I = entry["I"]; entry["ttl"] -= 1
            else:
                I = self._search(Q)
                self.cache[key] = {"I": I, "ttl": self.ttl}
                if len(self.cache) > 512: self.cache.pop(next(iter(self.cache)))
            self.out_q.put((step, I), block=False)
# -----------------------------------------------------------------------------
