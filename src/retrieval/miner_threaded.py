"""Threaded FAISS miner (no multiprocessing),
   with bounded queues and cooperative timeouts.
"""
from __future__ import annotations
import threading
import queue
import time
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np

@dataclass
class Job:
    qid: int
    queries: np.ndarray  # (B, 768)
    k: int

class ThreadedMiner:
    def __init__(self, faiss_index, n_workers: int = 2, max_qsize: int = 8, timeout_s: float = 1.0):
        self.index = faiss_index
        self.in_q: "queue.Queue[Job]" = queue.Queue(maxsize=max_qsize)
        self.out_q: "queue.Queue[Tuple[int, np.ndarray, np.ndarray]]" = queue.Queue(maxsize=max_qsize)
        self._stop = threading.Event()
        self.timeout_s = timeout_s
        self.workers = [threading.Thread(target=self._loop, daemon=True) for _ in range(n_workers)]
        for w in self.workers:
            w.start()

    def _loop(self):
        import faiss
        while not self._stop.is_set():
            try:
                job = self.in_q.get(timeout=self.timeout_s)
            except queue.Empty:
                continue
            D, I = self.index.search(job.queries.astype(np.float32), job.k)
            try:
                self.out_q.put((job.qid, I, D), timeout=self.timeout_s)
            except queue.Full:
                # Drop oldest silently; caller has TTL cache fallback
                pass
            finally:
                self.in_q.task_done()

    def submit(self, qid: int, queries: np.ndarray, k: int = 500) -> bool:
        try:
            self.in_q.put(Job(qid, queries, k), timeout=self.timeout_s)
            return True
        except queue.Full:
            return False

    def receive(self, timeout_s: Optional[float] = None):
        try:
            return self.out_q.get(timeout=timeout_s or self.timeout_s)
        except queue.Empty:
            return None

    def stop(self):
        self._stop.set()
        for _ in self.workers:
            try:
                self.in_q.put_nowait(Job(-1, np.zeros((1,768), np.float32), 1))
            except Exception:
                pass
        for w in self.workers:
            w.join(timeout=1.0)
