# Dual-Path System API Reference
**Version:** 1.0.0
**Date:** October 22, 2025
**Purpose:** Complete API documentation for all dual-path modules

---

## Table of Contents

1. [Decider Module](#decider-module) - Core decision logic
2. [Query Tower](#query-tower) - Neural query encoder
3. [FAISS Miners](#faiss-miners) - Sync/threaded retrieval
4. [Dual-Path Decoder](#dual-path-decoder) - Stateful wrapper
5. [Memory Profiling](#memory-profiling) - Diagnostics
6. [Configuration](#configuration) - Lane profiles & settings

---

## Decider Module

**Location:** `src/retrieval/decider.py`

### Data Classes

#### `LaneConfig`

Configuration for per-lane decision thresholds.

```python
@dataclass
class LaneConfig:
    tau_snap: float = 0.92      # Snap threshold (cosine ≥ tau_snap)
    tau_novel: float = 0.85      # Novel threshold (cosine ≤ tau_novel)
    lane_name: str = "neutral"   # Lane identifier
```

**Examples:**
```python
# Conservative (more grounding)
legal_lane = LaneConfig(tau_snap=0.94, tau_novel=0.88, lane_name="legal")

# Creative (more novelty)
creative_lane = LaneConfig(tau_snap=0.90, tau_novel=0.82, lane_name="creative")
```

---

#### `DecisionRecord`

Telemetry record for each decision step.

```python
@dataclass
class DecisionRecord:
    c_max: float                    # Max cosine to nearest neighbor
    decision: str                   # "SNAP" | "BLEND" | "NOVEL" | "NOVEL_DUP_DROP"
    neighbor_id: Optional[str]      # ID of selected neighbor (if any)
    alpha: Optional[float]          # Blending weight (if BLEND)
    lane: str                       # Lane name
    near_dup_drop: bool = False     # Whether duplicate was detected
```

**Example:**
```python
DecisionRecord(
    c_max=0.875,
    decision="BLEND",
    neighbor_id="cpe_12345",
    alpha=0.52,
    lane="neutral",
    near_dup_drop=False
)
```

---

### Core Functions

#### `choose_next_vector()`

Main dual-path decision function.

**Signature:**
```python
def choose_next_vector(
    v_hat: np.ndarray,                          # LVM-generated vector (768,)
    neighbors: Iterable[Tuple[str, np.ndarray, float]],  # (id, vec, cosine)
    lane_cfg: LaneConfig,                       # Lane configuration
    recent_ids: Iterable[str] = (),             # Recent IDs for dup detection
    near_dup_cos: float = 0.98,                 # Near-duplicate threshold
    near_dup_window: int = 8,                   # Recent window size
) -> Tuple[np.ndarray, DecisionRecord]:
    """
    Choose final vector per generation step: SNAP / BLEND / NOVEL.

    Args:
        v_hat: LVM-generated vector (768D, unit-norm)
        neighbors: Retriever candidates as (id, vec, cosine) tuples
        lane_cfg: Lane-specific thresholds
        recent_ids: Recently used neighbor IDs (for dup detection)
        near_dup_cos: Cosine threshold for duplicate detection
        near_dup_window: How many recent IDs to check

    Returns:
        (final_vector, decision_record)

    Decision Logic:
        - If c_max ≥ tau_snap → SNAP (use neighbor)
        - If c_max ≤ tau_novel → NOVEL (use LVM vector)
        - If tau_novel < c_max < tau_snap → BLEND (α-weighted mix)
        - If c_max > near_dup_cos AND id in recent → NOVEL_DUP_DROP

    Examples:
        >>> neighbors = [("n1", vec1, 0.93), ("n2", vec2, 0.85)]
        >>> v_out, rec = choose_next_vector(v_hat, neighbors, lane_cfg)
        >>> print(rec.decision)  # "SNAP" (0.93 > 0.92)
    """
```

---

#### `alpha_from_cos()`

Blending weight schedule.

**Signature:**
```python
def alpha_from_cos(c: float) -> float:
    """
    Compute blending weight α from cosine similarity.

    Schedule:
        c ≤ 0.86  → α = 0.3 (30% LVM, 70% bank)
        0.86-0.91 → α = 0.3 to 0.7 (linear)
        0.91-0.95 → α = 0.7 to 0.9 (linear)
        c ≥ 0.95  → α = 0.9 (90% LVM, 10% bank)

    Higher α = more LVM influence.

    Examples:
        >>> alpha_from_cos(0.85)
        0.3
        >>> alpha_from_cos(0.88)
        0.5
        >>> alpha_from_cos(0.92)
        0.725
        >>> alpha_from_cos(0.98)
        0.9
    """
```

---

#### `l2norm()`

L2 normalization helper.

**Signature:**
```python
def l2norm(x: np.ndarray) -> np.ndarray:
    """Unit-normalize vector (handles zero vectors)."""
```

---

## Query Tower

**Location:** `src/retrieval/query_tower.py`

### `QueryTower`

Neural query encoder for Two-Tower retrieval.

**Architecture:**
```
Input: (B, T, 768) context vectors
  ↓
GRU(768 → 768, 1 layer)
  ↓
Mean pooling over T
  ↓
LayerNorm
  ↓
L2 normalize
  ↓
Output: (B, 768) query vectors
```

**Signature:**
```python
class QueryTower(nn.Module):
    def __init__(self, hidden_size: int = 768, layers: int = 1):
        """
        Args:
            hidden_size: GRU hidden dimension (default: 768)
            layers: Number of GRU layers (default: 1)
        """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode context to query vector.

        Args:
            x: (B, T, 768) context vectors (GTR-T5 embeddings)

        Returns:
            (B, 768) unit-normalized query vectors

        Example:
            >>> tower = QueryTower()
            >>> ctx = torch.randn(4, 10, 768)  # Batch 4, context length 10
            >>> q = tower(ctx)
            >>> print(q.shape)  # torch.Size([4, 768])
            >>> print(torch.norm(q[0]))  # ~1.0 (unit-norm)
        """
```

**Parameters:**
- Total: 4,725,504 parameters
- GRU: 4,722,432 (768×768 + biases)
- LayerNorm: 1,536 (γ, β)
- Projection: 1,536 (bias only)

---

## FAISS Miners

### `SyncFaissMiner`

**Location:** `src/retrieval/miner_sync.py`

Synchronous FAISS search (no multiprocessing).

**Signature:**
```python
class SyncFaissMiner:
    def __init__(self, index: faiss.Index, nprobe: int = 8):
        """
        Synchronous FAISS miner for training stability.

        Args:
            index: FAISS index (IVF/HNSW/Flat)
            nprobe: Number of clusters to probe (IVF only)

        Example:
            >>> import faiss
            >>> index = faiss.read_index("artifacts/my_index.index")
            >>> miner = SyncFaissMiner(index, nprobe=8)
        """

    def search(
        self,
        queries: np.ndarray,  # (B, 768) float32
        k: int = 500
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search FAISS index (synchronous, main thread).

        Args:
            queries: (B, 768) float32 unit-norm query vectors
            k: Number of neighbors to retrieve

        Returns:
            (indices, distances) both (B, k)
            - indices: int64 neighbor indices
            - distances: float32 cosine similarities

        Latency: 2-5 ms (771k bank, IVF1024, nprobe=8, CPU)

        Example:
            >>> queries = np.random.randn(4, 768).astype(np.float32)
            >>> queries /= np.linalg.norm(queries, axis=1, keepdims=True)
            >>> I, D = miner.search(queries, k=100)
            >>> print(I.shape)  # (4, 100)
            >>> print(D.shape)  # (4, 100)
        """
```

---

### `ThreadedMiner`

**Location:** `src/retrieval/miner_threaded.py`

Threaded FAISS search (no multiprocessing, threads only).

**Signature:**
```python
class ThreadedMiner:
    def __init__(
        self,
        faiss_index: faiss.Index,
        n_workers: int = 2,
        max_qsize: int = 8,
        timeout_s: float = 1.0
    ):
        """
        Threaded FAISS miner (intermediate performance/stability).

        Args:
            faiss_index: FAISS index
            n_workers: Number of worker threads (default: 2)
            max_qsize: Max queue size (bounded, prevents memory explosion)
            timeout_s: Cooperative timeout for put/get operations

        Example:
            >>> miner = ThreadedMiner(index, n_workers=2, max_qsize=8)
        """

    def submit(self, qid: int, queries: np.ndarray, k: int = 500) -> bool:
        """
        Submit queries for async processing.

        Args:
            qid: Query ID (for tracking)
            queries: (B, 768) float32 queries
            k: Number of neighbors

        Returns:
            True if submitted, False if queue full

        Example:
            >>> success = miner.submit(qid=0, queries=q_np, k=500)
            >>> if not success:
            ...     print("Queue full, fallback to sync")
        """

    def receive(self, timeout_s: Optional[float] = None) -> Optional[Tuple]:
        """
        Receive results from worker threads.

        Args:
            timeout_s: Timeout (None = default timeout)

        Returns:
            (qid, indices, distances) or None if timeout

        Example:
            >>> result = miner.receive(timeout_s=1.0)
            >>> if result:
            ...     qid, I, D = result
        """

    def stop(self):
        """Stop worker threads gracefully."""
```

---

## Dual-Path Decoder

**Location:** `src/training/dual_path_decoder.py`

Stateful wrapper for per-step decoding.

**Signature:**
```python
class DualPathDecoder:
    def __init__(
        self,
        lane: str,
        tau_snap: float,
        tau_novel: float,
        near_dup_cos: float = 0.98,
        near_dup_window: int = 8
    ):
        """
        Stateful decoder managing recent IDs for duplicate detection.

        Args:
            lane: Lane name ("neutral", "conservative", "creative")
            tau_snap: Snap threshold
            tau_novel: Novel threshold
            near_dup_cos: Duplicate cosine threshold
            near_dup_window: Recent ID buffer size

        Example:
            >>> decoder = DualPathDecoder(
            ...     lane="neutral",
            ...     tau_snap=0.92,
            ...     tau_novel=0.85
            ... )
        """

    def step(
        self,
        v_hat: np.ndarray,
        neighbors: Iterable[Tuple[str, np.ndarray, float]]
    ) -> Tuple[np.ndarray, DecisionRecord]:
        """
        Process one generation step.

        Args:
            v_hat: LVM-generated vector (768,)
            neighbors: Retriever candidates

        Returns:
            (final_vector, decision_record)

        Side Effects:
            - Updates internal recent_ids buffer
            - Maintains last 64 neighbor IDs

        Example:
            >>> for step in range(max_len):
            ...     v_hat = lvm.forward(context)
            ...     neighbors = retriever.search(context, k=500)
            ...     v_out, rec = decoder.step(v_hat, neighbors)
            ...     print(f"Step {step}: {rec.decision}")
            ...     context.append(v_out)
        """
```

---

## Memory Profiling

**Location:** `src/utils/memprof.py`

### Functions

```python
def rss_mb() -> float:
    """Get current process RSS memory in MB."""

def log_mem(prefix="mem"):
    """
    Print formatted memory usage.

    Args:
        prefix: Log prefix (default: "mem")

    Output:
        [prefix] rss_mb=1234.5

    Example:
        >>> from src.utils.memprof import log_mem
        >>> log_mem("step_100")
        [step_100] rss_mb=1523.7
    """
```

---

## Configuration

**Location:** `configs/dual_path.yaml`

### Lane Profiles

```yaml
profiles:
  conservative:
    tau_snap: 0.94   # More grounding
    tau_novel: 0.88  # Wider blend band
    K: 500           # Number of candidates

  neutral:
    tau_snap: 0.92   # Balanced
    tau_novel: 0.85  # Default
    K: 500

  creative:
    tau_snap: 0.90   # More novelty
    tau_novel: 0.82  # Narrower blend band
    K: 300           # Fewer candidates (faster)
```

### FAISS Settings

```yaml
faiss:
  mode: sync            # sync | threaded
  index_type: ivf_flat  # IVF with flat quantization
  nlist: 1024           # Number of IVF clusters
  nprobe: 8             # Clusters to probe (4-32)
  qbatch: 1024          # Query batch size
  prefetch_depth: 0     # Prefetch queue depth
  lane_shard: true      # Shard index by lane
```

### Near-Duplicate Detection

```yaml
near_duplicate:
  cosine_threshold: 0.98  # Threshold for dup detection
  recent_window: 8        # Recent IDs to check
```

---

## Complete Usage Example

```python
import numpy as np
import faiss
from src.retrieval.query_tower import QueryTower
from src.retrieval.miner_sync import SyncFaissMiner
from src.training.dual_path_decoder import DualPathDecoder

# 1. Load resources
index = faiss.read_index("artifacts/my_index.index")
bank = np.load("artifacts/bank_vectors.npy")  # (N, 768)

# 2. Initialize components
query_tower = QueryTower()
query_tower.load_state_dict(torch.load("models/query_tower.pt"))
query_tower.eval()

miner = SyncFaissMiner(index, nprobe=8)

decoder = DualPathDecoder(
    lane="neutral",
    tau_snap=0.92,
    tau_novel=0.85
)

# 3. Generation loop
context = initial_context  # (T, 768) numpy array

for step in range(max_length):
    # LVM generates next vector
    v_hat = lvm.forward(context)  # (768,) numpy

    # Query tower encodes context
    ctx_torch = torch.from_numpy(context).unsqueeze(0)  # (1, T, 768)
    with torch.no_grad():
        q = query_tower(ctx_torch)  # (1, 768)
    q_np = q.cpu().numpy()[0]  # (768,)

    # Retriever finds candidates
    I, D = miner.search(q_np[None, :], k=500)  # (1, 500)
    neighbors = [
        (f"bank_{i}", bank[i], D[0, j])
        for j, i in enumerate(I[0])
    ]

    # Dual-path decision
    v_out, rec = decoder.step(v_hat, neighbors)

    # Log telemetry
    print(f"Step {step}: {rec.decision} (c_max={rec.c_max:.3f})")

    # Update context
    context = np.vstack([context, v_out])

    # Decode to text
    text = vec2text_decode(v_out)
    print(f"  → {text}")
```

---

## Performance Characteristics

### Latency (per step)

| Component | Latency | Notes |
|-----------|---------|-------|
| Query tower | 0.5-2 ms | (B=1, T=10, CPU) |
| Sync miner | 2-5 ms | (771k bank, IVF1024, nprobe=8, CPU) |
| Threaded miner | 1.5-4 ms | (2 workers, +20-40% speedup) |
| Decision | <0.1 ms | (negligible) |
| **Total retrieval** | **2.5-7 ms** | (excluding LVM/vec2text) |

### Memory Usage

| Component | Memory | Notes |
|-----------|--------|-------|
| Query tower | ~50 MB | (model weights) |
| Bank vectors | ~600 MB | (771k × 768 × 4 bytes) |
| FAISS index | ~2.2 GB | (IVF1024, flat quantization) |
| Decoder state | <1 MB | (64 recent IDs) |
| **Total** | **~2.9 GB** | (excluding LVM) |

---

## Error Handling

### Common Issues

**1. Near-duplicate overload**
```python
# Symptom: All decisions are NOVEL_DUP_DROP
# Cause: near_dup_window too large or near_dup_cos too low
# Fix: Increase near_dup_cos or decrease near_dup_window

decoder = DualPathDecoder(..., near_dup_cos=0.99, near_dup_window=4)
```

**2. FAISS index mismatch**
```python
# Symptom: IndexError in miner.search()
# Cause: Index size != bank size
# Fix: Rebuild index with correct vectors

assert index.ntotal == len(bank), "Index/bank size mismatch"
```

**3. Non-unit-norm vectors**
```python
# Symptom: Cosine similarities out of [-1, 1]
# Cause: Vectors not normalized
# Fix: Normalize before passing to choose_next_vector()

from src.retrieval.decider import l2norm
v_hat = l2norm(v_hat)
```

---

## See Also

- **PRD:** `docs/PRDs/PRD_LVM_plus_vecRAG_Dual_Path_Next_Vector_Generation_System.md`
- **Integration Report:** `docs/reports/Dual_Path_Integration_Report_2025-10-22.md`
- **Triage Guide:** `docs/ops/triage_playbook.md`
- **Training Patches:** `docs/ops/TRAINING_PATCHES.md`

---

**Generated:** October 22, 2025
**Maintainer:** Autonomous System
**Version:** 1.0.0
