# Ingest Timing Benchmark Commands (Oct 12, 2025)

This note captures the exact CLI invocations and context used to measure the current `/ingest` service performance after the batch-write fixes. All commands were executed from the project root with services warm (TinyLlama cache primed, in-proc GTR-T5 running).

## 1. Single-Chunk Warm-Up Probe

Five sequential calls against `/ingest` with a single chunk, 500 ms delay between calls. The final run's per-step timings were reported.

```
cd /Users/trentcarter/Artificial_Intelligence/AI_Projects/lnsp-phase-4 && python - <<'PY'
import time
import requests

payload = {
    "chunks": [
        {
            "text": "The water cycle describes movement of water across Earth's systems.",
            "source_document": "timing_doc",
            "chunk_index": 0
        }
    ],
    "dataset_source": "timing_probe_single",
    "skip_cpesh": True
}

url = "http://localhost:8004/ingest"
print("Running 5 warm-up iterations (single chunk)...")
last = None
for i in range(5):
    start = time.perf_counter()
    resp = requests.post(url, json=payload, timeout=60)
    elapsed = (time.perf_counter() - start) * 1000
    print(f"  Run {i+1}: status={resp.status_code} time={elapsed:.1f} ms")
    if resp.status_code != 200:
        print(resp.text)
        raise SystemExit(1)
    last = resp.json()
    if i < 4:
        time.sleep(0.5)

result = last['results'][0]
timings = result['timings_ms']
print("\nFinal run timings (ms):")
for key, val in timings.items():
    print(f"  {key}: {val:.3f}")
print(f"  total: {sum(timings.values()):.3f}")
print(f"Total processing_time_ms reported: {last['processing_time_ms']:.3f}")
PY
```

## 2. Multi-Chunk Batch Probe (Initial Verification)

Four-chunk payload, five iterations with a 500 ms delay. Prints per-chunk timings and step averages from the final response.

```
cd /Users/trentcarter/Artificial_Intelligence/AI_Projects/lnsp-phase-4 && python - <<'PY'
import time
import requests

chunks = [
    {"text": "The water cycle describes the continuous movement of water on, above, and below the surface of the Earth.", "source_document": "timing_doc", "chunk_index": 0},
    {"text": "Condensation occurs when water vapor cools and changes back into liquid droplets forming clouds.", "source_document": "timing_doc", "chunk_index": 1},
    {"text": "Precipitation happens when droplets in clouds become heavy enough to fall as rain, snow, or hail.", "source_document": "timing_doc", "chunk_index": 2},
    {"text": "Infiltration allows water to soak into the ground replenishing aquifers and soil moisture levels.", "source_document": "timing_doc", "chunk_index": 3}
]

payload = {
    "chunks": chunks,
    "dataset_source": "timing_probe_batch",
    "skip_cpesh": True
}

url = "http://localhost:8004/ingest"
print("Running 5 ingestion rounds (4 chunks) with 500ms spacing...")
last_response = None
for i in range(5):
    start = time.perf_counter()
    resp = requests.post(url, json=payload, timeout=120)
    elapsed = (time.perf_counter() - start) * 1000
    print(f"  Run {i+1}: HTTP {resp.status_code} in {elapsed:.1f} ms")
    if resp.status_code != 200:
        print(resp.text)
        raise SystemExit(1)
    last_response = resp.json()
    if i < 4:
        time.sleep(0.5)

results = last_response["results"]
print(f"\nChunks returned: {len(results)}")

rows = []
step_sums = {
    "cpesh_ms": 0.0,
    "tmd_ms": 0.0,
    "embedding_ms": 0.0,
    "tmd_encode_ms": 0.0,
    "fuse_ms": 0.0,
    "postgres_ms": 0.0,
}

for idx, res in enumerate(results):
    timings = res.get("timings_ms", {})
    row = {
        "chunk": idx,
        "cpesh_ms": timings.get("cpesh_ms", 0.0),
        "tmd_ms": timings.get("tmd_ms", 0.0),
        "embedding_ms": timings.get("embedding_ms", 0.0),
        "tmd_encode_ms": timings.get("tmd_encode_ms", 0.0),
        "fuse_ms": timings.get("fuse_ms", 0.0),
        "postgres_ms": timings.get("postgres_ms", 0.0),
    }
    row["total"] = sum(v for k, v in row.items() if k != "chunk")
    rows.append(row)
    for key in step_sums:
        step_sums[key] += row[key]

header = ("Chunk", "CPESH", "TMD", "Embedding", "TMD Encode", "Fuse", "Postgres", "Total")
print("Per-chunk timings (ms):")
print(" | ".join(f"{h:>12}" for h in header))
print("-" * 104)
for row in rows:
    print(" | ".join([
        f"{row['chunk']:>12}",
        f"{row['cpesh_ms']:.2f}".rjust(12),
        f"{row['tmd_ms']:.2f}".rjust(12),
        f"{row['embedding_ms']:.2f}".rjust(12),
        f"{row['tmd_encode_ms']:.2f}".rjust(12),
        f"{row['fuse_ms']:.2f}".rjust(12),
        f"{row['postgres_ms']:.2f}".rjust(12),
        f"{row['total']:.2f}".rjust(12),
    ]))

n = len(rows)
print("\nStep averages (ms):")
for label, key in [("CPESH", "cpesh_ms"), ("TMD", "tmd_ms"), ("Embedding", "embedding_ms"), ("TMD Encode", "tmd_encode_ms"), ("Fuse", "fuse_ms"), ("Postgres", "postgres_ms")]:
    avg = step_sums[key] / n if n else 0.0
    print(f"  {label:>12}: {avg:.2f}")
print(f"  {'Total':>12}: {sum(row['total'] for row in rows)/n if n else 0.0:.2f}")

print(f"\nAPI reported processing_time_ms: {last_response['processing_time_ms']:.2f} ms")
PY
```

## 3. Multi-Chunk Batch Probe (Confirmation Pass)

Same payload and iteration pattern as above, rerun after confirming the 500 error path was resolved. This produced the 651 ms average TMD timing and ~277 ms end-to-end per chunk.

```
cd /Users/trentcarter/Artificial_Intelligence/AI_Projects/lnsp-phase-4 && python - <<'PY'
import time
import requests

chunks = [
    {"text": "The water cycle describes the continuous movement of water on, above, and below the surface of the Earth.", "source_document": "timing_doc", "chunk_index": 0},
    {"text": "Condensation occurs when water vapor cools and changes back into liquid droplets forming clouds.", "source_document": "timing_doc", "chunk_index": 1},
    {"text": "Precipitation happens when droplets in clouds become heavy enough to fall as rain, snow, or hail.", "source_document": "timing_doc", "chunk_index": 2},
    {"text": "Infiltration allows water to soak into the ground replenishing aquifers and soil moisture levels.", "source_document": "timing_doc", "chunk_index": 3}
]

payload = {
    "chunks": chunks,
    "dataset_source": "timing_probe_batch",
    "skip_cpesh": True
}

url = "http://localhost:8004/ingest"
print("Running 5 ingestion rounds (4 chunks) with 500ms spacing...")
last_response = None
for i in range(5):
    start = time.perf_counter()
    resp = requests.post(url, json=payload, timeout=120)
    elapsed = (time.perf_counter() - start) * 1000
    print(f"  Run {i+1}: HTTP {resp.status_code} in {elapsed:.1f} ms")
    if resp.status_code != 200:
        print(resp.text)
        raise SystemExit(1)
    last_response = resp.json()
    if i < 4:
        time.sleep(0.5)

results = last_response["results"]
print(f"\nChunks returned: {len(results)}")

columns = [
    ("Chunk", "chunk"),
    ("CPESH", "cpesh_ms"),
    ("TMD", "tmd_ms"),
    ("Embedding", "embedding_ms"),
    ("TMD Encode", "tmd_encode_ms"),
    ("Fuse", "fuse_ms"),
    ("Postgres", "postgres_ms"),
    ("Total", None),
]

header = " | ".join(f"{name:>12}" for name, _ in columns)
print("Per-chunk timings (ms):")
print(header)
print("-" * len(header))

step_values = {key: [] for _, key in columns if key}
row_totals = []

for idx, res in enumerate(results):
    timings = res.get("timings_ms", {})
    row = {"chunk": idx}
    total = 0.0
    for label, key in columns:
        if key is None:
            continue
        val = timings.get(key, 0.0)
        row[key] = val
        total += val
        step_values[key].append(val)
    row_totals.append(total)
    print(" | ".join([
        f"{row['chunk']:>12}",
        f"{row['cpesh_ms']:.2f}".rjust(12),
        f"{row['tmd_ms']:.2f}".rjust(12),
        f"{row['embedding_ms']:.2f}".rjust(12),
        f"{row['tmd_encode_ms']:.2f}".rjust(12),
        f"{row['fuse_ms']:.2f}".rjust(12),
        f"{row['postgres_ms']:.2f}".rjust(12),
        f"{total:.2f}".rjust(12),
    ]))

print("\nStep averages (ms):")
for label, key in columns:
    if key is None:
        continue
    values = step_values[key]
    avg = sum(values) / len(values) if values else 0.0
    print(f"  {label:>12}: {avg:.2f}")
print(f"  {'Total':>12}: {sum(row_totals)/len(row_totals):.2f}")

print(f"\nAPI reported processing_time_ms: {last_response['processing_time_ms']:.2f} ms")
PY
```

---

**Observed Highlights**

- Warm single-chunk path: ~96 ms total per chunk (TMD ~74 ms, embeddings ~14 ms, Postgres ~8 ms).
- Warm four-chunk path: API `processing_time_ms` ~1.11 s (≈277 ms per chunk); TMD remains the dominant contributor (~651 ms average per chunk) even with cache hits.

These snippets can be reused verbatim to reproduce the measurements after future code or infrastructure changes.
