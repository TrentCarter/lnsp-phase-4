# VecRAG Test Suite Guide

This guide explains how to exercise the synthetic vecRAG regression suite that
ships with Sprint 10012025 S1.

## Components

- `tests/test_vecrag_e2e.py` – functional regression tests for the synthetic
  query→FAISS→Neo4j flow.
- `tests/benchmark_hop_reduction.py` – reproducible benchmark that measures the
  hop-count impact of 6-degree shortcuts.
- `tests/helpers/vecrag_fake_pipeline.py` – in-memory replacement for the
  production vecRAG stack (FAISS index + graph traversal) so tests run without
  external services.

## Running the Suite

Activate your environment and execute:

```bash
pytest tests/test_vecrag_e2e.py -v
```

Expected signals:

- 6 tests pass.
- `test_basic_retrieval` reports top-10 hits with similarity > 0.6.
- Shortcut tests verify mean hop counts drop from ~6 to <3.

To reproduce the benchmark:

```bash
PYTHONPATH=. python3 tests/benchmark_hop_reduction.py
```

Recent run (synthetic dataset):

```
Baseline hops: 6.22 ± 0.16
Shortcut hops: 2.33 ± 0.00
Reduction: 62.5%
```

The script asserts `reduction > 40%`; failure indicates shortcut behaviour
regressed.

## Extending the Suite

- Add new synthetic concepts/queries to `tests/helpers/vecrag_fake_pipeline.py`
  when additional retrieval scenarios are needed.
- Keep hop targets in `_GRAPH_EXPANSION_DATA` aligned with production models so
  the benchmark remains informative.
- Gate any heavy or integration-level checks behind `pytest.mark.heavy` to keep
  default CI runs light.
