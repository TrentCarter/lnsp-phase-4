# Day 3 Retrieval Report — 2025-09-23

## Hybrid API Run (baseline)
- Total queries: **20**
- Pass / Fail: **20 / 0**
- Echo pass: **100.0%**
- Mean latency: **4.62 ms**
- Ranking: **P@1 0.9000**, **P@5 0.2000**, **MRR 0.9375**, **Recall@k 1.0000**
- Lane distribution:
  | Lane | Queries | Pass | Pass % |
  |------|---------|------|--------|
  | L1_FACTOID | 20 | 20 | 100.0% |
- Query types: album, music_release, magazine, politics, album_history, motorsport×2, music_genre, mystery_plot×3, film, crime, policy, computing, album_review, poetry, religion, wildlife, honor
- Status codes: `200: 20`
- Artifacts: `eval/day3_results_fallback_final.jsonl`

## Independent Offline Sanity Check (lexical fallback)
- Command: `python3 -m src.eval_runner --queries eval/day3_eval.jsonl --offline-npz artifacts/fw1k_vectors.npz --top-k 5 --timeout 15 --out eval/day3_results_consultant.jsonl`
- Total queries: **20**
- Pass / Fail: **1 / 19**
- Echo pass: **5.0%**
- Mean latency: **0.01 ms**
- Ranking: **P@1 0.0500**, **P@5 0.0100**, **MRR 0.0500**, **Recall@k 0.0500**
- Lane distribution:
  | Lane | Queries | Pass | Pass % |
  |------|---------|------|--------|
  | L1_FACTOID | 20 | 1 | 5.0% |
- Status codes: `offline: 20`
- Artifacts: `eval/day3_results_consultant.jsonl`

## Notes
- Offline run validates the lexical fallback path in isolation; low pass rate is expected without the hybrid API.
- Samples refreshed in `eval/day3_samples/` (first-seen per type/lane) and remain aligned with the baseline hybrid results.
