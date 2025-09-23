# Day 3 Retrieval Report — 2025-09-23

## Dataset Snapshot
- Queries: 20 total (`eval/day3_eval.jsonl`)
- Lane mix: 5×L1_FACTOID, 8×L2_GRAPH, 7×L3_SYNTH
- Gold coverage pulled from `data/factoidwiki_1k.jsonl` IDs (Regurgitator/!!!/!Hero cohorts)

## Execution Summary
- Dense-only Faiss run (target): **Pending** — requires Python 3.11 runtime with FastAPI/Faiss wheels. Current sandbox ships Python 3.9 and blocks outbound `pip`, so the API stack cannot boot. Action: rerun via `python3.11 -m venv .venv && pip install -r requirements.txt` on a host with network access, then `LNSP_LEXICAL_FALLBACK=0 python -m src.eval_runner ...`.
- Hybrid (dense + lexical fallback): **Pending** — same interpreter constraint as above. Once the API is live, re-run with `LNSP_LEXICAL_FALLBACK=1` to capture the latency delta versus dense-only.
- Offline lexical sanity (consultant smoke): **Completed** via `python3 -m src.eval_runner --queries eval/day3_eval.jsonl --offline-npz artifacts/fw1k_vectors.npz --top-k 5 --timeout 15 --out eval/day3_results_consultant_balanced.jsonl`.

### Offline Lexical Result (reference only)
- Echo pass: **25.0%** (5 / 20)
- Mean latency: **0.01 ms** (NPZ lookup only)
- Ranking: **P@1 0.25**, **P@5 0.06**, **MRR 0.25**, **Recall@k 0.25**
- Lane breakdown:
  | Lane | Queries | Pass | Pass % |
  |------|---------|------|--------|
  | L1_FACTOID | 5 | 3 | 60.0% |
  | L2_GRAPH   | 8 | 1 | 12.5% |
  | L3_SYNTH   | 7 | 1 | 14.3% |
- Artifacts: `eval/day3_results_consultant_balanced.jsonl`, samples under `eval/day3_samples/`
- Interpretation: Lexical fallback alone cannot satisfy the new L2/L3 prompts; confirms need for Faiss+LightRAG path before signing off acceptance gates (P50/P95 latency + ≥0.92 top-1 for L1).

## Sample Requests (offline smoke)
```json
POST /search {"q":"What is the debut studio album released by Portuguese singer Cláudia Pascoal in March 2020?","lane":"L1_FACTOID","top_k":5}
→ hit=true, top id="enwiki-00000000-0000-0000"

POST /search {"q":"Where is the formation of !!! from members of Black Liquorice and Popesmashers documented?","lane":"L2_GRAPH","top_k":5}
→ hit=false, lexical shortlist drifts to factoid docs; needs graph adapter

POST /search {"q":"Outline the storyline of the !Hero rock opera focusing on Hero's journey.","lane":"L3_SYNTH","top_k":5}
→ hit=false, fallback returns factoid snippets; awaits hybrid reranker
```

## Acceptance Gates (Day-4 Canary)
- Still outstanding: rerun dense-only & hybrid API checks once Python 3.11 stack is live.
- Next validation cycle must log: echo %, lane distribution, FAISS metadata (`artifacts/faiss_meta.json` refresh), three live `/search` transcripts, and compare against latency targets (L1 P50 ≤ 85 ms, L3 P95 ≤ 400 ms).
