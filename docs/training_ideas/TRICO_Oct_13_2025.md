TRICO — Training Ideas (Vector-Only)

LNSP/LVM • Semantic GPS 768-D • October 13, 2025

Goal. Stop “garbage decodes” by making the model’s 768-D outputs both (a) on-manifold in your Semantic-GPS space and (b) decoder-compatible with vec2text. We’ll do that with four concrete, vector-native training tracks you can run in parallel and compare.

⸻

TL;DR (what to run)
	1.	E2E Vector Supervision (E2E-V): Simple cosine-align to targets; adds manifold control.
	2.	Iterative No-Grad→Grad (INGR): Teach “improve-by-recursion” with cheap compute.
	3.	Contrastive + Cycle (CONTRAST-CYCLE): Align to positives, repel hard negatives, and enforce vec2text round-trips.
	4.	Curriculum + Lane Hard-Negatives (CURR-LANE): Stage difficulty and TMD-aware negatives for stability.

Run all four; keep the one that makes decoded text coherent and lifts retrieval metrics.

⸻

Assumptions & Setup
	•	Space: 768-D Semantic GPS (unit-norm; use SGPS projector).
	•	Encoder/Decoder: Use vec2text for both directions (encoder and decoder) to avoid GTR-T5 compatibility headaches.
	•	Data: Wikipedia chunks (40–50k), CPESH labels when available.
	•	Lanes: TMD lanes (Factoid, Math-Deriv, Code-API, …) as categorical features and sampling buckets.

Common tricks (use in all tracks):
	•	Unit-norm outputs: ŷ ← y / ||y||.
	•	EMA teacher for stability (decay ≈ 0.999–0.9999).
	•	SGPS projection penalty: keep outputs near training distribution (e.g., distance to k-NN hull or small L2 to retrieval centroid).
	•	Vec-decoder compatibility loss (below) on a random 30–50% of batches.

⸻

Track 1 — E2E Vector Supervision (E2E-V)

Purpose: Baseline that already fixes many “garbage decode” cases by keeping vectors on-manifold and close to targets.

Batch I/O
	•	Input: q (question vec), optional retrieval pooled r̄, lane embedding t.
	•	Target: y* (CPESH Expected) or teacher vector (vec2text encoder of gold text).

Model (simple): ŷ = fθ(q, r̄, t) (Mamba or slim MLP); output 768-D.

Loss
	•	Alignment: L_align = 1 − cos(ŷ, y*).
	•	Manifold regularizer: L_mani = ||ŷ − Proj_SGPS(ŷ)||² or small penalty to drift from k nearest vectors in train set.
	•	Decoder-compat: decode and re-encode once:
	•	txt = vec2text.decode(ŷ)
	•	v' = vec2text.encode(txt)
	•	L_cycle = 1 − cos(ŷ, v')

Total: L = L_align + λ_mani L_mani + λ_cycle L_cycle.

Why it helps: Forces outputs to sit where vec2text is accurate, not just “somewhere” in 768-D.

⸻

Track 2 — Iterative No-Grad → Grad (INGR)

Purpose: Teach the head (or a tiny refiner) to improve by recursion without paying full backprop at every step.

Module: Tiny Refiner (CTR-SGPS) that updates (z, y) S times; last step trains.

(z0=0, y0 = pool(Y or initial head))
for s in 1..S-1:
    (zs, ys) = step_no_grad(zs-1, ys-1; q, r̄, t)
(zS, yS) = step_grad(zS-1, yS-1; q, r̄, t)

Loss (on final step only):
L = (1 − cos(yS, y*)) + α·BCE(p_halt,S, 𝟙{cos(yS,y*)≥τ}) + β·L_cycle(yS)

Notes
	•	Set S≈8–16 for baseline; cap latency with a halting head at inference.
	•	Use EMA of the refiner for evaluation.

Why it helps: You get the iterative behavior (big win for coherence) at almost the cost of a single forward.

⸻

Track 3 — Contrastive + Cycle (CONTRAST-CYCLE)

Purpose: Make vectors discriminative and decoder-friendly. Fixes “blurry” outputs that decode nonsensically.

Positives/Negatives
	•	Positive: y* or teacher vector for the same chunk.
	•	Negatives: (a) lane-hard: same lane, different article; (b) near-miss: high retrieval but wrong; (c) adversarial: lexically similar but semantically different.

Loss
	•	InfoNCE:
L_{\text{nce}} = -\log \frac{\exp(\cos(ŷ, y^+)/τ_c)}{\exp(\cos(ŷ, y^+)/τ_c)+\sum_j \exp(\cos(ŷ, y^-_j)/τ_c)}
	•	Cycle: same L_cycle as Track 1.
	•	Optional Mutual consistency: symmetrize with teacher: 1 − cos(fθ(q,…), stopgrad(y*)) + 1 − cos(stopgrad(fθ(q,…)), y*).

Total: L = L_nce + λ_cycle L_cycle.

Why it helps: Separates close concepts; keeps decodes sharp and faithful.

⸻

Track 4 — Curriculum + Lane Hard-Negatives (CURR-LANE)

Purpose: Stabilize training; expose the model to difficulty gradually; force lane-aware precision.

Stages
	1.	Stage A (Easy): Short, clean paragraphs; far negatives; batch size high.
	2.	Stage B (Medium): Normal chunks; lane-hard negatives; add L_cycle.
	3.	Stage C (Hard): Long/technical chunks; near-miss negatives; enable INGR (Track 2) for the last 30–50% of training.

Sampling
	•	Curriculum by readability/length and retrieval ambiguity.
	•	Maintain lane balance per epoch.

Scheduler
	•	Warmup → cosine decay; raise λ_cycle over time; increase negative count per batch over stages.

⸻

Implementation Notes (do these no matter what)
	•	Normalize everywhere: inputs, intermediate y, outputs.
	•	SGPS projector: if you have a PCA/autoencoder of the corpus vectors, project predictions back (light orthogonality helps).
	•	Adapters per lane: small FiLM/LoRA heads keyed by TMD → lowers interference.
	•	Batch shaping: mix 70% in-lane, 30% cross-lane examples to avoid collapse.
	•	Decode gating (for logging only): if cos(ŷ, v') < 0.7, mark as “decode-risky”; surface to eval dashboard.
	•	Teacher path (optional): keep a frozen EMA of a previously good checkpoint to generate soft targets y* when CPESH isn’t available.

⸻

Minimal Loss Recipes (copy/paste)

E2E-V

L = (1 - cos(ŷ, y*)) + 0.05 * ||ŷ - Proj_SGPS(ŷ)||^2 + 0.2 * (1 - cos(ŷ, vec2text.encode(vec2text.decode(ŷ))))

INGR

L = (1 - cos(yS, y*)) + 0.2 * BCE(p_halt,S, 𝟙{cos(yS,y*)≥0.85}) + 0.2 * (1 - cos(yS, vec2text.encode(vec2text.decode(yS))))

CONTRAST-CYCLE

L = InfoNCE(ŷ, y+, {y-}) + 0.2 * (1 - cos(ŷ, vec2text.encode(vec2text.decode(ŷ))))


⸻

Metrics & Gates (what “good” looks like)
	•	Vector alignment: cos(ŷ, y*) ↑; median ≥ 0.88 on val after Stage B.
	•	Decoder cycle: cos(ŷ, v') ↑; median ≥ 0.82.
	•	Retrieval synergy: nDCG@10 on re-query with ŷ ↑ vs baseline.
	•	Lane accuracy: per-lane pass rate (CPESH Expected within top-k when decoded) ↑.
	•	Halting efficiency (INGR): avg steps ≤ 8 with ≥ 85% inside S_max.
	•	Human sanity checks: 50-sample blind read—≥ 70% “sensible” after Stage C.

⸻

Failure Modes & Fast Fixes
	•	Nonsensical decodes despite high cosine to y*: raise λ_cycle; add near-miss negatives.
	•	Mode collapse (all vectors look alike): increase negative count; add lane adapters; up InfoNCE temperature τ_c.
	•	Over-halting (INGR stops too early): lower halt threshold τ or add penalty for early halts.
	•	Training stable but val decode poor: enable EMA eval; tighten SGPS projector; add small L2 to nearest-neighbor barycenter.

⸻

7-Day Plan (practical)

Day 1–2: Implement Tracks 1 & 3. Log cos(ŷ,y*), cos(ŷ,v'), nDCG@10.
Day 3: Add SGPS projector + EMA; run quick grid on λ_cycle ∈ {0.1,0.2,0.3}.
Day 4: Implement INGR (Track 2) with S=8; τ=0.85; halting head.
Day 5: Turn on CURR-LANE: Stage A→B scheduling + lane adapters (rank-8).
Day 6: Ablate: no cycle vs cycle, no contrast vs contrast, no INGR vs INGR.
Day 7: Pick winner by decode sanity + retrieval uplift; checkpoint + freeze.

⸻

Experiment Naming (so results aren’t chaos)

exp/<date>_<track>_S<steps>_CY<λcycle>_N<neg>_LANE<on|off>_EMA<on|off>
# example:
exp/2025-10-13_CONTRAST-CYCLE_S0_CY0.2_N64_LANEon_EMAon


⸻

Optional: Wire the Tiny Refiner (CTR-SGPS) cleanly
	•	Inputs: (q, r̄, Y_topk, t) → iterate (z,y); output (ŷ, p_halt).
	•	Use it inside Track 2 (training) and after the LVM head at inference for hard queries.
	•	Keep h=512, S_max=16, τ∈[0.82,0.9].

⸻

Final Notes (blunt and practical)
	•	If vec2text is the only reliable pair, standardize on it for both encode/decode.
	•	The single biggest lever against “garbage text” is CONTRAST-CYCLE with lane-hard negatives.
	•	The single cheapest quality bump is INGR (no-grad→grad) + EMA.
	•	Do not skip normalization and the SGPS projector; most drift bugs come from that.

If you want, I can turn this into a Makefile + runnable trainer skeleton next, with flags for --track, --lambda-cycle, --negatives, --ema, and --ingr-steps.



#
# UPDATED after critical Review:

You’re right to push back. Both reviews are fair. Here’s a tighter, low-risk plan that directly addresses each critique and keeps cost/infrastructure sane while we finish validating the baseline.

# What we change immediately

* **No parallel tracks.** We proceed **one experiment at a time** with hard stop/go gates.
* **No heavy cycle-loss in training (yet).** We use vec2text only for **diagnostics**, not as a training target, until we prove it’s helpful.
* **Defer CTR/INGR/halting, curriculum, lane adapters.** Those need infra we don’t have and add 10–100× cost. We’ll revisit only if simple fixes fail.

---

# Phase 0 — Baseline first (finish what’s running)

**Model:** simple LSTM (PID 11640) + **InfoNCE** with **in-batch negatives** only.
**Targets:** vec2text-encoded vectors (so train & eval live in the same latent).
**Report (single page):**

* `val_cosine` (mean/median)
* `test_cosine`
* Decode sanity on 100 val samples (**diagnostic only**): % with cycle_cos ≥ 0.8; % < 0.7; 10 example decodes.

**Exit criteria:**

* If `val_cosine ≥ 0.50` **and** ≥70% decodes judged coherent → **Ship baseline** and proceed to retrieval integration.
* If either fails → Phase 1.

---

# Phase 1 — Diagnose vec2text compatibility (cheap & decisive)

Goal: confirm whether “garbage decodes” come from **off-manifold predictions** or an **inherently brittle decoder**.

**D1. Latent compatibility sweep (no decoder calls)**

* Compute cosine to **nearest 8 train anchors** for each prediction; record mean/variance and **angular dispersion**.
* If off-manifold: you’ll see **low NN cosines** and **high dispersion** where decodes are bad.

**D2. Minimal decode probe (≤100 samples)**

* `y_pred → decode → encode → y_cycle`; record `cos(y_pred, y_cycle)` and human coherence (3-point scale).
* Correlate: low NN cosine ↔ low cycle_cos ↔ incoherent text? If yes → the model is off-manifold; if no → decoder is brittle even on-manifold.

**Decision:**

* If off-manifold → Phase 2A (cheap manifold alignment).
* If decoder brittle → Phase 2B (decoder-aware but **very light**).

---

# Phase 2A — Cheap manifold alignment (no decoder in the loop)

Cost multiplier: **≈1.05×** (tiny).

**A1. Anchor-MMD (mini-maximum mean discrepancy)**

* Precompute **1,024 anchor vectors** from train (random stratified by article).
* Add `L_mmd = MMD_RBF(y_pred, anchors_batch)` with **fixed tiny weight** (λ_mmd=0.02 to start).
* Effect: pulls predictions toward the empirical latent distribution **without** any PCA/projector infra.

**A2. Mean/variance matching (batch-level)**

* Maintain running **per-lane** (or global, if no lanes) mean/var stats of train embeddings.
* Add a **soft penalty** to match batch mean/var of `y_pred` to these stats (λ_stat=0.01).

**Train once**, measure:

* Δ`val_cosine`, Δdecode coherence on 100 samples.
* If both improve (or remain equal with cleaner decodes) → keep A1/A2; else drop them.

---

# Phase 2B — Decoder-aware training (lightweight, gated)

Only if Phase 1 showed decoder brittleness **even on-manifold**.

**B1. Sparse cycle audits (training-time)**

* Apply cycle loss to **5% of batches**, **vec2text --steps=1 only**.
* Weight tiny: λ_cycle=0.05.
* Budget: expect ≤**1.5×** slowdown (vs 5–10× previously).
* If throughput tanks, cut to 2% and/or stagger (every Nth step).

**B2. Cache & reuse** (no infra changes)

* Keep a **FIFO cache** of `(y_pred → text → y_cycle)` for recently seen patterns; reuse within the epoch to avoid repeated decodes on near-duplicates.

**Exit gate:** Only keep cycle if we see **measurable** uplift in:

* % with `cycle_cos ≥ 0.8` (≥+10 pts) **and**
* Human coherence on the 100-sample panel (≥+10 pts absolute)
  at ≤2× wall-time.

---

# Phase 3 — Stronger contrastive, still simple

If we need more discriminability and Phase 2 didn’t get us there.

**C1. In-batch + “same-article” negatives**

* Free hard negatives: for each anchor, sample negatives from **the same article** (different chunk). No lane infra needed.
* Keep temperature sweep tight: τ ∈ {0.05, 0.07, 0.1}.
* Log: train loss curve, `val_cosine`, and decode sanity.
* If it helps, keep; otherwise revert—no sunk cost.

---

# What we are NOT doing (yet)

* No **INGR/halting** (multi-step refiners) — agrees with both reviews: too expensive, infra heavy.
* No **curriculum** — needs readability metadata & pipelines we don’t have.
* No **lane adapters** or lane-aware negatives — later, when lane batching exists.
* No **SGPS projector** — replaced by **Anchor-MMD** + mean/var matching (no PCA/AE buildout).

---

# Concrete numbers (to remove hand-waving)

* **λ_mmd = 0.02**, RBF σ chosen as median pairwise distance on anchors (compute once).
* **λ_stat = 0.01** (mean/var match).
* **Cycle weight λ_cycle = 0.05** on **5%** of batches, **--steps 1**.
* **Contrastive τ** grid: 0.05, 0.07, 0.10 (pick 1 by val cosine).
* **Success gates** (realistic):

  * Baseline: `val_cosine ≥ 0.50`.
  * After Phase 2: `+0.02` absolute on `val_cosine` **or** +10 pts on 100-sample decode-coherence, with ≤1.5× time.
  * After Phase 3: further +0.02 on `val_cosine` or +5 pts decode-coherence.

---

# Why this directly answers the critiques

**Premature optimization:** We run **baseline → diagnose → one small change** at a time.
**Compute cost:** Cycle is **off** by default; when on, it’s 5% batches, `--steps 1`, λ small. Anchor-MMD/mean-var are ~free.
**Missing infra:** We avoid projector, lanes, curriculum, INGR. Everything here is runnable in the current loop.
**Vec2text bottleneck:** Used sparingly for **diagnostics**; if ever in the loss, it’s **light** and **cached**.
**Metrics realism:** Gates are tied to current 0.75-ish cycle baselines; no fantasy targets.
**Actionable:** Each phase is 1–2 days, not weeks, and can be abandoned if no gain.

---

# If baseline already wins

If the LSTM + InfoNCE already delivers acceptable decode quality: **stop**. Integrate retrieval and move to evaluation at system level. Save fancy methods for when the data or task demands them.

---

If you want, I’ll package Phase-1 diagnostics (anchors, dispersion, 100-sample decode probe) and Phase-2A losses as drop-in functions you can toggle with flags like:

```
--mmd_anchors 1024 --lambda_mmd 0.02 --lambda_stat 0.01 \
--cycle_pct 0.05 --cycle_lambda 0.05 --cycle_steps 1
```

This keeps experiments surgical and measurable.
