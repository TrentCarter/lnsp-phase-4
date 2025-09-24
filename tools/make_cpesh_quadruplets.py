#!/usr/bin/env python3
"""
Creates a 10× REAL CPESH training set: (anchor/probe, expected, soft_neg?, hard_neg?)
Produces:
- artifacts/train/cpesh_10x.jsonl  (text-only quadruplets with IDs)
- artifacts/train/cpesh_10x_vectors.npz  (optional 768D vectors if --embed)
"""
from __future__ import annotations
import os, sys, json, random, argparse, pathlib
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer

BAD_HEADINGS = {"track listing","discography","overview","biography","reception","plot","references","external links"}

def cpesh_ok(probe, expected, soft, hard):
    if not expected or expected.strip().lower() in BAD_HEADINGS:
        return False
    # basic type hints from probe word
    p = (probe or "").strip().lower()
    if p.startswith("who") and not any(ch.isalpha() for ch in expected):
        return False
    if "track" in p and expected.strip().lower() in BAD_HEADINGS:
        return False
    # negatives must differ from expected
    if soft and soft.strip().lower() == expected.strip().lower():
        return False
    if hard and hard.strip().lower() == expected.strip().lower():
        return False
    return True

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="artifacts/fw10k_vectors_768.npz")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--embed", action="store_true", help="also save 768D vectors for A/P/S/H")
    ap.add_argument("--out-jsonl", default="artifacts/train/cpesh_10x.jsonl")
    ap.add_argument("--out-npz", default="artifacts/train/cpesh_10x_vectors.npz")
    args = ap.parse_args()

    # os.environ["TRANSFORMERS_OFFLINE"]="1"; os.environ["HF_HUB_OFFLINE"]="1"
    model_dir = os.getenv("LNSP_EMBED_MODEL_DIR","sentence-transformers/gtr-t5-base")
    enc = SentenceTransformer(model_dir)

    D = np.load(args.npz, allow_pickle=True)
    V = D["vectors"]; C = D["concept_texts"]; doc_ids = D["doc_ids"]
    N = V.shape[0]
    random.seed(31)
    ix = random.sample(range(N), min(args.n, N))

    out = []
    A = []; P = []; S = []; H = []; IDS=[]
    # ensure we can import the local llama client from src/llm
    repo_root = pathlib.Path(__file__).resolve().parents[1]
    src_dir = repo_root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    try:
        from llm.local_llama_client import LocalLlamaClient  # type: ignore
    except Exception as exc:  # pragma: no cover - explicit failure path
        raise SystemExit(
            "Missing local Llama client. Ensure src/llm/local_llama_client.py exists."
            f" Import error: {exc}"
        )

    for i in ix:
        fact = str(C[i]); doc_id=str(doc_ids[i])
        # Use your local Llama to get CPESH
        llm = LocalLlamaClient(
            os.getenv("LNSP_LLM_ENDPOINT","http://localhost:11434"),
            os.getenv("LNSP_LLM_MODEL","llama3.1:8b")
        )
        prompt = f"""Generate a JSON object for CPESH data extraction from this factoid.

Factoid: "{fact}"

Return a JSON object with these exact fields:
- concept: The main concept or entity from the factoid
- probe: A question that this factoid answers
- expected: The expected answer text
- soft_negative: A plausible but incorrect alternative answer
- hard_negative: A very similar but wrong answer
- insufficient_evidence: false if we have enough info, true otherwise

IMPORTANT CONSTRAINTS:
- Answer type constraint: Make expected the concrete entity that answers the probe (e.g., a person, track title, year). Do not return section labels like 'Track listing', 'Biography', or categories.
- Span preference: Prefer an exact span from the provided text. If you cannot justify the exact answer from the text/context, set insufficient_evidence=true and set the field to null.

Return only valid JSON, no other text:"""

        # Try up to 3 times to get a good CPESH response
        attempts = 1
        cpesh = llm.complete_json(prompt, timeout_s=15)  # must return parsed JSON
        while not cpesh_ok(cpesh.get("probe"), cpesh.get("expected"),
                           cpesh.get("soft_negative"), cpesh.get("hard_negative")) and attempts < 3:
            cpesh = llm.complete_json(prompt, timeout_s=15)
            attempts += 1
        rec = {"doc_id":doc_id, "concept":cpesh.get("concept"), "probe":cpesh.get("probe"),
               "expected":cpesh.get("expected"), "soft_negative":cpesh.get("soft_negative"),
               "hard_negative":cpesh.get("hard_negative")}
        out.append(rec)

        if args.embed:
            a = cpesh.get("probe") or fact
            p = cpesh.get("expected") or fact
            s = cpesh.get("soft_negative") or ""
            h = cpesh.get("hard_negative") or ""
            va = enc.encode([a], normalize_embeddings=True)[0].astype(np.float32)
            vp = enc.encode([p], normalize_embeddings=True)[0].astype(np.float32)
            vs = enc.encode([s], normalize_embeddings=True)[0].astype(np.float32) if s else np.zeros(768, np.float32)
            vh = enc.encode([h], normalize_embeddings=True)[0].astype(np.float32) if h else np.zeros(768, np.float32)

            # Apply quality gates based on cosine similarity
            exp_sim = float(vp @ va)                  # expected vs anchor
            soft_sim = float(vs @ va) if s else 0.0   # soft vs anchor
            hard_sim = float(vh @ va) if h else 0.0   # hard vs anchor

            if not (exp_sim >= 0.45 and hard_sim <= 0.55 and (soft_sim < exp_sim or abs(soft_sim - exp_sim) < 0.05)):
                # Skip this record; doesn't meet quality thresholds
                print(f"Skipping {doc_id}: exp={exp_sim:.3f}, soft={soft_sim:.3f}, hard={hard_sim:.3f}")
                out.pop()  # Remove the record we just added
                continue

            A.append(va); P.append(vp); S.append(vs); H.append(vh); IDS.append(doc_id)

    Path(args.out_jsonl).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_jsonl,"w") as w:
        for r in out: w.write(json.dumps(r, ensure_ascii=False) + "\n")

    if args.embed:
        np.savez(args.out_npz,
                 anchor=np.stack(A) if A else np.zeros((0,768),np.float32),
                 positive=np.stack(P) if P else np.zeros((0,768),np.float32),
                 soft=np.stack(S) if S else np.zeros((0,768),np.float32),
                 hard=np.stack(H) if H else np.zeros((0,768),np.float32),
                 doc_ids=np.array(IDS, dtype=object))
    print(f"Wrote: {args.out_jsonl}" + (f" and {args.out_npz}" if args.embed else ""))

if __name__ == "__main__":
    main()
