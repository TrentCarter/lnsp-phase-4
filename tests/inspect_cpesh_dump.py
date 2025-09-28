#!/usr/bin/env python
"""
CPESH Data Inspector (offline)

Purpose:
  Emit 1..N normalized CPESH records with ALL text + metadata fields for
  human validation. Includes TMD decoded to text names, timestamps, counters,
  and source provenance. No AI. No network.

Sources:
  - Active JSONL: artifacts/cpesh_active.jsonl (or legacy cpesh_cache.jsonl)
  - Parquet Segments (optional): via artifacts/cpesh_manifest.jsonl

Run:
  PYTHONPATH=src python -m tests.inspect_cpesh_dump --limit 10
  PYTHONPATH=src python -m tests.inspect_cpesh_dump --limit 10 --no-active --segments
  # Save newline-delimited JSON:
  PYTHONPATH=src python -m tests.inspect_cpesh_dump --limit 25 --out artifacts/cpesh_sample_dump.jsonl

Make sure:
  - If you want segment sampling, install pyarrow (pip install pyarrow).
"""

from __future__ import annotations
import os, json, sys, argparse, glob, random, gzip
from typing import Dict, Any, Iterable, List, Optional

ART = "artifacts"
ACTIVE = os.path.join(ART, "cpesh_active.jsonl")
LEGACY = os.path.join(ART, "cpesh_cache.jsonl")
MANIFEST = os.path.join(ART, "cpesh_manifest.jsonl")

# --- Optional imports: TMD + Names -------------------------------------------------
try:
    from src.utils.tmd import unpack_tmd  # -> (domain_code, task_code, modifier_code)
except Exception:
    def unpack_tmd(bits: Optional[int]):
        if bits is None:
            return (None, None, None)
        # best-effort unpack stub (depends on your real packing; replace if needed)
        d = (bits >> 11) & 0x1F
        t = (bits >> 6)  & 0x1F
        m = (bits >> 0)  & 0x3F
        return (d, t, m)

try:
    # Optional name maps. If absent, we’ll fall back to code tokens.
    from src.enums import DOMAIN_NAMES, TASK_NAMES, MODIFIER_NAMES  # dict[int]->str
except Exception:
    DOMAIN_NAMES, TASK_NAMES, MODIFIER_NAMES = {}, {}, {}

# --- Optional Parquet (segments) ---------------------------------------------------
try:
    import pyarrow.dataset as ds
    import pyarrow.parquet as pq  # noqa: F401  (import validates presence)
    HAS_ARROW = True
except Exception:
    HAS_ARROW = False

# --- Helpers -----------------------------------------------------------------------
def _short(s: Optional[str], n: int = 80) -> str:
    if not isinstance(s, str):
        return ""
    return s if len(s) <= n else s[: max(0, n - 1)] + "…"

def _decode_tmd(bits: Optional[int]) -> Dict[str, Any]:
    d, t, m = unpack_tmd(bits) if bits is not None else (None, None, None)
    dn = DOMAIN_NAMES.get(d, f"domain_{d}" if d is not None else "—")
    tn = TASK_NAMES.get(t, f"task_{t}" if t is not None else "—")
    mn = MODIFIER_NAMES.get(m, f"modifier_{m}" if m is not None else "—")
    return {
        "tmd_bits": bits,
        "tmd_domain_code": d, "tmd_task_code": t, "tmd_modifier_code": m,
        "tmd_domain": dn, "tmd_task": tn, "tmd_modifier": mn,
        "tmd_text": f"{dn}/{tn}/{mn}"
    }

def _flatten_cpesh_row(raw: Dict[str, Any], source_hint: str) -> Dict[str, Any]:
    """
    Accepts either shape:
      - flat CPESH fields at top level, or
      - { "cpesh": {...}, "access_count": int, ... } legacy envelope
    """
    env_access = raw.get("access_count")
    env_created = raw.get("created_at")
    env_last = raw.get("last_accessed")
    env_id = raw.get("cpe_id") or raw.get("doc_id")

    obj = raw.get("cpesh", raw)  # prefer nested cpesh if present
    rec: Dict[str, Any] = {}

    # ID & audit
    rec["cpe_id"]        = obj.get("cpe_id") or env_id or obj.get("doc_id")
    rec["created_at"]    = obj.get("created_at") or env_created
    rec["last_accessed"] = obj.get("last_accessed") or env_last
    rec["access_count"]  = obj.get("access_count", env_access if isinstance(env_access, int) else 0)

    # CPESH text
    rec["concept_text"]     = obj.get("concept_text")
    rec["probe_question"]   = obj.get("probe_question")
    rec["expected_answer"]  = obj.get("expected_answer")
    rec["soft_negative"]    = obj.get("soft_negative")
    rec["hard_negative"]    = obj.get("hard_negative")
    rec["mission_text"]     = obj.get("mission_text")
    rec["dataset_source"]   = obj.get("dataset_source")
    rec["content_type"]     = obj.get("content_type")

    # Routing / quality
    rec["lane_index"]            = obj.get("lane_index")
    rec["tmd_bits"]              = obj.get("tmd_bits")
    rec["quality"]               = obj.get("quality")
    rec["echo_score"]            = obj.get("echo_score")
    rec["insufficient_evidence"] = bool(obj.get("insufficient_evidence", False))

    # Optional chunk position
    cp = obj.get("chunk_position") or {}
    rec["doc_id"]     = cp.get("doc_id") or obj.get("doc_id")
    rec["chunk_start"] = cp.get("start")
    rec["chunk_end"]   = cp.get("end")

    # Relations (stringified preview)
    rels = obj.get("relations_text")
    if isinstance(rels, list):
        rec["relations_preview"] = _short(json.dumps(rels)[:400], 120)
    else:
        rec["relations_preview"] = ""

    # TMD decode
    rec.update(_decode_tmd(rec.get("tmd_bits")))

    # Provenance
    rec["__source"] = source_hint
    return rec

def _iter_active_lines(path: str, limit: int) -> Iterable[Dict[str, Any]]:
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            if i > limit:
                break
            try:
                j = json.loads(line)
                yield _flatten_cpesh_row(j, source_hint="active_jsonl")
            except Exception:
                continue

def _iter_segments(manifest_path: str, limit: int) -> Iterable[Dict[str, Any]]:
    if not HAS_ARROW or not os.path.exists(manifest_path):
        return
    # Read all segment paths
    segs: List[str] = []
    with open(manifest_path, "r", encoding="utf-8") as mf:
        for line in mf:
            try:
                j = json.loads(line)
                p = j.get("path")
                if p and os.path.exists(p):
                    segs.append(p)
            except Exception:
                continue
    if not segs:
        return

    # Sample uniformly across segments
    per = max(1, limit // len(segs))
    taken = 0
    cols = [
        "cpe_id","concept_text","probe_question","expected_answer",
        "soft_negative","hard_negative","mission_text","dataset_source","content_type",
        "lane_index","tmd_bits","quality","echo_score","insufficient_evidence",
        "created_at","last_accessed","access_count","doc_id","relations_text","chunk_position"
    ]
    for seg in segs:
        try:
            dset = ds.dataset(seg, format="parquet")
            count = dset.count()
            if count == 0:
                continue
            # take first 'per' rows (cheap, deterministic). Switch to random sampling if preferred.
            idx = list(range(0, min(per, count)))
            tbl = dset.to_table(columns=[c for c in cols if c in dset.schema.names]).take(idx)
            for row in tbl.to_pylist():
                # Align shape with _flatten_cpesh_row expectations
                # Parquet rows are already "flat CPESH"; relations might be nested.
                row_norm = _flatten_cpesh_row(row, source_hint=os.path.basename(seg))
                yield row_norm
                taken += 1
                if taken >= limit:
                    return
        except Exception:
            continue

def collect_cpesh_samples(limit: int = 10, include_active: bool = True, include_segments: bool = False, seed: Optional[int] = 42) -> List[Dict[str, Any]]:
    """Return up to 'limit' normalized CPESH dicts from active and/or segments."""
    rng = random.Random(seed)
    out: List[Dict[str, Any]] = []

    if include_active:
        path = ACTIVE if os.path.exists(ACTIVE) else (LEGACY if os.path.exists(LEGACY) else None)
        if path:
            out.extend(list(_iter_active_lines(path, limit)))

    if include_segments and len(out) < limit:
        out.extend(list(_iter_segments(MANIFEST, limit - len(out))))

    # deterministic shuffle of mixed sources (so output isn't biased)
    rng.shuffle(out)
    return out[:limit]

# -------------- ASCII table printer (human check) ----------------------------------
def ascii_table(headers: List[str], rows: List[List[str]]) -> str:
    widths = [len(h) for h in headers]
    for r in rows:
        for i, c in enumerate(r):
            widths[i] = max(widths[i], len(str(c)))
    line = "+" + "+".join("-" * (w + 2) for w in widths) + "+"
    out = [line, "| " + " | ".join(h.ljust(widths[i]) for i, h in enumerate(headers)) + " |", line]
    for r in rows:
        out.append("| " + " | ".join(str(r[i]).ljust(widths[i]) for i in range(len(headers))) + " |")
    out.append(line)
    return "\n".join(out)

def print_samples_table(samples: List[Dict[str, Any]]):
    cols = [
        "cpe_id","created_at","last_accessed","access_count","lane_index",
        "tmd_text","tmd_bits","doc_id","dataset_source",
        "quality","echo_score","insufficient_evidence","__source",
        "concept_text","probe_question","expected_answer"
    ]
    rows = []
    for s in samples:
        rows.append([
            s.get("cpe_id",""),
            s.get("created_at",""),
            s.get("last_accessed",""),
            s.get("access_count",""),
            s.get("lane_index",""),
            s.get("tmd_text",""),
            s.get("tmd_bits",""),
            _short(s.get("doc_id",""), 24),
            _short(s.get("dataset_source",""), 18),
            s.get("quality",""),
            s.get("echo_score",""),
            s.get("insufficient_evidence",""),
            s.get("__source",""),
            _short(s.get("concept_text",""), 40),
            _short(s.get("probe_question",""), 40),
            _short(s.get("expected_answer",""), 40),
        ])
    print(ascii_table(cols, rows))

# -------------- CLI ----------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=10, help="Number of datapoints to emit")
    ap.add_argument("--segments", action="store_true", help="Include Parquet segments (needs pyarrow)")
    ap.add_argument("--no-active", action="store_true", help="Exclude active/legacy JSONL")
    ap.add_argument("--out", type=str, default="", help="Write JSONL to this path as well")
    ap.add_argument("--no-table", action="store_true", help="Do not print ASCII table; JSONL only")
    args = ap.parse_args()

    samples = collect_cpesh_samples(
        limit=max(1, args.limit),
        include_active=not args.no_active,
        include_segments=args.segments
    )

    if not args.no_table:
        print_samples_table(samples)

    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            for s in samples:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
        print(f"[inspect] wrote {len(samples)} rows to {args.out}")

    # Also print machine-readable JSON to stdout (last line) for scripting
    print(json.dumps({"count": len(samples)}))

if __name__ == "__main__":
    main()
