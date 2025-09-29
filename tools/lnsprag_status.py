#!/usr/bin/env python
import argparse
import os, sys, json, time, glob, gzip, math
from pathlib import Path
from statistics import median
from typing import Any, Dict

API = None

ART = "artifacts"
INDEX_META = os.path.join(ART, "index_meta.json")
ACTIVE = os.path.join(ART, "cpesh_active.jsonl")
LEGACY = os.path.join(ART, "cpesh_cache.jsonl")
WARM_GLOB = os.path.join(ART, "cpesh_warm_*.jsonl.gz")
GATING_FILE = os.path.join(ART, "gating_decisions.jsonl")
METRICS_GATING = os.path.join(ART, "metrics_gating.json")
MANIFEST = os.path.join(ART, "cpesh_manifest.jsonl")
FAISS_GLOB = os.path.join(ART, "*.index")

# Matrix-aware paths (disk only; no sockets)
ART_DIR = Path(ART)
ACTIVE_JSONL = ART_DIR / "cpesh_active.jsonl"
ACTIVE_JSONL_FIXED = ART_DIR / "cpesh_active_fixed.jsonl"
ACTIVE_LEGACY = ART_DIR / "cpesh_cache.jsonl"
SEGMENT_DIR = ART_DIR / "cpesh_segments"
SQLITE_IDX = ART_DIR / "cpesh_index.db"
NPZ_GLOB = str(ART_DIR / "*.npz")
FAISS_META_CANDIDATES = [ART_DIR / "index_meta.json", ART_DIR / "faiss_meta.json"]
FAISS_INDEX_GLOB = str(ART_DIR / "*.index")
GATING_DECISIONS = Path(GATING_FILE)
SLO_JSON = ART_DIR / "metrics_slo.json"

def sizeof(n):
    for u in ["B","KB","MB","GB","TB"]:
        if n < 1024.0: return f"{n:,.1f} {u}"
        n/=1024.0
    return f"{n:.1f} PB"

def pct(x, total): return ("—" if total==0 else f"{(100.0*x/total):.1f}%")

def percentile(vals, p):
    if not vals: return None
    s = sorted(vals)
    k = (len(s)-1)*p
    f = math.floor(k); c = math.ceil(k)
    if f==c: return s[int(k)]
    return s[f] + (s[c]-s[f])*(k-f)

def ascii_table(headers, rows):
    cols = len(headers)
    widths = [len(h) for h in headers]
    for r in rows:
        for i,c in enumerate(r):
            widths[i] = max(widths[i], len(str(c)))
    def line(sep_left="+", sep_mid="+", sep_right="+", pad="-"):
        return sep_left + sep_mid.join(pad*(w+2) for w in widths) + sep_right
    out = [line(), "| " + " | ".join(h.ljust(widths[i]) for i,h in enumerate(headers)) + " |", line("+","+","+","-")]
    for r in rows:
        out.append("| " + " | ".join(str(r[i]).ljust(widths[i]) for i in range(cols)) + " |")
    out.append(line())
    return "\n".join(out)


def _tick(flag: bool) -> str:
    return "✓" if flag else "—"


def _first_existing(paths):
    for p in paths:
        try:
            if p.exists():
                return p
        except OSError:
            continue
    return paths[0]


def _presence_from_disk() -> Dict[str, Any]:
    active_candidates = [ACTIVE_JSONL, ACTIVE_JSONL_FIXED, ACTIVE_LEGACY]
    active_exists = any(p.exists() for p in active_candidates)
    active_path = _first_existing(active_candidates)

    parquet_pattern = SEGMENT_DIR / "*.parquet"
    parquet_exists = SEGMENT_DIR.exists() and any(SEGMENT_DIR.glob("*.parquet"))

    sqlite_exists = SQLITE_IDX.exists()

    npz_files = glob.glob(NPZ_GLOB)
    npz_exists = bool(npz_files)
    npz_path = npz_files[0] if npz_files else NPZ_GLOB

    faiss_meta_path = _first_existing(FAISS_META_CANDIDATES)
    faiss_meta_exists = faiss_meta_path.exists()
    faiss_index_files = glob.glob(FAISS_INDEX_GLOB)
    faiss_present = faiss_meta_exists or bool(faiss_index_files)
    faiss_path = faiss_index_files[0] if faiss_index_files else str(faiss_meta_path)

    gating_present = GATING_DECISIONS.exists()
    slo_present = SLO_JSON.exists()

    present_pg = bool(os.getenv("PG_DSN") or os.getenv("PGDATABASE") or os.getenv("PGHOST") or os.getenv("PGUSER"))
    present_neo4j = bool(os.getenv("NEO4J_URI") or (os.getenv("NEO4J_USER") and os.getenv("NEO4J_PASSWORD")))

    return {
        "active_jsonl": active_exists,
        "active_path": str(active_path),
        "parquet": parquet_exists,
        "parquet_pattern": str(parquet_pattern),
        "pg": present_pg,
        "neo4j": present_neo4j,
        "npz": npz_exists,
        "npz_path": str(npz_path),
        "faiss": faiss_present,
        "faiss_path": str(faiss_path),
        "sqlite": sqlite_exists,
        "gating": gating_present,
        "slo": slo_present,
    }


def _fmt_table(headers, rows):
    return ascii_table(headers, [[str(c) for c in row] for row in rows])


def _print_matrix(p: Dict[str, Any]) -> None:
    col_active = _tick(p["active_jsonl"])
    col_parquet = _tick(p["parquet"])
    col_pg_entry = _tick(p["pg"])
    col_pg_vec = _tick(p["pg"])
    col_npz = _tick(p["npz"])
    col_faiss = _tick(p["faiss"])
    col_neo_concept = _tick(p["neo4j"])
    col_graph_edge = _tick(p["neo4j"])

    headers1 = [
        "Field",
        "Active JSONL",
        "Parquet Segments",
        "Postgres cpe_entry",
        "Postgres cpe_vectors",
        "Neo4j Concept",
        "Graph edges",
    ]
    rows1 = [
        ["cpe_id",              col_active, col_parquet, col_pg_entry, col_pg_vec,  col_neo_concept, "—"],
        ["doc_id",              col_active, col_parquet, col_pg_entry, "—",        col_neo_concept,  col_graph_edge],
        ["concept_text",        col_active, col_parquet, col_pg_entry, "—",        col_neo_concept,  "—"],
        ["probe_question",      col_active, col_parquet, col_pg_entry, "—",        "—",              "—"],
        ["expected_answer",     col_active, col_parquet, col_pg_entry, "—",        "—",              "—"],
        ["soft_negatives[3]",   col_active, col_parquet, col_pg_entry, "—",        "—",              "—"],
        ["hard_negatives[3]",   col_active, col_parquet, col_pg_entry, "—",        "—",              "—"],
        ["tmd_bits",            col_active, col_parquet, col_pg_entry, "—",        col_neo_concept,  col_graph_edge],
        ["tmd_text (decoded)",  "∆",        "∆",          "∆",         "—",        "∆",             "∆"],
        ["lane_index",          col_active, col_parquet, col_pg_entry, "—",        col_neo_concept,  col_graph_edge],
        ["created_at",          col_active, col_parquet, col_pg_entry, "—",        col_neo_concept,  col_graph_edge],
        ["last_accessed",       col_active, col_parquet, col_pg_entry, "—",        col_neo_concept,  col_graph_edge],
        ["access_count",        col_active, col_parquet, col_pg_entry, "—",        col_neo_concept,  col_graph_edge],
        ["quality / echo_score",col_active, col_parquet, col_pg_entry, "—",        col_neo_concept,  col_graph_edge],
        ["insufficient_evidence",col_active, col_parquet, col_pg_entry, "—",       col_neo_concept,  col_graph_edge],
        ["dataset_source",      col_active, col_parquet, col_pg_entry, "—",        col_neo_concept,  col_graph_edge],
        ["content_type",        col_active, col_parquet, col_pg_entry, "—",        col_neo_concept,  "—"],
        ["chunk_position{...}", col_active, col_parquet, col_pg_entry, "—",        col_neo_concept,  "—"],
        ["relations_text",      col_active, col_parquet, "—",          "—",        "→ edges",        col_graph_edge],
    ]

    headers2 = [
        "Vector / Index",
        "Active JSONL",
        "Parquet Segments",
        "Postgres cpe_vectors",
        "NPZ Vector Store",
        "FAISS Index",
    ]
    rows2 = [
        ["concept_vec 768D",  "—", "—", col_pg_vec, col_npz, "used"],
        ["question_vec 768D", "—", "—", col_pg_vec, col_npz, "used"],
        ["tmd_dense 16D",     "—", "—", col_pg_vec, col_npz, "fused"],
        ["fused_vec 784D",    "—", "—", col_pg_vec, col_npz, col_faiss],
        ["index meta",        "—", "—", "—",           "—",     col_faiss],
    ]

    headers3 = ["Artifact", "Path", "Present"]
    rows3 = [
        ["Active lake (JSONL)",       p["active_path"],        col_active],
        ["Warm segments (Parquet)",    p["parquet_pattern"],    col_parquet],
        ["SQLite locator",             str(SQLITE_IDX),         _tick(p["sqlite"])],
        ["FAISS meta/index",           p["faiss_path"],         col_faiss],
        ["Gating decisions (JSONL)",   str(GATING_DECISIONS),   _tick(p["gating"])],
        ["SLO snapshot (JSON)",        str(SLO_JSON),           _tick(p["slo"])],
        ["NPZ vector store",           p["npz_path"],          col_npz],
    ]

    print("\nLNSP RAG — Storage Matrix\n")
    print(_fmt_table(headers1, rows1))
    print(_fmt_table(headers2, rows2))
    print(_fmt_table(headers3, rows3))
    print("\nLegend: ✓ present / configured   — absent   ∆ decoded at read-time   → materialized into graph edges\n")

def read_json(path):
    try:
        with open(path) as f: return json.load(f)
    except Exception:
        return None

def count_lines(path, limit=None, parse=False):
    cnt = 0; sample=[]
    if not os.path.exists(path): return 0, sample
    with open(path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            if parse and (limit is None or len(sample)<limit):
                try: sample.append(json.loads(line))
                except Exception: pass
            cnt += 1
    return cnt, sample

def count_gz_lines(path, limit=None):
    cnt=0
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for _ in f: 
            cnt+=1
            if limit and cnt>=limit: break
    return cnt

def get_api_json(path):
    if not API: return None
    import urllib.request
    try:
        with urllib.request.urlopen(API.rstrip("/") + path) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception:
        return None

def get_slo(api):
    if not api: return None
    try: return get_api_json("/metrics/slo")
    except Exception: return None

def gating_stats():
    # Prefer live endpoint, fallback to file
    g = get_api_json("/metrics/gating")
    if not g and os.path.exists(METRICS_GATING):
        g = read_json(METRICS_GATING)
    total = int(g.get("total",0)) if g else 0
    used  = int(g.get("used_cpesh",0)) if g else 0

    # latency & nprobe distribution from decision log
    lat_cp, lat_fb, nprobe_hist = [], [], {}
    if os.path.exists(GATING_FILE):
        with open(GATING_FILE) as f:
            for line in f:
                try:
                    j = json.loads(line)
                    nprobe_hist[j.get("chosen_nprobe")] = nprobe_hist.get(j.get("chosen_nprobe"),0)+1
                    if j.get("used_cpesh"):
                        if (lm:=j.get("latency_ms")) is not None: lat_cp.append(float(lm))
                    else:
                        if (lm:=j.get("latency_ms")) is not None: lat_fb.append(float(lm))
                except Exception:
                    pass
    def lat_summary(xs):
        if not xs: return ("—","—","—")
        p50 = percentile(xs,0.5); p95 = percentile(xs,0.95)
        return (f"{len(xs)}", f"{p50:.1f}", f"{p95:.1f}")
    cp_n, cp_p50, cp_p95 = lat_summary(lat_cp)
    fb_n, fb_p50, fb_p95 = lat_summary(lat_fb)
    return total, used, nprobe_hist, (cp_n, cp_p50, cp_p95), (fb_n, fb_p50, fb_p95)

def cpesh_stats():
    # active
    active_path = ACTIVE if os.path.exists(ACTIVE) else (LEGACY if os.path.exists(LEGACY) else None)
    active_lines = 0; q_vals=[]; insuff=0; created_ats=[]
    # training pairs = rows with concept/probe/expected strings present
    pairs = 0
    if active_path:
        active_lines, sample = count_lines(active_path, limit=20000, parse=True)
        for e in sample:
            obj = e.get("cpesh", e)
            if isinstance(obj.get("concept_text"), str) and isinstance(obj.get("probe_question"), str) and isinstance(obj.get("expected_answer"), str):
                pairs += 1
            q = e.get("cpesh",{}).get("quality") if "cpesh" in e else e.get("quality")
            if isinstance(q,(int,float)): q_vals.append(float(q))
            if (e.get("cpesh",{}).get("insufficient_evidence") if "cpesh" in e else e.get("insufficient_evidence")): insuff += 1
            ca = e.get("cpesh",{}).get("created_at") if "cpesh" in e else e.get("created_at")
            if isinstance(ca,str): created_ats.append(ca)
    # warm
    warm_files = glob.glob(WARM_GLOB)
    warm_count = len(warm_files)
    warm_bytes = sum(os.path.getsize(p) for p in warm_files) if warm_files else 0
    warm_lines_est = 0
    for p in warm_files[:5]:  # sample a few to estimate
        try:
            warm_lines_est += count_gz_lines(p, limit=None)
        except Exception:
            pass
    # quantiles
    q_median = f"{median(q_vals):.3f}" if q_vals else "—"
    q_p10 = f"{percentile(q_vals,0.10):.3f}" if q_vals else "—"
    q_p90 = f"{percentile(q_vals,0.90):.3f}" if q_vals else "—"
    return {
        "active_path": active_path or "—",
        "active_lines": active_lines,
        "warm_segments": warm_count,
        "warm_size": sizeof(warm_bytes),
        "warm_lines_est": warm_lines_est if warm_count else 0,
        "quality_median": q_median,
        "quality_p10": q_p10,
        "quality_p90": q_p90,
        "insufficient_in_sample": insuff,
        "created_at_min": min(created_ats) if created_ats else "—",
        "created_at_max": max(created_ats) if created_ats else "—",
        "pairs_in_sample": pairs
    }

def segment_stats():
    """Read cpesh_manifest.jsonl to get segment statistics"""
    if not os.path.exists(MANIFEST):
        return {"segments": 0, "rows": 0, "latest": "—", "storage": "—"}

    segments = []
    total_rows = 0
    latest_created = None

    with open(MANIFEST, "r", encoding="utf-8") as f:
        for line in f:
            try:
                entry = json.loads(line)
                segments.append(entry)
                total_rows += entry.get("rows", 0)
                created_utc = entry.get("created_utc", "")
                if created_utc and (not latest_created or created_utc > latest_created):
                    latest_created = created_utc
            except Exception:
                pass

    # Format latest timestamp
    if latest_created:
        try:
            # Convert from YYYYMMDD_HHMMSS to ISO format
            latest_iso = f"{latest_created[:4]}-{latest_created[4:6]}-{latest_created[6:8]}T{latest_created[9:11]}:{latest_created[11:13]}:{latest_created[13:15]}Z"
        except Exception:
            latest_iso = latest_created
    else:
        latest_iso = "—"

    return {
        "segments": len(segments),
        "rows": f"{total_rows/1000000:.1f} M" if total_rows >= 1000000 else f"{total_rows/1000:.1f} k" if total_rows >= 1000 else str(total_rows),
        "latest": latest_iso,
        "storage": "Parquet (ZSTD)" if segments else "—"
    }

def shard_stats():
    """Read FAISS index files to get vector shard information"""
    faiss_files = glob.glob(FAISS_GLOB)
    if not faiss_files:
        return []

    shards = []
    for fpath in faiss_files:
        try:
            # Try to read basic info about the FAISS index
            # Note: This is a simplified approach - in practice, you'd use faiss library
            shard_name = os.path.basename(fpath).replace(".index", "")
            size_bytes = os.path.getsize(fpath)

            # For now, we'll use heuristics based on file size and name
            # In a real implementation, you'd load the index to get accurate stats
            estimated_vectors = size_bytes // 768 // 4  # rough estimate for 768D float32

            shards.append({
                "shard": shard_name,
                "type": "IVF_FLAT",  # default assumption
                "ntotal": f"{estimated_vectors/1000:.0f}k" if estimated_vectors >= 1000 else str(estimated_vectors),
                "nlist": "—"  # would need to load index to get actual value
            })
        except Exception:
            pass

    return shards

def main():
    global API

    parser = argparse.ArgumentParser(description="LNSP RAG status")
    parser.add_argument("--api", help="Base URL for live API status probes")
    parser.add_argument("--matrix", action="store_true",
                        help="Print storage-location matrix (disk + env) and exit")
    args, unknown = parser.parse_known_args()

    # Preserve compatibility with callers that pass extra args (ignored here)
    if unknown:
        sys.argv = [sys.argv[0]] + unknown
    else:
        sys.argv = [sys.argv[0]]

    API = args.api or os.getenv("LNSP_STATUS_API")

    if args.matrix:
        _print_matrix(_presence_from_disk())
        return

    print("\nLNSP RAG — System Status\n")

    # Index
    meta = read_json(INDEX_META) or {}
    # Handle nested structure - get the most recent index
    if meta and isinstance(meta, dict) and all(isinstance(v, dict) for v in meta.values()):
        # It's nested - get the latest one by build time
        latest_key = max(meta.keys(), key=lambda k: meta[k].get("build_seconds", 0))
        meta = meta[latest_key]

    idx_rows = [[
        meta.get("type","—"),
        meta.get("metric","—"),
        meta.get("count","—"),
        meta.get("nlist","—"),
        meta.get("requested_nlist","—"),
        meta.get("max_safe_nlist","—"),
        meta.get("nprobe","—"),
        meta.get("build_seconds","—"),
    ]]
    print(ascii_table(
        ["IndexType","Metric","Vectors","nlist","requested","max_safe","nprobe","build_s"],
        idx_rows
    ))

    # Check for 40x rule discrepancy
    N = meta.get("count")
    safe_meta = meta.get("max_safe_nlist")
    safe_expected = (N // 40) if isinstance(N, int) and N else None
    if isinstance(safe_meta, int) and isinstance(safe_expected, int) and safe_meta != safe_expected:
        print(f"⚠️  Warning: max_safe_nlist={safe_meta} (from metadata) vs {safe_expected} (expected from N={N} using 40× rule)")
        if safe_meta > safe_expected:
            print(f"   Note: max_safe may be derived from ntrain instead of N")

    # FAISS health (live API best-effort)
    h = get_api_json("/health/faiss")
    if h:
        print(ascii_table(
            ["loaded","trained","ntotal","dim","nlist","type","metric","error"],
            [[h.get("loaded"),h.get("trained"),h.get("ntotal"),h.get("dim"),h.get("nlist"),h.get("type"),h.get("metric"),h.get("error","") or ""]]
        ))

    # CPESH datastore
    ds = cpesh_stats()
    print(ascii_table(
        ["active_file","active_lines","#warm","warm_size","warm_lines(est)","q_med","q_p10","q_p90","insuff_in_sample","created_min","created_max"],
        [[ds["active_path"], ds["active_lines"], ds["warm_segments"], ds["warm_size"], ds["warm_lines_est"],
          ds["quality_median"], ds["quality_p10"], ds["quality_p90"], ds["insufficient_in_sample"],
          ds["created_at_min"], ds["created_at_max"]]]
    ))

    # Segments
    seg = segment_stats()
    print(ascii_table(
        ["segments","rows","latest","storage"],
        [[seg["segments"], seg["rows"], seg["latest"], seg["storage"]]]
    ))

    # Vector Shards
    shards = shard_stats()
    if shards:
        shard_rows = [[s["shard"], s["type"], s["ntotal"], s["nlist"]] for s in shards]
        print(ascii_table(
            ["shard","type","ntotal","nlist"],
            shard_rows
        ))
    else:
        print(ascii_table(
            ["shard","type","ntotal","nlist"],
            [["—", "—", "—", "—"]]
        ))

    # SLO snapshot (if present)
    slo = get_slo(API)
    if slo and slo.get("present", True):
        print(ascii_table(
            ["queries","Hit@1","Hit@3","p50_ms","p95_ms","notes","as_of"],
            [[slo.get("queries","—"), slo.get("hit_at_1","—"), slo.get("hit_at_3","—"),
              slo.get("p50_ms","—"), slo.get("p95_ms","—"), (slo.get("notes") or "")[:36], slo.get("timestamp_utc","—")]]
        ))

    # Gating usage & latency
    total, used, nprobe_hist, cp, fb = gating_stats()
    print(ascii_table(
        ["gating_total","used_cpesh","usage_rate"],
        [[total, used, pct(used,total)]]
    ))
    hist_rows = sorted([(k or "—", v) for k,v in nprobe_hist.items()], key=lambda x: (0 if x[0]=="—" else 1, x[0]))
    if hist_rows:
        print(ascii_table(["nprobe","count"], hist_rows))
    print(ascii_table(
        ["slice","n","p50_ms","p95_ms"],
        [["cpesh", *cp], ["fallback", *fb]]
    ))

    print(ascii_table(
        ["training_pairs(sample)","note"],
        [[ds["pairs_in_sample"], "Sampled from active; Parquet counting coming next."]]
    ))

    # GraphRAG Health (when enabled)
    if API and os.getenv("LNSP_GRAPHRAG_ENABLED", "0") == "1":
        graph_health = get_api_json("/graph/health")
        if graph_health:
            print(ascii_table(
                ["concepts", "edges", "status"],
                [[
                    f"{graph_health.get('concepts', 0):,}",
                    f"{graph_health.get('edges', 0):,}",
                    graph_health.get('status', 'unknown')
                ]]
            ))
        else:
            print(ascii_table(
                ["concepts", "edges", "status"],
                [["—", "—", "GraphRAG API unavailable"]]
            ))

    if API:
        print(f"\nNote: live API read from {API}")
    print("Done.")

if __name__ == "__main__":
    main()
