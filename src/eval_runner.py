# src/eval_runner.py
# P4: Day 3 eval runner — API-driven
# Usage:
#   python3 -m src.eval_runner \
#     --queries eval/day3_eval.jsonl \
#     --api http://localhost:8080/search \
#     --top-k 8 \
#     --timeout 15 \
#     --out eval/day3_results.jsonl

from __future__ import annotations
import argparse, json, time, os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
import requests
from collections import Counter, defaultdict
import numpy as np
import re

# LLM integration setup
USE_LLM = os.getenv("LNSP_USE_LLM", "false").lower() == "true"
try:
    from .llm_bridge import annotate_with_llama
except Exception:
    annotate_with_llama = None

def _load_npz_catalog(npz_path: str):
    import numpy as np
    d = np.load(npz_path, allow_pickle=True)
    ids = d["doc_id"].tolist()
    txt = d["concept_text"].tolist()
    return {i:t for i,t in zip(ids, txt)}

CATALOG = None
if os.getenv("LNSP_OFFLINE_NPZ"):
    try:
        CATALOG = _load_npz_catalog(os.getenv("LNSP_OFFLINE_NPZ"))
    except Exception:
        CATALOG = None

# Defaults (can be overridden by CLI)
DEFAULT_RESULTS = Path("eval/day3_results.jsonl")
REPORT_FILE = Path("eval/day3_report.md")
SAMPLES_DIR = Path("eval/day3_samples")
CHAT_LOG = Path("chats/conversation_09222025_P4.md")


def _tokenize(text: str) -> Tuple[str, ...]:
    if not text:
        return tuple()
    return tuple(re.findall(r"\w+", text.lower()))


def _pick_method_from_response(resp: Dict[str, Any]) -> str:
    method = resp.get("mode")
    if isinstance(method, str) and method:
        return method
    arr = resp.get("items") or resp.get("results") or []
    if isinstance(arr, list) and arr:
        first = arr[0]
        if isinstance(first, dict):
            retriever = first.get("retriever")
            if isinstance(retriever, str) and retriever:
                return retriever.upper()
    return "UNKNOWN"


def _proposition_from_item(item: Dict[str, Any]) -> str:
    return (
        item.get("proposition")
        or item.get("query")
        or item.get("probe")
        or ""
    )


def _tmd_from_item_and_resp(
    item: Dict[str, Any],
    resp: Dict[str, Any],
    default_domain: str = "FACTOIDWIKI",
) -> Dict[str, Any]:
    tmd = item.get("tmd") or {}
    task = tmd.get("task") or "RETRIEVE"
    method = tmd.get("method") or _pick_method_from_response(resp)
    domain = tmd.get("domain") or item.get("domain") or default_domain
    return {"task": task, "method": method, "domain": domain}


def _cpe_from_item(item: Dict[str, Any]) -> Dict[str, Any]:
    cpe = item.get("cpe") or {}
    concept = cpe.get("concept") or item.get("concept")
    probe = cpe.get("probe") or item.get("query") or item.get("probe") or ""
    expected = cpe.get("expected") or item.get("gold") or []
    return {"concept": concept, "probe": probe, "expected": expected}


class LocalSearcher:
    """Fallback lexical retriever that mirrors the API's zero-vector path."""

    def __init__(self, npz_path: Path) -> None:
        self.npz_path = npz_path
        self.catalog: List[Dict[str, Any]] = []
        with np.load(npz_path) as npz:
            cpe_ids = npz.get("cpe_ids", [])
            doc_ids = npz.get("doc_ids", [])
            concepts = npz.get("concept_texts", [])
            size = len(cpe_ids)
            for idx in range(size):
                cpe_id = str(cpe_ids[idx])
                doc_id = str(doc_ids[idx]) if len(doc_ids) > idx else ""
                concept = str(concepts[idx]) if len(concepts) > idx else ""
                tokens = set(_tokenize(concept))
                self.catalog.append({
                    "cpe_id": cpe_id,
                    "doc_id": doc_id,
                    "concept": concept,
                    "tokens": tokens,
                })

    def search(self, query: str, topk: int) -> Dict[str, Any]:
        query_tokens = set(_tokenize(query))
        scored: List[Dict[str, Any]] = []
        if query_tokens:
            for item in self.catalog:
                overlap = len(query_tokens & item["tokens"])
                if overlap:
                    scored.append({
                        "cpe_id": item["cpe_id"],
                        "doc_id": item["doc_id"],
                        "score": float(overlap),
                        "retriever": "lexical",
                        "lane_index": 0,
                        "metadata": {
                            "concept_text": item["concept"],
                            "doc_id": item["doc_id"],
                        },
                    })

        if not scored:
            for item in self.catalog[:topk]:
                scored.append({
                    "cpe_id": item["cpe_id"],
                    "doc_id": item["doc_id"],
                    "score": 0.0,
                    "retriever": "lexical",
                    "lane_index": 0,
                    "metadata": {
                        "concept_text": item["concept"],
                        "doc_id": item["doc_id"],
                    },
                })

        scored.sort(key=lambda x: (-x["score"], x.get("doc_id") or ""))
        results = []
        for rank, item in enumerate(scored[:topk], start=1):
            item = dict(item)
            item["rank"] = rank
            results.append(item)

        return {"query": query, "lane_index": 0, "results": results}

def _read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)

def _write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def _safe_get_hits(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Accepts multiple response shapes:
    - {"items":[{"id":"doc_...","score":...}, ...], "lane":"L1_FACTOID", "mode":"DENSE"}
    - {"results":[{"id":"..."}, ...]}
    - {"results":[{"doc_id":"..."}]}
    - {"items":[{"doc_id":"..."}]}
    Returns a normalized list of dicts with at least {"id": "..."}.
    """
    candidates = []
    arr = None
    if isinstance(payload, dict):
        if isinstance(payload.get("items"), list):
            arr = payload["items"]
        elif isinstance(payload.get("results"), list):
            arr = payload["results"]
    if not isinstance(arr, list):
        return candidates

    for x in arr:
        if not isinstance(x, dict):
            continue
        _id = x.get("id") or x.get("doc_id") or x.get("cpe_id")
        if _id is None:
            # sometimes nested
            meta = x.get("meta") or {}
            _id = meta.get("id") or meta.get("doc_id") or meta.get("cpe_id")
        if _id is None:
            # nothing identifiable
            continue
        # preserve possible score/why/doc_id for samples
        out = {"id": _id}
        if "score" in x: out["score"] = x["score"]
        if "why" in x: out["why"] = x["why"]
        if "doc_id" in x: out["doc_id"] = x["doc_id"]
        candidates.append(out)
    return candidates

def _detect_lane(payload: Dict[str, Any], fallback: str = "unknown") -> str:
    # prefer response-advertised lane; else mode/route; else fallback
    lane = payload.get("lane")
    if isinstance(lane, str) and lane:
        return lane
    # sometimes embedded in meta
    meta = payload.get("meta") or {}
    lane = meta.get("lane")
    if isinstance(lane, str) and lane:
        return lane
    return fallback

def _intersects(gold: List[str], hits: List[Dict[str, Any]]) -> bool:
    if not gold or not hits:
        return False
    gold_set = {g.strip() for g in gold if isinstance(g, str) and g.strip()}
    if not gold_set:
        return False
    for h in hits:
        # Check both id and doc_id fields for matches
        _id = h.get("id")
        _doc_id = h.get("doc_id")
        if (isinstance(_id, str) and _id in gold_set) or (isinstance(_doc_id, str) and _doc_id in gold_set):
            return True
    return False

def _mrr(gold_set: set, hit_ids: List[str]) -> float:
    """Compute Mean Reciprocal Rank for a single query."""
    for i, hid in enumerate(hit_ids, 1):
        if hid in gold_set:
            return 1.0 / i
    return 0.0

def _precision_at_k(gold_set: set, hit_ids: List[str], k: int) -> float:
    """Compute Precision@k for a single query."""
    if k <= 0:
        return 0.0
    relevant_in_top_k = sum(1 for hid in hit_ids[:k] if hid in gold_set)
    return relevant_in_top_k / k

def _recall_at_k(gold_set: set, hit_ids: List[str], k: int) -> float:
    """Compute Recall@k for a single query."""
    if not gold_set or k <= 0:
        return 0.0
    relevant_in_top_k = sum(1 for hid in hit_ids[:k] if hid in gold_set)
    return relevant_in_top_k / len(gold_set)

def _compute_ranking_metrics(gold: List[str], hits: List[Dict[str, Any]], top_k: int) -> Dict[str, float]:
    """Compute P@1, P@5, MRR, and Recall@k metrics for a single query."""
    if not gold or not hits:
        return {"p_at_1": 0.0, "p_at_5": 0.0, "mrr": 0.0, "recall_at_k": 0.0}

    gold_set = {g.strip() for g in gold if isinstance(g, str) and g.strip()}
    if not gold_set:
        return {"p_at_1": 0.0, "p_at_5": 0.0, "mrr": 0.0, "recall_at_k": 0.0}

    # Use doc_id if available, fallback to id
    hit_ids = [h.get("doc_id") or h.get("id") for h in hits if h.get("doc_id") or h.get("id")]

    return {
        "p_at_1": _precision_at_k(gold_set, hit_ids, 1),
        "p_at_5": _precision_at_k(gold_set, hit_ids, 5),
        "mrr": _mrr(gold_set, hit_ids),
        "recall_at_k": _recall_at_k(gold_set, hit_ids, top_k),
    }

def _mk_sample_name(qid: str, lane: str) -> str:
    lane_slim = lane.replace("L", "L").replace("/", "_")
    return f"{qid}_{lane_slim}.json"

def evaluate(
    queries_path: Path,
    api_url: Optional[str],
    top_k: int,
    timeout: int,
    out_path: Path,
    offline: Optional[LocalSearcher] = None,
) -> Dict[str, Any]:
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

    results: List[Dict[str, Any]] = []
    lane_counter = Counter()
    lane_pass = Counter()
    types_counter = Counter()
    latencies_ms: List[float] = []
    status_counter = Counter()

    # New ranking metrics
    all_p_at_1: List[float] = []
    all_p_at_5: List[float] = []
    all_mrr: List[float] = []
    all_recall_at_k: List[float] = []

    total = 0
    pass_count = 0
    fail_count = 0

    for item in _read_jsonl(queries_path):
        total += 1
        qid   = item.get("id") or f"Q{total}"
        lane  = item.get("lane") or "L1_FACTOID"
        query = item.get("query") or item.get("probe") or ""
        qtype = item.get("type") or "unknown"
        gold  = item.get("gold") or []  # list[str]

        # Shape the request as POST with JSON body for the new API contract
        t0 = time.perf_counter()
        if offline is not None:
            data = offline.search(query, top_k)
            status_counter["offline"] += 1
        else:
            payload = {"q": query, "top_k": top_k}
            if lane:
                payload["lane"] = lane
            try:
                assert api_url is not None, "api_url must be provided when offline search is disabled"
                r = requests.post(api_url, json=payload, timeout=timeout)
                status_counter[r.status_code] += 1
                r.raise_for_status()
                data = r.json()
            except Exception as e:
                data = {"error": str(e)}
                status_counter["exception"] += 1
        t1 = time.perf_counter()

        latency_ms = (t1 - t0) * 1000.0
        latencies_ms.append(latency_ms)

        # normalize
        hits = _safe_get_hits(data)
        # prefer API-advertised lane; fall back to requested
        resp_lane = _detect_lane(data, fallback=lane)
        lane_counter[resp_lane] += 1
        types_counter[qtype] += 1

        hit = _intersects(gold, hits)
        if hit:
            pass_count += 1
            lane_pass[resp_lane] += 1
        else:
            fail_count += 1

        # Compute ranking metrics
        ranking_metrics = _compute_ranking_metrics(gold, hits, top_k)
        all_p_at_1.append(ranking_metrics["p_at_1"])
        all_p_at_5.append(ranking_metrics["p_at_5"])
        all_mrr.append(ranking_metrics["mrr"])
        all_recall_at_k.append(ranking_metrics["recall_at_k"])

        # --- existing deterministic fill ---
        proposition = _proposition_from_item(item)
        tmd = _tmd_from_item_and_resp(item, data)
        cpe = _cpe_from_item(item)

        # --- LLM override (opt-in) ---
        if USE_LLM and annotate_with_llama:
            # prepare a few snippets for the LLM (best-effort)
            top_items = data.get("items") or data.get("results") or []
            top_docs = []
            for it in top_items[:3]:
                did = (it.get("doc_id") or (it.get("metadata") or {}).get("doc_id"))
                snippet = None
                if CATALOG and did:
                    snippet = CATALOG.get(did)
                elif it.get("metadata") and it["metadata"].get("concept_text"):
                    snippet = it["metadata"]["concept_text"]
                top_docs.append({"doc_id": did, "text": snippet or ""})
            try:
                anno = annotate_with_llama(query=query,
                                           top_docs=top_docs,
                                           method_hint=tmd["method"],
                                           concept_hint=(cpe.get("concept") or ""),
                                           expected_ids=(cpe.get("expected") or []))
                proposition = anno.get("proposition", proposition) or proposition
                tmd = anno.get("tmd", tmd) or tmd
                cpe = anno.get("cpe", cpe) or cpe
            except Exception:
                # fall back silently if LLM not reachable
                pass

        row = {
            "id": qid,
            "lane_req": lane,
            "lane_resp": resp_lane,
            "type": qtype,
            "query": query,
            "gold": gold,
            "hit": hit,
            "latency_ms": round(latency_ms, 2),
            "response": data,
            "top_k": top_k,
            "status": None,
            "p_at_1": round(ranking_metrics["p_at_1"], 4),
            "p_at_5": round(ranking_metrics["p_at_5"], 4),
            "mrr": round(ranking_metrics["mrr"], 4),
            "recall_at_k": round(ranking_metrics["recall_at_k"], 4),
            "proposition": proposition,
            "tmd": tmd,
            "cpe": cpe,
        }
        # keep compact status
        if isinstance(data, dict) and "error" in data:
            row["status"] = "error"
        else:
            row["status"] = "ok"

        results.append(row)

        # write a sample file for first few of each type (max 1 per type per lane for readability)
        key = (qtype, resp_lane)
        # allow up to 1 sample per (type,lane); keep deterministic first seen
        sample_gate_name = f"__seen_{qtype}__{resp_lane}"
        if not globals().get(sample_gate_name):
            globals()[sample_gate_name] = True
            sample_name = _mk_sample_name(qid, resp_lane)
            with (SAMPLES_DIR / sample_name).open("w", encoding="utf-8") as sf:
                json.dump(
                    {
                        "query": query,
                        "lane_req": lane,
                        "lane_resp": resp_lane,
                        "gold": gold,
                        "hit": hit,
                        "latency_ms": round(latency_ms, 2),
                        "top_k": top_k,
                        "results": hits,
                        "raw": data,
                        "proposition": proposition,
                        "tmd": tmd,
                        "cpe": cpe,
                    },
                    sf, ensure_ascii=False, indent=2
                )

    # persist results
    _write_jsonl(out_path, results)

    # metrics
    mean_latency = sum(latencies_ms) / max(1, len(latencies_ms))
    p_pass = (pass_count / max(1, total)) * 100.0

    # Compute mean ranking metrics
    mean_p_at_1 = sum(all_p_at_1) / max(1, len(all_p_at_1))
    mean_p_at_5 = sum(all_p_at_5) / max(1, len(all_p_at_5))
    mean_mrr = sum(all_mrr) / max(1, len(all_mrr))
    mean_recall_at_k = sum(all_recall_at_k) / max(1, len(all_recall_at_k))

    metrics = {
        "total": total,
        "pass": pass_count,
        "fail": fail_count,
        "echo_pass_pct": round(p_pass, 2),
        "mean_latency_ms": round(mean_latency, 2),
        "mean_p_at_1": round(mean_p_at_1, 4),
        "mean_p_at_5": round(mean_p_at_5, 4),
        "mean_mrr": round(mean_mrr, 4),
        "mean_recall_at_k": round(mean_recall_at_k, 4),
        "lane_counts": dict(lane_counter),
        "lane_pass": dict(lane_pass),
        "types_counts": dict(types_counter),
        "statuses": {str(k): v for k, v in status_counter.items()},
    }

    # render report
    _render_report(metrics, out_path)
    _log_chat_status(metrics, out_path)

    return metrics

def _render_report(metrics: Dict[str, Any], results_path: Path) -> None:
    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    lines.append(f"# Day 3 Retrieval Report — {time.strftime('%Y-%m-%d')}\n")
    lines.append("## Summary\n")
    lines.append(f"- Total queries: **{metrics['total']}**")
    lines.append(f"- Pass: **{metrics['pass']}**  |  Fail: **{metrics['fail']}**")
    lines.append(f"- Echo pass: **{metrics['echo_pass_pct']}%**")
    lines.append(f"- Mean latency: **{metrics['mean_latency_ms']} ms**")
    lines.append("")
    lines.append("## Ranking Metrics")
    lines.append(f"- **P@1**: {metrics['mean_p_at_1']:.4f}")
    lines.append(f"- **P@5**: {metrics['mean_p_at_5']:.4f}")
    lines.append(f"- **MRR**: {metrics['mean_mrr']:.4f}")
    lines.append(f"- **Recall@k**: {metrics['mean_recall_at_k']:.4f}")
    lines.append("")
    lines.append("## Lane Distribution")
    lines.append("| Lane | Queries | Pass | Pass % |")
    lines.append("|------|---------|------|--------|")
    for lane, cnt in sorted(metrics["lane_counts"].items()):
        lp = metrics["lane_pass"].get(lane, 0)
        pct = (lp / cnt * 100.0) if cnt else 0.0
        lines.append(f"| {lane} | {cnt} | {lp} | {pct:.1f}% |")
    lines.append("")
    lines.append("## Query Types")
    for t, c in sorted(metrics["types_counts"].items()):
        lines.append(f"- {t}: {c}")
    lines.append("")
    lines.append("## Status Codes")
    for k, v in sorted(metrics["statuses"].items()):
        lines.append(f"- {k}: {v}")
    lines.append("")
    lines.append("## Artifacts")
    lines.append(f"- Results: `{results_path}`")
    lines.append(f"- Samples: `{SAMPLES_DIR}/`")
    lines.append("")
    REPORT_FILE.write_text("\n".join(lines), encoding="utf-8")

def _log_chat_status(metrics: Dict[str, Any], results_path: Path) -> None:
    CHAT_LOG.parent.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y-%m-%dT%H:%M:%S")
    msg = (
        f"[Consultant] eval_runner: {stamp} — "
        f"total={metrics['total']} pass={metrics['pass']} "
        f"echo={metrics['echo_pass_pct']}% results={results_path}\n"
    )
    with CHAT_LOG.open("a", encoding="utf-8") as f:
        f.write(msg)

def main(argv: List[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Run retrieval API evaluation")
    p.add_argument("--queries", required=True, help="Path to evaluation JSONL queries")
    p.add_argument("--api", help="Base URL for the /search endpoint (omit when using --offline-npz)")
    p.add_argument("--top-k", type=int, default=8, help="Top-k to request per query")
    p.add_argument("--timeout", type=int, default=15, help="HTTP timeout in seconds")
    p.add_argument("--out", required=True, help="Path to write results JSONL")
    p.add_argument("--offline-npz", help="Optional NPZ path for offline lexical evaluation")
    args = p.parse_args(argv)

    queries_path = Path(args.queries)
    out_path = Path(args.out)

    # sanity
    if not queries_path.exists():
        raise SystemExit(f"Queries file not found: {queries_path}")

    offline = None
    api_url = args.api.rstrip("/") if args.api else None

    if args.offline_npz:
        npz_path = Path(args.offline_npz)
        if not npz_path.exists():
            raise SystemExit(f"Offline NPZ not found: {npz_path}")
        offline = LocalSearcher(npz_path)
    elif not api_url:
        raise SystemExit("Provide --api or --offline-npz")

    metrics = evaluate(
        queries_path=queries_path,
        api_url=api_url,
        top_k=args.top_k,
        timeout=args.timeout,
        out_path=out_path,
        offline=offline,
    )
    print(json.dumps(metrics, indent=2))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
