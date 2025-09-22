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
from typing import Any, Dict, Iterable, List, Tuple
import requests
from collections import Counter, defaultdict

# Defaults (can be overridden by CLI)
DEFAULT_RESULTS = Path("eval/day3_results.jsonl")
REPORT_FILE = Path("eval/day3_report.md")
SAMPLES_DIR = Path("eval/day3_samples")
CHAT_LOG = Path("chats/conversation_09222025_P4.md")

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
        # preserve possible score/why for samples
        out = {"id": _id}
        if "score" in x: out["score"] = x["score"]
        if "why" in x: out["why"] = x["why"]
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
        _id = h.get("id")
        if isinstance(_id, str) and _id in gold_set:
            return True
    return False

def _mk_sample_name(qid: str, lane: str) -> str:
    lane_slim = lane.replace("L", "L").replace("/", "_")
    return f"{qid}_{lane_slim}.json"

def evaluate(
    queries_path: Path,
    api_url: str,
    top_k: int,
    timeout: int,
    out_path: Path,
) -> Dict[str, Any]:
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

    results: List[Dict[str, Any]] = []
    lane_counter = Counter()
    lane_pass = Counter()
    types_counter = Counter()
    latencies_ms: List[float] = []
    status_counter = Counter()

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

        # Shape the request. We POST JSON so FastAPI can evolve without querystring hell.
        payload = {"q": query, "lane": lane, "top_k": top_k}
        t0 = time.perf_counter()
        try:
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
        if not globals().__dict__.get(sample_gate_name):
            globals().__dict__[sample_gate_name] = True
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
                    },
                    sf, ensure_ascii=False, indent=2
                )

    # persist results
    _write_jsonl(out_path, results)

    # metrics
    mean_latency = sum(latencies_ms) / max(1, len(latencies_ms))
    p_pass = (pass_count / max(1, total)) * 100.0

    metrics = {
        "total": total,
        "pass": pass_count,
        "fail": fail_count,
        "echo_pass_pct": round(p_pass, 2),
        "mean_latency_ms": round(mean_latency, 2),
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
    p.add_argument("--api", required=True, help="Base URL for the /search endpoint")
    p.add_argument("--top-k", type=int, default=8, help="Top-k to request per query")
    p.add_argument("--timeout", type=int, default=15, help="HTTP timeout in seconds")
    p.add_argument("--out", required=True, help="Path to write results JSONL")
    args = p.parse_args(argv)

    queries_path = Path(args.queries)
    out_path = Path(args.out)

    # sanity
    if not queries_path.exists():
        raise SystemExit(f"Queries file not found: {queries_path}")

    metrics = evaluate(
        queries_path=queries_path,
        api_url=args.api.rstrip("/"),
        top_k=args.top_k,
        timeout=args.timeout,
        out_path=out_path,
    )
    print(json.dumps(metrics, indent=2))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
