#!/usr/bin/env python3
"""Consultant evaluation harness for CPESH latency and quality study."""

from __future__ import annotations

import sys
import os
from pathlib import Path

# Add the parent directory to sys.path for direct execution
if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)

import json
import math
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from src.llm.local_llama_client import LocalLlamaClient
from src.prompt_extractor import extract_cpe_from_text
from src.vectorizer import EmbeddingBackend


LNSP_CPESH_CACHE = os.getenv("LNSP_CPESH_CACHE", "artifacts/cpesh_cache.jsonl")
LANE_TO_INDEX = {"L1_FACTOID": 0, "L2_GRAPH": 1, "L3_SYNTH": 2}
INDEX_TO_LANE = {v: k for k, v in LANE_TO_INDEX.items()}
TOP_K = 5
CANDIDATE_POOL = 64


@dataclass(frozen=True)
class QueryConfig:
    query: str
    lane: str
    gold: Sequence[str]
    source: Optional[str]
    repeat: Optional[int]


class CPESHCache:
    def __init__(self, path: str) -> None:
        self.path = Path(path)
        self.data: Dict[str, Dict[str, object]] = {}
        if self.path.exists():
            with self.path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    record = json.loads(line)
                    doc_id = str(record.get("doc_id"))
                    cpesh = record.get("cpesh") or {}
                    self.data[doc_id] = cpesh

    def get(self, doc_id: str) -> Optional[Dict[str, object]]:
        return self.data.get(doc_id)

    def put(self, doc_id: str, cpesh: Dict[str, object]) -> None:
        if doc_id in self.data:
            return
        self.data[doc_id] = cpesh
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"doc_id": doc_id, "cpesh": cpesh}, ensure_ascii=False) + "\n")


def load_queries(path: Path, limit: int = 100) -> List[QueryConfig]:
    records: List[QueryConfig] = []
    with path.open("r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            if idx >= limit:
                break
            entry = json.loads(line)
            lane = entry.get("lane") or entry.get("lane_req") or "L1_FACTOID"
            records.append(
                QueryConfig(
                    query=entry["query"],
                    lane=lane,
                    gold=[str(x) for x in entry.get("gold", [])],
                    source=entry.get("source"),
                    repeat=entry.get("repeat"),
                )
            )
    return records


def percentile(values: Sequence[float], pct: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    k = (len(ordered) - 1) * pct
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return ordered[int(k)]
    return ordered[f] + (ordered[c] - ordered[f]) * (k - f)


def compute_tmd_code(tmd_dense: Optional[np.ndarray], idx: int) -> Optional[str]:
    if tmd_dense is None or idx >= len(tmd_dense):
        return None
    vec = tmd_dense[idx]
    if len(vec) < 3:
        return None
    domain = int(vec[0])
    task = int(vec[1])
    modifier = int(vec[2])
    return f"{domain}.{task}.{modifier}"


def select_candidates(scores: np.ndarray, mask: np.ndarray, pool: int) -> np.ndarray:
    masked = np.where(mask, scores, -1e9)
    if masked.size <= pool:
        return np.argsort(masked)[::-1]
    idx = np.argpartition(masked, -pool)[-pool:]
    return idx[np.argsort(masked[idx])[::-1]]


def build_items(
    indices: Sequence[int],
    scores: np.ndarray,
    quality: np.ndarray,
    doc_ids: np.ndarray,
    concept_texts: np.ndarray,
    tmd_dense: Optional[np.ndarray],
    *,
    w_cos: float,
    w_quality: float,
    use_quality: bool,
    top_k: int = TOP_K,
) -> List[Dict[str, object]]:
    items: List[Dict[str, object]] = []
    for idx in indices:
        score = float(scores[idx])
        qval = float(quality[idx])
        final = w_cos * score + w_quality * qval if use_quality else score
        items.append(
            {
                "doc_id": str(doc_ids[idx]),
                "score": score,
                "quality": qval,
                "final_score": final,
                "concept_text": str(concept_texts[idx]),
                "tmd_code": compute_tmd_code(tmd_dense, idx),
            }
        )
    items.sort(key=lambda x: x["final_score"], reverse=True)
    return items[:top_k]


def compute_hits(records: Iterable[Dict[str, object]]) -> Tuple[int, int]:
    hit1 = 0
    hit3 = 0
    for record in records:
        gold = set(record["gold"])
        ranked = record["ranked"]
        if not gold or not ranked:
            continue
        if ranked[0] in gold:
            hit1 += 1
        if any(doc in gold for doc in ranked[:3]):
            hit3 += 1
    return hit1, hit3


def main() -> None:
    queries_path = Path("eval/day_s5_queries.jsonl")
    queries = load_queries(queries_path, limit=100)
    if not queries:
        raise SystemExit(f"No queries found at {queries_path}")

    npz = np.load("artifacts/fw10k_vectors_768.npz", allow_pickle=True)
    vectors = np.asarray(npz["vectors"], dtype=np.float32)
    doc_ids = np.asarray(npz["doc_ids"], dtype=object)
    concept_texts = np.asarray(npz["concept_texts"], dtype=object)
    lane_indices = np.asarray(npz["lane_indices"], dtype=int)
    tmd_dense = npz.get("tmd_dense")

    quality_path = Path("artifacts/id_quality.jsonl")
    quality_map: Dict[str, float] = {}
    if quality_path.exists():
        with quality_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                quality_map[str(record["doc_id"])] = float(record.get("quality", 0.5))
    quality_array = np.array([quality_map.get(str(doc), 0.5) for doc in doc_ids], dtype=np.float32)

    embedder = EmbeddingBackend(model_name=os.getenv("LNSP_EMBED_MODEL_DIR", "sentence-transformers/gtr-t5-base"))
    llm = LocalLlamaClient()
    call_counter = {"count": 0}
    original_complete = llm.complete_json

    def counted_complete(prompt: str, timeout_s: Optional[float] = None):  # type: ignore[override]
        call_counter["count"] += 1
        return original_complete(prompt, timeout_s=timeout_s)

    llm.complete_json = counted_complete  # type: ignore

    cache = CPESHCache(LNSP_CPESH_CACHE)

    w_cos = float(os.getenv("LNSP_W_COS", "0.85"))
    w_quality = float(os.getenv("LNSP_W_QUALITY", "0.15"))
    cpesh_k = max(0, int(os.getenv("LNSP_CPESH_MAX_K", "2")))

    unique_keys = sorted({(q.query, q.lane) for q in queries})
    concept_inputs = [extract_cpe_from_text(key[0])["concept"] for key in unique_keys]
    concept_embeddings = embedder.encode(concept_inputs, batch_size=16)
    query_embeddings = embedder.encode([key[0] for key in unique_keys], batch_size=16)

    query_info: Dict[Tuple[str, str], Dict[str, object]] = {}
    for idx, key in enumerate(unique_keys):
        query_text, lane = key
        extracted = extract_cpe_from_text(query_text)
        query_info[key] = {
            "concept_text": extracted["concept"],
            "concept_vec": concept_embeddings[idx],
            "query_vec": query_embeddings[idx],
            "lane_index": LANE_TO_INDEX.get(lane, 0),
        }

    cold_latencies: List[float] = []
    warm_latencies: List[float] = []
    warm_cache: Dict[Tuple[str, str], Dict[str, object]] = {}
    records_with: List[Dict[str, object]] = []

    for qc in queries:
        qkey = (qc.query, qc.lane)
        info = query_info[qkey]
        concept_vec = info["concept_vec"]
        lane_index = info["lane_index"]

        scores = vectors @ concept_vec
        mask = lane_indices == lane_index
        candidate_idx = select_candidates(scores, mask, CANDIDATE_POOL)
        items = build_items(
            candidate_idx,
            scores,
            quality_array,
            doc_ids,
            concept_texts,
            tmd_dense,
            w_cos=w_cos,
            w_quality=w_quality,
            use_quality=True,
            top_k=TOP_K,
        )

        start = time.perf_counter()
        enriched_items: List[Dict[str, object]] = []
        for rank, item in enumerate(items, start=1):
            entry = dict(item)
            if rank <= cpesh_k:
                cached = cache.get(entry["doc_id"])
                if cached is None:
                    prompt = (
                        "Return JSON only for CPESH_EXTRACT.\n"
                        f'Factoid: "{entry["concept_text"]}"\n'
                        '{"concept":"...","probe":"...","expected":"...","soft_negative":"...","hard_negative":"...","insufficient_evidence":false}'
                    )
                    try:
                        cpesh_resp = llm.complete_json(prompt)
                    except Exception as exc:
                        cpesh_resp = {
                            "concept": "",
                            "probe": qc.query[:64],
                            "expected": "",
                            "soft_negative": "",
                            "hard_negative": "",
                            "insufficient_evidence": True,
                            "error": str(exc),
                        }
                    cache.put(entry["doc_id"], cpesh_resp)
                    cached = cpesh_resp
                cpesh_payload = dict(cached)
                query_vec = info["query_vec"]
                if cpesh_payload.get("soft_negative"):
                    soft_vec = embedder.encode([cpesh_payload["soft_negative"]], batch_size=1)[0]
                    cpesh_payload["soft_sim"] = float(np.dot(query_vec, soft_vec))
                if cpesh_payload.get("hard_negative"):
                    hard_vec = embedder.encode([cpesh_payload["hard_negative"]], batch_size=1)[0]
                    cpesh_payload["hard_sim"] = float(np.dot(query_vec, hard_vec))
                entry["cpesh"] = cpesh_payload
            enriched_items.append(entry)

        latency_ms = (time.perf_counter() - start) * 1000.0
        cold_latencies.append(latency_ms)

        ranked_doc_ids = [item["doc_id"] for item in enriched_items]
        records_with.append(
            {
                "query": qc.query,
                "gold": list(qc.gold),
                "ranked": ranked_doc_ids,
                "items": enriched_items,
            }
        )
        warm_cache.setdefault(qkey, {
            "items": enriched_items,
            "ranked": ranked_doc_ids,
            "gold": list(qc.gold),
        })

    for qc in queries:
        key = (qc.query, qc.lane)
        start = time.perf_counter()
        _ = warm_cache[key]
        warm_latencies.append((time.perf_counter() - start) * 1000.0)

    no_cpesh_records: List[Dict[str, object]] = []
    for qc in queries:
        qkey = (qc.query, qc.lane)
        info = query_info[qkey]
        concept_vec = info["concept_vec"]
        lane_index = info["lane_index"]
        scores = vectors @ concept_vec
        mask = lane_indices == lane_index
        candidate_idx = select_candidates(scores, mask, CANDIDATE_POOL)
        items = build_items(
            candidate_idx,
            scores,
            quality_array,
            doc_ids,
            concept_texts,
            tmd_dense,
            w_cos=1.0,
            w_quality=0.0,
            use_quality=False,
            top_k=TOP_K,
        )
        ranked_doc_ids = [item["doc_id"] for item in items]
        no_cpesh_records.append(
            {
                "query": qc.query,
                "gold": list(qc.gold),
                "ranked": ranked_doc_ids,
                "items": items,
            }
        )

    hit1_with, hit3_with = compute_hits(records_with)
    hit1_without, hit3_without = compute_hits(no_cpesh_records)
    total = len(queries)

    cpesh_examples = []
    for record in records_with:
        for rank, item in enumerate(record["items"], start=1):
            cpesh = item.get("cpesh")
            if not cpesh:
                continue
            if cpesh.get("soft_sim") is None and cpesh.get("hard_sim") is None:
                continue
            cpesh_examples.append(
                {
                    "query": record["query"],
                    "doc_id": item["doc_id"],
                    "rank": rank,
                    "cpesh": cpesh,
                }
            )

    seen_pairs = set()
    unique_examples = []
    for example in cpesh_examples:
        key = (example["query"], example["doc_id"])
        if key in seen_pairs:
            continue
        seen_pairs.add(key)
        unique_examples.append(example)

    def example_score(ex: Dict[str, object]) -> float:
        cp = ex["cpesh"]
        soft = cp.get("soft_sim")
        hard = cp.get("hard_sim")
        vals = [v for v in (soft, hard) if isinstance(v, (int, float))]
        if not vals:
            return -1.0
        if len(vals) == 1:
            return abs(vals[0])
        return abs(vals[0] - vals[1])

    unique_examples.sort(key=example_score, reverse=True)
    selected_examples = unique_examples[:3]

    cold_p50 = percentile(cold_latencies, 0.50)
    cold_p95 = percentile(cold_latencies, 0.95)
    warm_p50 = percentile(warm_latencies, 0.50)
    warm_p95 = percentile(warm_latencies, 0.95)

    report_lines = []
    report_lines.append("# Day S5 Consultant Report\n")
    report_lines.append(f"Generated: {time.strftime('%Y-%m-%dT%H:%M:%S')}\n")
    report_lines.append("\n## Settings\n")
    report_lines.append(f"- Queries: {total} (unique: {len(unique_keys)})\n")
    report_lines.append(f"- LNSP_CPESH_MAX_K={cpesh_k}\n")
    report_lines.append(f"- LNSP_CPESH_TIMEOUT_S={os.getenv('LNSP_CPESH_TIMEOUT_S', '12')}\n")
    report_lines.append(f"- LNSP_W_COS={w_cos}, LNSP_W_QUALITY={w_quality}\n")
    report_lines.append(f"- Cache path: {LNSP_CPESH_CACHE}\n")

    report_lines.append("\n## Latency (ms)\n")
    report_lines.append("| State | P50 | P95 | Mean | Notes |\n")
    report_lines.append("|---|---|---|---|---|\n")
    report_lines.append(f"| Cold (cache miss) | {cold_p50:.2f} | {cold_p95:.2f} | {sum(cold_latencies)/len(cold_latencies):.2f} | Includes CPESH generation |\n")
    report_lines.append(f"| Warm (cache hit) | {warm_p50:.5f} | {warm_p95:.5f} | {sum(warm_latencies)/len(warm_latencies):.5f} | Dictionary lookup simulation |\n")
    report_lines.append(f"| Δ (Cold - Warm) | {cold_p50 - warm_p50:.2f} | {cold_p95 - warm_p95:.2f} | {(sum(cold_latencies)/len(cold_latencies)) - (sum(warm_latencies)/len(warm_latencies)):.2f} | Cached avoids LLM |\n")

    report_lines.append("\nLLM complete_json calls:\n")
    report_lines.append(f"- Cold pass total: {call_counter['count']}\n")

    report_lines.append("\n## Retrieval Quality\n")
    report_lines.append("| Setting | Hit@1 | Hit@3 |\n")
    report_lines.append("|---|---|---|\n")
    report_lines.append(f"| Quality + CPESH | {100.0 * hit1_with / total:.1f}% | {100.0 * hit3_with / total:.1f}% |\n")
    report_lines.append(f"| No quality, no CPESH | {100.0 * hit1_without / total:.1f}% | {100.0 * hit3_without / total:.1f}% |\n")

    report_lines.append("\n## CPESH Examples\n")
    if selected_examples:
        for idx, example in enumerate(selected_examples, start=1):
            cp = example["cpesh"]
            report_lines.append(f"{idx}. **Query:** {example['query']}\n")
            report_lines.append(f"   - Rank {example['rank']} doc `{example['doc_id']}`\n")
            report_lines.append(f"   - Expected: {cp.get('expected')}\n")
            report_lines.append(f"   - Soft negative: {cp.get('soft_negative')} (sim={cp.get('soft_sim')})\n")
            report_lines.append(f"   - Hard negative: {cp.get('hard_negative')} (sim={cp.get('hard_sim')})\n")
            report_lines.append(f"   - Probe: {cp.get('probe')}\n")
    else:
        report_lines.append("- No CPESH responses with similarity scores captured.\n")

    report_lines.append("\n## Observations\n")
    report_lines.append(f"- Hit@1 successes: {hit1_with}/{total}\n")
    miss_at3 = total - hit3_with
    report_lines.append(f"- Missed Hit@3 cases: {miss_at3}\n")
    report_lines.append("- Warm-path timings simulate cache retrieval; real API should confirm with live server.\n")

    report_lines.append("\n### Next Knobs\n")
    report_lines.append("- Persist scored warm cache inside API to avoid recompute per request.\n")
    report_lines.append("- Expand evaluation set beyond recycled queries for broader coverage.\n")
    report_lines.append("- Investigate CPESH responses lacking similarity metrics to tighten prompts.\n")

    report_path = Path("eval/day_s5_report.md")
    report_path.write_text("".join(report_lines), encoding="utf-8")

    summary = {
        "queries_total": total,
        "queries_unique": len(unique_keys),
        "latency": {
            "cold_p50": cold_p50,
            "cold_p95": cold_p95,
            "warm_p50": warm_p50,
            "warm_p95": warm_p95,
        },
        "hit_with": {
            "hit1": hit1_with / total,
            "hit3": hit3_with / total,
        },
        "hit_without": {
            "hit1": hit1_without / total,
            "hit3": hit3_without / total,
        },
        "llm_calls": call_counter["count"],
        "examples": selected_examples,
        "report_path": str(report_path),
    }

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
