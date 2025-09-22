#!/usr/bin/env bash
# Evaluate retrieval quality and echo metrics for LNSP.
#
# Modes:
#   1. JSONL evaluation (default) — hits running API, compares to gold, writes report.
#      Usage: scripts/eval_echo.sh --queries eval/day3_eval.jsonl --api http://localhost:8080/search
#      Optional flags: --top-k 8 --out eval/day3_results.jsonl --report eval/day3_report.md --samples eval/day3_samples
#   2. Legacy NPZ mode — retains Day 2 vector norm check.
#      Usage: scripts/eval_echo.sh --npz artifacts/fw1k_vectors.npz --threshold 0.82 --report eval/day2_report.md

set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT_DIR"

MODE="queries"
QUERIES="eval/day3_eval.jsonl"
API_URL="http://localhost:8080/search"
TOP_K=8
RESULTS_PATH="eval/day3_results.jsonl"
REPORT_PATH=""
SAMPLES_DIR="eval/day3_samples"
NPZ_PATH=""
THRESH=0.82

usage() {
  cat <<'USAGE'
Usage:
  scripts/eval_echo.sh [--queries FILE --api URL --top-k N --out FILE --report FILE --samples DIR]
  scripts/eval_echo.sh --npz FILE [--threshold 0.82] [--report FILE]
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --queries)
      MODE="queries"
      QUERIES="$2"
      shift 2
      ;;
    --api)
      API_URL="$2"
      shift 2
      ;;
    --top-k)
      TOP_K="$2"
      shift 2
      ;;
    --out)
      RESULTS_PATH="$2"
      shift 2
      ;;
    --report)
      REPORT_PATH="$2"
      shift 2
      ;;
    --samples)
      SAMPLES_DIR="$2"
      shift 2
      ;;
    --npz)
      MODE="npz"
      NPZ_PATH="$2"
      shift 2
      ;;
    --threshold)
      THRESH="$2"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
 done

if [[ "$MODE" == "npz" ]]; then
  if [[ -z "$NPZ_PATH" ]]; then
    echo "❌ Missing --npz path" >&2
    exit 1
  fi
  if [[ -z "$REPORT_PATH" ]]; then
    REPORT_PATH="eval/day2_report.md"
  fi
  mkdir -p "$(dirname "$REPORT_PATH")"
  export NPZ_PATH REPORT_PATH THRESH
  python3 - <<'PY'
import numpy as np
import os
from pathlib import Path

npz_path = Path(os.environ['NPZ_PATH']).expanduser().resolve()
threshold = float(os.environ['THRESH'])
report_path = Path(os.environ['REPORT_PATH']).expanduser().resolve()
if not npz_path.exists():
    raise SystemExit(f"❌ Missing vectors NPZ: {npz_path}")
npz = np.load(npz_path)
norms = npz["norms"].astype("float32")
passed = int((norms >= threshold).sum())
failed = int(norms.size - passed)
ratio = passed / max(1, norms.size)
report_path.parent.mkdir(parents=True, exist_ok=True)
with report_path.open("w", encoding="utf-8") as handle:
    handle.write("# Day 2 Echo Evaluation\n")
    handle.write(f"Vectors: {npz_path}  \n")
    handle.write(f"Threshold: {threshold}\n\n")
    handle.write("## Results\n")
    handle.write(f"- Total: {norms.size}\n")
    handle.write(f"- Passed: {passed}\n")
    handle.write(f"- Failed: {failed}\n")
    handle.write(f"- Pass Ratio: {ratio:.3f}\n")
print(f"✅ Wrote {report_path}")
PY
else
  if [[ -z "$REPORT_PATH" ]]; then
    REPORT_PATH="eval/day3_report.md"
  fi
  export QUERIES RESULTS_PATH REPORT_PATH API_URL SAMPLES_DIR
  export TOP_K THRESH
  mkdir -p "$(dirname "$RESULTS_PATH")"
  mkdir -p "$(dirname "$REPORT_PATH")"
  mkdir -p "$SAMPLES_DIR"
  python3 -m src.eval_runner \
    --queries "$QUERIES" \
    --api "$API_URL" \
    --top-k "$TOP_K" \
    --out "$RESULTS_PATH"
  python3 - <<'PY'
import json
import os
from collections import Counter
from pathlib import Path

queries_path = Path(os.environ['QUERIES']).expanduser()
results_path = Path(os.environ['RESULTS_PATH']).expanduser()
report_path = Path(os.environ['REPORT_PATH']).expanduser()
samples_dir = Path(os.environ['SAMPLES_DIR']).expanduser()
if not queries_path.exists() or not results_path.exists():
    raise SystemExit("❌ Missing inputs for metrics computation")

with queries_path.open("r", encoding="utf-8") as handle:
    queries = [json.loads(line) for line in handle if line.strip()]
lookup = {row['id']: row for row in queries}

records = []
with results_path.open("r", encoding="utf-8") as handle:
    for line in handle:
        if line.strip():
            records.append(json.loads(line))

summary = {
    'total': len(records),
    'passes': 0,
    'p_at_1': 0,
    'p_at_5': 0,
    'lane_counts': Counter(),
    'lane_passes': Counter(),
    'lane_p1': Counter(),
    'lane_p5': Counter(),
    'mode_mix': Counter(),
    'latencies_ms': [],
}

samples = []

for record in records:
    qid = record['id']
    query_info = lookup.get(qid, {})
    gold = set(query_info.get('gold', []))
    lane = query_info.get('lane', 'UNKNOWN') or 'UNKNOWN'
    summary['lane_counts'][lane] += 1
    status = record.get('status', 'error')
    items = []
    if status == 'ok':
        payload = record.get('response') or {}
        items = list(payload.get('items', []))
        mode = payload.get('mode')
        if lane == 'L3_SYNTH' and mode:
            summary['mode_mix'][mode] += 1
        summary['latencies_ms'].append(record.get('elapsed_ms', 0.0))
    else:
        payload = {}

    hit_ids = [item.get('id') for item in items if item.get('id')]
    passed = any(doc_id in gold for doc_id in hit_ids)
    p1 = 1 if hit_ids[:1] and hit_ids[0] in gold else 0
    p5 = 1 if any(doc_id in gold for doc_id in hit_ids[:5]) else 0

    if passed:
        summary['passes'] += 1
        summary['lane_passes'][lane] += 1
    summary['p_at_1'] += p1
    summary['p_at_5'] += p5
    summary['lane_p1'][lane] += p1
    summary['lane_p5'][lane] += p5

    if len(samples) < 5:
        samples.append({
            'id': qid,
            'request_url': record.get('request_url'),
            'lane': lane,
            'query': query_info.get('query'),
            'response': record.get('response') or {},
            'status': status,
        })

report_path.parent.mkdir(parents=True, exist_ok=True)

average_latency = sum(summary['latencies_ms']) / len(summary['latencies_ms']) if summary['latencies_ms'] else 0.0

lines = []
lines.append("# Day 3 Retrieval Report\n")
lines.append(f"Total queries: {summary['total']}  ")
lines.append(f"Echo pass rate: {summary['passes']}/{summary['total']} ({(summary['passes']/summary['total']*100 if summary['total'] else 0):.1f}%)\n")
lines.append(f"Mean latency: {average_latency:.1f} ms\n")

lines.append("\n## Lane Metrics\n")
lines.append("| Lane | Queries | Pass | P@1 | P@5 |\n")
lines.append("|------|---------|------|-----|-----|\n")
for lane in sorted(summary['lane_counts']):
    total = summary['lane_counts'][lane]
    passes = summary['lane_passes'][lane]
    p1 = summary['lane_p1'][lane]
    p5 = summary['lane_p5'][lane]
    lines.append(f"| {lane} | {total} | {passes} | {p1} | {p5} |\n")

if summary['mode_mix']:
    lines.append("\n## L3 Mode Mix\n")
    total_modes = sum(summary['mode_mix'].values())
    for mode, count in summary['mode_mix'].most_common():
        frac = (count / total_modes * 100) if total_modes else 0
        lines.append(f"- {mode}: {count} ({frac:.1f}%)\n")

with report_path.open("w", encoding="utf-8") as handle:
    handle.writelines(lines)

for sample in samples:
    sample_path = samples_dir / f"{sample['id']}.json"
    with sample_path.open("w", encoding="utf-8") as handle:
        json.dump(sample, handle, indent=2)

print(f"✅ Wrote report to {report_path}")
print(f"✅ Wrote {len(samples)} samples to {samples_dir}")
PY
fi
