#!/usr/bin/env bash
set -euo pipefail

usage(){ cat <<'USAGE'
eval_phase2.sh --ckpt PATH --vecdb PATH --k 1 5 10 [--lane on|off]
USAGE
}

CKPT=""
VECDB=""
LANE=on
KS=(1 5 10)

while [[ $# -gt 0 ]]; do
  case "$1" in
    --ckpt) CKPT=$2; shift 2;;
    --vecdb) VECDB=$2; shift 2;;
    --k) shift; KS=(); while [[ $# -gt 0 && $1 != --* ]]; do KS+=("$1"); shift; done;;
    --lane) LANE=$2; shift 2;;
    -h|--help) usage; exit 0;;
    *) echo "Unknown arg: $1"; usage; exit 2;;
  esac
done

[[ -z "$CKPT" || -z "$VECDB" ]] && { echo "Missing --ckpt/--vecdb"; exit 2; }

python3 eval_hitk.py \
  --ckpt "$CKPT" --vector-bank "$VECDB" \
  --k ${KS[*]} --lane-filter "$LANE" \
  --normalize l2 --delta-reconstruct on
