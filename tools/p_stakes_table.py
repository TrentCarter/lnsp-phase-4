#!/usr/bin/env python3
"""
P-Stakes ASCII status table for LNSP (P1..P17)

- Parses docs/PRDs/lnsp_lrag_tmd_cpe_pipeline.md
- For each Px section, extracts candidate artifact paths mentioned in the text
  (e.g., src/... , scripts/... , docs/... , eval/... , artifacts/... , data/... , tests/...)
- Runs a dynamic "existence test":
    PASS (✅) if any referenced artifact exists (files or globs) under --root
    FAIL (❌) otherwise
- Prints an ASCII table with columns: Stake | Title | Status | Artifact
- Optionally writes the table to --out and returns non-zero exit if any FAIL.

Usage:
  python tools/p_stakes_table.py \
      --md docs/PRDs/lnsp_lrag_tmd_cpe_pipeline.md \
      --root . \
      --out docs/status/p_stakes_table.txt

Tips:
- You can override/augment discovered artifacts per-step via a tiny JSON file
  mapping, e.g.:
    {
      "P4": ["eval/day3_report.md", "eval/day3_results.jsonl"]
    }
  Then pass it with --extra-mapping tests/p_stakes_overrides.json
"""

from __future__ import annotations
import argparse
import json
import os
import re
import sys
from glob import glob
from pathlib import Path
from typing import Dict, List, Tuple, Optional

GREEN_CHECK = "✅"
RED_X = "❌"

P_RANGE = list(range(1, 18))  # P1..P17

# Paths we consider "artifacts" when scanning text
ARTIFACT_PREFIXES = (
    "src/", "scripts/", "docs/", "eval/", "artifacts/", "data/", "tests/", "chats/"
)

# Regex to find P-sections and artifact-ish paths inside them
RE_P_HEADER = re.compile(r"^(?:\s*#{1,6}\s*)?P(?P<num>\d{1,2})\b(?P<title>[^\n]*)$", re.MULTILINE)
RE_PATH = re.compile(
    r"(?P<path>\b(?:src|scripts|docs|eval|artifacts|data|tests|chats)/[A-Za-z0-9_.\-\/]+)",
    re.IGNORECASE,
)

def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        sys.stderr.write(f"[p-stakes] Spec not found: {path}\n")
        return ""

def split_sections(md: str) -> Dict[str, Dict[str, object]]:
    """Return dict like {'P1': {'title': ' ...', 'body': '...'}, ...}"""
    sections: Dict[str, Dict[str, object]] = {}
    if not md:
        return sections

    matches = list(RE_P_HEADER.finditer(md))
    for i, m in enumerate(matches):
        num = int(m.group("num"))
        key = f"P{num}"
        title_line = f"P{num}{m.group('title')}".strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(md)
        body = md[start:end]
        sections[key] = {"title": title_line, "body": body}
    return sections

def extract_paths(text: str) -> List[str]:
    paths = []
    for pm in RE_PATH.finditer(text):
        p = pm.group("path").strip().rstrip(").,` ")
        paths.append(p)
    # de-dup preserve order
    seen = set()
    out = []
    for p in paths:
        if p not in seen:
            out.append(p)
            seen.add(p)
    return out

def test_artifacts(root: Path, candidates: List[str]) -> Tuple[bool, Optional[str]]:
    """
    PASS if any candidate artifact exists.
    Supports glob patterns too (e.g., data/index/*).
    Returns (pass, first_existing_path_rel).
    """
    for cand in candidates:
        # If the candidate contains invalid chars (like trailing punctuation), strip last char heuristically:
        c = cand.strip()
        # Try both as-is and as a glob
        for mode in ("exact", "glob"):
            if mode == "exact":
                p = (root / c).resolve()
                if p.exists():
                    return True, str(Path(c))
            else:
                matches = [Path(m) for m in glob(str(root / c), recursive=True)]
                for m in matches:
                    if m.exists():
                        # Return path relative to root
                        try:
                            rel = m.resolve().relative_to(root.resolve())
                        except Exception:
                            rel = m
                        return True, str(rel)
    return False, None

def truncate(s: str, n: int) -> str:
    return s if len(s) <= n else s[: max(0, n - 1 - 3)] + "..."

def build_table(rows: List[Tuple[str, str, str, str]],
                max_width: int = 120,
                col_widths: Optional[Tuple[int, int, int, int]] = None) -> str:
    """
    rows: list of (Stake, Title, Status, Artifact)
    """
    # default widths (tunable)
    if col_widths is None:
        # allocate roughly: stake 6, status 7, rest split
        w_stake = 6
        w_status = 7
        w_art = 40
        w_title = max_width - (w_stake + w_status + w_art + 5)  # +5 for pipes
        col_widths = (w_stake, w_title, w_status, w_art)

    w1, w2, w3, w4 = col_widths
    h = "+{}+{}+{}+{}+".format("-"*w1, "-"*w2, "-"*w3, "-"*w4)
    def row_line(a,b,c,d):
        return "|{:<{w1}}|{:<{w2}}|{:<{w3}}|{:<{w4}}|".format(
            truncate(a, w1), truncate(b, w2), truncate(c, w3), truncate(d, w4),
            w1=w1, w2=w2, w3=w3, w4=w4
        )

    out = [h, row_line("Stake","Title","Status","Artifact"), h]
    for stake, title, status, artifact in rows:
        out.append(row_line(stake, title, status, artifact or "—"))
    out.append(h)
    return "\n".join(out)

def load_extra_mapping(path: Optional[Path]) -> Dict[str, List[str]]:
    if not path:
        return {}
    if not path.exists():
        sys.stderr.write(f"[p-stakes] Extra mapping file not found: {path}\n")
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        # Normalize to list[str]
        out: Dict[str, List[str]] = {}
        for k, v in data.items():
            if isinstance(v, str):
                out[k] = [v]
            elif isinstance(v, list):
                out[k] = [str(x) for x in v]
        return out
    except Exception as e:
        sys.stderr.write(f"[p-stakes] Failed to parse extra mapping JSON: {e}\n")
        return {}

def main() -> int:
    ap = argparse.ArgumentParser(description="Generate P1..P17 ASCII status table from pipeline spec.")
    ap.add_argument("--md", required=True, help="Path to lnsp_lrag_tmd_cpe_pipeline.md")
    ap.add_argument("--root", default=".", help="Project root to resolve artifacts")
    ap.add_argument("--out", default=None, help="Write table to this file (optional)")
    ap.add_argument("--max-width", type=int, default=120, help="Max table width")
    ap.add_argument("--extra-mapping", default=None,
                    help="Optional JSON file mapping e.g. {\"P4\": [\"eval/day3_report.md\"]}")
    args = ap.parse_args()

    md_path = Path(args.md)
    root = Path(args.root)
    extra = load_extra_mapping(Path(args.extra_mapping)) if args.extra_mapping else {}

    md = read_text(md_path)
    sections = split_sections(md)
    rows: List[Tuple[str, str, str, str]] = []
    any_fail = False

    for n in P_RANGE:
        key = f"P{n}"
        if key in sections:
            title = sections[key]["title"] or key
            body = sections[key]["body"] or ""
            artifacts = extract_paths(title + "\n" + body)
        else:
            title = key
            artifacts = []

        # Augment with explicit overrides, if provided
        if key in extra:
            artifacts = list(dict.fromkeys(artifacts + extra[key]))  # de-dup, preserve order

        passed, art = test_artifacts(root, artifacts)
        status = GREEN_CHECK if passed else RED_X
        if not passed:
            any_fail = True
        rows.append((key, str(title).strip(), status, art or ""))

    table = build_table(rows, max_width=args.max_width)
    print(table)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(table + "\n", encoding="utf-8")

    # Return non-zero if any fail (useful for CI)
    return 0 if not any_fail else 2

if __name__ == "__main__":
    raise SystemExit(main())
