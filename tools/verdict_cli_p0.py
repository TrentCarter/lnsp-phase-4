#!/usr/bin/env python3
"""
Verdict: PAS Gateway CLI (P0 Production Scaffold)

Usage:
    verdict --help
    verdict health
    verdict send --title "..." --goal "..." --entry-file "..."
    verdict status --run-id <uuid>
    verdict list
"""
import argparse
import json
import os
import sys
import urllib.request
import urllib.error

DEFAULT_GATEWAY = os.environ.get("PAS_GATEWAY_URL", "http://127.0.0.1:6120")


def _get(url):
    """GET request"""
    try:
        with urllib.request.urlopen(url, timeout=15) as r:
            return json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        print(f"HTTP Error {e.code}: {e.reason}", file=sys.stderr)
        return None
    except urllib.error.URLError as e:
        print(f"URL Error: {e.reason}", file=sys.stderr)
        return None


def _post(url, data):
    """POST request"""
    try:
        req = urllib.request.Request(
            url,
            data=json.dumps(data).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        print(f"HTTP Error {e.code}: {e.reason}", file=sys.stderr)
        try:
            error_body = e.read().decode("utf-8")
            print(f"Error details: {error_body}", file=sys.stderr)
        except:
            pass
        return None
    except urllib.error.URLError as e:
        print(f"URL Error: {e.reason}", file=sys.stderr)
        return None


def cmd_health(args):
    """GET /health"""
    res = _get(f"{args.gateway}/health")
    if res:
        print(json.dumps(res, indent=2))
        return 0
    return 1


def cmd_send(args):
    """POST /prime_directives (send Prime Directive)"""
    # Build payload
    payload = {
        "title": args.title,
        "description": args.description,
        "repo_root": args.repo_root,
        "goal": args.goal,
        "entry_files": args.entry_file,
    }

    # Add optional budget constraints
    if args.budget_tokens:
        payload["budget_tokens_max"] = args.budget_tokens
    if args.budget_cost:
        payload["budget_cost_usd_max"] = args.budget_cost

    res = _post(f"{args.gateway}/prime_directives", payload)
    if not res:
        return 1

    # Pretty result
    print(json.dumps(res, indent=2))
    print(f"\n✓ Prime Directive submitted: {res.get('run_id')}")
    print(f"  Status: {res.get('status')}")
    print(f"\nCheck status with: verdict status --run-id {res.get('run_id')}")
    return 0


def cmd_status(args):
    """GET /runs/{run_id} (check run status)"""
    res = _get(f"{args.gateway}/runs/{args.run_id}")
    if not res:
        return 1

    # Pretty result
    print(json.dumps(res, indent=2))
    print(f"\n✓ Run ID: {res.get('run_id')}")
    print(f"  Status: {res.get('status')}")
    if res.get("message"):
        print(f"  Message: {res.get('message')}")
    if res.get("duration_s"):
        print(f"  Duration: {res.get('duration_s')}s")

    # Check artifacts
    if res.get("status") == "completed":
        run_id = res.get("run_id")
        print(f"\nArtifacts: artifacts/runs/{run_id}/")

    return 0


def cmd_list(args):
    """GET /runs (list recent runs)"""
    params = f"?limit={args.limit}"
    if args.status:
        params += f"&status={args.status}"

    res = _get(f"{args.gateway}/runs{params}")
    if not res:
        return 1

    # Pretty result
    runs = res.get("runs", [])
    print(f"Recent runs (showing {len(runs)}):\n")
    for r in runs:
        print(f"  {r['run_id'][:8]}...  {r['status']:10s}  {r.get('title', 'N/A')}")

    return 0


def main(argv=None):
    p = argparse.ArgumentParser(
        prog="verdict",
        description="Verdict CLI - Submit Prime Directives to PAS Gateway",
    )
    p.add_argument(
        "--gateway",
        default=DEFAULT_GATEWAY,
        help="PAS Gateway base URL (default: %(default)s)",
    )
    sub = p.add_subparsers(dest="cmd")

    # health command
    p_h = sub.add_parser("health", help="GET /health - Check service status")
    p_h.set_defaults(func=cmd_health)

    # send command
    p_s = sub.add_parser("send", help="POST /prime_directives - Submit Prime Directive")
    p_s.add_argument("--title", required=True, help="Short title for Prime Directive")
    p_s.add_argument("--description", default="CLI submission", help="Description")
    p_s.add_argument(
        "--repo-root",
        default=os.getcwd(),
        help="Repository root path (default: current directory)",
    )
    p_s.add_argument("--goal", required=True, help="Natural language goal")
    p_s.add_argument(
        "--entry-file",
        action="append",
        default=[],
        help="Entry file(s) for Aider (can specify multiple times)",
    )
    p_s.add_argument(
        "--budget-tokens", type=int, help="Max tokens budget (optional)"
    )
    p_s.add_argument("--budget-cost", type=float, help="Max cost in USD (optional)")
    p_s.set_defaults(func=cmd_send)

    # status command
    p_st = sub.add_parser("status", help="GET /runs/{run_id} - Check run status")
    p_st.add_argument("--run-id", required=True, help="Run ID to check")
    p_st.set_defaults(func=cmd_status)

    # list command
    p_l = sub.add_parser("list", help="GET /runs - List recent runs")
    p_l.add_argument("--limit", type=int, default=50, help="Max runs to show")
    p_l.add_argument(
        "--status", help="Filter by status (queued, running, completed, error)"
    )
    p_l.set_defaults(func=cmd_list)

    args = p.parse_args(argv)
    if not args.cmd:
        p.print_help()
        return 2
    return args.func(args) or 0


if __name__ == "__main__":
    raise SystemExit(main())
