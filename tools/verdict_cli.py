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
import pathlib
import urllib.request
import urllib.error
import webbrowser

# P0: Use Gateway as default endpoint
DEFAULT_GATEWAY = os.environ.get("PAS_GATEWAY_URL", "http://127.0.0.1:6120")
COST_DIR = pathlib.Path(os.environ.get("PAS_COST_DIR", "artifacts/costs"))
HMI_URL = os.environ.get("PAS_HMI_URL", "http://localhost:6101")


def _get(url):
    """GET request"""
    try:
        with urllib.request.urlopen(url) as r:
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
        with urllib.request.urlopen(req) as r:
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
    res = _get(f"{args.rpc}/health")
    if res:
        print(json.dumps(res, indent=2))
        return 0
    return 1


def cmd_describe(args):
    """GET /describe"""
    res = _get(f"{args.rpc}/describe")
    if res:
        print(json.dumps(res, indent=2))
        return 0
    return 1


def cmd_invoke(args):
    """POST /invoke (wrapped Aider)"""
    payload = {
        "target": {"name": args.name, "type": "agent", "role": args.role},
        "payload": {
            "message": args.message,
            "files": args.files,
            "auto_commit": not args.no_commit,
        },
        "policy": {"timeout_s": args.timeout_s, "require_caps": args.require_caps},
        "run_id": args.run_id or "",
    }

    res = _post(f"{args.rpc}/invoke", payload)
    if not res:
        return 1

    if args.json:
        print(json.dumps(res, indent=2))
        return 0

    # Pretty result
    rr = res.get("routing_receipt", {})
    print(f"✓ Status: {rr.get('status')}")
    print(f"  Run ID: {rr.get('run_id')}")
    print(f"  Latency: {rr.get('timings_ms',{}).get('total')} ms")

    # Point to receipt file
    rid = rr.get("run_id")
    if rid:
        receipt_path = COST_DIR / f"{rid}.json"
        print(f"  Receipt: {receipt_path}")

    # Show upstream status
    upstream = res.get("upstream", {}).get("outputs", {})
    if upstream.get("status") == "ok":
        receipt = upstream.get("receipt", {})
        kpis = receipt.get("kpis", {})
        print(f"\n✓ Task completed:")
        print(f"  Files changed: {kpis.get('files_changed', 0)}")
        print(f"  Duration: {kpis.get('duration_seconds', 0):.2f}s")
        print(f"  Cost: ${receipt.get('cost', {}).get('total_cost', 0):.4f}")

        # Open HMI if requested
        if args.open_hmi:
            hmi_actions_url = f"{HMI_URL}/actions"
            print(f"\n🌐 Opening HMI: {hmi_actions_url}")
            webbrowser.open(hmi_actions_url)
    else:
        print(f"\n✗ Task failed:")
        print(f"  Error: {upstream.get('error', 'Unknown error')}")
        return 1

    return 0


def main(argv=None):
    p = argparse.ArgumentParser(
        prog="verdict",
        description="PAS-wrapped Aider CLI - Calls Aider-LCO RPC server with guardrails",
    )
    p.add_argument(
        "--rpc",
        default=DEFAULT_RPC,
        help="Aider-LCO RPC base URL (default: %(default)s)",
    )
    sub = p.add_subparsers(dest="cmd")

    # health command
    p_h = sub.add_parser("health", help="GET /health - Check service status")
    p_h.set_defaults(func=cmd_health)

    # describe command
    p_d = sub.add_parser("describe", help="GET /describe - Get service capabilities")
    p_d.set_defaults(func=cmd_describe)

    # invoke command
    p_i = sub.add_parser("invoke", help="POST /invoke - Execute Aider task")
    p_i.add_argument("--message", required=True, help="Editing instruction for Aider")
    p_i.add_argument("--files", nargs="+", required=True, help="Target files")
    p_i.add_argument("--run-id", default="", help="Optional run ID (for receipts)")
    p_i.add_argument("--timeout-s", type=int, default=120, help="Timeout in seconds")
    p_i.add_argument("--name", default="Aider-LCO", help="Target service name")
    p_i.add_argument("--role", default="execution", help="Target service role")
    p_i.add_argument(
        "--require-caps",
        nargs="*",
        default=["git-edit"],
        help="Required capabilities",
    )
    p_i.add_argument(
        "--no-commit",
        action="store_true",
        help="Disable auto-commit (default: auto-commit enabled)",
    )
    p_i.add_argument("--json", action="store_true", help="Print full JSON response")
    p_i.add_argument(
        "--open-hmi",
        action="store_true",
        help="Open HMI dashboard after successful execution",
    )
    p_i.set_defaults(func=cmd_invoke)

    args = p.parse_args(argv)
    if not args.cmd:
        p.print_help()
        return 2
    return args.func(args) or 0


if __name__ == "__main__":
    raise SystemExit(main())
