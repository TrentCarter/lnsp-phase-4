
#!/usr/bin/env python3
import argparse, json, os, pathlib, urllib.request, webbrowser

DEFAULT_RPC = os.environ.get("CLAUDIA_RPC", "http://127.0.0.1:6151")
HMI_URL = os.environ.get("PAS_HMI_URL", "http://127.0.0.1:6101")
COST_DIR = pathlib.Path(os.environ.get("PAS_COST_DIR", "artifacts/costs"))

def _get(url):
    with urllib.request.urlopen(url) as r:
        return json.loads(r.read().decode("utf-8"))

def _post(url, data):
    req = urllib.request.Request(
        url, data=json.dumps(data).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(req) as r:
        return json.loads(r.read().decode("utf-8"))

def cmd_health(args):
    print(json.dumps(_get(f"{args.rpc}/health"), indent=2))

def cmd_describe(args):
    print(json.dumps(_get(f"{args.rpc}/describe"), indent=2))

def cmd_invoke(args):
    payload = {
        "target": {"name":"Claude-LCO","type":"host","role":"execution"},
        "payload": {"message": args.message, "files": args.files, "dry_run": args.dry_run},
        "policy": {"timeout_s": args.timeout_s, "require_caps": ["git-edit"]},
        "run_id": args.run_id or ""
    }
    res = _post(f"{args.rpc}/invoke", payload)
    if args.json:
        print(json.dumps(res, indent=2))
    else:
        rr = res.get("routing_receipt", {})
        print(f"status: {rr.get('status')}")
        print(f"run_id: {rr.get('run_id')}")
        print(f"latency_ms: {rr.get('timings_ms',{}).get('total')}")
        rid = rr.get("run_id")
        if rid:
            print(f"receipt: {COST_DIR / f'{rid}.json'}")
    if args.open_hmi:
        try:
            webbrowser.open(f"{HMI_URL}/actions")
        except Exception:
            pass

def main(argv=None):
    p = argparse.ArgumentParser(prog="claudia", description="Claude-LCO RPC CLI")
    p.add_argument("--rpc", default=DEFAULT_RPC, help="Claude-LCO RPC base URL")
    sub = p.add_subparsers(dest="cmd")

    p_h = sub.add_parser("health", help="GET /health"); p_h.set_defaults(func=cmd_health)
    p_d = sub.add_parser("describe", help="GET /describe"); p_d.set_defaults(func=cmd_describe)

    p_i = sub.add_parser("invoke", help="POST /invoke (Claude host)")
    p_i.add_argument("--message", required=True); p_i.add_argument("--files", nargs="+", required=True)
    p_i.add_argument("--run-id", default=""); p_i.add_argument("--timeout-s", type=int, default=120)
    p_i.add_argument("--dry-run", action="store_true"); p_i.add_argument("--json", action="store_true")
    p_i.add_argument("--open-hmi", action="store_true")
    p_i.set_defaults(func=cmd_invoke)

    args = p.parse_args(argv)
    if not args.cmd:
        p.print_help(); return 2
    return args.func(args) or 0

if __name__ == "__main__":
    import sys
    raise SystemExit(main(sys.argv[1:]))
