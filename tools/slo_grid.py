#!/usr/bin/env python
import os, sys, json, time, urllib.request, urllib.error

def post(url, obj):
    data = json.dumps(obj).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type":"application/json"})
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode("utf-8"))

def get(url):
    with urllib.request.urlopen(url) as resp:
        return json.loads(resp.read().decode("utf-8"))

def main():
    api = os.getenv("API", "http://127.0.0.1:8094")
    qfile = os.getenv("QUERIES", "eval/100q.jsonl")
    top_k = int(os.getenv("TOPK", "5"))

    queries = []
    try:
        with open(qfile) as f:
            for line in f:
                try: queries.append(json.loads(line))
                except: pass
    except FileNotFoundError:
        # fallback synthetic tiny set
        queries = [{"id": i, "q": f"query {i}", "lane":"L1_FACTOID"} for i in range(20)]

    total = len(queries)
    hit1 = hit3 = 0
    lat = []

    for q in queries:
        body = {"q": q.get("q"), "lane": q.get("lane","L1_FACTOID"), "top_k": top_k, "compact": True}
        t0 = time.time()
        try:
            r = post(api.rstrip("/") + "/search", body)
        except Exception as e:
            continue
        dt = (time.time()-t0)*1000.0
        lat.append(dt)

        # optional scoring if ground truth doc ids exist
        gts = set(q.get("doc_ids_gt", []))
        if gts and isinstance(r, dict) and "items" in r:
            ids = [it.get("doc_id") for it in r["items"] if it.get("doc_id")]
            if any(i in gts for i in ids[:1]): hit1 += 1
            if any(i in gts for i in ids[:3]): hit3 += 1

    p50 = sorted(lat)[int(0.5*(len(lat)-1))] if lat else None
    p95 = sorted(lat)[int(0.95*(len(lat)-1))] if lat else None

    snap = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "queries": total,
        "hit_at_1": round(hit1/total, 4) if total and hit1 else 0.0,
        "hit_at_3": round(hit3/total, 4) if total and hit3 else 0.0,
        "p50_ms": round(p50, 1) if p50 else None,
        "p95_ms": round(p95, 1) if p95 else None,
        "notes": os.getenv("SLO_NOTES", "")
    }

    # save locally AND POST to API
    os.makedirs("artifacts", exist_ok=True)
    with open("artifacts/metrics_slo.json","w") as f: json.dump(snap, f)
    try:
        post(api.rstrip("/") + "/metrics/slo", snap)
    except Exception:
        pass
    print(json.dumps(snap))

if __name__ == "__main__":
    main()