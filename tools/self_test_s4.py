#!/usr/bin/env python3
import os, sys, json, time, argparse, pathlib, requests
from pathlib import Path

def ok(label, cond, msg=""):
    print(("✅" if cond else "❌"), label, (f"— {msg}" if msg else ""))
    return cond

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default="http://localhost:8092")
    ap.add_argument("--ollama", default="http://localhost:11434")
    ap.add_argument("--model_dir", default="data/teacher_models/gtr-t5-base")
    args = ap.parse_args()

    # 0) offline flags
    os.environ.setdefault("TRANSFORMERS_OFFLINE","1")
    os.environ.setdefault("HF_HUB_OFFLINE","1")

    # 1) GTR model loads offline
    try:
        from sentence_transformers import SentenceTransformer
        t0=time.time(); m=SentenceTransformer(args.model_dir)
        dim = m.get_sentence_embedding_dimension()
        v = m.encode(["ping"], normalize_embeddings=True)[0]
        ok("GTR offline load", dim==768 and v.shape[0]==768, f"dim={dim}, ||v||≈{(v**2).sum()**0.5:.2f}")
    except Exception as e:
        ok("GTR offline load", False, str(e))

    # 2) Ollama reachability
    try:
        r = requests.get(f"{args.ollama}/api/tags", timeout=3)
        names = [m.get("name") for m in r.json().get("models",[])]
        ok("Ollama /api/tags", r.ok and any("llama3" in (n or "") for n in names), f"models={names[:4]}")
        # minimal JSON-only generate
        payload = {"model": "llama3.1:8b", "prompt": "Return JSON only: {\"ok\": true}", "options": {"num_predict": 20}}
        r2 = requests.post(f"{args.ollama}/api/generate", json=payload, timeout=6)
        jlines = [json.loads(l.split(":",1)[1]) for l in r2.text.splitlines() if l.startswith('{"response"')]
        text = "".join(d.get("response","") for d in jlines)
        ok("Ollama /api/generate", "{\"ok\": true}" in text.replace(" ",""), text[:80])
    except Exception as e:
        ok("Ollama", False, str(e))

    # 3) LNSP API CPESH-full
    try:
        q = {"q":"Which ocean is largest?","top_k":5,"lane":"L1_FACTOID",
             "use_quality": True, "return_cpesh": True, "cpesh_mode":"full"}
        r = requests.post(f"{args.api}/search", json=q, timeout=8)
        data = r.json()
        items = data.get("items", [])
        cpesh_ok = bool(items) and ("cpesh" in items[0])
        ok("API /search", r.ok and cpesh_ok, f"{len(items)} items; cpesh keys on first={list(items[0].get('cpesh',{}).keys()) if items else []}")
    except Exception as e:
        ok("API /search", False, str(e))

    # 4) GraphRAG runner import via module mode
    try:
        sys.path.insert(0, "src")
        import importlib
        importlib.import_module("adapters.lightrag.graphrag_runner")
        ok("GraphRAG runner import", True)
    except Exception as e:
        ok("GraphRAG runner import", False, str(e))

if __name__ == "__main__":
    main()
