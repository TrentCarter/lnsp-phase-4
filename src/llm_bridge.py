import os, json, requests
from typing import List, Dict, Any

# Backend selection -----------------------------------------------------------
# Prefer explicit; otherwise infer from available envs.
BACKEND = (os.getenv("LNSP_LLM_BACKEND") or "").lower()  # "ollama" | "openai"
if not BACKEND:
    if os.getenv("OLLAMA_HOST") or os.getenv("OLLAMA_URL"):
        BACKEND = "ollama"
    elif os.getenv("OPENAI_BASE_URL"):
        BACKEND = "openai"
    else:
        BACKEND = "ollama"  # sensible local default

# Common knobs (align with your how-to)
MODEL = os.getenv("LNSP_LLM_MODEL", "llama3:8b")  # e.g., "llama3:8b" (Ollama) or "qwen2.5" (OpenAI-compatible)
DOMAIN_DEFAULT = os.getenv("LNSP_DOMAIN_DEFAULT", "FACTOIDWIKI")

# Backend endpoints
OLLAMA_HOST = os.getenv("OLLAMA_URL") or os.getenv("OLLAMA_HOST") or "http://localhost:11434"
OLLAMA_CHAT = f"{OLLAMA_HOST.rstrip('/')}/api/chat"

OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "http://localhost:8000/v1")  # vLLM/LM Studio/OpenRouter/etc.
OPENAI_KEY = os.getenv("OPENAI_API_KEY", "sk-local")
OPENAI_CHAT = f"{OPENAI_BASE_URL.rstrip('/')}/chat/completions"

SYS = ("You are LNSP's annotation assistant. Given a user query and a few top documents, "
       "produce JSON with: proposition (concise truth-statement), "
       "tmd {task, method, domain}, and cpe {concept, probe, expected}. "
       "Return ONLY valid JSON (no prose, no code fences).")

PROMPT = """Query: {query}

Top docs (id → snippet):
{docs}

Return JSON:
{{
  "proposition": "...",
  "tmd": {{"task":"RETRIEVE","method":"{method_hint}","domain":"{domain}"}},
  "cpe": {{"concept": "{concept_hint}", "probe": "{probe_hint}", "expected": {expected_ids} }}
}}"""

def _strip_fences(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        s = s.strip("` \n")
        if "\n" in s:
            s = s.split("\n", 1)[1]
    return s

def annotate_with_llm(query: str,
                      top_docs: List[Dict[str, Any]],
                      method_hint: str = "HYBRID",
                      concept_hint: str = "",
                      expected_ids: List[str] = None,
                      domain: str = DOMAIN_DEFAULT) -> Dict[str, Any]:
    expected_ids = expected_ids or []
    docs_str = "\n".join(f"- {d.get('doc_id','?')}: {d.get('text','')[:300]}"
                         for d in top_docs if d)

    user_msg = PROMPT.format(
        query=query,
        docs=docs_str or "(no snippets available)",
        method_hint=method_hint,
        concept_hint=concept_hint,
        probe_hint=query,
        expected_ids=json.dumps(expected_ids),
        domain=domain,
    )

    if BACKEND == "openai":
        headers = {"Authorization": f"Bearer {OPENAI_KEY}"}
        body = {"model": MODEL, "messages": [{"role":"system","content":SYS},{"role":"user","content":user_msg}],
                "temperature": 0, "stream": False}
        r = requests.post(OPENAI_CHAT, json=body, headers=headers, timeout=60)
        r.raise_for_status()
        content = r.json()["choices"][0]["message"]["content"]
    else:  # ollama
        body = {"model": MODEL, "messages": [{"role":"system","content":SYS},{"role":"user","content":user_msg}],
                "stream": False, "options": {"temperature": 0}}
        r = requests.post(OLLAMA_CHAT, json=body, timeout=60)
        r.raise_for_status()
        content = r.json().get("message", {}).get("content", "")

    return json.loads(_strip_fences(content) or "{}")