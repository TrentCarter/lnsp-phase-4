import os
import json
import requests
from typing import List, Dict, Any

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/chat")
LLM_MODEL = os.getenv("LNSP_LLM_MODEL", "llama3:8b")  # e.g., llama3:8b, llama3.1:70b

SYS = ("You are LNSP's annotation assistant. Given a user query and a few top documents, "
       "produce JSON with: proposition (1 concise truth-statement answering the query), "
       "tmd {task, method, domain}, and cpe {concept, probe, expected}.\n"
       "Return ONLY valid JSON.")

PROMPT_TMPL = """Query: {query}

Top docs (id → snippet):
{docs}

Return JSON:
{{
  "proposition": "...",
  "tmd": {{"task":"RETRIEVE","method":"{method_hint}","domain":"FACTOIDWIKI"}},
  "cpe": {{"concept": "{concept_hint}", "probe": "{probe_hint}", "expected": {expected_ids} }}
}}"""


def annotate_with_llama(query: str,
                        top_docs: List[Dict[str, Any]],
                        method_hint: str = "HYBRID",
                        concept_hint: str = "",
                        expected_ids: List[str] = None) -> Dict[str, Any]:
    expected_ids = expected_ids or []
    docs_str = "\n".join(f"- {d.get('doc_id','?')}: {d.get('text','')[:300]}"
                         for d in top_docs if d)
    body = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": SYS},
            {"role": "user", "content": PROMPT_TMPL.format(
                query=query,
                docs=docs_str or "(no snippets available)",
                method_hint=method_hint,
                concept_hint=concept_hint,
                probe_hint=query,
                expected_ids=json.dumps(expected_ids),
            )}
        ],
        "stream": False
    }
    r = requests.post(OLLAMA_URL, json=body, timeout=60)
    r.raise_for_status()
    content = r.json().get("message", {}).get("content", "").strip()
    # be forgiving if the model wraps code fences
    if content.startswith("```"):
        content = content.strip("` \n")
        if "\n" in content:
            content = content.split("\n", 1)[1]
    return json.loads(content)