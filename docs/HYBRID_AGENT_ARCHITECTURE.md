# Hybrid Agent Architecture — Three-Tier Implementation Guide

**Date:** 2025-11-06
**Status:** Design Complete
**Related PRDs:** PRD_Polyglot_Agent_Swarm.md, PRD_Human_Machine_Interface_HMI.md

---

## 1. Overview

This architecture unifies three types of agents into a single orchestrated swarm:

1. **Claude Code Sub-Agents** (Tier 1) — Tool-heavy, free, context-aware
2. **Local LLM Services** (Tier 2) — FastAPI wrappers, no API cost, fast inference
3. **External LLM APIs** (Tier 3) — Vendor APIs, high quality, cost per use

**Key Principle:** All agents register as services using a **unified contract**, enabling dynamic routing based on capability, cost, and SLA requirements.

---

## 2. Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                        ORCHESTRATION LAYER                          │
│                                                                      │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐         │
│  │  Architect   │───▶│  Directors   │───▶│  Managers    │          │
│  │  (Claude)    │    │  (Claude)    │    │  (Claude)    │          │
│  └──────────────┘    └──────────────┘    └──────────────┘         │
│         │                    │                    │                 │
│         └────────────────────┼────────────────────┘                 │
│                              ▼                                      │
├─────────────────────────────────────────────────────────────────────┤
│                         GATEWAY & REGISTRY                          │
│                                                                      │
│  ┌───────────────────────────────────────────────────────────────┐ │
│  │  Gateway (6120): Single entrypoint, routing, receipts        │ │
│  │  Registry (6121): Service discovery, heartbeats, TTL         │ │
│  │  Router (6103): Capability matching, cost optimization       │ │
│  └───────────────────────────────────────────────────────────────┘ │
│                              ▼                                      │
├─────────────────────────────────────────────────────────────────────┤
│                        EXECUTION LAYER                              │
│                                                                      │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐ │
│  │   TIER 1         │  │   TIER 2         │  │   TIER 3         │ │
│  │   Claude Agents  │  │   Local LLMs     │  │   External APIs  │ │
│  ├──────────────────┤  ├──────────────────┤  ├──────────────────┤ │
│  │ Corpus Auditor   │  │ llama-3.1-8b     │  │ OpenAI GPT-5     │ │
│  │ Graph Builder    │  │ @ :8050          │  │ (API)            │ │
│  │ Evaluator        │  │                  │  │                  │ │
│  │ Release Coord    │  │ TinyLlama-1.1B   │  │ Anthropic Claude │ │
│  │ Report Writer    │  │ @ :8051          │  │ (API)            │ │
│  │                  │  │                  │  │                  │ │
│  │ (19 agents)      │  │ TLC Classifier   │  │ Google Gemini    │ │
│  │                  │  │ @ :8052          │  │ (API)            │ │
│  │                  │  │                  │  │                  │ │
│  │                  │  │                  │  │ xAI Grok         │ │
│  │                  │  │                  │  │ (API)            │ │
│  └──────────────────┘  └──────────────────┘  └──────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 3. Tier Specifications

### Tier 1: Claude Code Sub-Agents (.claude/agents/)

**Format:** Markdown with YAML frontmatter

**Example:** `.claude/agents/corpus-auditor.md`

```markdown
---
name: corpus-auditor
description: Source checks, licensing validation, and dataset statistics
tools: Read, Glob, Grep, Bash, Write
model: inherit
---

You are the Corpus Auditor agent responsible for validating data sources.

**Primary tasks:**
1. Check file licensing and attribution
2. Validate encoding and format consistency
3. Generate dataset statistics reports
4. Flag PII or sensitive content

**Constraints:**
- Token budget: 0.30/0.50 (target/hard)
- Must emit heartbeat every 60s
- Write reports to artifacts/corpus_reports/
- Request approval before any deletions
```

**Invocation:**
```python
# From Claude (me) or parent agent
task_result = Task(
    subagent_type="corpus-auditor",
    description="Audit Wikipedia batch 7",
    prompt="Audit the dataset at data/wikipedia_batch_7.jsonl for licensing, encoding issues, and generate stats report."
)
```

**Strengths:**
- ✅ Full tool access (Read, Write, Bash, Git, Grep, Glob, etc.)
- ✅ Free (no API cost)
- ✅ Context-aware of codebase
- ✅ Fast local execution

**Best for:**
- File operations (read, write, search, glob)
- Git operations (commit, PR, tag)
- Code generation and refactoring
- Planning and orchestration
- Report writing and documentation

---

### Tier 2: Local LLM Services (FastAPI)

**Format:** FastAPI service that registers with Registry

**Example:** `services/local_llm/llama_service.py`

```python
#!/usr/bin/env python3
"""
Local LLM Service — llama-3.1-8b wrapper with Registry integration
Port: 8050
"""
import httpx
import uuid
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
import asyncio

app = FastAPI()

# Service configuration
SERVICE_CONFIG = {
    "service_id": str(uuid.uuid4()),
    "name": "llama-3.1-8b",
    "type": "model",
    "role": "production",
    "url": "http://127.0.0.1:8050",
    "caps": ["infer", "classify"],
    "labels": {
        "space": "local",
        "domain": ["general"],
        "ctx": "32k"
    },
    "ctx_limit": 32768,
    "cost_hint": {"usd_per_1k": 0.00},
    "heartbeat_interval_s": 60,
    "ttl_s": 90
}

# Registry client
REGISTRY_URL = "http://localhost:6121"

class InferRequest(BaseModel):
    prompt: str
    max_tokens: int = 512
    temperature: float = 0.7

class InferResponse(BaseModel):
    text: str
    tokens_used: int
    model: str

@app.on_event("startup")
async def register_with_registry():
    """Register this service on startup"""
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                f"{REGISTRY_URL}/register",
                json=SERVICE_CONFIG,
                timeout=5.0
            )
            response.raise_for_status()
            print(f"✓ Registered with Registry: {response.json()}")
        except Exception as e:
            print(f"✗ Failed to register: {e}")

async def send_heartbeat():
    """Send heartbeat every 60s"""
    while True:
        await asyncio.sleep(60)
        async with httpx.AsyncClient() as client:
            try:
                await client.put(
                    f"{REGISTRY_URL}/heartbeat",
                    json={
                        "service_id": SERVICE_CONFIG["service_id"],
                        "status": "ok",
                        "p95_ms": 500,
                        "queue_depth": 0,
                        "load": 0.2
                    },
                    timeout=5.0
                )
            except Exception as e:
                print(f"✗ Heartbeat failed: {e}")

@app.on_event("startup")
async def start_heartbeat():
    """Start background heartbeat task"""
    asyncio.create_task(send_heartbeat())

@app.post("/infer", response_model=InferResponse)
async def infer(request: InferRequest):
    """Run inference via Ollama llama-3.1-8b"""
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": "llama3.1:8b",
                    "prompt": request.prompt,
                    "stream": False,
                    "options": {
                        "num_predict": request.max_tokens,
                        "temperature": request.temperature
                    }
                },
                timeout=60.0
            )
            response.raise_for_status()
            data = response.json()

            return InferResponse(
                text=data["response"],
                tokens_used=data.get("eval_count", 0),
                model="llama-3.1-8b"
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "ok", "model": "llama-3.1-8b", "ctx_limit": 32768}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8050)
```

**Invocation:**
```python
# Via Gateway
response = requests.post("http://localhost:6120/invoke", json={
    "target": {
        "name": "llama-3.1-8b",
        "type": "model",
        "role": "production"
    },
    "payload": {
        "prompt": "Classify this domain: 'quantum entanglement physics'",
        "max_tokens": 50
    }
})
```

**Strengths:**
- ✅ Zero API cost
- ✅ Fast inference (local GPU/CPU)
- ✅ Good for batch operations
- ✅ Privacy (no external calls)

**Best for:**
- Domain classification (L0/Lpath tagging)
- PII/license flagging
- Lightweight content extraction
- Batch processing (thousands of items)

---

### Tier 3: External LLM API Adapters

**Format:** FastAPI adapter service that wraps external APIs

**Example:** `services/external_llm/openai_adapter.py`

```python
#!/usr/bin/env python3
"""
External LLM Adapter — OpenAI GPT-5 wrapper with Registry integration
Port: 8100
"""
import os
import uuid
import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import asyncio
from openai import AsyncOpenAI

app = FastAPI()

# Load credentials from .env
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL_NAME", "gpt-5-codex")

client = AsyncOpenAI(api_key=OPENAI_API_KEY)

# Service configuration
SERVICE_CONFIG = {
    "service_id": str(uuid.uuid4()),
    "name": "OpenAI-GPT-5",
    "type": "model",
    "role": "production",
    "url": "http://127.0.0.1:8100",
    "caps": ["plan", "code_write", "reasoning"],
    "labels": {
        "vendor": "openai",
        "quality": "high"
    },
    "ctx_limit": 128000,
    "cost_hint": {"usd_per_1k": 0.03},  # Adjust based on actual pricing
    "heartbeat_interval_s": 60,
    "ttl_s": 90
}

REGISTRY_URL = "http://localhost:6121"

class InferRequest(BaseModel):
    messages: List[dict]
    max_tokens: Optional[int] = 1024
    temperature: Optional[float] = 0.7

class InferResponse(BaseModel):
    text: str
    tokens_used: dict
    model: str
    cost_estimate: float

@app.on_event("startup")
async def register_with_registry():
    """Register this service on startup"""
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                f"{REGISTRY_URL}/register",
                json=SERVICE_CONFIG,
                timeout=5.0
            )
            response.raise_for_status()
            print(f"✓ Registered OpenAI adapter: {response.json()}")
        except Exception as e:
            print(f"✗ Failed to register: {e}")

async def send_heartbeat():
    """Send heartbeat every 60s"""
    while True:
        await asyncio.sleep(60)
        async with httpx.AsyncClient() as client:
            try:
                await client.put(
                    f"{REGISTRY_URL}/heartbeat",
                    json={
                        "service_id": SERVICE_CONFIG["service_id"],
                        "status": "ok",
                        "p95_ms": 1200,
                        "queue_depth": 0
                    },
                    timeout=5.0
                )
            except Exception as e:
                print(f"✗ Heartbeat failed: {e}")

@app.on_event("startup")
async def start_heartbeat():
    asyncio.create_task(send_heartbeat())

@app.post("/infer", response_model=InferResponse)
async def infer(request: InferRequest):
    """Run inference via OpenAI API"""
    try:
        response = await client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=request.messages,
            max_tokens=request.max_tokens,
            temperature=request.temperature
        )

        # Calculate cost estimate
        prompt_tokens = response.usage.prompt_tokens
        completion_tokens = response.usage.completion_tokens
        cost_estimate = (prompt_tokens * 0.03 / 1000) + (completion_tokens * 0.06 / 1000)

        return InferResponse(
            text=response.choices[0].message.content,
            tokens_used={
                "prompt": prompt_tokens,
                "completion": completion_tokens,
                "total": response.usage.total_tokens
            },
            model=OPENAI_MODEL,
            cost_estimate=cost_estimate
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "ok",
        "model": OPENAI_MODEL,
        "vendor": "openai",
        "api_key_set": bool(OPENAI_API_KEY)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8100)
```

**Invocation:**
```python
# Via Gateway (automatic routing based on capability)
response = requests.post("http://localhost:6120/invoke", json={
    "target": {
        "type": "model",
        "role": "production",
        "require_caps": ["plan"]
    },
    "payload": {
        "messages": [
            {"role": "system", "content": "You are a planning expert."},
            {"role": "user", "content": "Plan the implementation of feature X."}
        ]
    },
    "policy": {
        "prefer": "cheapest_that_meets_sla",
        "sla_ms_p95": 2000
    }
})
```

**Strengths:**
- ✅ Highest quality reasoning
- ✅ Large context windows (128k+)
- ✅ Specialized capabilities (code, vision, etc.)
- ✅ Reliable availability

**Best for:**
- Complex planning and reasoning
- Code generation (high quality)
- Cross-vendor PR reviews
- Research and analysis tasks

---

## 4. Unified Service Contract

All services (Tier 1, 2, 3) register using the **same schema**:

**Service Registration Schema:**
```json
{
  "service_id": "uuid-4",
  "name": "llama-3.1-8b",
  "type": "model",
  "role": "production",
  "url": "http://127.0.0.1:8050",
  "caps": ["infer", "classify"],
  "labels": {
    "space": "local",
    "domain": ["general"]
  },
  "ctx_limit": 32768,
  "cost_hint": {"usd_per_1k": 0.00},
  "heartbeat_interval_s": 60,
  "ttl_s": 90
}
```

**Job Card Schema (all tiers):**
```json
{
  "job_id": "J-2025-11-06-001",
  "agent": "TLC-Domain-Classifier",
  "task": "classify_domain",
  "capability": "classify_light",
  "inputs": {
    "text": "quantum entanglement physics"
  },
  "outputs": {},
  "priority": 5,
  "created_at": "2025-11-06T10:30:00Z"
}
```

---

## 5. Routing Policy (Provider Matrix)

**File:** `config/providers.matrix.json`

```json
{
  "capabilities": {
    "plan": [
      {"provider": "openai", "model": "gpt-5-codex", "tier": 3},
      {"provider": "anthropic", "model": "claude-sonnet-4-5", "tier": 3},
      {"provider": "gemini", "model": "gemini-2.5-pro", "tier": 3},
      {"provider": "local", "model": "deepseek-r1:7b", "tier": 2}
    ],
    "code_write": [
      {"provider": "anthropic", "model": "claude-sonnet-4-5", "tier": 3},
      {"provider": "openai", "model": "gpt-5-codex", "tier": 3},
      {"provider": "claude-agent", "agent": "code-writer", "tier": 1}
    ],
    "classify_light": [
      {"provider": "local", "model": "llama-3.1-8b", "tier": 2},
      {"provider": "local", "model": "TinyLlama-1.1B", "tier": 2}
    ],
    "eval_review": [
      {"provider": "gemini", "model": "gemini-2.5-pro", "tier": 3},
      {"provider": "anthropic", "model": "claude-sonnet-4-5", "tier": 3}
    ],
    "tool_heavy": [
      {"provider": "claude-agent", "agent": "corpus-auditor", "tier": 1},
      {"provider": "claude-agent", "agent": "graph-builder", "tier": 1},
      {"provider": "claude-agent", "agent": "evaluator", "tier": 1}
    ]
  },
  "routing_policy": {
    "prefer": "cheapest_that_meets_sla",
    "sla_ms_p95": 1500,
    "fallback_order": ["tier1", "tier2", "tier3"]
  }
}
```

---

## 6. Implementation Checklist

### Phase 1: Core Infrastructure (2-3 days)
- [ ] Create Registry service (6121) with SQLite backend
- [ ] Create Gateway service (6120) with routing logic
- [ ] Create Router service (6103) with capability matching
- [ ] Implement heartbeat monitoring (6109)
- [ ] Create contract schemas (service_registration, job_card, etc.)

### Phase 2: Local LLM Services (1-2 days)
- [ ] Create llama-3.1-8b wrapper service (8050)
- [ ] Create TinyLlama wrapper service (8051)
- [ ] Create TLC Domain Classifier service (8052)
- [ ] Test registration and heartbeat flow
- [ ] Validate inference via Gateway

### Phase 3: External LLM Adapters (1-2 days)
- [ ] Create OpenAI adapter (8100)
- [ ] Create Anthropic adapter (8101)
- [ ] Create Gemini adapter (8102)
- [ ] Create Grok adapter (8103)
- [ ] Load credentials from .env
- [ ] Test routing and cost receipts

### Phase 4: Claude Sub-Agents (1 day)
- [ ] Create 23 Coordinator/System agents in .claude/agents/
- [ ] Create 19 Execution agents in .claude/agents/
- [ ] Test invocation via Task tool
- [ ] Validate tool access and permissions

### Phase 5: Integration & Testing (1-2 days)
- [ ] End-to-end test: Architect → Gateway → Local/External/Claude
- [ ] Validate routing policy (prefer local → fallback external)
- [ ] Test cross-tier communication
- [ ] Verify cost receipts and logging
- [ ] Run sample multi-agent workflow

---

## 7. Example Multi-Tier Workflow

**Scenario:** Process Wikipedia batch with domain classification and quality checks

```python
# Step 1: Architect (Claude Tier 1) plans the work
architect_plan = """
1. Audit corpus (Corpus Auditor - Claude Tier 1)
2. Classify domains for 10k entries (TLC - Local Tier 2)
3. Generate quality report (Report Writer - Claude Tier 1)
4. Cross-vendor review (Gemini - External Tier 3)
"""

# Step 2: Route to Corpus Auditor (Claude agent via Task tool)
audit_result = invoke_task(
    agent="corpus-auditor",
    task="audit_wikipedia_batch",
    inputs={"path": "data/wikipedia_batch_7.jsonl"}
)

# Step 3: Route to TLC Domain Classifier (Local LLM via Gateway)
classify_result = gateway_invoke(
    capability="classify_light",
    inputs={"texts": audit_result["entries"]},
    policy={"prefer": "local"}  # Forces Tier 2
)

# Step 4: Route to Report Writer (Claude agent via Task tool)
report_result = invoke_task(
    agent="report-writer",
    task="generate_quality_report",
    inputs={
        "audit": audit_result,
        "classifications": classify_result
    }
)

# Step 5: Route to Gemini for review (External API via Gateway)
review_result = gateway_invoke(
    capability="eval_review",
    inputs={"report": report_result["markdown"]},
    policy={"prefer": "gemini"}  # Forces Tier 3, vendor=gemini
)
```

**Cost breakdown:**
- Architect planning: $0 (Claude Tier 1)
- Corpus audit: $0 (Claude Tier 1)
- Domain classification: $0 (Local Tier 2, 10k items)
- Report generation: $0 (Claude Tier 1)
- Gemini review: ~$0.05 (External Tier 3)

**Total: ~$0.05 for entire workflow**

---

## 8. Benefits of Hybrid Architecture

✅ **Cost Optimization:**
- Tier 1 (Claude): Free for 90% of tool-heavy work
- Tier 2 (Local): Zero API cost for classification/batch work
- Tier 3 (External): Only when needed for high-quality reasoning

✅ **Performance:**
- Claude agents: Instant (local execution)
- Local LLMs: Fast (GPU/CPU inference, <500ms)
- External APIs: Variable (1-3s typical)

✅ **Reliability:**
- Tier 1 always available (Claude)
- Tier 2 always available (local services)
- Tier 3 fallback when Tier 1/2 can't handle task

✅ **Privacy:**
- Sensitive operations stay on Tier 1/2 (no external calls)
- Only reviewed/sanitized data goes to Tier 3

✅ **Flexibility:**
- Add new agents to any tier without changing architecture
- Easy A/B testing (canary role + Registry)
- Cross-vendor reviews (enforce different vendors per role)

---

## 9. Next Steps

1. **Review this architecture** with stakeholders
2. **Create Registry + Gateway services** (Phase 1)
3. **Wrap existing local LLMs** (Phase 2)
4. **Create external adapters** (Phase 3)
5. **Generate Claude sub-agents** (Phase 4)
6. **Integration testing** (Phase 5)
7. **Deploy HMI** (Web UI @ 6101 for monitoring)

---

## 10. Open Questions

- [ ] Should Registry use SQLite or PostgreSQL for persistence?
- [ ] Default timeout values for each tier?
- [ ] How to handle quota breaches (rate limits on external APIs)?
- [ ] Should we cache external API responses (semantic caching)?
- [ ] TTS backend for HMI narration (local vs vendor)?

---

**END OF DOCUMENT**
