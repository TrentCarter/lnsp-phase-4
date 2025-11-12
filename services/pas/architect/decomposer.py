#!/usr/bin/env python3
"""
Task Decomposer - LLM-Powered PRD Decomposition

Uses Claude Sonnet 4.5 to decompose Prime Directives into lane job cards

Contract: Based on docs/contracts/ARCHITECT_SYSTEM_PROMPT.md
"""
import os
import json
from typing import Dict, List, Any, Optional
import httpx


class TaskDecomposer:
    """
    LLM-powered PRD decomposer for Architect

    Uses Claude Sonnet 4.5 (or fallback LLM) to analyze PRD and generate:
    - Executive summary
    - Lane allocations (Code, Models, Data, DevSecOps, Docs)
    - Dependency graph (Graphviz DOT)
    - Resource reservations
    - Acceptance gates per lane
    """

    def __init__(self):
        """Initialize decomposer with LLM configuration"""
        # In test mode, default to Ollama instead of Anthropic
        test_mode = os.getenv("LNSP_TEST_MODE", "0") == "1"
        default_provider = "ollama" if test_mode else "anthropic"

        self.llm_provider = os.getenv("ARCHITECT_LLM_PROVIDER", default_provider)
        self.llm_model = os.getenv("ARCHITECT_LLM", "claude-sonnet-4-5-20250929")
        self.llm_endpoint = self._get_llm_endpoint()

    def _get_llm_endpoint(self) -> str:
        """Get LLM endpoint based on provider"""
        if self.llm_provider == "anthropic":
            return "https://api.anthropic.com/v1/messages"
        elif self.llm_provider == "ollama":
            return os.getenv("LNSP_LLM_ENDPOINT", "http://localhost:11434") + "/api/chat"
        elif self.llm_provider == "openai":
            return "https://api.openai.com/v1/chat/completions"
        else:
            raise ValueError(f"Unknown LLM provider: {self.llm_provider}")

    async def decompose(
        self,
        prd: str,
        title: str,
        entry_files: List[str],
        budget: Dict[str, Any],
        policy: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Decompose PRD using LLM

        Args:
            prd: Full PRD text
            title: User-provided title
            entry_files: Entry files specified by user
            budget: Budget constraints (tokens_max, duration_max_mins, cost_usd_max)
            policy: Policy settings (require_cross_vendor_review, protected_paths, etc.)

        Returns:
            Dict with:
                executive_summary: str
                lane_allocations: Dict[lane_name, allocation_dict]
                dependency_graph: str (Graphviz DOT)
                resource_reservations: Dict
                acceptance_gates: Dict[lane_name, List[gate_str]]
        """
        # Build prompt for LLM
        prompt = self._build_decomposition_prompt(prd, title, entry_files, budget, policy)

        # Call LLM
        response = await self._call_llm(prompt)

        # Parse response
        result = self._parse_llm_response(response)

        return result

    def _build_decomposition_prompt(
        self,
        prd: str,
        title: str,
        entry_files: List[str],
        budget: Dict[str, Any],
        policy: Dict[str, Any]
    ) -> str:
        """Build decomposition prompt for LLM"""
        return f"""You are the Architect in a Polyglot Agent Swarm (PAS). Your role is to decompose Prime Directives into lane-specific job cards.

# Prime Directive

**Title:** {title}

**PRD:**
{prd}

**Entry Files:**
{json.dumps(entry_files, indent=2)}

**Budget:**
{json.dumps(budget, indent=2)}

**Policy:**
{json.dumps(policy, indent=2)}

# Your Task

Analyze this Prime Directive and decompose it into lane-specific job cards. You have 5 lanes available:

1. **Code** - Implementation, testing, reviews, builds
2. **Models** - Training, evaluation, model lifecycle
3. **Data** - Ingestion, validation, schema management
4. **DevSecOps** - CI/CD, SBOM, scans, deploys
5. **Docs** - Documentation, reviews, completeness

For each relevant lane, create a job card with:
- Task description (what needs to be done)
- Inputs (files/data needed)
- Expected artifacts (outputs)
- Acceptance criteria (tests, checks, gates)
- Risks (potential blockers)
- Budget allocation (tokens, time)

# Output Format

Respond with a JSON object in this EXACT format:

```json
{{
  "executive_summary": "1-2 sentence summary of what needs to be accomplished",
  "lane_allocations": {{
    "Code": {{
      "task": "Detailed task description for Code lane",
      "inputs": [{{"path": "file/path"}}],
      "expected_artifacts": [{{"path": "artifacts/path"}}],
      "acceptance": [{{"check": "pytest>=0.90"}}, {{"check": "lint==0"}}, {{"check": "coverage>=0.85"}}],
      "risks": ["Risk 1", "Risk 2"],
      "budget": {{"tokens_target_ratio": 0.50, "tokens_hard_ratio": 0.75}},
      "metadata": {{}}
    }},
    "Docs": {{
      "task": "Documentation task",
      "inputs": [],
      "expected_artifacts": [],
      "acceptance": [{{"check": "completeness"}}],
      "risks": [],
      "budget": {{"tokens_target_ratio": 0.30}},
      "metadata": {{}}
    }}
  }},
  "dependency_graph": "digraph {{ PASRoot -> Architect; Architect -> DirCode; Architect -> DirDocs; DirCode -> DirDevSecOps; DirDocs -> DirDevSecOps; DirDevSecOps -> Architect; }}",
  "resource_reservations": {{
    "cpu_cores": 4,
    "ram_gb": 16,
    "disk_gb": 10,
    "tokens_max": 50000
  }},
  "acceptance_gates": {{
    "Code": ["pytest>=0.90", "lint==0", "coverage>=0.85"],
    "Docs": ["completeness", "review_pass"]
  }}
}}
```

# Guidelines

- Only include lanes that are ACTUALLY needed for this task
- If it's just code changes → Code lane only (maybe Docs)
- If it involves training → Code + Models + Data lanes
- DevSecOps is usually added at the end for deployment
- Docs is optional unless docs updates explicitly requested

- Acceptance criteria must be measurable
- Budget tokens_target_ratio should sum to ≤ 1.0 across all lanes
- Dependency graph in Graphviz DOT format

Respond ONLY with the JSON object, no additional text.
"""

    async def _call_llm(self, prompt: str) -> str:
        """Call LLM with prompt"""
        if self.llm_provider == "anthropic":
            return await self._call_anthropic(prompt)
        elif self.llm_provider == "ollama":
            return await self._call_ollama(prompt)
        else:
            raise ValueError(f"LLM provider {self.llm_provider} not implemented")

    async def _call_anthropic(self, prompt: str) -> str:
        """Call Anthropic Claude API"""
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not set")

        headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        }

        payload = {
            "model": self.llm_model,
            "max_tokens": 4096,
            "messages": [
                {"role": "user", "content": prompt}
            ]
        }

        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(self.llm_endpoint, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()
            return data["content"][0]["text"]

    async def _call_ollama(self, prompt: str) -> str:
        """Call Ollama local LLM"""
        payload = {
            "model": os.getenv("LNSP_LLM_MODEL", "llama3.1:8b"),
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "stream": False
        }

        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(self.llm_endpoint, json=payload)
            response.raise_for_status()
            data = response.json()
            return data["message"]["content"]

    def _parse_llm_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM JSON response"""
        # Extract JSON from response (handle markdown code blocks)
        response = response.strip()

        if response.startswith("```json"):
            response = response[7:]  # Remove ```json
        if response.startswith("```"):
            response = response[3:]  # Remove ```
        if response.endswith("```"):
            response = response[:-3]  # Remove trailing ```

        response = response.strip()

        # Parse JSON
        try:
            result = json.loads(response)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse LLM response as JSON: {e}\nResponse: {response}")

        # Validate required fields
        required_fields = ["executive_summary", "lane_allocations", "dependency_graph", "resource_reservations", "acceptance_gates"]
        for field in required_fields:
            if field not in result:
                raise ValueError(f"Missing required field in LLM response: {field}")

        return result
