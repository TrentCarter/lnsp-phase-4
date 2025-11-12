#!/usr/bin/env python3
"""
Manager Task Decomposer - LLM-Powered Job Card Decomposition

Uses Gemini 2.5 Flash (or fallback) to decompose Docs lane job cards into Manager tasks

Contract: Based on docs/contracts/DIRECTOR_DOCS_SYSTEM_PROMPT.md
"""
import os
import json
from typing import Dict, List, Any, Optional
import httpx


class ManagerTaskDecomposer:
    """
    LLM-powered job card decomposer for Director-Code

    Uses Gemini 2.5 Flash (or fallback LLM) to analyze job cards and generate:
    - Manager-level documentation task breakdown
    - Test requirements
    - Acceptance gates per Manager
    - Resource allocations
    - Dependencies
    """

    def __init__(self):
        """Initialize decomposer with LLM configuration"""
        self.llm_provider = os.getenv("DIR_DOCS_LLM_PROVIDER", "anthropic")
        self.llm_model = os.getenv("DIR_DOCS_LLM", "claude-sonnet-4-5-20250929")
        self.llm_endpoint = self._get_llm_endpoint()

    def _get_llm_endpoint(self) -> str:
        """Get LLM endpoint based on provider"""
        if self.llm_provider == "anthropic":
            return "https://generativelanguage.anthropicapis.com/v1beta/models/" + self.llm_model + ":generateContent"
        elif self.llm_provider == "anthropic":
            return "https://api.anthropic.com/v1/messages"
        elif self.llm_provider == "ollama":
            return os.getenv("LNSP_LLM_ENDPOINT", "http://localhost:11434") + "/api/chat"
        else:
            raise ValueError(f"Unknown LLM provider: {self.llm_provider}")

    async def decompose(
        self,
        job_card_id: str,
        task: str,
        inputs: List[Dict[str, Any]],
        expected_artifacts: List[Dict[str, Any]],
        acceptance: List[Dict[str, Any]],
        risks: List[str],
        budget: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Decompose job card into Manager tasks using LLM

        Args:
            job_card_id: Parent job card ID
            task: Task description from Architect
            inputs: Input files/data
            expected_artifacts: Expected output artifacts
            acceptance: Acceptance criteria
            risks: Known risks
            budget: Budget constraints

        Returns:
            Dict with:
                manager_tasks: List[Dict] - One dict per Manager
                    - manager_id: str (e.g., "Mgr-Code-01")
                    - task: str
                    - files: List[str] (target files)
                    - inputs: List[Dict]
                    - expected_artifacts: List[Dict]
                    - acceptance: List[Dict]
                    - budget: Dict
                    - programmers: List[str] (LLM models to use)
                dependencies: List[Dict] - Manager dependencies
                    - from: str (manager_id)
                    - to: str (manager_id)
                    - reason: str
        """
        # Build prompt for LLM
        prompt = self._build_decomposition_prompt(
            job_card_id, task, inputs, expected_artifacts,
            acceptance, risks, budget
        )

        # Call LLM
        response = await self._call_llm(prompt)

        # Parse response
        result = self._parse_llm_response(response)

        return result

    def _build_decomposition_prompt(
        self,
        job_card_id: str,
        task: str,
        inputs: List[Dict[str, Any]],
        expected_artifacts: List[Dict[str, Any]],
        acceptance: List[Dict[str, Any]],
        risks: List[str],
        budget: Dict[str, Any]
    ) -> str:
        """Build decomposition prompt for LLM"""
        return f"""You are Dir-Docs, the Director of the Docs lane in a Polyglot Agent Swarm (PAS).

Your role is to decompose Docs lane job cards into Manager-level documentation tasks. Each Manager will coordinate 1-5 Programmers to complete their task.

# Job Card from Architect

**Job Card ID:** {job_card_id}

**Task:**
{task}

**Inputs:**
{json.dumps(inputs, indent=2)}

**Expected Artifacts:**
{json.dumps(expected_artifacts, indent=2)}

**Acceptance Criteria:**
{json.dumps(acceptance, indent=2)}

**Risks:**
{json.dumps(risks, indent=2)}

**Budget:**
{json.dumps(budget, indent=2)}

# Your Task

Decompose this job card into Manager-level documentation tasks. Follow these rules:

1. **One Manager per module/subsystem**
   - Example: Mgr-Code-01 for implementation, Mgr-Code-02 for tests
   - Each Manager should focus on 1-3 related files

2. **Surgical task breakdown**
   - Each Manager gets a CLEAR, FOCUSED task
   - "Implement OAuth2 login in app/services/auth.py" (good)
   - "Fix all the code" (bad)

3. **Test coverage**
   - If implementation changes → Need test Manager
   - If refactoring → Need test validation Manager
   - Test Managers depend on implementation Managers

4. **Protected paths** (require cross-vendor review)
   - app/ (core runtime)
   - contracts/ (schemas)
   - scripts/ (automation)
   - docs/PRDs/ (specs)

5. **Budget allocation**
   - tokens_target_ratio should sum to ≤ 1.0 across all Managers
   - Implementation typically gets 40-50%
   - Tests get 30-40%
   - Reviews/lint get 10-20%

# Output Format

Respond with a JSON object in this EXACT format:

```json
{{
  "manager_tasks": [
    {{
      "manager_id": "Mgr-Code-01",
      "task": "Implement OAuth2 login/logout/refresh in app/services/auth.py",
      "files": ["app/services/auth.py"],
      "inputs": [{{"path": "docs/PRDs/PRD_OAuth2.md"}}, {{"path": "app/services/auth.py"}}],
      "expected_artifacts": [
        {{"path": "artifacts/runs/{{RUN_ID}}/code/diffs/auth.py.diff"}},
        {{"path": "artifacts/runs/{{RUN_ID}}/code/test_results.json"}}
      ],
      "acceptance": [
        {{"check": "ruff app/services/auth.py==0"}},
        {{"check": "mypy app/services/auth.py==0"}}
      ],
      "budget": {{
        "tokens_target_ratio": 0.50,
        "tokens_hard_ratio": 0.75
      }},
      "programmers": ["Prog-Qwen-001", "Prog-Claude-001"],
      "requires_review": true
    }},
    {{
      "manager_id": "Mgr-Code-02",
      "task": "Write comprehensive tests for OAuth2 flow in tests/test_auth.py",
      "files": ["tests/test_auth.py"],
      "inputs": [{{"path": "app/services/auth.py"}}],
      "expected_artifacts": [
        {{"path": "artifacts/runs/{{RUN_ID}}/code/diffs/test_auth.py.diff"}},
        {{"path": "artifacts/runs/{{RUN_ID}}/code/test_results.json"}},
        {{"path": "artifacts/runs/{{RUN_ID}}/code/coverage.json"}}
      ],
      "acceptance": [
        {{"check": "pytest tests/test_auth.py>=0.95"}},
        {{"check": "coverage app/services/auth.py>=0.90"}},
        {{"check": "ruff tests/test_auth.py==0"}}
      ],
      "budget": {{
        "tokens_target_ratio": 0.40,
        "tokens_hard_ratio": 0.60
      }},
      "programmers": ["Prog-Qwen-002"],
      "requires_review": false
    }}
  ],
  "dependencies": [
    {{
      "from": "Mgr-Code-02",
      "to": "Mgr-Code-01",
      "reason": "Tests require implementation to be complete"
    }}
  ]
}}
```

# Guidelines

- Typically 2-4 Managers per job card (don't over-split)
- Implementation + Tests is the most common pattern
- Add review Manager if protected paths touched
- Dependencies: Tests depend on implementation, reviews depend on tests
- Programmer LLMs: Qwen 2.5 Coder 7B for primary, Claude Haiku for fallback
- requires_review: true for protected paths (app/, contracts/, scripts/, docs/PRDs/)

Respond ONLY with the JSON object, no additional text.
"""

    async def _call_llm(self, prompt: str) -> str:
        """Call LLM with prompt"""
        if self.llm_provider == "anthropic":
            return await self._call_anthropic(prompt)
        elif self.llm_provider == "anthropic":
            return await self._call_anthropic(prompt)
        elif self.llm_provider == "ollama":
            return await self._call_ollama(prompt)
        else:
            raise ValueError(f"LLM provider {self.llm_provider} not implemented")

    async def _call_anthropic(self, prompt: str) -> str:
        """Call Google Gemini API"""
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not set")

        url = f"{self.llm_endpoint}?key={api_key}"

        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": prompt}
                    ]
                }
            ],
            "generationConfig": {
                "temperature": 0.7,
                "maxOutputTokens": 4096
            }
        }

        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(url, json=payload)
            response.raise_for_status()
            data = response.json()
            return data["candidates"][0]["content"]["parts"][0]["text"]

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
            "model": "claude-sonnet-4-5-20250929",
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
        required_fields = ["manager_tasks", "dependencies"]
        for field in required_fields:
            if field not in result:
                raise ValueError(f"Missing required field in LLM response: {field}")

        # Validate manager_tasks structure
        for task in result["manager_tasks"]:
            required_task_fields = ["manager_id", "task", "files", "acceptance", "budget", "programmers"]
            for field in required_task_fields:
                if field not in task:
                    raise ValueError(f"Missing required field in manager task: {field}")

        return result
