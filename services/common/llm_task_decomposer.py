#!/usr/bin/env python3
"""
LLM-Powered Task Decomposition Service

Provides intelligent task decomposition for Managers to break down
Director-level job cards into surgical Programmer tasks.

Uses local LLM (Ollama) to analyze the task and generate optimal
decomposition based on:
- File dependencies
- Task complexity
- Operation types (create, modify, delete, refactor)
- Parallel execution opportunities
"""
import os
import json
import requests
from typing import List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class ProgrammerTask:
    """Surgical, atomic task for a Programmer"""
    task: str
    files: List[str]
    operation: str  # 'create', 'modify', 'delete', 'refactor'
    context: List[str]  # Related files for context
    acceptance: List[Dict[str, Any]]
    budget: Dict[str, Any]
    timeout_s: int = 300


class LLMTaskDecomposer:
    """LLM-powered task decomposition service"""

    def __init__(self):
        self.backend = os.getenv("LNSP_LLM_BACKEND", "ollama").lower()
        self.model = os.getenv("LNSP_LLM_MODEL", "llama3.1:8b")
        self.ollama_host = os.getenv("LNSP_LLM_ENDPOINT", "http://localhost:11434")
        self.ollama_chat = f"{self.ollama_host.rstrip('/')}/api/chat"
        self.enabled = os.getenv("LNSP_LLM_DECOMPOSITION", "true").lower() == "true"

    def decompose(
        self,
        job_card: Dict[str, Any],
        max_tasks: int = 5,
        fallback: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Decompose job card into Programmer tasks

        Args:
            job_card: Director job card with task, inputs, acceptance criteria
            max_tasks: Maximum number of Programmer tasks to generate
            fallback: Use simple 1:1 decomposition if LLM fails

        Returns:
            List of Programmer task dictionaries
        """
        if not self.enabled:
            return self._simple_decomposition(job_card)

        try:
            return self._llm_decomposition(job_card, max_tasks)
        except Exception as e:
            print(f"⚠️  LLM decomposition failed: {e}")
            if fallback:
                print("   Falling back to simple decomposition")
                return self._simple_decomposition(job_card)
            raise

    def _llm_decomposition(
        self,
        job_card: Dict[str, Any],
        max_tasks: int
    ) -> List[Dict[str, Any]]:
        """Use LLM to intelligently decompose task"""

        task = job_card.get("task", "")
        files = job_card.get("inputs", [])
        file_paths = [f["path"] for f in files if f.get("type") == "file"]
        acceptance = job_card.get("acceptance", [])
        budget = job_card.get("budget", {})

        # Build LLM prompt
        system_prompt = """You are a task decomposition expert for a Multi-Tier Programmer AI System.

Your job is to break down high-level coding tasks into surgical, atomic subtasks that can be executed in parallel by junior programmer agents.

Guidelines:
1. Each subtask should be independent and atomic (single file or single operation)
2. Identify dependencies between subtasks
3. Prefer parallel execution where possible
4. Each subtask should take < 5 minutes
5. Use operations: 'create', 'modify', 'delete', 'refactor'
6. Provide context files that each subtask may need to read

Return ONLY valid JSON array (no prose, no code fences):
[
  {
    "task": "Specific task description",
    "files": ["file/to/modify.py"],
    "operation": "modify|create|delete|refactor",
    "context": ["related/file1.py", "related/file2.py"],
    "priority": 1,
    "dependencies": []
  }
]"""

        user_prompt = f"""Task: {task}

Files involved: {', '.join(file_paths) if file_paths else 'No specific files (create new)'}

Acceptance criteria:
{json.dumps(acceptance, indent=2) if acceptance else 'None specified'}

Break this down into {max_tasks} or fewer surgical subtasks that can be executed by junior programmers.
Focus on atomic operations that can run in parallel where possible."""

        # Call LLM
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "stream": False,
            "options": {"temperature": 0.3}  # Slightly creative but mostly deterministic
        }

        response = requests.post(
            self.ollama_chat,
            json=body,
            timeout=30
        )
        response.raise_for_status()

        content = response.json().get("message", {}).get("content", "")
        content = self._strip_fences(content)

        # Parse LLM response
        try:
            tasks = json.loads(content)
        except json.JSONDecodeError as e:
            print(f"⚠️  Failed to parse LLM response: {e}")
            print(f"   Response: {content[:200]}")
            raise

        # Convert to standard format
        programmer_tasks = []
        for task_def in tasks[:max_tasks]:
            programmer_tasks.append({
                "task": task_def.get("task", ""),
                "files": task_def.get("files", []),
                "operation": task_def.get("operation", "modify"),
                "context": task_def.get("context", file_paths),
                "acceptance": acceptance,
                "budget": budget,
                "timeout_s": 300,
                "priority": task_def.get("priority", 1),
                "dependencies": task_def.get("dependencies", [])
            })

        # If LLM returned empty list, fall back
        if not programmer_tasks:
            raise ValueError("LLM returned empty task list")

        return programmer_tasks

    def _simple_decomposition(self, job_card: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Simple 1:1 file decomposition (fallback)

        Creates one Programmer task per file, or a single generic task if no files.
        """
        task = job_card.get("task", "")
        files = job_card.get("inputs", [])
        file_paths = [f["path"] for f in files if f.get("type") == "file"]
        acceptance = job_card.get("acceptance", [])
        budget = job_card.get("budget", {})

        programmer_tasks = []
        for file_path in file_paths:
            programmer_tasks.append({
                "task": f"{task} in {file_path}",
                "files": [file_path],
                "operation": "modify",
                "context": file_paths,
                "acceptance": acceptance,
                "budget": budget,
                "timeout_s": 300
            })

        # If no files, create a single generic task
        if not programmer_tasks:
            programmer_tasks.append({
                "task": task,
                "files": [],
                "operation": "create",
                "context": [],
                "acceptance": acceptance,
                "budget": budget,
                "timeout_s": 300
            })

        return programmer_tasks

    def _strip_fences(self, s: str) -> str:
        """Remove markdown code fences from LLM output"""
        s = s.strip()
        if s.startswith("```"):
            s = s.strip("` \n")
            if "\n" in s:
                s = s.split("\n", 1)[1]
        if s.endswith("```"):
            s = s[:-3].strip()
        return s


# Singleton instance
_decomposer_instance: Optional[LLMTaskDecomposer] = None


def get_task_decomposer() -> LLMTaskDecomposer:
    """Get singleton task decomposer instance"""
    global _decomposer_instance
    if _decomposer_instance is None:
        _decomposer_instance = LLMTaskDecomposer()
    return _decomposer_instance


# Convenience function for direct use
def decompose_task(
    job_card: Dict[str, Any],
    max_tasks: int = 5,
    fallback: bool = True
) -> List[Dict[str, Any]]:
    """
    Decompose job card into Programmer tasks (convenience function)

    Args:
        job_card: Director job card
        max_tasks: Maximum number of subtasks
        fallback: Use simple decomposition if LLM fails

    Returns:
        List of Programmer task dictionaries
    """
    return get_task_decomposer().decompose(job_card, max_tasks, fallback)
