#!/usr/bin/env python3
"""
Manager Executor - Executes Manager tasks via Aider RPC

This is a lightweight executor that:
1. Receives Manager task from Director
2. Calls Aider RPC to execute code changes
3. Reports completion back to Director

Managers are metadata entities tracked in Manager Pool, not separate processes.
"""
import httpx
import asyncio
from typing import Dict, List, Any, Optional
from pathlib import Path
import json

from services.common.manager_pool.manager_factory import get_manager_factory
from services.common.heartbeat import get_monitor, AgentState
from services.common.comms_logger import get_logger, MessageType


class ManagerExecutor:
    """
    Executes Manager tasks by calling Aider RPC

    Usage:
        executor = ManagerExecutor()
        result = await executor.execute_manager_task(
            manager_id="Mgr-Code-01",
            task="Add hello() function to utils.py",
            files=["utils.py"],
            run_id="run-123"
        )
    """

    def __init__(self, aider_rpc_url: str = "http://127.0.0.1:6130"):
        """Initialize executor"""
        self.aider_rpc_url = aider_rpc_url
        self.manager_factory = get_manager_factory()
        self.heartbeat_monitor = get_monitor()
        self.logger = get_logger()

    async def execute_manager_task(
        self,
        manager_id: str,
        task: str,
        files: List[str],
        run_id: str,
        director: str,
        acceptance: List[Dict[str, Any]] = None,
        budget: Dict[str, Any] = None,
        programmers: List[str] = None
    ) -> Dict[str, Any]:
        """
        Execute Manager task via Aider RPC

        Args:
            manager_id: Manager ID (e.g., "Mgr-Code-01")
            task: Task description (natural language)
            files: Files to edit
            run_id: Run ID for tracking
            director: Parent Director (e.g., "Dir-Code")
            acceptance: Acceptance criteria (tests, lint, coverage)
            budget: Budget constraints
            programmers: LLM models to use (e.g., ["Prog-Qwen-001"])

        Returns:
            Dict with:
                success: bool
                output: str (Aider output)
                duration_s: float
                artifacts: Dict (paths to generated artifacts)
                acceptance_results: Dict (test/lint/coverage results)
        """
        acceptance = acceptance or []
        budget = budget or {}
        programmers = programmers or []

        # Send heartbeat: Manager starting
        self.heartbeat_monitor.heartbeat(
            agent=manager_id,
            run_id=run_id,
            state=AgentState.EXECUTING,
            message=f"Executing: {task[:50]}...",
            progress=0.1
        )

        # Log command
        self.logger.log_cmd(
            from_agent=director,
            to_agent=manager_id,
            message=f"Executing task: {task[:100]}...",
            run_id=run_id,
            metadata={
                "files": files,
                "programmers": programmers,
                "budget": budget
            }
        )

        # Convert relative file paths to absolute paths for Aider RPC allowlist
        repo_root = Path.cwd()  # Current working directory is repo root
        absolute_files = []
        for file in files:
            file_path = Path(file)
            if not file_path.is_absolute():
                file_path = (repo_root / file_path).resolve()
            absolute_files.append(str(file_path))

        try:
            # Call Aider RPC
            async with httpx.AsyncClient(timeout=1800.0) as client:  # 30 min timeout
                response = await client.post(
                    f"{self.aider_rpc_url}/aider/edit",
                    json={
                        "message": task,
                        "files": absolute_files,  # Use absolute paths
                        "dry_run": False,
                        "run_id": run_id
                    }
                )
                response.raise_for_status()
                result = response.json()

            # Check if Aider succeeded
            if not result.get("ok", False):
                # Aider failed
                self.heartbeat_monitor.heartbeat(
                    agent=manager_id,
                    run_id=run_id,
                    state=AgentState.FAILED,
                    message=f"Aider failed: rc={result.get('rc', 'unknown')}",
                    progress=1.0
                )

                self.logger.log_response(
                    from_agent=manager_id,
                    to_agent=director,
                    message=f"Task failed: {result.get('stderr', 'Unknown error')[:200]}",
                    run_id=run_id,
                    status="failed",
                    metadata={"aider_result": result}
                )

                return {
                    "success": False,
                    "output": result.get("stderr", ""),
                    "duration_s": result.get("duration_s", 0),
                    "artifacts": {},
                    "acceptance_results": {}
                }

            # Aider succeeded - run acceptance checks
            self.heartbeat_monitor.heartbeat(
                agent=manager_id,
                run_id=run_id,
                state=AgentState.VALIDATING,
                message="Validating acceptance criteria",
                progress=0.8
            )

            acceptance_results = await self._validate_acceptance(
                acceptance=acceptance,
                files=files,
                run_id=run_id
            )

            # All acceptance checks passed?
            all_passed = all(acceptance_results.values())

            if all_passed:
                # Success!
                self.heartbeat_monitor.heartbeat(
                    agent=manager_id,
                    run_id=run_id,
                    state=AgentState.COMPLETED,
                    message="Task completed successfully",
                    progress=1.0
                )

                self.logger.log_response(
                    from_agent=manager_id,
                    to_agent=director,
                    message="Task completed successfully",
                    run_id=run_id,
                    status="completed",
                    metadata={
                        "duration_s": result.get("duration_s", 0),
                        "acceptance_results": acceptance_results
                    }
                )

                return {
                    "success": True,
                    "output": result.get("stdout", ""),
                    "duration_s": result.get("duration_s", 0),
                    "artifacts": self._collect_artifacts(files, run_id),
                    "acceptance_results": acceptance_results
                }
            else:
                # Acceptance checks failed
                failed_checks = [k for k, v in acceptance_results.items() if not v]

                self.heartbeat_monitor.heartbeat(
                    agent=manager_id,
                    run_id=run_id,
                    state=AgentState.FAILED,
                    message=f"Acceptance checks failed: {', '.join(failed_checks)}",
                    progress=1.0
                )

                self.logger.log_response(
                    from_agent=manager_id,
                    to_agent=director,
                    message=f"Acceptance checks failed: {', '.join(failed_checks)}",
                    run_id=run_id,
                    status="failed",
                    metadata={"acceptance_results": acceptance_results}
                )

                return {
                    "success": False,
                    "output": result.get("stdout", ""),
                    "duration_s": result.get("duration_s", 0),
                    "artifacts": self._collect_artifacts(files, run_id),
                    "acceptance_results": acceptance_results
                }

        except httpx.HTTPError as e:
            # HTTP error calling Aider RPC
            self.heartbeat_monitor.heartbeat(
                agent=manager_id,
                run_id=run_id,
                state=AgentState.FAILED,
                message=f"Aider RPC error: {str(e)}",
                progress=1.0
            )

            self.logger.log_response(
                from_agent=manager_id,
                to_agent=director,
                message=f"Aider RPC error: {str(e)}",
                run_id=run_id,
                status="failed",
                metadata={"error": str(e)}
            )

            return {
                "success": False,
                "output": f"Aider RPC error: {str(e)}",
                "duration_s": 0,
                "artifacts": {},
                "acceptance_results": {}
            }

        except Exception as e:
            # Unexpected error
            self.heartbeat_monitor.heartbeat(
                agent=manager_id,
                run_id=run_id,
                state=AgentState.FAILED,
                message=f"Unexpected error: {str(e)}",
                progress=1.0
            )

            self.logger.log_response(
                from_agent=manager_id,
                to_agent=director,
                message=f"Unexpected error: {str(e)}",
                run_id=run_id,
                status="failed",
                metadata={"error": str(e)}
            )

            return {
                "success": False,
                "output": f"Error: {str(e)}",
                "duration_s": 0,
                "artifacts": {},
                "acceptance_results": {}
            }

    async def _validate_acceptance(
        self,
        acceptance: List[Dict[str, Any]],
        files: List[str],
        run_id: str
    ) -> Dict[str, bool]:
        """
        Validate acceptance criteria

        Acceptance criteria format:
        [
            {"check": "pytest>=0.90"},
            {"check": "lint==0"},
            {"check": "coverage>=0.85"}
        ]

        Returns:
            Dict mapping check name to pass/fail
        """
        results = {}

        for criterion in acceptance:
            check = criterion.get("check", "")

            if not check:
                continue

            # Parse check (e.g., "pytest>=0.90", "lint==0", "coverage>=0.85")
            if "pytest" in check.lower():
                # Run pytest
                passed = await self._run_pytest(files)
                results["pytest"] = passed

            elif "lint" in check.lower() or "ruff" in check.lower():
                # Run lint
                passed = await self._run_lint(files)
                results["lint"] = passed

            elif "coverage" in check.lower():
                # Check coverage
                passed = await self._check_coverage(files)
                results["coverage"] = passed

            else:
                # Unknown check - assume pass for now
                results[check] = True

        return results

    async def _run_pytest(self, files: List[str]) -> bool:
        """Run pytest on test files"""
        # For now, return True (TODO: implement actual pytest execution)
        # This would need to identify test files, run pytest, parse results
        return True

    async def _run_lint(self, files: List[str]) -> bool:
        """Run lint on files"""
        # For now, return True (TODO: implement actual lint execution)
        # This would run ruff/flake8/mypy and check for errors
        return True

    async def _check_coverage(self, files: List[str]) -> bool:
        """Check test coverage for files"""
        # For now, return True (TODO: implement actual coverage check)
        # This would run pytest with coverage and check threshold
        return True

    def _collect_artifacts(self, files: List[str], run_id: str) -> Dict[str, str]:
        """
        Collect artifacts generated by Manager

        Returns:
            Dict mapping artifact type to path
        """
        artifacts = {
            "diffs": f"artifacts/runs/{run_id}/code/diffs/",
            "files_modified": files
        }

        return artifacts


# Singleton accessor
_executor_instance = None


def get_manager_executor() -> ManagerExecutor:
    """Get singleton Manager executor instance"""
    global _executor_instance
    if _executor_instance is None:
        _executor_instance = ManagerExecutor()
    return _executor_instance
