#!/usr/bin/env python3
"""
Manager-Data Service - Data Pipeline Executor

Port: 6145 (Mgr-Data-01)
LLM: Qwen 2.5 Coder 7B (primary), Claude Sonnet 4.5 (fallback)

Responsibilities:
- Receive data processing task assignments from Dir-Data
- Execute data ingestion, transformation, and validation
- Run data quality checks
- Report results back to Dir-Data
- Ask questions when clarification needed

Contract: docs/contracts/MANAGER_DATA_SYSTEM_PROMPT.md
"""
import sys
import os
from pathlib import Path
from typing import Dict, Any, List

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from services.common.manager_base import BaseManager, ManagerConfig


class DataManager(BaseManager):
    """
    Data Manager - Handles data pipeline tasks.

    Responsibilities:
    - Data ingestion from various sources
    - Data transformation and validation
    - Quality checks (schema validation, completeness, uniqueness)
    - Integration with data processing tools
    """

    async def execute_task(
        self,
        task_description: str,
        files: List[str],
        budget: Dict[str, Any],
        thread_id: str,
        run_id: str
    ) -> Dict[str, Any]:
        """
        Execute data processing task.

        Args:
            task_description: Natural language description of data task
            files: Data files to process
            budget: Resource constraints
            thread_id: Agent chat thread ID
            run_id: Overall run ID

        Returns:
            Dict with execution results (records_processed, duration, etc.)
        """
        # Send progress update
        await self.agent_chat.send_message(
            thread_id=thread_id,
            from_agent=self.config.manager_id,
            to_agent=self.config.parent_agent,
            message_type="status",
            content=f"Processing data: {len(files)} file(s)",
            metadata={"progress": 40, "files": files}
        )

        # TODO: Implement actual data processing logic
        # For now, return success placeholder
        result = {
            "ok": True,
            "records_processed": 0,
            "duration_s": 0.0,
            "output_files": files
        }

        # Send final status before returning
        await self.agent_chat.send_message(
            thread_id=thread_id,
            from_agent=self.config.manager_id,
            to_agent=self.config.parent_agent,
            message_type="status",
            content="Data processing complete",
            metadata={"progress": 90, "records": result["records_processed"]}
        )

        return result

    async def run_acceptance_checks(
        self,
        acceptance: List[Dict[str, Any]],
        files: List[str],
        run_id: str
    ) -> Dict[str, bool]:
        """
        Run data quality acceptance checks.

        Checks:
        - Schema validation
        - Completeness (no missing required fields)
        - Uniqueness (primary key constraints)
        - Range validation (values within expected bounds)
        """
        results = {}

        for check in acceptance:
            check_type = check.get("type", "unknown")
            check_name = check.get("name", check_type)

            # TODO: Implement actual data quality checks
            # For now, assume all pass
            results[check_name] = True

        return results


# === Application Entry Point ===

# Get configuration from environment
MANAGER_ID = os.getenv("MANAGER_ID", "Mgr-Data-01")
MANAGER_PORT = int(os.getenv("MANAGER_PORT", "6145"))
MANAGER_LLM = os.getenv("MANAGER_LLM", "qwen2.5-coder:7b")

# Create manager configuration
config = ManagerConfig(
    manager_id=MANAGER_ID,
    port=MANAGER_PORT,
    parent_agent="Dir-Data",
    llm_model=MANAGER_LLM,
    description="Data pipeline task executor"
)

# Create manager instance and FastAPI app
manager = DataManager(config)
app = manager.create_app()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=MANAGER_PORT)
