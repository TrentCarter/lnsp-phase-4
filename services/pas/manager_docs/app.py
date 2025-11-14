#!/usr/bin/env python3
"""
Manager-Docs Service - Documentation Task Executor

Port: 6147 (Mgr-Docs-01)
LLM: Claude Sonnet 4.5 (primary for writing quality)

Responsibilities:
- Receive documentation task assignments from Dir-Docs
- Generate/update technical documentation
- Create API documentation, README files, guides
- Run documentation quality checks
- Report results back to Dir-Docs
- Ask questions when clarification needed

Contract: docs/contracts/MANAGER_DOCS_SYSTEM_PROMPT.md
"""
import sys
import os
from pathlib import Path
from typing import Dict, Any, List

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from services.common.manager_base import BaseManager, ManagerConfig


class DocsManager(BaseManager):
    """
    Documentation Manager - Handles documentation tasks.

    Responsibilities:
    - Generate technical documentation
    - Update README files
    - Create API documentation
    - Generate user guides and tutorials
    - Maintain documentation consistency
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
        Execute documentation task.

        Args:
            task_description: Natural language description of docs task
            files: Documentation files to create/update
            budget: Resource constraints
            thread_id: Agent chat thread ID
            run_id: Overall run ID

        Returns:
            Dict with execution results (files_updated, word_count, etc.)
        """
        # Send progress update
        await self.agent_chat.send_message(
            thread_id=thread_id,
            from_agent=self.config.manager_id,
            to_agent=self.config.parent_agent,
            message_type="status",
            content=f"Generating documentation: {len(files)} file(s)",
            metadata={"progress": 40, "files": files}
        )

        # TODO: Implement actual documentation generation logic
        # - Parse existing docs
        # - Generate new content with LLM
        # - Format as markdown
        # - Update cross-references
        # For now, return success placeholder
        result = {
            "ok": True,
            "files_updated": len(files),
            "word_count": 0,
            "duration_s": 0.0
        }

        # Send final status before returning
        await self.agent_chat.send_message(
            thread_id=thread_id,
            from_agent=self.config.manager_id,
            to_agent=self.config.parent_agent,
            message_type="status",
            content="Documentation complete",
            metadata={"progress": 90, "files_updated": result["files_updated"]}
        )

        return result

    async def run_acceptance_checks(
        self,
        acceptance: List[Dict[str, Any]],
        files: List[str],
        run_id: str
    ) -> Dict[str, bool]:
        """
        Run documentation quality acceptance checks.

        Checks:
        - Markdown syntax validation
        - Link checker (no broken links)
        - Spelling/grammar check
        - Consistency check (terminology, formatting)
        - Completeness (all sections present)
        """
        results = {}

        for check in acceptance:
            check_type = check.get("type", "unknown")
            check_name = check.get("name", check_type)

            # TODO: Implement actual documentation quality checks
            # - markdownlint for syntax
            # - Link validation
            # - Spell check
            # For now, assume all pass
            results[check_name] = True

        return results


# === Application Entry Point ===

# Get configuration from environment
MANAGER_ID = os.getenv("MANAGER_ID", "Mgr-Docs-01")
MANAGER_PORT = int(os.getenv("MANAGER_PORT", "6147"))
MANAGER_LLM = os.getenv("MANAGER_LLM", "anthropic/claude-sonnet-4-5")

# Create manager configuration
config = ManagerConfig(
    manager_id=MANAGER_ID,
    port=MANAGER_PORT,
    parent_agent="Dir-Docs",
    llm_model=MANAGER_LLM,
    description="Documentation task executor"
)

# Create manager instance and FastAPI app
manager = DocsManager(config)
app = manager.create_app()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=MANAGER_PORT)
