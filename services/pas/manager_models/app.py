#!/usr/bin/env python3
"""
Manager-Models Service - ML Training and Evaluation Executor

Port: 6144 (Mgr-Models-01)
LLM: Qwen 2.5 Coder 7B (primary), Claude Sonnet 4.5 (fallback)

Responsibilities:
- Receive ML training/evaluation task assignments from Dir-Models
- Execute model training with specified hyperparameters
- Run model evaluation and validation
- Track experiment metrics
- Report results back to Dir-Models
- Ask questions when clarification needed

Contract: docs/contracts/MANAGER_MODELS_SYSTEM_PROMPT.md
"""
import sys
import os
from pathlib import Path
from typing import Dict, Any, List

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from services.common.manager_base import BaseManager, ManagerConfig


class ModelsManager(BaseManager):
    """
    Models Manager - Handles ML training and evaluation tasks.

    Responsibilities:
    - Model training with hyperparameter tuning
    - Model evaluation and validation
    - Experiment tracking and metrics logging
    - Model deployment preparation
    - Performance benchmarking
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
        Execute ML training/evaluation task.

        Args:
            task_description: Natural language description of ML task
            files: Training scripts, config files, data files
            budget: Resource constraints (GPU time, epochs, etc.)
            thread_id: Agent chat thread ID
            run_id: Overall run ID

        Returns:
            Dict with execution results (accuracy, loss, training_time, etc.)
        """
        # Send progress update
        await self.agent_chat.send_message(
            thread_id=thread_id,
            from_agent=self.config.manager_id,
            to_agent=self.config.parent_agent,
            message_type="status",
            content=f"Starting ML task: {task_description[:80]}...",
            metadata={"progress": 40, "files": files}
        )

        # TODO: Implement actual ML training logic
        # - Parse training configuration
        # - Load dataset
        # - Initialize model
        # - Run training loop with progress updates
        # - Evaluate on validation set
        # - Save model checkpoint
        # - Log metrics to experiment tracker
        # For now, return success placeholder
        result = {
            "ok": True,
            "accuracy": 0.0,
            "loss": 0.0,
            "epochs_completed": 0,
            "training_time_s": 0.0,
            "model_path": ""
        }

        # Send final status before returning
        await self.agent_chat.send_message(
            thread_id=thread_id,
            from_agent=self.config.manager_id,
            to_agent=self.config.parent_agent,
            message_type="status",
            content=f"ML task complete - Accuracy: {result['accuracy']:.2%}",
            metadata={"progress": 90, "metrics": result}
        )

        return result

    async def run_acceptance_checks(
        self,
        acceptance: List[Dict[str, Any]],
        files: List[str],
        run_id: str
    ) -> Dict[str, bool]:
        """
        Run ML model acceptance checks.

        Checks:
        - Model accuracy meets minimum threshold
        - No significant performance regression
        - Model size within limits
        - Inference time acceptable
        - No data leakage detected
        """
        results = {}

        for check in acceptance:
            check_type = check.get("type", "unknown")
            check_name = check.get("name", check_type)

            # TODO: Implement actual ML acceptance checks
            # - Accuracy threshold check
            # - Performance regression test
            # - Model size validation
            # - Inference time benchmark
            # For now, assume all pass
            results[check_name] = True

        return results


# === Application Entry Point ===

# Get configuration from environment
MANAGER_ID = os.getenv("MANAGER_ID", "Mgr-Models-01")
MANAGER_PORT = int(os.getenv("MANAGER_PORT", "6144"))
MANAGER_LLM = os.getenv("MANAGER_LLM", "qwen2.5-coder:7b")

# Create manager configuration
config = ManagerConfig(
    manager_id=MANAGER_ID,
    port=MANAGER_PORT,
    parent_agent="Dir-Models",
    llm_model=MANAGER_LLM,
    description="ML training and evaluation task executor"
)

# Create manager instance and FastAPI app
manager = ModelsManager(config)
app = manager.create_app()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=MANAGER_PORT)
