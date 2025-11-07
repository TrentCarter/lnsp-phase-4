#!/usr/bin/env python3
"""
Cost & KPI Receipt Generation for Aider RPC

Tracks token usage, costs, and quality metrics for integration with
PAS Token Governor and Experiment Ledger.

Based on PAS PRD receipt schema.
"""
from __future__ import annotations

import time
import tempfile
from dataclasses import dataclass, asdict
from typing import Dict, Optional, Any
from pathlib import Path
import json


@dataclass
class TokenUsage:
    """Token usage breakdown"""
    input_tokens: int = 0
    output_tokens: int = 0
    thinking_tokens: int = 0  # For models with extended thinking
    total_tokens: int = 0

    def update_total(self):
        """Recalculate total from components"""
        self.total_tokens = self.input_tokens + self.output_tokens + self.thinking_tokens


@dataclass
class CostEstimate:
    """Cost estimation in USD"""
    input_cost: float = 0.0
    output_cost: float = 0.0
    total_cost: float = 0.0
    model: str = "unknown"
    pricing_date: str = ""  # ISO date when pricing was retrieved

    def calculate_total(self):
        """Recalculate total from components"""
        self.total_cost = self.input_cost + self.output_cost


@dataclass
class ProviderSnapshot:
    """Provider configuration snapshot for replay passports"""
    provider: str = "ollama"  # ollama|openai|anthropic|bedrock|etc
    model: str = "qwen2.5-coder:7b-instruct"
    temperature: float = 0.7
    max_tokens: int = 4096
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    api_base: Optional[str] = None
    api_version: Optional[str] = None


@dataclass
class KPIMetrics:
    """Quality KPIs for lane-specific validation"""
    test_pass_rate: Optional[float] = None  # % of tests passing after change
    lint_errors: int = 0                     # Number of lint errors introduced
    type_errors: int = 0                     # Number of type errors introduced
    files_changed: int = 0
    lines_added: int = 0
    lines_removed: int = 0
    commits_created: int = 0
    duration_seconds: float = 0.0


@dataclass
class AiderReceipt:
    """
    Complete receipt for an Aider execution.

    Aligns with PAS receipt schema and includes:
    - Token usage & costs
    - Provider snapshot (for replay)
    - KPI metrics
    - Timing breakdown
    """
    run_id: str
    job_id: Optional[str] = None
    task: str = "unknown"
    status: str = "ok"  # ok|error|timeout|cancelled

    # Token & cost tracking
    usage: TokenUsage = None
    cost: CostEstimate = None
    provider: ProviderSnapshot = None

    # Quality metrics
    kpis: KPIMetrics = None

    # Timing breakdown (milliseconds)
    timings_ms: Dict[str, float] = None

    # Metadata
    created_at: str = ""  # ISO timestamp
    completed_at: str = ""  # ISO timestamp
    artifacts: list = None  # List of artifact paths (diffs, logs, etc.)

    def __post_init__(self):
        """Initialize nested dataclasses if not provided"""
        if self.usage is None:
            self.usage = TokenUsage()
        if self.cost is None:
            self.cost = CostEstimate()
        if self.provider is None:
            self.provider = ProviderSnapshot()
        if self.kpis is None:
            self.kpis = KPIMetrics()
        if self.timings_ms is None:
            self.timings_ms = {}
        if self.artifacts is None:
            self.artifacts = []

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return {
            "run_id": self.run_id,
            "job_id": self.job_id,
            "task": self.task,
            "status": self.status,
            "usage": asdict(self.usage),
            "cost": asdict(self.cost),
            "provider": asdict(self.provider),
            "kpis": asdict(self.kpis),
            "timings_ms": self.timings_ms,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
            "artifacts": self.artifacts,
        }

    def save(self, path: Path | str):
        """Save receipt to JSON file (atomic write via tmp→rename)"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Atomic write: tmp file → rename (prevents partial writes)
        tmp_fd, tmp_path = tempfile.mkstemp(
            dir=path.parent,
            prefix=".tmp_receipt_",
            suffix=".json"
        )

        try:
            with open(tmp_fd, "w") as f:
                json.dump(self.to_dict(), f, indent=2)

            # Atomic rename (POSIX guarantees atomicity)
            Path(tmp_path).rename(path)
        except Exception:
            # Clean up tmp file on failure
            try:
                Path(tmp_path).unlink()
            except Exception:
                pass
            raise

    @classmethod
    def load(cls, path: Path | str) -> "AiderReceipt":
        """Load receipt from JSON file"""
        with open(path, "r") as f:
            data = json.load(f)

        # Reconstruct nested dataclasses
        usage = TokenUsage(**data.get("usage", {}))
        cost = CostEstimate(**data.get("cost", {}))
        provider = ProviderSnapshot(**data.get("provider", {}))
        kpis = KPIMetrics(**data.get("kpis", {}))

        return cls(
            run_id=data["run_id"],
            job_id=data.get("job_id"),
            task=data.get("task", "unknown"),
            status=data.get("status", "ok"),
            usage=usage,
            cost=cost,
            provider=provider,
            kpis=kpis,
            timings_ms=data.get("timings_ms", {}),
            created_at=data.get("created_at", ""),
            completed_at=data.get("completed_at", ""),
            artifacts=data.get("artifacts", []),
        )


def estimate_cost(tokens: TokenUsage, model: str) -> CostEstimate:
    """
    Estimate cost based on token usage and model pricing.

    Pricing as of 2025-01 (adjust as needed):
    """
    pricing_table = {
        # Ollama local models (free)
        "qwen2.5-coder:7b-instruct": (0.0, 0.0),
        "llama3.1:8b": (0.0, 0.0),
        "phi-4:14b": (0.0, 0.0),

        # OpenAI
        "gpt-4-turbo": (0.01, 0.03),  # $10/$30 per 1M tokens
        "gpt-4o": (0.005, 0.015),     # $5/$15 per 1M tokens
        "gpt-3.5-turbo": (0.0005, 0.0015),  # $0.50/$1.50 per 1M tokens

        # Anthropic
        "claude-3-5-sonnet-20241022": (0.003, 0.015),  # $3/$15 per 1M tokens
        "claude-3-opus-20240229": (0.015, 0.075),      # $15/$75 per 1M tokens
        "claude-3-haiku-20240307": (0.00025, 0.00125), # $0.25/$1.25 per 1M tokens

        # Default fallback
        "unknown": (0.001, 0.005),
    }

    input_per_1k, output_per_1k = pricing_table.get(model, pricing_table["unknown"])

    input_cost = (tokens.input_tokens / 1000.0) * input_per_1k
    output_cost = (tokens.output_tokens / 1000.0) * output_per_1k

    cost = CostEstimate(
        input_cost=round(input_cost, 6),
        output_cost=round(output_cost, 6),
        model=model,
        pricing_date="2025-01-01"
    )
    cost.calculate_total()

    return cost


def now_iso() -> str:
    """Get current timestamp in ISO format"""
    import datetime as _dt
    return _dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


if __name__ == "__main__":
    # Self-test: create and save a sample receipt
    print("Receipt Generation Self-Test:")

    receipt = AiderReceipt(
        run_id="test-run-001",
        job_id="DOC-REF-001",
        task="doc_update",
        status="ok",
        created_at=now_iso(),
        completed_at=now_iso(),
    )

    # Simulate token usage
    receipt.usage.input_tokens = 5000
    receipt.usage.output_tokens = 1500
    receipt.usage.update_total()

    # Estimate cost
    receipt.provider.model = "qwen2.5-coder:7b-instruct"
    receipt.cost = estimate_cost(receipt.usage, receipt.provider.model)

    # Add KPIs
    receipt.kpis.files_changed = 5
    receipt.kpis.lines_added = 120
    receipt.kpis.lines_removed = 80
    receipt.kpis.duration_seconds = 12.5

    # Save receipt
    test_path = Path("/tmp/test_aider_receipt.json")
    receipt.save(test_path)
    print(f"✓ Receipt saved to {test_path}")

    # Load and verify
    loaded = AiderReceipt.load(test_path)
    print(f"✓ Receipt loaded: run_id={loaded.run_id}, tokens={loaded.usage.total_tokens}, cost=${loaded.cost.total_cost:.4f}")

    # Print summary
    print("\nReceipt Summary:")
    print(f"  Run ID: {receipt.run_id}")
    print(f"  Task: {receipt.task}")
    print(f"  Tokens: {receipt.usage.total_tokens:,} ({receipt.usage.input_tokens:,} in, {receipt.usage.output_tokens:,} out)")
    print(f"  Cost: ${receipt.cost.total_cost:.4f} ({receipt.provider.model})")
    print(f"  Changes: {receipt.kpis.files_changed} files, +{receipt.kpis.lines_added}/-{receipt.kpis.lines_removed} lines")
    print(f"  Duration: {receipt.kpis.duration_seconds:.2f}s")
