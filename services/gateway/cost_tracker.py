"""
Cost Tracker - Handles cost calculation, tracking, and receipt generation

This module:
- Calculates per-request costs based on token usage
- Tracks rolling cost windows (per-minute, per-hour, per-day)
- Generates LDJSON receipts to artifacts/costs/
- Provides budget alerts and threshold monitoring
"""

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional, List
from decimal import Decimal, ROUND_HALF_UP
from collections import defaultdict


class CostTracker:
    """Tracks costs and generates receipts for routed requests"""

    def __init__(self, receipts_dir: str = "artifacts/costs"):
        """Initialize cost tracker with receipts directory"""
        self.receipts_dir = Path(receipts_dir)
        self.receipts_dir.mkdir(parents=True, exist_ok=True)

        # In-memory rolling windows (for real-time metrics)
        self.costs_by_minute: Dict[str, List[Dict]] = defaultdict(list)
        self.costs_by_hour: Dict[str, List[Dict]] = defaultdict(list)
        self.costs_by_day: Dict[str, List[Dict]] = defaultdict(list)

        # Budget tracking
        self.budgets: Dict[str, Decimal] = {}  # run_id -> budget
        self.spent: Dict[str, Decimal] = defaultdict(Decimal)  # run_id -> spent

    def calculate_cost(
        self,
        input_tokens: int,
        output_tokens: int,
        cost_per_input_token: float,
        cost_per_output_token: float
    ) -> Decimal:
        """
        Calculate precise cost using Decimal for financial accuracy

        Args:
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            cost_per_input_token: Cost per input token (USD)
            cost_per_output_token: Cost per output token (USD)

        Returns:
            Total cost in USD as Decimal (rounded to 6 decimal places)
        """
        input_cost = Decimal(str(input_tokens)) * Decimal(str(cost_per_input_token))
        output_cost = Decimal(str(output_tokens)) * Decimal(str(cost_per_output_token))
        total_cost = input_cost + output_cost

        # Round to 6 decimal places (microdollars)
        return total_cost.quantize(Decimal('0.000001'), rounding=ROUND_HALF_UP)

    def record_request(
        self,
        request_id: str,
        run_id: Optional[str],
        agent: str,
        provider: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cost_per_input_token: float,
        cost_per_output_token: float,
        latency_ms: int,
        status: str,
        error_message: Optional[str] = None,
        fallback_from: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> Dict:
        """
        Record a request and generate receipt

        Args:
            request_id: Unique request identifier
            run_id: Optional run identifier for grouping
            agent: Agent that made the request
            provider: Provider that handled the request
            model: Model used
            input_tokens: Input token count
            output_tokens: Output token count
            cost_per_input_token: Input token cost
            cost_per_output_token: Output token cost
            latency_ms: Request latency in milliseconds
            status: Request status (success/error/timeout/fallback)
            error_message: Optional error message
            fallback_from: Original provider if fallback
            metadata: Additional metadata

        Returns:
            Receipt dictionary
        """
        timestamp = datetime.utcnow()
        cost_usd = self.calculate_cost(
            input_tokens, output_tokens,
            cost_per_input_token, cost_per_output_token
        )

        receipt = {
            "request_id": request_id,
            "run_id": run_id,
            "agent": agent,
            "timestamp": timestamp.isoformat(),
            "provider": provider,
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost_usd": float(cost_usd),
            "latency_ms": latency_ms,
            "status": status
        }

        if error_message:
            receipt["error_message"] = error_message

        if fallback_from:
            receipt["fallback_from"] = fallback_from

        if metadata:
            receipt["metadata"] = metadata

        # Write receipt to file (LDJSON format)
        self._write_receipt(receipt, run_id or "default")

        # Update rolling windows
        self._update_rolling_windows(receipt, timestamp)

        # Update budget tracking
        if run_id:
            self.spent[run_id] += cost_usd

        return receipt

    def _write_receipt(self, receipt: Dict, run_id: str):
        """Write receipt to LDJSON file"""
        receipt_file = self.receipts_dir / f"{run_id}.jsonl"

        with open(receipt_file, 'a') as f:
            f.write(json.dumps(receipt) + '\n')

    def _update_rolling_windows(self, receipt: Dict, timestamp: datetime):
        """Update rolling cost windows"""
        minute_key = timestamp.strftime('%Y-%m-%d %H:%M')
        hour_key = timestamp.strftime('%Y-%m-%d %H')
        day_key = timestamp.strftime('%Y-%m-%d')

        self.costs_by_minute[minute_key].append(receipt)
        self.costs_by_hour[hour_key].append(receipt)
        self.costs_by_day[day_key].append(receipt)

        # Clean old entries (keep last 60 minutes, 24 hours, 7 days)
        self._cleanup_old_entries(timestamp)

    def _cleanup_old_entries(self, current_time: datetime):
        """Remove old entries from rolling windows"""
        # Keep last 60 minutes
        minute_cutoff = (current_time - timedelta(minutes=60)).strftime('%Y-%m-%d %H:%M')
        self.costs_by_minute = {
            k: v for k, v in self.costs_by_minute.items() if k >= minute_cutoff
        }

        # Keep last 24 hours
        hour_cutoff = (current_time - timedelta(hours=24)).strftime('%Y-%m-%d %H')
        self.costs_by_hour = {
            k: v for k, v in self.costs_by_hour.items() if k >= hour_cutoff
        }

        # Keep last 7 days
        day_cutoff = (current_time - timedelta(days=7)).strftime('%Y-%m-%d')
        self.costs_by_day = {
            k: v for k, v in self.costs_by_day.items() if k >= day_cutoff
        }

    def get_metrics(self, window: str = "minute") -> Dict:
        """
        Get cost metrics for a rolling window

        Args:
            window: Time window ("minute", "hour", "day")

        Returns:
            Dictionary with cost metrics
        """
        if window == "minute":
            data = self.costs_by_minute
        elif window == "hour":
            data = self.costs_by_hour
        elif window == "day":
            data = self.costs_by_day
        else:
            raise ValueError(f"Invalid window: {window}")

        all_receipts = []
        for receipts in data.values():
            all_receipts.extend(receipts)

        if not all_receipts:
            return {
                "window": window,
                "total_cost_usd": 0.0,
                "total_requests": 0,
                "total_tokens": 0,
                "cost_per_request": 0.0,
                "requests_per_provider": {},
                "cost_per_provider": {}
            }

        total_cost = sum(Decimal(str(r['cost_usd'])) for r in all_receipts)
        total_tokens = sum(r['input_tokens'] + r['output_tokens'] for r in all_receipts)

        # Per-provider metrics
        provider_requests = defaultdict(int)
        provider_costs = defaultdict(Decimal)

        for r in all_receipts:
            provider_requests[r['provider']] += 1
            provider_costs[r['provider']] += Decimal(str(r['cost_usd']))

        return {
            "window": window,
            "total_cost_usd": float(total_cost),
            "total_requests": len(all_receipts),
            "total_tokens": total_tokens,
            "cost_per_request": float(total_cost / len(all_receipts)),
            "requests_per_provider": dict(provider_requests),
            "cost_per_provider": {k: float(v) for k, v in provider_costs.items()}
        }

    def set_budget(self, run_id: str, budget_usd: float):
        """Set budget for a run"""
        self.budgets[run_id] = Decimal(str(budget_usd))

    def get_budget_status(self, run_id: str) -> Dict:
        """
        Get budget status with alert levels

        Args:
            run_id: Run identifier

        Returns:
            Budget status with alerts
        """
        if run_id not in self.budgets:
            return {
                "run_id": run_id,
                "budget_set": False,
                "message": "No budget set for this run"
            }

        budget = self.budgets[run_id]
        spent = self.spent.get(run_id, Decimal('0'))
        remaining = budget - spent
        percent_used = (spent / budget * 100) if budget > 0 else 0

        alert_level = "ok"
        if percent_used >= 100:
            alert_level = "critical"  # Block
        elif percent_used >= 90:
            alert_level = "warning"   # Alert
        elif percent_used >= 75:
            alert_level = "caution"   # Warn

        return {
            "run_id": run_id,
            "budget_set": True,
            "budget_usd": float(budget),
            "spent_usd": float(spent),
            "remaining_usd": float(remaining),
            "percent_used": float(percent_used),
            "alert_level": alert_level,
            "can_proceed": alert_level != "critical"
        }

    def get_all_receipts(self, run_id: str) -> List[Dict]:
        """Read all receipts for a run from LDJSON file"""
        receipt_file = self.receipts_dir / f"{run_id}.jsonl"

        if not receipt_file.exists():
            return []

        receipts = []
        with open(receipt_file, 'r') as f:
            for line in f:
                if line.strip():
                    receipts.append(json.loads(line))

        return receipts
