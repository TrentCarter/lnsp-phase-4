# 20250817T130205_V1.2
"""
Compact MLflow training stats exporter.
- Prefers reading embedded MLflow metadata from a checkpoint (.pth)
- Optional auto-discovery of newest best_model.pth under output/
- Minimal JSON (or Markdown) for sharing
"""
from __future__ import annotations

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any, Optional

def _extract_mlflow_from_checkpoint(checkpoint_path: str) -> Optional[Dict[str, Any]]:
    """Lightweight extractor: read 'mlflow_metadata' from a checkpoint without validation."""
    try:
        if not os.path.exists(checkpoint_path):
            return None
        import torch  # local import to avoid hard dependency at import time
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        # Check for new mlflow_metadata format first, then fallback to old mlops_metadata
        mlflow_meta = ckpt.get("mlflow_metadata")
        if mlflow_meta and isinstance(mlflow_meta, dict):
            return mlflow_meta
        mlops = ckpt.get("mlops_metadata")
        return mlops if isinstance(mlops, dict) else None
    except (FileNotFoundError, RuntimeError, Exception):
        return None


def _fallback_stats_from_path(checkpoint_path: str) -> Dict[str, Any]:
    """When no embedded metadata exists, infer minimal identity from filename."""
    try:
        import re
        name = os.path.basename(checkpoint_path)
        sn_match = re.search(r"SN(\d+)", name)
        sn = f"SN{sn_match.group(1)}" if sn_match else None
        return {
            "identity": {"experiment_id": None, "run_id": None, "run_name": Path(checkpoint_path).stem, "sn": sn},
            "status": "unknown",
            "params": {},
            "metrics": {},
            "artifacts": {"checkpoint_path": checkpoint_path},
        }
    except Exception:
        return {"error": "fallback_inference_failed", "artifacts": {"checkpoint_path": checkpoint_path}}

IMPORTANT_PARAM_KEYS = {
    "epochs",
    "batch_size",
    "learning_rate",
    "teacher_dim",
    "student_dim",
    "num_layers",
    "num_experts",
    "lora_rank",
}

IMPORTANT_METRIC_KEYS = {
    "train/loss_final",
    "val/loss_best",
    "val/acc_best",
    "val/cos_best",
    "epoch_time_avg",
}


def find_newest_checkpoint(root: str = "output", filename: str = "best_model.pth") -> Optional[str]:
    """Return path to newest matching checkpoint under root, else None."""
    try:
        root_path = Path(root)
        if not root_path.exists():
            return None
        newest_path = None
        newest_mtime = -1.0
        for p in root_path.rglob(filename):
            try:
                mtime = p.stat().st_mtime
                if mtime > newest_mtime:
                    newest_mtime = mtime
                    newest_path = p
            except Exception:
                continue
        return str(newest_path) if newest_path else None
    except Exception:
        return None


def _subset_params(params: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k in IMPORTANT_PARAM_KEYS:
        if k in params:
            out[k] = params[k]
    return out


def _subset_metrics(metrics: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k in IMPORTANT_METRIC_KEYS:
        if k in metrics:
            out[k] = metrics[k]
    return out


def generate_compact_stats_from_checkpoint(checkpoint_path: str) -> Dict[str, Any]:
    """Create compact stats dict from a checkpoint's embedded MLflow metadata."""
    try:
        mlops = _extract_mlflow_from_checkpoint(checkpoint_path)
        if not mlops:
            return _fallback_stats_from_path(checkpoint_path)

        # Handle new mlflow_metadata format (direct MLFlow run info)
        if "run_id" in mlops:
            # New format - direct MLFlow run information
            result: Dict[str, Any] = {
                "identity": {
                    "experiment_id": mlops.get("experiment_id"),
                    "run_id": mlops.get("run_id"),
                    "run_name": mlops.get("run_name"),
                    "sn": None,  # Could extract from filename if needed
                },
                "status": mlops.get("status"),
                "start_time": mlops.get("start_time"),
                "artifact_uri": mlops.get("artifact_uri"),
                "params": {},  # Would need to fetch from MLFlow DB
                "metrics": {},  # Would need to fetch from MLFlow DB  
                "artifacts": {
                    "checkpoint_path": checkpoint_path,
                },
            }
        else:
            # Old format - nested info/data structure
            info = mlops.get("info", {})
            data = mlops.get("data", {})
            tags = info.get("tags", {})
            params = data.get("params", {})
            metrics = data.get("metrics", {})

            result: Dict[str, Any] = {
                "identity": {
                    "experiment_id": info.get("experiment_id"),
                    "run_id": info.get("run_id"),
                    "run_name": info.get("run_name"),
                    "sn": tags.get("ln.serial_number"),
                },
                "status": info.get("status"),
                "params": _subset_params(params),
                "metrics": _subset_metrics(metrics),
                "artifacts": {
                    "checkpoint_path": checkpoint_path,
                },
            }
        return result
    except (FileNotFoundError, ValueError) as e:
        return {
            "error": str(e),
            "artifacts": {"checkpoint_path": checkpoint_path},
        }
    except Exception as e:
        return {"error": f"Unexpected error: {e}"}


def _to_markdown(stats: Dict[str, Any]) -> str:
    try:
        if "error" in stats:
            return f"## MLflow Compact Stats\n\nError: {stats['error']}\n"
        idt = stats.get("identity", {})
        params = stats.get("params", {})
        metrics = stats.get("metrics", {})
        arts = stats.get("artifacts", {})
        lines = [
            "## MLflow Compact Stats",
            "",
            f"- experiment_id: {idt.get('experiment_id')}",
            f"- run_id: {idt.get('run_id')}",
            f"- run_name: {idt.get('run_name')}",
            f"- sn: {idt.get('sn')}",
            f"- status: {stats.get('status')}",
            f"- checkpoint: {arts.get('checkpoint_path')}",
            "",
            "### Params",
        ]
        for k, v in params.items():
            lines.append(f"- {k}: {v}")
        lines.append("")
        lines.append("### Metrics")
        for k, v in metrics.items():
            lines.append(f"- {k}: {v}")
        return "\n".join(lines)
    except Exception as e:
        return f"## MLflow Compact Stats\n\nError rendering markdown: {e}\n"


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Emit compact MLflow training stats")
    parser.add_argument("--checkpoint", help="Path to checkpoint .pth; auto if omitted", default=None)
    parser.add_argument("--root", help="Search root for auto mode", default="output")
    parser.add_argument("--format", choices=["json", "md"], default="json")
    parser.add_argument("--output", help="Write to file instead of stdout", default=None)
    args = parser.parse_args(argv)

    try:
        ckpt = args.checkpoint or find_newest_checkpoint(args.root)
        if not ckpt:
            payload = {"error": f"No checkpoint found under {args.root}"}
        else:
            payload = generate_compact_stats_from_checkpoint(ckpt)
            # ensure SN runtime naming if writing to file handled outside
        if args.format == "json":
            text = json.dumps(payload, indent=2)
        else:
            text = _to_markdown(payload)
        if args.output:
            Path(Path(args.output).parent).mkdir(parents=True, exist_ok=True)
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(text)
        else:
            print(text)
        return 0
    except Exception as e:
        print(json.dumps({"error": f"CLI failure: {e}"}))
        return 1


if __name__ == "__main__":
    sys.exit(main())
