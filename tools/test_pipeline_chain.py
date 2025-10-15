#!/usr/bin/env python3
"""
LNSP Pipeline Chain Tester

Tests the complete TMD-LS pipeline across all FastAPI services:
1. Text → Chunker (8001) → Chunks
2. Chunk → TMD Router (8002) → TMD codes + Lane
3. Concept → GTR-T5 (8765) → 768D vector
4. Vector sequence → LVM (8003) → Predicted vector
5. Predicted vector → Vec2Text (8766) → Decoded text

Beautiful visualization of each stage's input/output.
"""

import argparse
import json
import sys
import time
from typing import Dict, List, Any
import requests

# Try to import rich for beautiful output
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.syntax import Syntax
    from rich import print as rprint
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    print("⚠️  Install 'rich' for beautiful output: pip install rich")


# ============================================================================
# Configuration
# ============================================================================

SERVICES = {
    "chunker": "http://localhost:8001",
    "tmd_router": "http://localhost:8002",
    "gtr_t5": "http://localhost:8767",
    "lvm": "http://localhost:8003",
    "vec2text": "http://localhost:8766"
}


# ============================================================================
# Service Health Checks
# ============================================================================

def check_service_health(name: str, url: str) -> bool:
    """Check if a service is running"""
    try:
        response = requests.get(f"{url}/health", timeout=2)
        return response.status_code == 200
    except Exception:
        return False


def check_all_services() -> Dict[str, bool]:
    """Check health of all services"""
    return {name: check_service_health(name, url) for name, url in SERVICES.items()}


# ============================================================================
# Pipeline Stages
# ============================================================================

class PipelineResult:
    """Result from a pipeline stage"""
    def __init__(self, stage: str, success: bool, data: Any = None, error: str = None, latency_ms: int = 0):
        self.stage = stage
        self.success = success
        self.data = data
        self.error = error
        self.latency_ms = latency_ms


def stage_1_chunker(text: str, mode: str = "semantic", breakpoint_threshold: int = 50, min_chunk_size: int = 10) -> PipelineResult:
    """Stage 1: Text → Chunks"""
    try:
        start = time.time()
        response = requests.post(
            f"{SERVICES['chunker']}/chunk",
            json={
                "text": text,
                "mode": mode,
                "max_chunk_size": 320,
                "breakpoint_threshold": breakpoint_threshold,
                "min_chunk_size": min_chunk_size
            },
            timeout=10
        )
        latency = int((time.time() - start) * 1000)

        if response.status_code == 200:
            data = response.json()
            return PipelineResult("Chunker", True, data, latency_ms=latency)
        else:
            return PipelineResult("Chunker", False, error=f"HTTP {response.status_code}")
    except Exception as e:
        return PipelineResult("Chunker", False, error=str(e))


def stage_2_tmd_router(concept_text: str) -> PipelineResult:
    """Stage 2: Concept → TMD codes + Lane"""
    try:
        start = time.time()
        response = requests.post(
            f"{SERVICES['tmd_router']}/route",
            json={"concept_text": concept_text},
            timeout=10
        )
        latency = int((time.time() - start) * 1000)

        if response.status_code == 200:
            data = response.json()
            return PipelineResult("TMD Router", True, data, latency_ms=latency)
        else:
            return PipelineResult("TMD Router", False, error=f"HTTP {response.status_code}")
    except Exception as e:
        return PipelineResult("TMD Router", False, error=str(e))


def stage_3_gtr_t5(texts: List[str]) -> PipelineResult:
    """Stage 3: Text → 768D vectors"""
    try:
        start = time.time()
        response = requests.post(
            f"{SERVICES['gtr_t5']}/embed",
            json={"texts": texts, "normalize": True},
            timeout=10
        )
        latency = int((time.time() - start) * 1000)

        if response.status_code == 200:
            data = response.json()
            return PipelineResult("GTR-T5 Embedder", True, data, latency_ms=latency)
        else:
            return PipelineResult("GTR-T5 Embedder", False, error=f"HTTP {response.status_code}")
    except Exception as e:
        return PipelineResult("GTR-T5 Embedder", False, error=str(e))


def stage_4_lvm(vector_sequence: List[List[float]], tmd_codes: List[int]) -> PipelineResult:
    """Stage 4: Vector sequence → Predicted vector"""
    try:
        start = time.time()
        response = requests.post(
            f"{SERVICES['lvm']}/infer",
            json={
                "vector_sequence": vector_sequence,
                "tmd_codes": tmd_codes,
                "use_mock": True  # Use mock since we don't have trained model yet
            },
            timeout=10
        )
        latency = int((time.time() - start) * 1000)

        if response.status_code == 200:
            data = response.json()
            return PipelineResult("LVM Inference", True, data, latency_ms=latency)
        else:
            return PipelineResult("LVM Inference", False, error=f"HTTP {response.status_code}")
    except Exception as e:
        return PipelineResult("LVM Inference", False, error=str(e))


def stage_5_vec2text(vector: List[float]) -> PipelineResult:
    """Stage 5: Vector → Decoded text"""
    try:
        start = time.time()
        response = requests.post(
            f"{SERVICES['vec2text']}/decode",
            json={
                "vectors": [vector],
                "subscribers": "jxe,ielab",
                "steps": 1
            },
            timeout=30
        )
        latency = int((time.time() - start) * 1000)

        if response.status_code == 200:
            data = response.json()
            return PipelineResult("Vec2Text Decoder", True, data, latency_ms=latency)
        else:
            return PipelineResult("Vec2Text Decoder", False, error=f"HTTP {response.status_code}")
    except Exception as e:
        return PipelineResult("Vec2Text Decoder", False, error=str(e))


# ============================================================================
# Display Functions
# ============================================================================

def display_rich(results: List[PipelineResult], input_text: str):
    """Display results using rich library"""
    console = Console()

    # Title
    console.print()
    console.print(Panel.fit(
        "[bold cyan]LNSP Pipeline Chain Test[/bold cyan]\n"
        "Text → Chunker → TMD Router → GTR-T5 → LVM → Vec2Text",
        border_style="cyan"
    ))
    console.print()

    # Input
    console.print(Panel(
        f"[bold]Input Text:[/bold]\n{input_text}",
        border_style="green",
        title="📝 Input"
    ))
    console.print()

    # Each stage
    for i, result in enumerate(results, 1):
        if result.success:
            # Success panel
            title = f"✅ Stage {i}: {result.stage} ({result.latency_ms}ms)"
            border = "green"

            # Format data based on stage
            if result.stage == "Chunker":
                chunks = result.data.get("chunks", [])
                content = f"[bold]Chunks:[/bold] {len(chunks)}\n\n"
                for j, chunk in enumerate(chunks[:3], 1):  # Show first 3
                    content += f"[dim]Chunk {j}:[/dim] {chunk['text'][:100]}...\n"
                if len(chunks) > 3:
                    content += f"[dim]... and {len(chunks) - 3} more[/dim]"

            elif result.stage == "TMD Router":
                content = (
                    f"[bold]Domain:[/bold] {result.data['domain_name']} (code: {result.data['domain_code']})\n"
                    f"[bold]Task:[/bold] {result.data['task_code']}\n"
                    f"[bold]Modifier:[/bold] {result.data['modifier_code']}\n"
                    f"[bold]Lane Model:[/bold] {result.data['lane_model']}\n"
                    f"[bold]Lane Port:[/bold] {result.data['lane_port']}\n"
                    f"[bold]Cache Hit:[/bold] {result.data['cache_hit']}"
                )

            elif result.stage == "GTR-T5 Embedder":
                content = (
                    f"[bold]Embeddings:[/bold] {result.data['count']}\n"
                    f"[bold]Dimension:[/bold] {result.data['dimension']}\n"
                    f"[bold]Sample vector:[/bold] [{result.data['embeddings'][0][0]:.4f}, "
                    f"{result.data['embeddings'][0][1]:.4f}, ...]"
                )

            elif result.stage == "LVM Inference":
                content = (
                    f"[bold]Predicted vector:[/bold] [{result.data['predicted_vector'][0]:.4f}, "
                    f"{result.data['predicted_vector'][1]:.4f}, ...]\n"
                    f"[bold]Confidence:[/bold] {result.data['confidence']:.2f}\n"
                    f"[bold]Model:[/bold] {result.data['model_version']}\n"
                    f"[bold]Mock mode:[/bold] {result.data['is_mock']}"
                )

            elif result.stage == "Vec2Text Decoder":
                decoded = result.data['results'][0] if result.data['results'] else {}
                jxe_text = decoded.get('jxe', {}).get('decoded_text', 'N/A')
                ielab_text = decoded.get('ielab', {}).get('decoded_text', 'N/A')
                content = (
                    f"[bold]JXE decoder:[/bold] {jxe_text}\n"
                    f"[bold]IELab decoder:[/bold] {ielab_text}"
                )

            else:
                content = json.dumps(result.data, indent=2)

            console.print(Panel(content, border_style=border, title=title))

        else:
            # Error panel
            title = f"❌ Stage {i}: {result.stage} FAILED"
            console.print(Panel(
                f"[bold red]Error:[/bold red] {result.error}",
                border_style="red",
                title=title
            ))

        console.print()

    # Summary
    total_latency = sum(r.latency_ms for r in results if r.success)
    successful = sum(1 for r in results if r.success)

    summary = Table(title="Summary", border_style="cyan")
    summary.add_column("Metric", style="cyan")
    summary.add_column("Value", style="green")

    summary.add_row("Stages completed", f"{successful}/{len(results)}")
    summary.add_row("Total latency", f"{total_latency}ms")
    summary.add_row("Avg latency/stage", f"{total_latency // len(results)}ms")

    console.print(summary)
    console.print()


def display_simple(results: List[PipelineResult], input_text: str):
    """Display results using simple text (no rich)"""
    print("\n" + "=" * 80)
    print("LNSP PIPELINE CHAIN TEST")
    print("=" * 80)
    print(f"\nInput: {input_text}\n")

    for i, result in enumerate(results, 1):
        print(f"\n--- Stage {i}: {result.stage} ---")
        if result.success:
            print(f"✅ SUCCESS ({result.latency_ms}ms)")
            print(f"Data: {json.dumps(result.data, indent=2)[:500]}...")
        else:
            print(f"❌ FAILED")
            print(f"Error: {result.error}")

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    total_latency = sum(r.latency_ms for r in results if r.success)
    successful = sum(1 for r in results if r.success)
    print(f"Stages completed: {successful}/{len(results)}")
    print(f"Total latency: {total_latency}ms")
    print()


# ============================================================================
# Main Pipeline
# ============================================================================

def run_pipeline(text: str, mode: str = "semantic", breakpoint_threshold: int = 50, min_chunk_size: int = 10) -> List[PipelineResult]:
    """Run the complete pipeline"""
    results = []

    # Stage 1: Chunker
    result = stage_1_chunker(text, mode, breakpoint_threshold, min_chunk_size)
    results.append(result)
    if not result.success:
        return results

    # Get first chunk for subsequent stages
    chunks = result.data.get("chunks", [])
    if not chunks:
        results.append(PipelineResult("Pipeline", False, error="No chunks produced"))
        return results

    first_chunk = chunks[0]["text"]

    # Stage 2: TMD Router
    result = stage_2_tmd_router(first_chunk)
    results.append(result)
    if not result.success:
        return results

    tmd_codes = [
        result.data["domain_code"],
        result.data["task_code"],
        result.data["modifier_code"]
    ]

    # Stage 3: GTR-T5 (embed first chunk)
    result = stage_3_gtr_t5([first_chunk])
    results.append(result)
    if not result.success:
        return results

    embeddings = result.data["embeddings"]

    # Stage 4: LVM (predict next vector)
    # Use first 3 chunks as sequence if available
    chunk_texts = [c["text"] for c in chunks[:3]]
    embed_result = stage_3_gtr_t5(chunk_texts)
    if embed_result.success:
        vector_sequence = embed_result.data["embeddings"]
    else:
        vector_sequence = embeddings

    result = stage_4_lvm(vector_sequence, tmd_codes)
    results.append(result)
    if not result.success:
        return results

    predicted_vector = result.data["predicted_vector"]

    # Stage 5: Vec2Text (decode predicted vector)
    result = stage_5_vec2text(predicted_vector)
    results.append(result)

    return results


def main():
    parser = argparse.ArgumentParser(description="Test LNSP pipeline chain")
    parser.add_argument("text", nargs="?", default="Photosynthesis is the process by which plants convert sunlight into chemical energy, using chlorophyll to capture light and produce glucose.")
    parser.add_argument("--mode", default="semantic", choices=["simple", "semantic", "proposition", "hybrid"])
    parser.add_argument("--breakpoint-threshold", type=int, default=50, help="Semantic boundary sensitivity (50-99, lower=more chunks)")
    parser.add_argument("--min-chunk-size", type=int, default=10, help="Minimum characters per chunk")
    parser.add_argument("--check-health", action="store_true", help="Check service health only")
    parser.add_argument("--no-rich", action="store_true", help="Disable rich output")

    args = parser.parse_args()

    # Check health
    if args.check_health or RICH_AVAILABLE:
        print("\n🔍 Checking service health...\n")
        health = check_all_services()

        if RICH_AVAILABLE and not args.no_rich:
            console = Console()
            table = Table(title="Service Health", border_style="cyan")
            table.add_column("Service", style="cyan")
            table.add_column("URL", style="dim")
            table.add_column("Status", style="bold")

            for name, url in SERVICES.items():
                status = "✅ Running" if health[name] else "❌ Down"
                style = "green" if health[name] else "red"
                table.add_row(name.capitalize(), url, f"[{style}]{status}[/{style}]")

            console.print(table)
            console.print()
        else:
            for name, url in SERVICES.items():
                status = "✅ Running" if health[name] else "❌ Down"
                print(f"{name.capitalize():15} {url:30} {status}")
            print()

        if args.check_health:
            return

        # Check if all required services are running
        required = ["chunker", "tmd_router", "gtr_t5", "lvm"]
        missing = [name for name in required if not health[name]]
        if missing:
            print(f"❌ Required services not running: {', '.join(missing)}")
            print("\nStart missing services:")
            for name in missing:
                port = SERVICES[name].split(":")[-1]
                if name == "chunker":
                    print(f"  ./.venv/bin/uvicorn app.api.chunking:app --host 127.0.0.1 --port {port}")
                elif name == "tmd_router":
                    print(f"  ./.venv/bin/uvicorn app.api.tmd_router:app --host 127.0.0.1 --port {port}")
                elif name == "gtr_t5":
                    print(f"  ./.venv/bin/uvicorn app.api.gtr_embedding_server:app --host 127.0.0.1 --port {port}")
                elif name == "lvm":
                    print(f"  ./.venv/bin/uvicorn app.api.lvm_server:app --host 127.0.0.1 --port {port}")
            print()
            return

    # Run pipeline
    print(f"\n🚀 Running pipeline with input text ({len(args.text)} chars)...")
    print(f"   Mode: {args.mode}")
    print(f"   Breakpoint threshold: {args.breakpoint_threshold}")
    print(f"   Min chunk size: {args.min_chunk_size} chars\n")
    results = run_pipeline(args.text, args.mode, args.breakpoint_threshold, args.min_chunk_size)

    # Display results
    if RICH_AVAILABLE and not args.no_rich:
        display_rich(results, args.text)
    else:
        display_simple(results, args.text)


if __name__ == "__main__":
    main()
