#!/usr/bin/env python3
"""
Complete Pipeline Walkthrough with Detailed Timing

Tests the full pipeline:
1. Text Input → Chunker API (port 8001)
2. Chunks → Ingest API (port 8004)
   - Phase 1: Parallel TMD extraction
   - Phase 2: Batch embeddings (CPU or MPS)
   - Phase 3: Parallel database writes

Runs 5 iterations with 500ms cooldown between runs.
Reports detailed timing from the 5th (warmed-up) iteration.

Usage:
    ./tools/pipeline_walkthrough.py
    ./tools/pipeline_walkthrough.py --input data/samples/sample_prompts_1.json
    ./tools/pipeline_walkthrough.py --mps  # Test with MPS enabled
"""

import argparse
import json
import os
import sys
import time
import requests
from pathlib import Path
from typing import Dict, List, Any

# Configuration
CHUNKER_API = "http://localhost:8001"
INGEST_API = "http://localhost:8004"
DEFAULT_PROMPT_FILE = "data/samples/sample_prompts_1.json"

# ANSI colors
GREEN = "\033[92m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
CYAN = "\033[96m"
RED = "\033[91m"
BOLD = "\033[1m"
RESET = "\033[0m"


def print_header(text: str):
    """Print colored header"""
    print(f"\n{BOLD}{CYAN}{'=' * 80}{RESET}")
    print(f"{BOLD}{CYAN}{text}{RESET}")
    print(f"{BOLD}{CYAN}{'=' * 80}{RESET}\n")


def print_step(step: str, text: str):
    """Print colored step"""
    print(f"{BOLD}{BLUE}[{step}]{RESET} {text}")


def print_timing(label: str, ms: float, extra: str = ""):
    """Print timing info"""
    extra_str = f" {YELLOW}({extra}){RESET}" if extra else ""
    print(f"   {GREEN}⏱{RESET}  {label:30s}: {BOLD}{ms:7.1f}ms{RESET}{extra_str}")


def print_success(text: str):
    """Print success message"""
    print(f"{GREEN}✓{RESET} {text}")


def print_warning(text: str):
    """Print warning message"""
    print(f"{YELLOW}⚠{RESET}  {text}")


def print_error(text: str):
    """Print error message"""
    print(f"{RED}✗{RESET} {text}")


def check_services():
    """Check if all required services are running"""
    print_step("CHECK", "Verifying services...")

    services = {
        "Chunker API": CHUNKER_API,
        "Ingest API": INGEST_API
    }

    all_healthy = True
    for name, url in services.items():
        try:
            response = requests.get(f"{url}/health", timeout=2)
            if response.status_code == 200:
                print_success(f"{name:20s} → {url}")
            else:
                print_error(f"{name:20s} → {url} (HTTP {response.status_code})")
                all_healthy = False
        except Exception as e:
            print_error(f"{name:20s} → {url} (not responding)")
            all_healthy = False

    if not all_healthy:
        print()
        print_error("Some services are not running. Start them with:")
        print("   ./.venv/bin/python tools/launch_fastapis.py")
        sys.exit(1)

    print()


def load_prompt(prompt_file: str) -> str:
    """Load prompt from JSON file"""
    print_step("LOAD", f"Reading prompt from {prompt_file}")

    try:
        with open(prompt_file, 'r') as f:
            prompts = json.load(f)
            prompt_text = prompts[0] if isinstance(prompts, list) else prompts

        char_count = len(prompt_text)
        word_count = len(prompt_text.split())
        print(f"   📄 Loaded: {char_count} characters, {word_count} words")
        print()
        return prompt_text

    except Exception as e:
        print_error(f"Failed to load prompt: {e}")
        sys.exit(1)


def call_chunker(text: str, iteration: int, chunker_config: Dict[str, Any]) -> Dict[str, Any]:
    """Send text to Chunker API"""
    mode = chunker_config.get("mode", "semantic")
    breakpoint = chunker_config.get("breakpoint_threshold", 95)

    print_step("CHUNK", f"Sending to Chunker API (iteration {iteration}/5) [mode={mode}, breakpoint={breakpoint}]...")

    payload = {
        "text": text,
        **chunker_config  # Merge all chunker config parameters
    }

    start_time = time.perf_counter()
    try:
        response = requests.post(
            f"{CHUNKER_API}/chunk",
            json=payload,
            timeout=30
        )
        response.raise_for_status()
        elapsed_ms = (time.perf_counter() - start_time) * 1000

        result = response.json()
        chunk_count = result["total_chunks"]
        server_time = result["processing_time_ms"]

        if iteration == 5:
            print_timing("Chunker API call", elapsed_ms, f"client time")
            print_timing("Chunker processing", server_time, f"server time")
            print(f"   📦 Created {chunk_count} chunks")
        else:
            print(f"   Warmup iteration {iteration}: {chunk_count} chunks in {elapsed_ms:.1f}ms")

        print()
        return result

    except Exception as e:
        print_error(f"Chunker API failed: {e}")
        sys.exit(1)


def call_ingest(chunks: List[Dict], iteration: int) -> Dict[str, Any]:
    """Send chunks to Ingest API"""
    print_step("INGEST", f"Sending {len(chunks)} chunks to Ingest API (iteration {iteration}/5)...")

    payload = {
        "chunks": chunks,
        "dataset_source": f"pipeline_walkthrough_iter{iteration}",
        "skip_cpesh": True  # Fast mode for benchmarking
    }

    start_time = time.perf_counter()
    try:
        response = requests.post(
            f"{INGEST_API}/ingest",
            json=payload,
            timeout=120
        )
        response.raise_for_status()
        elapsed_ms = (time.perf_counter() - start_time) * 1000

        result = response.json()

        if iteration == 5:
            print_timing("Ingest API call", elapsed_ms, f"client time")
            print_timing("Ingest processing", result["processing_time_ms"], f"server time")
            print(f"   ✅ Successful: {result['successful']}/{result['total_chunks']} chunks")

            # Show per-chunk timing breakdown
            if result["results"] and len(result["results"]) > 0:
                first_result = result["results"][0]
                if "timings_ms" in first_result:
                    print()
                    print(f"   {BOLD}Per-Chunk Timing Breakdown (first chunk):{RESET}")
                    for step, ms in first_result["timings_ms"].items():
                        print_timing(step, ms)
        else:
            print(f"   Warmup iteration {iteration}: {result['successful']} chunks in {elapsed_ms:.1f}ms")

        print()
        return result

    except Exception as e:
        print_error(f"Ingest API failed: {e}")
        if hasattr(e, 'response') and e.response:
            try:
                error_detail = e.response.json()
                print(f"   Error details: {json.dumps(error_detail, indent=2)}")
            except:
                print(f"   Response text: {e.response.text[:500]}")
        sys.exit(1)


def run_pipeline(prompt_text: str, iteration: int, chunker_config: Dict[str, Any]) -> tuple:
    """Run complete pipeline: Text → Chunker → Ingest"""

    # Step 1: Chunk the text
    chunk_result = call_chunker(prompt_text, iteration, chunker_config)
    chunks = chunk_result["chunks"]
    chunking_time = chunk_result["processing_time_ms"]

    # Step 2: Ingest chunks
    ingest_result = call_ingest(chunks, iteration)
    ingest_time = ingest_result["processing_time_ms"]

    # Calculate total pipeline time
    total_time = chunking_time + ingest_time

    return chunk_result, ingest_result, total_time


def generate_summary_report(chunk_result: Dict, ingest_result: Dict, total_time: float):
    """Generate final summary report"""
    print_header("PIPELINE SUMMARY (Iteration 5 - Warmed Up)")

    # Overview
    chunk_count = chunk_result["total_chunks"]
    chunking_time = chunk_result["processing_time_ms"]
    ingest_time = ingest_result["processing_time_ms"]

    print(f"{BOLD}Pipeline Stages:{RESET}")
    print_timing("1. Chunking", chunking_time, f"{chunk_count} chunks created")
    print_timing("2. Ingestion", ingest_time, f"{chunk_count} chunks processed")
    print_timing("TOTAL PIPELINE", total_time, f"end-to-end")
    print()

    # Throughput
    chunks_per_sec = chunk_count / (total_time / 1000) if total_time > 0 else 0
    ms_per_chunk = total_time / chunk_count if chunk_count > 0 else 0
    print(f"{BOLD}Throughput:{RESET}")
    print(f"   📊 {chunks_per_sec:.2f} chunks/second")
    print(f"   📊 {ms_per_chunk:.1f} ms/chunk (average)")
    print()

    # Ingestion breakdown
    if ingest_result["results"] and len(ingest_result["results"]) > 0:
        first_result = ingest_result["results"][0]
        if "timings_ms" in first_result and first_result["timings_ms"]:
            print(f"{BOLD}Ingestion Phase Breakdown (per chunk):{RESET}")
            timings = first_result["timings_ms"]

            # Calculate phase times from individual chunk timings
            # Note: These are amortized times for batch processing
            for phase, ms in sorted(timings.items(), key=lambda x: x[1], reverse=True):
                percentage = (ms / ingest_time * 100) if ingest_time > 0 else 0
                print(f"   {phase:25s}: {ms:6.1f}ms ({percentage:5.1f}% of ingest time)")
            print()

    # Device info
    if ingest_result["results"] and len(ingest_result["results"]) > 0:
        first_result = ingest_result["results"][0]
        if "backends" in first_result:
            backends = first_result["backends"]
            print(f"{BOLD}Backends Used:{RESET}")
            for backend_name, backend_value in backends.items():
                print(f"   {backend_name:15s}: {backend_value}")
            print()

    # Quality metrics
    if ingest_result["results"] and len(ingest_result["results"]) > 0:
        first_result = ingest_result["results"][0]
        if "quality_metrics" in first_result:
            metrics = first_result["quality_metrics"]
            print(f"{BOLD}Quality Metrics (first chunk):{RESET}")
            print(f"   Vector dimension: {metrics.get('vector_dimension', 'N/A')}")
            print(f"   Vector norm: {metrics.get('vector_norm', 0):.4f}")
            print(f"   Text length: {metrics.get('text_length', 0)} chars")
            print(f"   Concept length: {metrics.get('concept_length', 0)} chars")
            print()

    # Success rate
    success_rate = (ingest_result["successful"] / ingest_result["total_chunks"] * 100) if ingest_result["total_chunks"] > 0 else 0
    print(f"{BOLD}Success Rate:{RESET}")
    print(f"   {ingest_result['successful']}/{ingest_result['total_chunks']} chunks ingested successfully ({success_rate:.1f}%)")
    print()


def main():
    parser = argparse.ArgumentParser(description="Complete pipeline walkthrough with timing")
    parser.add_argument(
        "--input",
        type=str,
        default=DEFAULT_PROMPT_FILE,
        help=f"Input prompt file (default: {DEFAULT_PROMPT_FILE})"
    )
    parser.add_argument(
        "--mps",
        action="store_true",
        help="Test with MPS enabled (sets LNSP_FORCE_T5_MPS=1)"
    )
    # Chunking parameters
    parser.add_argument(
        "--mode",
        type=str,
        default="semantic",
        choices=["simple", "semantic", "proposition", "hybrid"],
        help="Chunking mode (default: semantic)"
    )
    parser.add_argument(
        "--breakpoint",
        type=int,
        default=95,
        help="Semantic breakpoint threshold 50-99 (default: 95). Lower=more chunks"
    )
    parser.add_argument(
        "--min-chunk-size",
        type=int,
        default=100,
        help="Minimum chunk size in characters (default: 100)"
    )
    parser.add_argument(
        "--max-chunk-size",
        type=int,
        default=320,
        help="Maximum chunk size in words (default: 320)"
    )
    args = parser.parse_args()

    # Set MPS if requested
    if args.mps:
        os.environ["LNSP_FORCE_T5_MPS"] = "1"
        print_warning("MPS mode enabled (LNSP_FORCE_T5_MPS=1)")
        print_warning("Make sure to restart APIs for this to take effect!")
        print()

    # Build chunker config from args
    chunker_config = {
        "mode": args.mode,
        "breakpoint_threshold": args.breakpoint,
        "min_chunk_size": args.min_chunk_size,
        "max_chunk_size": args.max_chunk_size
    }

    # Banner
    print_header("LNSP Pipeline Walkthrough")
    print(f"   Test file: {args.input}")
    print(f"   Chunking mode: {args.mode}")
    print(f"   Breakpoint: {args.breakpoint} (lower=more chunks)")
    print(f"   Chunk size: {args.min_chunk_size}-{args.max_chunk_size}")
    print(f"   Iterations: 5 (warmup: 4, reporting: 1)")
    print(f"   Cooldown: 500ms between iterations")
    print()

    # Check services
    check_services()

    # Load prompt
    prompt_text = load_prompt(args.input)

    # Run pipeline 5 times
    results = []
    for i in range(1, 6):
        print_header(f"Iteration {i}/5")

        chunk_result, ingest_result, total_time = run_pipeline(prompt_text, i, chunker_config)
        results.append((chunk_result, ingest_result, total_time))

        # Cooldown between iterations (except after last one)
        if i < 5:
            print(f"   {YELLOW}⏸{RESET}  Cooldown: 500ms...\n")
            time.sleep(0.5)

    # Report on 5th iteration (warmed up)
    chunk_result, ingest_result, total_time = results[-1]
    generate_summary_report(chunk_result, ingest_result, total_time)

    print_header("Pipeline Walkthrough Complete!")
    print(f"{GREEN}✓{RESET} All iterations completed successfully")
    print(f"{GREEN}✓{RESET} Final timing from iteration 5 (warmed-up state)")
    print()


if __name__ == "__main__":
    main()
