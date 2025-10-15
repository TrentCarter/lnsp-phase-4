#!/usr/bin/env python3
"""
Full Pipeline Benchmark - Run complete chain multiple times
Shows warm-up effects and cache performance
"""

import requests
import time
from typing import List, Dict, Any

SERVICES = {
    "chunker": "http://localhost:8001",
    "tmd_router": "http://localhost:8002",
    "gtr_t5": "http://localhost:8767",
    "lvm": "http://localhost:8003",
    "vec2text": "http://localhost:8766"
}

TEST_TEXT = "Photosynthesis is the process by which plants convert sunlight into chemical energy using chlorophyll. This amazing process occurs in specialized organelles called chloroplasts, which are found in plant cells and some algae."


def run_single_pipeline(run_number: int) -> Dict[str, Any]:
    """Run complete pipeline once and collect metrics"""

    print(f"\n{'='*80}")
    print(f"RUN #{run_number}")
    print(f"{'='*80}\n")

    stages = {}
    total_start = time.time()

    # Stage 1: Chunker
    print("Stage 1: Chunker...")
    start = time.time()
    try:
        response = requests.post(
            f"{SERVICES['chunker']}/chunk",
            json={
                "text": TEST_TEXT,
                "mode": "semantic",
                "breakpoint_threshold": 50,
                "min_chunk_size": 10,
                "max_chunk_size": 320
            },
            timeout=10
        )
        latency = (time.time() - start) * 1000

        if response.status_code == 200:
            data = response.json()
            chunks = data.get("chunks", [])
            first_chunk = chunks[0]["text"] if chunks else ""

            stages["chunker"] = {
                "success": True,
                "latency_ms": latency,
                "server_time_ms": data.get("processing_time_ms", 0),
                "num_chunks": len(chunks),
                "first_chunk": first_chunk
            }
            print(f"  ✅ {latency:.1f}ms (server: {data.get('processing_time_ms', 0):.1f}ms)")
            print(f"  → {len(chunks)} chunks\n")
        else:
            stages["chunker"] = {"success": False, "error": f"HTTP {response.status_code}"}
            print(f"  ❌ HTTP {response.status_code}\n")
            return {"stages": stages, "total_ms": (time.time() - total_start) * 1000}

    except Exception as e:
        stages["chunker"] = {"success": False, "error": str(e)}
        print(f"  ❌ {e}\n")
        return {"stages": stages, "total_ms": (time.time() - total_start) * 1000}

    # Stage 2: TMD Router
    print("Stage 2: TMD Router...")
    start = time.time()
    try:
        response = requests.post(
            f"{SERVICES['tmd_router']}/route",
            json={"concept_text": stages["chunker"]["first_chunk"]},
            timeout=15
        )
        latency = (time.time() - start) * 1000

        if response.status_code == 200:
            data = response.json()
            stages["tmd_router"] = {
                "success": True,
                "latency_ms": latency,
                "domain": data.get("domain_name"),
                "domain_code": data.get("domain_code"),
                "lane_model": data.get("lane_model"),
                "lane_port": data.get("lane_port"),
                "cache_hit": data.get("cache_hit")
            }
            cache_status = "🔥 cache hit" if data.get("cache_hit") else "❄️  cache miss"
            print(f"  ✅ {latency:.1f}ms ({cache_status})")
            print(f"  → Domain: {data.get('domain_name')} → {data.get('lane_model')}\n")
        else:
            stages["tmd_router"] = {"success": False, "error": f"HTTP {response.status_code}"}
            print(f"  ❌ HTTP {response.status_code}\n")
            return {"stages": stages, "total_ms": (time.time() - total_start) * 1000}

    except Exception as e:
        stages["tmd_router"] = {"success": False, "error": str(e)}
        print(f"  ❌ {e}\n")
        return {"stages": stages, "total_ms": (time.time() - total_start) * 1000}

    # Stage 3: GTR-T5 Embeddings
    print("Stage 3: GTR-T5 Embeddings...")
    start = time.time()
    try:
        response = requests.post(
            f"{SERVICES['gtr_t5']}/embed",
            json={
                "texts": [stages["chunker"]["first_chunk"]],
                "normalize": True
            },
            timeout=10
        )
        latency = (time.time() - start) * 1000

        if response.status_code == 200:
            data = response.json()
            embeddings = data.get("embeddings", [])
            stages["gtr_t5"] = {
                "success": True,
                "latency_ms": latency,
                "dimension": data.get("dimension"),
                "count": len(embeddings),
                "embedding": embeddings[0] if embeddings else None
            }
            print(f"  ✅ {latency:.1f}ms")
            print(f"  → {data.get('dimension')}D vector\n")
        else:
            stages["gtr_t5"] = {"success": False, "error": f"HTTP {response.status_code}"}
            print(f"  ❌ HTTP {response.status_code}\n")
            return {"stages": stages, "total_ms": (time.time() - total_start) * 1000}

    except Exception as e:
        stages["gtr_t5"] = {"success": False, "error": str(e)}
        print(f"  ❌ {e}\n")
        return {"stages": stages, "total_ms": (time.time() - total_start) * 1000}

    # Stage 4: LVM Inference
    print("Stage 4: LVM Inference...")
    start = time.time()
    try:
        tmd_codes = [
            stages["tmd_router"]["domain_code"],
            stages["tmd_router"].get("task_code", 0),
            stages["tmd_router"].get("modifier_code", 0)
        ]

        response = requests.post(
            f"{SERVICES['lvm']}/infer",
            json={
                "vector_sequence": [stages["gtr_t5"]["embedding"]],
                "tmd_codes": tmd_codes,
                "use_mock": True
            },
            timeout=10
        )
        latency = (time.time() - start) * 1000

        if response.status_code == 200:
            data = response.json()
            stages["lvm"] = {
                "success": True,
                "latency_ms": latency,
                "confidence": data.get("confidence"),
                "is_mock": data.get("is_mock"),
                "predicted_vector": data.get("predicted_vector")
            }
            mode = "mock" if data.get("is_mock") else "real"
            print(f"  ✅ {latency:.1f}ms ({mode} mode)")
            print(f"  → Confidence: {data.get('confidence'):.2f}\n")
        else:
            stages["lvm"] = {"success": False, "error": f"HTTP {response.status_code}"}
            print(f"  ❌ HTTP {response.status_code}\n")
            return {"stages": stages, "total_ms": (time.time() - total_start) * 1000}

    except Exception as e:
        stages["lvm"] = {"success": False, "error": str(e)}
        print(f"  ❌ {e}\n")
        return {"stages": stages, "total_ms": (time.time() - total_start) * 1000}

    # Stage 5: Vec2Text (optional - may not be running)
    print("Stage 5: Vec2Text Decoder...")
    start = time.time()
    try:
        response = requests.post(
            f"{SERVICES['vec2text']}/decode",
            json={
                "vectors": [stages["lvm"]["predicted_vector"]],
                "subscribers": "jxe,ielab",
                "steps": 1
            },
            timeout=30
        )
        latency = (time.time() - start) * 1000

        if response.status_code == 200:
            data = response.json()
            stages["vec2text"] = {
                "success": True,
                "latency_ms": latency,
                "results": data.get("results", [])
            }
            print(f"  ✅ {latency:.1f}ms")
            if data.get("results"):
                result = data["results"][0]
                if "jxe" in result:
                    print(f"  → JXE: {result['jxe'].get('decoded_text', 'N/A')}")
                if "ielab" in result:
                    print(f"  → IELab: {result['ielab'].get('decoded_text', 'N/A')}")
            print()
        else:
            stages["vec2text"] = {"success": False, "error": f"HTTP {response.status_code}"}
            print(f"  ⚠️  Not running (HTTP {response.status_code})\n")

    except Exception as e:
        stages["vec2text"] = {"success": False, "error": str(e)}
        print(f"  ⚠️  Not running ({str(e)[:50]}...)\n")

    total_ms = (time.time() - total_start) * 1000

    return {
        "stages": stages,
        "total_ms": total_ms
    }


def main():
    print("\n" + "="*80)
    print("FULL PIPELINE BENCHMARK - 3 CONSECUTIVE RUNS")
    print("="*80)
    print("\nThis will run the complete pipeline 3 times to observe:")
    print("  1. Cold start performance (first run)")
    print("  2. Warm cache performance (subsequent runs)")
    print("  3. Cache hit rates and speedup\n")

    # Check services
    print("Checking services...")
    all_ok = True
    for name, url in SERVICES.items():
        try:
            response = requests.get(f"{url}/health", timeout=2)
            status = "✅" if response.status_code == 200 else f"❌ HTTP {response.status_code}"
        except:
            status = "❌ Down"
            if name in ["chunker", "tmd_router", "gtr_t5", "lvm"]:
                all_ok = False

        print(f"  {name:15} {status}")

    if not all_ok:
        print("\n❌ Required services not running. Start them first.")
        return

    print()

    # Run 3 times
    results = []
    for i in range(1, 4):
        result = run_single_pipeline(i)
        results.append(result)
        time.sleep(0.5)  # Brief pause between runs

    # Summary
    print("\n" + "="*80)
    print("SUMMARY - LATENCY COMPARISON")
    print("="*80 + "\n")

    # Table header
    print(f"{'Stage':<20} {'Run 1':<12} {'Run 2':<12} {'Run 3':<12} {'Speedup':<12}")
    print("-" * 80)

    # Compare each stage
    stage_names = ["chunker", "tmd_router", "gtr_t5", "lvm", "vec2text"]
    for stage_name in stage_names:
        latencies = []
        for result in results:
            stage = result["stages"].get(stage_name, {})
            if stage.get("success"):
                latencies.append(stage.get("latency_ms", 0))
            else:
                latencies.append(None)

        if latencies[0] is not None:
            run1 = f"{latencies[0]:.1f}ms"
            run2 = f"{latencies[1]:.1f}ms" if latencies[1] is not None else "N/A"
            run3 = f"{latencies[2]:.1f}ms" if latencies[2] is not None else "N/A"

            if latencies[1] is not None and latencies[0] > 0:
                speedup = f"{latencies[0] / latencies[1]:.1f}x"
            else:
                speedup = "N/A"

            print(f"{stage_name:<20} {run1:<12} {run2:<12} {run3:<12} {speedup:<12}")

    print("-" * 80)

    # Total
    total_1 = results[0]["total_ms"]
    total_2 = results[1]["total_ms"]
    total_3 = results[2]["total_ms"]
    speedup_total = total_1 / total_2 if total_2 > 0 else 0

    print(f"{'TOTAL':<20} {total_1:.1f}ms      {total_2:.1f}ms      {total_3:.1f}ms      {speedup_total:.1f}x")

    # Cache analysis
    print("\n" + "="*80)
    print("CACHE ANALYSIS")
    print("="*80 + "\n")

    for i, result in enumerate(results, 1):
        tmd = result["stages"].get("tmd_router", {})
        if tmd.get("success"):
            cache_status = "🔥 HIT" if tmd.get("cache_hit") else "❄️  MISS"
            print(f"Run {i} - TMD Router: {cache_status}")

    print()


if __name__ == "__main__":
    main()
