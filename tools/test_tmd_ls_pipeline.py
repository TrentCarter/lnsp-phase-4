#!/usr/bin/env python3
"""
TMD-LS Pipeline Test

Demonstrates the complete TMD-LS (Task-Modifier-Domain Lane Specialist)
pipeline including routing, lane selection, and Echo Loop validation.
"""
import sys
import os
from pathlib import Path

# Add src to path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.tmd_router import route_concept, get_cache_stats, get_lane_prompt
from src.echo_loop_validator import validate_cpesh, get_validator
import json


def test_tmd_router():
    """Test TMD router with sample concepts."""
    print("=" * 80)
    print("TEST 1: TMD ROUTER")
    print("=" * 80)
    print()

    test_concepts = [
        "oxidoreductase activity",
        "Python programming language",
        "World War II",
        "cardiac arrest",
        "machine learning algorithm",
        "quantum entanglement",
        "supply and demand",
        "photosynthesis"
    ]

    print(f"Testing {len(test_concepts)} concepts...\n")

    results = []
    for concept in test_concepts:
        print(f"🔍 Routing: '{concept}'")
        result = route_concept(concept)

        print(f"   ├─ Domain: {result['domain_name']} (code: {result['domain_code']})")
        print(f"   ├─ Lane Model: {result['lane_model']} (port {result['lane_port']})")
        print(f"   ├─ Fallback: {result['is_fallback']}")
        print(f"   └─ Cache Hit: {result['cache_hit']}")
        print()

        results.append(result)

    # Show cache stats
    print("📊 Cache Statistics:")
    stats = get_cache_stats()
    print(f"   ├─ Size: {stats['size']} / {stats['maxsize']}")
    print(f"   ├─ Hits: {stats['hits']}")
    print(f"   ├─ Misses: {stats['misses']}")
    print(f"   └─ Hit Rate: {stats['hit_rate']:.1%}")
    print()

    return results


def test_echo_loop_validator():
    """Test Echo Loop validator with sample CPESH structures."""
    print("=" * 80)
    print("TEST 2: ECHO LOOP VALIDATOR")
    print("=" * 80)
    print()

    # Test CPESH structures (good and bad)
    test_cases = [
        {
            "concept_text": "photosynthesis",
            "cpesh": {
                "concept": "photosynthesis",
                "probe": "What is the process by which plants convert sunlight into energy?",
                "expected": "Photosynthesis converts light energy into chemical energy stored in glucose",
                "soft_negatives": ["cellular respiration", "chemosynthesis", "glycolysis"],
                "hard_negatives": ["transpiration", "germination", "pollination"]
            },
            "expected_quality": "high"
        },
        {
            "concept_text": "machine learning",
            "cpesh": {
                "concept": "machine learning",
                "probe": "What type of AI allows systems to learn from data?",
                "expected": "Machine learning enables systems to improve performance through experience",
                "soft_negatives": ["deep learning", "neural networks", "supervised learning"],
                "hard_negatives": ["data mining", "statistical analysis", "business intelligence"]
            },
            "expected_quality": "high"
        },
        {
            "concept_text": "quantum entanglement",
            "cpesh": {
                "concept": "quantum mechanics",  # Drift! Wrong concept
                "probe": "What is a weird quantum thing?",  # Poor probe
                "expected": "Stuff happens",  # Poor expected answer
                "soft_negatives": ["physics", "science"],  # Poor negatives
                "hard_negatives": ["chemistry", "biology"]
            },
            "expected_quality": "low"
        }
    ]

    validator = get_validator(threshold=0.82)

    for i, test in enumerate(test_cases, 1):
        concept = test["concept_text"]
        cpesh = test["cpesh"]
        expected_quality = test["expected_quality"]

        print(f"Test Case {i}: '{concept}' (expected: {expected_quality} quality)")
        print(f"  CPESH Concept: {cpesh['concept']}")
        print(f"  CPESH Expected: {cpesh['expected']}")
        print()

        result = validator.validate_cpesh(concept, cpesh)

        status = "✅" if result['valid'] else "❌"
        print(f"{status} Validation Result:")
        print(f"   ├─ Valid: {result['valid']}")
        print(f"   ├─ Cosine Similarity: {result['cosine_similarity']:.4f}")
        print(f"   ├─ Threshold: {result['threshold']:.4f}")
        print(f"   ├─ Action: {result['action']}")
        print(f"   └─ Reason: {result['reason']}")
        print()

    # Show validation stats
    print("📊 Validation Statistics:")
    stats = validator.stats()
    print(f"   ├─ Total Validations: {stats['validations']}")
    print(f"   ├─ Accepts: {stats['accepts']} ({stats['accept_rate']:.1%})")
    print(f"   ├─ Re-queues: {stats['requeues']} ({stats['requeue_rate']:.1%})")
    print(f"   └─ Escalates: {stats['escalates']} ({stats['escalate_rate']:.1%})")
    print()


def test_integrated_pipeline():
    """Test TMD router + Echo Loop validator together."""
    print("=" * 80)
    print("TEST 3: INTEGRATED PIPELINE (Router + Echo Loop)")
    print("=" * 80)
    print()

    # Example: Route concept and validate a generated CPESH
    concept_text = "cellular respiration"

    print(f"📍 Step 1: Route concept '{concept_text}'")
    route_result = route_concept(concept_text)

    print(f"   ├─ Domain: {route_result['domain_name']} (code: {route_result['domain_code']})")
    print(f"   ├─ Lane Model: {route_result['lane_model']}")
    print(f"   ├─ Port: {route_result['lane_port']}")
    print(f"   └─ Specialist Prompt: {route_result['specialist_prompt_id']}")
    print()

    # Get the specialist prompt template
    prompt_template = get_lane_prompt(route_result['specialist_prompt_id'])
    if prompt_template:
        print(f"📝 Step 2: Lane Specialist Prompt Preview")
        print(f"   Template length: {len(prompt_template)} chars")
        print(f"   First 200 chars: {prompt_template[:200]}...")
        print()

    # Simulate generated CPESH (in real system, this would come from lane specialist LLM)
    simulated_cpesh = {
        "concept": "cellular respiration",
        "probe": "What is the process cells use to convert glucose into ATP?",
        "expected": "Cellular respiration breaks down glucose to produce ATP energy",
        "soft_negatives": ["photosynthesis", "fermentation", "glycolysis"],
        "hard_negatives": ["digestion", "metabolism", "oxidation"]
    }

    print(f"🤖 Step 3: Simulated CPESH Generation")
    print(f"   (In production, this would come from {route_result['lane_model']})")
    print(f"   Concept: {simulated_cpesh['concept']}")
    print(f"   Expected: {simulated_cpesh['expected']}")
    print()

    # Validate with Echo Loop
    print(f"🔄 Step 4: Echo Loop Validation")
    validation_result = validate_cpesh(concept_text, simulated_cpesh)

    status = "✅ PASS" if validation_result['valid'] else "❌ FAIL"
    print(f"   {status}")
    print(f"   ├─ Cosine Similarity: {validation_result['cosine_similarity']:.4f}")
    print(f"   ├─ Threshold: {validation_result['threshold']:.4f}")
    print(f"   ├─ Action: {validation_result['action']}")
    print(f"   └─ Reason: {validation_result['reason']}")
    print()

    # Decision logic
    print(f"💡 Step 5: Decision")
    if validation_result['action'] == 'accept':
        print(f"   ✅ CPESH accepted - store in database")
    elif validation_result['action'] == 're_queue':
        print(f"   🔄 Re-queue for improvement")
    elif validation_result['action'] == 'escalate':
        print(f"   ⚠️  Escalate to fallback model (Llama 3.1:8b)")
    print()


def show_architecture_summary():
    """Display TMD-LS architecture summary."""
    print("=" * 80)
    print("TMD-LS ARCHITECTURE SUMMARY")
    print("=" * 80)
    print()

    print("📋 Components Implemented:")
    print("   ✅ TMD Router (src/tmd_router.py)")
    print("      - Extracts Domain/Task/Modifier codes from concepts")
    print("      - Routes to appropriate lane specialist")
    print("      - LRU cache for TMD extractions (10k items)")
    print("      - Fallback to Llama 3.1 if primary model unavailable")
    print()

    print("   ✅ Echo Loop Validator (src/echo_loop_validator.py)")
    print("      - Validates CPESH quality via cosine similarity")
    print("      - Threshold: 0.82 (configurable)")
    print("      - Actions: accept / re_queue / escalate")
    print("      - Tracks validation statistics")
    print()

    print("   ✅ Prompt Configuration (configs/llm_prompts/llm_prompts_master.json)")
    print("      - TMD router prompt")
    print("      - 16 lane specialist prompts (one per domain)")
    print("      - Output smoothing prompt")
    print("      - Ontology ingestion prompt")
    print("      - RL ingestion prompt")
    print()

    print("📊 Lane Specialist Assignments:")
    print("   Port 11434 (llama3.1:8b):  Philosophy, Law")
    print("   Port 11435 (tinyllama:1.1b): Science, Engineering, History, Economics, Environment")
    print("   Port 11436 (phi3:mini): Mathematics, Medicine, Art, Politics, Software")
    print("   Port 11437 (granite3-moe:1b): Technology, Psychology, Literature, Education")
    print()

    print("🎯 Performance Targets:")
    print("   Throughput: 1,100 tok/s (266% improvement vs baseline)")
    print("   Echo validation pass rate: ≥ 93%")
    print("   TMD cache hit rate: ≥ 80%")
    print("   Cost reduction: -72% vs monolithic 7B model")
    print()


def main():
    """Run all tests."""
    print()
    show_architecture_summary()
    print()

    # Check if Llama is running
    print("🔧 Checking Prerequisites...")
    try:
        from src.llm.local_llama_client import call_local_llama_simple
        call_local_llama_simple("test", timeout=5)
        print("   ✅ Ollama LLM is running")
    except Exception as e:
        print(f"   ⚠️  Warning: Ollama may not be running: {e}")
        print("   Tip: Start with 'ollama serve' in another terminal")
    print()

    try:
        # Run tests
        test_tmd_router()
        test_echo_loop_validator()
        test_integrated_pipeline()

        print("=" * 80)
        print("✅ ALL TESTS COMPLETED")
        print("=" * 80)
        print()
        print("Next Steps:")
        print("  1. Review implementation document: docs/PRDs/PRD_TMD-LS_Implementation.md")
        print("  2. Test with real lane specialists (requires models running on ports)")
        print("  3. Integrate with P5 (LLM Interrogation) for production use")
        print("  4. Build 784D vector fusion (768D + 16D TMD)")
        print()

    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
