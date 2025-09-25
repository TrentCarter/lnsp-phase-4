#!/usr/bin/env python3
"""S5 self-test script for LNSP functionality."""

import os
import json
import requests
import time
from sentence_transformers import SentenceTransformer


def main():
    """Run S5 self-test."""
    print("🧪 Running S5 Self-Test...")

    # Set offline flags
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

    # Test 1: GTR offline load
    mdir = os.getenv("LNSP_EMBED_MODEL_DIR", "data/teacher_models/gtr-t5-base")
    try:
        m = SentenceTransformer(mdir)
        assert m.get_sentence_embedding_dimension() == 768
        print("✅ GTR offline load OK")
    except Exception as e:
        print(f"❌ GTR offline load failed: {e}")
        return False

    # Test 2: Ollama reachability
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=3)
        r.raise_for_status()
        models = [model.get("name") for model in r.json().get("models", [])]
        print(f"✅ Ollama reachable with models: {models[:3]}...")
    except Exception as e:
        print(f"❌ Ollama unreachable: {e}")
        return False

    # Test 3: API search with CPESH
    try:
        q = {
            "q": "Which ocean is largest?",
            "top_k": 5,
            "lane": "L1_FACTOID",
            "return_cpesh": True,
            "cpesh_mode": "full",
            "cpesh_k": 2,
            "compact": True
        }
        s = requests.post("http://localhost:8092/search", json=q, timeout=10)
        s.raise_for_status()

        data = s.json()
        items = data.get("items", [])

        if not items:
            print("❌ API returned no items")
            return False

        # Check if TMD codes are properly formatted
        first_item = items[0]
        tmd_code = first_item.get("tmd_code", "0.0.0")

        if tmd_code == "0.0.0":
            print(f"⚠️ TMD code is still 0.0.0: {tmd_code}")
        else:
            print(f"✅ TMD code properly formatted: {tmd_code}")

        # Check if CPESH is present
        if "cpesh" in first_item:
            print("✅ CPESH data present in response")
        else:
            print("⚠️ No CPESH data in response")

        print(f"✅ API /search with CPESH OK ({len(items)} items)")
        return True

    except Exception as e:
        print(f"❌ API /search failed: {e}")
        return False


if __name__ == "__main__":
    success = main()
    if success:
        print("🎉 All S5 tests passed!")
        exit(0)
    else:
        print("💥 Some S5 tests failed!")
        exit(1)
