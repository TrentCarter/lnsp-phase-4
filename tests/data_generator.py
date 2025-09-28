#!/usr/bin/env python3
"""
Test Data Generator

This script provides a function to generate synthetic data points with all
text-based fields populated for validation purposes.
"""

import random
import sys
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Dict, Any

# Add project root to the Python path to allow importing from 'src'
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

# As per the sprint, using the V2 extractors for domain/task/modifier names
from src.tmd_extractor_v2 import DOMAIN_PATTERNS, TASK_PATTERNS, MODIFIER_PATTERNS

# Pre-defined plausible text for CPESH fields for consistency
CONCEPT_EXAMPLES = [
    "The theory of relativity, developed by Albert Einstein, revolutionized our understanding of space, time, and gravity.",
    "Photosynthesis is a process used by plants, algae, and certain bacteria to convert light energy into chemical energy.",
    "The Industrial Revolution was the transition to new manufacturing processes in Europe and the United States, in the period from about 1760 to sometime between 1820 and 1840.",
    "A blockchain is a decentralized, distributed, and oftentimes public, digital ledger consisting of records called blocks."
]

PROBE_EXAMPLES = [
    "Who developed the theory of relativity?",
    "What is the primary function of photosynthesis?",
    "When did the Industrial Revolution take place?",
    "What is a blockchain?"
]

EXPECTED_EXAMPLES = [
    "Albert Einstein.",
    "To convert light energy into chemical energy.",
    "From about 1760 to sometime between 1820 and 1840.",
    "A decentralized, distributed digital ledger."
]


def generate_synthetic_data_points(num_points: int = 1) -> List[Dict[str, Any]]:
    """
    Generates 1 to N data points with all text-based data populated.

    Args:
        num_points: The number of data points to generate.

    Returns:
        A list of dictionaries, where each dictionary is a synthetic data point.
    """
    if num_points < 1:
        raise ValueError("Number of points must be at least 1.")

    data_points = []
    for _ in range(num_points):
        # Choose random TMD codes and get their names
        domain_code = random.choice(list(DOMAIN_PATTERNS.keys()))
        task_code = random.choice(list(TASK_PATTERNS.keys()))
        modifier_code = random.choice(list(MODIFIER_PATTERNS.keys()))

        domain_name = DOMAIN_PATTERNS[domain_code]['name']
        task_name = TASK_PATTERNS[task_code]['name']
        modifier_name = MODIFIER_PATTERNS[modifier_code]['name']

        # Generate timestamps
        now = datetime.now(timezone.utc)
        created_at = (now - timedelta(days=random.randint(1, 365))).isoformat() + "Z"
        last_accessed = (now - timedelta(minutes=random.randint(1, 1440))).isoformat() + "Z"

        # Select plausible CPESH text
        concept = random.choice(CONCEPT_EXAMPLES)
        probe = random.choice(PROBE_EXAMPLES)
        expected = random.choice(EXPECTED_EXAMPLES)

        point = {
            "cpe_id": str(uuid.uuid4()),
            "doc_id": str(uuid.uuid4()),
            "created_at": created_at,
            "last_accessed": last_accessed,
            "access_count": random.randint(0, 1000),
            "cpesh": {
                "concept_text": concept,
                "probe_question": probe,
                "expected_answer": expected,
                "soft_negative": f"A related but incorrect statement about {domain_name}.",
                "hard_negative": f"A completely unrelated statement about {task_name}."
            },
            "tmd": {
                "domain_code": domain_code,
                "task_code": task_code,
                "modifier_code": modifier_code,
                "domain_name": domain_name,
                "task_name": task_name,
                "modifier_name": modifier_name
            },
            "word_count": random.randint(180, 320),
            "tmd_confidence": round(random.uniform(0.75, 0.99), 2)
        }
        data_points.append(point)

    return data_points


if __name__ == '__main__':
    # Example of how to use the function
    generated_data = generate_synthetic_data_points(3)
    import json
    print(json.dumps(generated_data, indent=2))
