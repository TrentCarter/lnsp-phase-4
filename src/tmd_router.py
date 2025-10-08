#!/usr/bin/env python3
"""
TMD Router Service

Routes concepts to appropriate lane specialist based on Domain/Task/Modifier codes.
Implements caching, load balancing, and fallback logic.
"""
import json
import os
from pathlib import Path
from typing import Dict, Optional, Tuple
from functools import lru_cache
import sys

# Add src to path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.llm_tmd_extractor import extract_tmd_with_llm
import requests
from hashlib import md5


# ============================================================================
# Configuration Loading
# ============================================================================

@lru_cache(maxsize=1)
def load_prompt_config() -> Dict:
    """Load prompts from configs/llm_prompts/llm_prompts_master.json"""
    config_path = ROOT / "configs" / "llm_prompts" / "llm_prompts_master.json"

    if not config_path.exists():
        raise FileNotFoundError(f"Prompt config not found: {config_path}")

    with open(config_path, 'r') as f:
        return json.load(f)


# ============================================================================
# TMD Extraction Cache
# ============================================================================

class TMDCache:
    """LRU cache for TMD extractions to avoid re-computation."""

    def __init__(self, maxsize: int = 10000):
        self._cache = {}
        self._maxsize = maxsize
        self._lru_order = []
        self._hits = 0
        self._misses = 0

    def _make_key(self, concept_text: str) -> str:
        """Create cache key from concept text."""
        # Use MD5 hash for consistent short keys
        return md5(concept_text.strip().lower().encode('utf-8')).hexdigest()

    def get(self, concept_text: str) -> Optional[Dict]:
        """Get cached TMD extraction."""
        key = self._make_key(concept_text)
        if key in self._cache:
            # Move to end (most recently used)
            self._lru_order.remove(key)
            self._lru_order.append(key)
            self._hits += 1
            return self._cache[key]
        self._misses += 1
        return None

    def put(self, concept_text: str, tmd_dict: Dict) -> None:
        """Cache TMD extraction result."""
        key = self._make_key(concept_text)

        # Remove least recently used items if cache is full
        while len(self._cache) >= self._maxsize and key not in self._cache:
            oldest_key = self._lru_order.pop(0)
            del self._cache[oldest_key]

        # Add/update the cache
        if key in self._cache:
            self._lru_order.remove(key)
        self._cache[key] = tmd_dict
        self._lru_order.append(key)

    def stats(self) -> Dict:
        """Get cache statistics."""
        total = self._hits + self._misses
        hit_rate = self._hits / total if total > 0 else 0.0
        return {
            'size': len(self._cache),
            'maxsize': self._maxsize,
            'hits': self._hits,
            'misses': self._misses,
            'hit_rate': hit_rate
        }

    def clear(self) -> None:
        """Clear the cache."""
        self._cache.clear()
        self._lru_order.clear()
        self._hits = 0
        self._misses = 0


# Global cache instance
_tmd_cache = TMDCache(maxsize=10000)


# ============================================================================
# Model Availability Check
# ============================================================================

def check_model_availability(port: int, timeout: float = 2.0) -> bool:
    """
    Check if model is running and accepting requests.

    Args:
        port: Port number to check
        timeout: Request timeout in seconds

    Returns:
        True if model is available, False otherwise
    """
    try:
        response = requests.get(
            f"http://localhost:{port}/api/tags",
            timeout=timeout
        )
        return response.status_code == 200
    except Exception:
        return False


# ============================================================================
# Lane Selection Logic
# ============================================================================

def select_lane(
    domain_code: int,
    task_code: int,
    modifier_code: int,
    allow_fallback: bool = True
) -> Dict:
    """
    Select lane specialist model based on TMD codes.

    Args:
        domain_code: Domain code (0-15)
        task_code: Task code (0-31)
        modifier_code: Modifier code (0-63)
        allow_fallback: Whether to fallback to Llama 3.1 if primary unavailable

    Returns:
        {
            'model': str (e.g., 'tinyllama:1.1b'),
            'port': int,
            'specialist_prompt_id': str,
            'temperature': float,
            'max_tokens': int,
            'domain_name': str,
            'is_fallback': bool
        }
    """
    config = load_prompt_config()
    metadata = config.get('metadata', {})
    model_assignments = metadata.get('model_assignments', {})

    # Domain name lookup
    domain_names = [
        "Science", "Mathematics", "Technology", "Engineering", "Medicine",
        "Psychology", "Philosophy", "History", "Literature", "Art",
        "Economics", "Law", "Politics", "Education", "Environment", "Software"
    ]
    domain_name = domain_names[domain_code] if 0 <= domain_code < len(domain_names) else "Unknown"

    # Routing table (matches llm_prompts_master.json metadata)
    routing_table = {
        0: ("tinyllama:1.1b", 11435),    # Science
        1: ("phi3:mini", 11436),          # Mathematics
        2: ("granite3-moe:1b", 11437),    # Technology
        3: ("tinyllama:1.1b", 11435),    # Engineering
        4: ("phi3:mini", 11436),          # Medicine
        5: ("granite3-moe:1b", 11437),    # Psychology
        6: ("llama3.1:8b", 11434),        # Philosophy
        7: ("tinyllama:1.1b", 11435),    # History
        8: ("granite3-moe:1b", 11437),    # Literature
        9: ("phi3:mini", 11436),          # Art
        10: ("tinyllama:1.1b", 11435),   # Economics
        11: ("llama3.1:8b", 11434),       # Law
        12: ("phi3:mini", 11436),         # Politics
        13: ("granite3-moe:1b", 11437),   # Education
        14: ("tinyllama:1.1b", 11435),   # Environment
        15: ("phi3:mini", 11436),         # Software
    }

    # Get primary model assignment
    primary_model, primary_port = routing_table.get(
        domain_code,
        ("llama3.1:8b", 11434)  # Default fallback
    )

    # Check primary model availability
    is_available = check_model_availability(primary_port)
    is_fallback = False

    if not is_available and allow_fallback:
        # Fallback to Llama 3.1:8b
        primary_model = "llama3.1:8b"
        primary_port = 11434
        is_fallback = True

    # Get specialist prompt ID
    specialist_prompt_id = f"lane_specialist_{domain_name.lower()}"

    # Get temperature and max_tokens from prompt config
    prompts = config.get('prompts', {})
    prompt_config = prompts.get(specialist_prompt_id, {})
    temperature = prompt_config.get('temperature', 0.3)
    max_tokens = prompt_config.get('max_tokens', 200)

    return {
        'model': primary_model,
        'port': primary_port,
        'specialist_prompt_id': specialist_prompt_id,
        'temperature': temperature,
        'max_tokens': max_tokens,
        'domain_name': domain_name,
        'domain_code': domain_code,
        'task_code': task_code,
        'modifier_code': modifier_code,
        'is_fallback': is_fallback
    }


# ============================================================================
# Main Router Function
# ============================================================================

def route_concept(
    concept_text: str,
    use_cache: bool = True,
    llm_endpoint: Optional[str] = None,
    llm_model: str = "llama3.1:8b"
) -> Dict:
    """
    Extract TMD codes and route to appropriate lane specialist.

    Args:
        concept_text: Concept to route
        use_cache: Whether to use TMD cache (default True)
        llm_endpoint: LLM endpoint for TMD extraction (default: env LNSP_LLM_ENDPOINT)
        llm_model: LLM model for TMD extraction (default: llama3.1:8b)

    Returns:
        {
            'concept_text': str,
            'domain_code': int (0-15),
            'task_code': int (0-31),
            'modifier_code': int (0-63),
            'domain_name': str,
            'lane_model': str,
            'lane_port': int,
            'specialist_prompt_id': str,
            'temperature': float,
            'max_tokens': int,
            'is_fallback': bool,
            'cache_hit': bool
        }
    """
    # Try to get from cache
    cache_hit = False
    if use_cache:
        cached_tmd = _tmd_cache.get(concept_text)
        if cached_tmd is not None:
            tmd_dict = cached_tmd
            cache_hit = True
        else:
            # Extract TMD codes using LLM
            tmd_dict = extract_tmd_with_llm(
                text=concept_text,
                llm_endpoint=llm_endpoint,
                llm_model=llm_model
            )
            # Cache the result
            _tmd_cache.put(concept_text, tmd_dict)
    else:
        # Extract without caching
        tmd_dict = extract_tmd_with_llm(
            text=concept_text,
            llm_endpoint=llm_endpoint,
            llm_model=llm_model
        )

    # Select lane specialist
    lane_info = select_lane(
        domain_code=tmd_dict['domain_code'],
        task_code=tmd_dict['task_code'],
        modifier_code=tmd_dict['modifier_code']
    )

    # Combine TMD and lane info
    result = {
        'concept_text': concept_text,
        'domain_code': tmd_dict['domain_code'],
        'task_code': tmd_dict['task_code'],
        'modifier_code': tmd_dict['modifier_code'],
        'domain_name': lane_info['domain_name'],
        'lane_model': lane_info['model'],
        'lane_port': lane_info['port'],
        'specialist_prompt_id': lane_info['specialist_prompt_id'],
        'temperature': lane_info['temperature'],
        'max_tokens': lane_info['max_tokens'],
        'is_fallback': lane_info['is_fallback'],
        'cache_hit': cache_hit
    }

    return result


# ============================================================================
# Utility Functions
# ============================================================================

def get_cache_stats() -> Dict:
    """Get TMD cache statistics."""
    return _tmd_cache.stats()


def clear_cache() -> None:
    """Clear TMD cache."""
    _tmd_cache.clear()


def get_lane_prompt(specialist_prompt_id: str) -> Optional[str]:
    """
    Get prompt template for a lane specialist.

    Args:
        specialist_prompt_id: Prompt ID (e.g., 'lane_specialist_science')

    Returns:
        Prompt template string, or None if not found
    """
    config = load_prompt_config()
    prompts = config.get('prompts', {})
    prompt_config = prompts.get(specialist_prompt_id, {})
    return prompt_config.get('template')


# ============================================================================
# CLI for Testing
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="TMD Router - Route concepts to lane specialists")
    parser.add_argument('concept', nargs='*', help='Concept text to route')
    parser.add_argument('--no-cache', action='store_true', help='Disable TMD cache')
    parser.add_argument('--stats', action='store_true', help='Show cache statistics')
    parser.add_argument('--clear-cache', action='store_true', help='Clear TMD cache')

    args = parser.parse_args()

    if args.clear_cache:
        clear_cache()
        print("✅ TMD cache cleared")
        sys.exit(0)

    if args.stats:
        stats = get_cache_stats()
        print("📊 TMD Cache Statistics")
        print(f"  Size: {stats['size']} / {stats['maxsize']}")
        print(f"  Hits: {stats['hits']}")
        print(f"  Misses: {stats['misses']}")
        print(f"  Hit Rate: {stats['hit_rate']:.1%}")
        sys.exit(0)

    if not args.concept:
        parser.print_help()
        sys.exit(1)

    # Route concept
    concept_text = ' '.join(args.concept)
    use_cache = not args.no_cache

    print(f"🔍 Routing concept: '{concept_text}'")
    print()

    result = route_concept(concept_text, use_cache=use_cache)

    print("✅ Routing Result")
    print(f"  Concept: {result['concept_text']}")
    print(f"  Domain: {result['domain_name']} (code: {result['domain_code']})")
    print(f"  Task: code {result['task_code']}")
    print(f"  Modifier: code {result['modifier_code']}")
    print()
    print(f"  Lane Model: {result['lane_model']}")
    print(f"  Lane Port: {result['lane_port']}")
    print(f"  Specialist Prompt: {result['specialist_prompt_id']}")
    print(f"  Temperature: {result['temperature']}")
    print(f"  Max Tokens: {result['max_tokens']}")
    print()
    print(f"  Is Fallback: {result['is_fallback']}")
    print(f"  Cache Hit: {result['cache_hit']}")
