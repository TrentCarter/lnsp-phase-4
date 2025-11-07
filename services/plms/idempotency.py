"""
PLMS Idempotency - Redis-backed with Replay Headers

Replaces in-memory cache with Redis for multi-pod safety.

Key design:
- Hash: method + path + body_sha + user_id
- TTL: 24 hours
- Replay header: Idempotent-Replay: true
"""

import hashlib
import json
from typing import Optional, Dict, Any
from datetime import timedelta

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False


class IdempotencyCache:
    """Redis-backed idempotency cache with 24h TTL."""

    def __init__(self, redis_url: str = "redis://localhost:6379/0"):
        """
        Initialize Redis client.

        Args:
            redis_url: Redis connection URL (default: localhost:6379/0)
        """
        if not REDIS_AVAILABLE:
            raise ImportError("Redis not installed. Run: pip install redis")

        self.redis_client = redis.from_url(redis_url, decode_responses=True)
        self.ttl = int(timedelta(hours=24).total_seconds())

    def compute_key(self, method: str, path: str, body: Dict, user_id: str) -> str:
        """
        Compute idempotency key from request components.

        Args:
            method: HTTP method (POST, PUT, etc.)
            path: Request path (e.g., /api/projects/42/start)
            body: Request body dict
            user_id: User identifier

        Returns:
            Idempotency key: "idem:<hash>"
        """
        # Serialize body to stable JSON (sorted keys)
        body_json = json.dumps(body, sort_keys=True)
        body_sha = hashlib.sha256(body_json.encode()).hexdigest()[:16]

        # Combine method + path + body_sha + user_id
        key_components = f"{method}:{path}:{body_sha}:{user_id}"
        key_hash = hashlib.sha256(key_components.encode()).hexdigest()[:16]

        return f"idem:{key_hash}"

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        """
        Get cached response for idempotency key.

        Args:
            key: Idempotency key

        Returns:
            Cached response dict or None if not found
        """
        cached = self.redis_client.get(key)
        if cached:
            return json.loads(cached)
        return None

    def set(self, key: str, response: Dict[str, Any]):
        """
        Store response in Redis with TTL.

        Args:
            key: Idempotency key
            response: Response dict to cache
        """
        self.redis_client.setex(
            key,
            self.ttl,
            json.dumps(response)
        )

    def check_and_store(
        self,
        method: str,
        path: str,
        body: Dict,
        user_id: str,
        response: Optional[Dict] = None
    ) -> Optional[Dict]:
        """
        Check cache and optionally store new response.

        Args:
            method: HTTP method
            path: Request path
            body: Request body
            user_id: User identifier
            response: Response to store (if provided)

        Returns:
            Cached response if exists, None otherwise
        """
        key = self.compute_key(method, path, body, user_id)

        # Check cache first
        cached = self.get(key)
        if cached:
            return cached

        # Store new response if provided
        if response:
            self.set(key, response)

        return None


# Global instance (lazy initialization)
_cache: Optional[IdempotencyCache] = None


def get_cache(redis_url: str = "redis://localhost:6379/0") -> IdempotencyCache:
    """
    Get global idempotency cache instance.

    Args:
        redis_url: Redis connection URL

    Returns:
        IdempotencyCache instance
    """
    global _cache
    if _cache is None:
        _cache = IdempotencyCache(redis_url)
    return _cache


# Fallback in-memory cache for dev/test (not production-safe!)
class InMemoryIdempotencyCache:
    """
    In-memory fallback (NOT production-safe).

    Only use for local dev/testing when Redis unavailable.
    """

    def __init__(self):
        self._cache: Dict[str, Dict] = {}

    def compute_key(self, method: str, path: str, body: Dict, user_id: str) -> str:
        """Compute idempotency key (same as Redis version)."""
        body_json = json.dumps(body, sort_keys=True)
        body_sha = hashlib.sha256(body_json.encode()).hexdigest()[:16]
        key_components = f"{method}:{path}:{body_sha}:{user_id}"
        key_hash = hashlib.sha256(key_components.encode()).hexdigest()[:16]
        return f"idem:{key_hash}"

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        """Get cached response."""
        return self._cache.get(key)

    def set(self, key: str, response: Dict[str, Any]):
        """Store response (no TTL in memory)."""
        self._cache[key] = response

    def check_and_store(
        self,
        method: str,
        path: str,
        body: Dict,
        user_id: str,
        response: Optional[Dict] = None
    ) -> Optional[Dict]:
        """Check cache and optionally store."""
        key = self.compute_key(method, path, body, user_id)
        cached = self.get(key)
        if cached:
            return cached
        if response:
            self.set(key, response)
        return None
