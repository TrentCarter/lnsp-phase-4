from functools import lru_cache
from typing import Tuple, Any, List
from ..schemas import SearchItem, Lane

@lru_cache(maxsize=2048)
def cached_search_key(lane: str, q: str, top_k: int) -> Tuple[str, str, int]:
    """Create a cache key for search requests."""
    return (lane, q.strip().lower(), top_k)

class SearchCache:
    """Simple LRU cache for search results."""

    def __init__(self, maxsize: int = 2048):
        self._cache = {}
        self._maxsize = maxsize
        self._lru_order = []

    def get(self, lane: str, query: str, top_k: int) -> List[SearchItem] | None:
        """Get cached search results."""
        key = cached_search_key(lane, query, top_k)
        if key in self._cache:
            # Move to end (most recently used)
            self._lru_order.remove(key)
            self._lru_order.append(key)
            return self._cache[key]
        return None

    def put(self, lane: str, query: str, top_k: int, items: List[SearchItem]) -> None:
        """Cache search results."""
        key = cached_search_key(lane, query, top_k)

        # Remove least recently used items if cache is full
        while len(self._cache) >= self._maxsize and key not in self._cache:
            oldest_key = self._lru_order.pop(0)
            del self._cache[oldest_key]

        # Add/update the cache
        if key in self._cache:
            self._lru_order.remove(key)
        self._cache[key] = items
        self._lru_order.append(key)

    def clear(self) -> None:
        """Clear the cache."""
        self._cache.clear()
        self._lru_order.clear()

    def size(self) -> int:
        """Get current cache size."""
        return len(self._cache)

# Global cache instance
_search_cache = SearchCache(maxsize=2048)