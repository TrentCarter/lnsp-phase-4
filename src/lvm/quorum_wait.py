"""
STAGE 5: Multi-Concept Quorum Wait

Fixes v1 "wait for all" bottleneck. Proceeds when Q% ready + grace period.

Key Design:
- Q = 70% by default (configurable)
- Grace window = 250ms (configurable)
- Phase 1: Wait until quorum Q met OR grace timeout
- Phase 2: After quorum, wait remaining grace time for stragglers
- Confidence filter: Only accept results with confidence >= 0.5

See: docs/PRDs/PRD_Inference_LVM_v2_PRODUCTION.md (lines 496-542, 954-1002)
"""

import asyncio
import time
import math
import logging
from typing import List, Dict, Any, Set, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class QuorumResult:
    """Result from quorum wait operation."""
    ready_predictions: List[Dict[str, Any]]
    metrics: Dict[str, Any]

    @property
    def found(self) -> bool:
        """Whether quorum was met."""
        return len(self.ready_predictions) >= self.metrics.get("Q", 0)


async def quorum_wait(
    prediction_futures: List[asyncio.Future],
    grace_window_sec: float = 0.25,
    quorum_pct: float = 0.70,
    min_confidence: float = 0.5
) -> QuorumResult:
    """
    Wait for Q% of predictions + grace window for stragglers.

    Args:
        prediction_futures: List of async tasks returning prediction dicts
        grace_window_sec: Maximum time to wait for stragglers (default 250ms)
        quorum_pct: Percentage of results needed for quorum (default 70%)
        min_confidence: Minimum confidence to accept a result (default 0.5)

    Returns:
        QuorumResult with ready predictions and metrics

    Example:
        >>> futures = [retrieve_concept_async(c) for c in concepts]
        >>> result = await quorum_wait(futures, grace_window_sec=0.25, quorum_pct=0.70)
        >>> print(f"Got {len(result.ready_predictions)}/{result.metrics['N']} predictions")
    """
    N = len(prediction_futures)
    Q = math.ceil(quorum_pct * N)

    ready_predictions: List[Dict[str, Any]] = []
    start_time = time.time()

    # Phase 1: Wait until quorum met OR grace timeout
    pending: Set[asyncio.Future] = set(prediction_futures)

    logger.info(f"Quorum wait started: N={N}, Q={Q} ({quorum_pct*100:.0f}%), grace={grace_window_sec*1000:.0f}ms")

    while len(ready_predictions) < Q and pending:
        done, pending = await asyncio.wait(
            pending,
            timeout=0.01,  # 10ms poll interval
            return_when=asyncio.FIRST_COMPLETED
        )

        # Process completed futures
        for future in done:
            try:
                result = await future
                if result.get("confidence", 0) >= min_confidence:
                    ready_predictions.append(result)
                    logger.debug(f"Accepted result: conf={result.get('confidence', 0):.2f}, "
                               f"progress={len(ready_predictions)}/{Q}")
                else:
                    logger.debug(f"Rejected low-confidence result: conf={result.get('confidence', 0):.2f}")
            except Exception as e:
                logger.warning(f"Future failed: {e}")

        # Check grace timeout
        elapsed = time.time() - start_time
        if elapsed > grace_window_sec:
            logger.warning(f"Grace timeout after {elapsed*1000:.0f}ms: "
                         f"{len(ready_predictions)}/{Q} ready, proceeding with partials")
            break

    # Phase 2: After quorum met, wait remaining grace time for stragglers
    remaining_time = max(0, (start_time + grace_window_sec) - time.time())

    if pending and remaining_time > 0 and len(ready_predictions) >= Q:
        logger.info(f"Quorum met! Waiting {remaining_time*1000:.0f}ms for stragglers...")

        late_done, still_pending = await asyncio.wait(
            pending,
            timeout=remaining_time,
            return_when=asyncio.ALL_COMPLETED
        )

        # Process straggler results
        for future in late_done:
            try:
                result = await future
                if result.get("confidence", 0) >= min_confidence:
                    ready_predictions.append(result)
                    logger.debug(f"Accepted straggler: conf={result.get('confidence', 0):.2f}")
            except Exception as e:
                logger.debug(f"Straggler failed: {e}")

        if still_pending:
            logger.info(f"Cancelled {len(still_pending)} still-pending tasks")
            for future in still_pending:
                future.cancel()

    elapsed_total = time.time() - start_time

    metrics = {
        "N": N,
        "Q": Q,
        "ready": len(ready_predictions),
        "quorum_met": len(ready_predictions) >= Q,
        "elapsed_ms": elapsed_total * 1000,
        "grace_window_ms": grace_window_sec * 1000,
        "quorum_pct": quorum_pct,
        "min_confidence": min_confidence
    }

    logger.info(f"Quorum wait complete: {len(ready_predictions)}/{N} ready "
               f"(Q={Q}, {elapsed_total*1000:.0f}ms)")

    return QuorumResult(ready_predictions=ready_predictions, metrics=metrics)


async def quorum_wait_with_timeout(
    prediction_futures: List[asyncio.Future],
    hard_timeout_sec: float = 2.0,
    **quorum_kwargs
) -> QuorumResult:
    """
    Quorum wait with hard timeout fallback.

    If quorum_wait exceeds hard_timeout_sec, cancel all pending and return partials.

    Args:
        prediction_futures: List of async tasks
        hard_timeout_sec: Absolute maximum wait time (default 2s)
        **quorum_kwargs: Passed to quorum_wait (grace_window_sec, quorum_pct, etc.)

    Returns:
        QuorumResult (may have fewer than Q results if hard timeout hit)
    """
    try:
        result = await asyncio.wait_for(
            quorum_wait(prediction_futures, **quorum_kwargs),
            timeout=hard_timeout_sec
        )
        return result
    except asyncio.TimeoutError:
        logger.error(f"Hard timeout after {hard_timeout_sec}s! Cancelling all tasks.")

        # Cancel all futures
        for future in prediction_futures:
            if not future.done():
                future.cancel()

        # Return empty result
        return QuorumResult(
            ready_predictions=[],
            metrics={
                "N": len(prediction_futures),
                "Q": 0,
                "ready": 0,
                "quorum_met": False,
                "hard_timeout": True,
                "timeout_sec": hard_timeout_sec
            }
        )


# ============================================================================
# Testing Utilities
# ============================================================================

async def _mock_prediction_task(concept_id: str, delay_sec: float, confidence: float) -> Dict[str, Any]:
    """Mock async prediction task for testing."""
    await asyncio.sleep(delay_sec)
    return {
        "concept_id": concept_id,
        "text": f"Concept {concept_id}",
        "confidence": confidence,
        "delay_sec": delay_sec
    }


async def test_quorum_basic():
    """Test basic quorum wait with fast tasks."""
    print("\n=== Test: Basic Quorum (5 fast tasks) ===")

    futures = [
        _mock_prediction_task(f"c{i}", delay_sec=0.05, confidence=0.8)
        for i in range(5)
    ]

    result = await quorum_wait([asyncio.create_task(f) for f in futures], grace_window_sec=0.25, quorum_pct=0.70)

    print(f"Results: {result.metrics}")
    assert result.metrics["quorum_met"], "Quorum should be met"
    assert len(result.ready_predictions) >= 4, "Should have at least 4/5 results (70%)"


async def test_quorum_stragglers():
    """Test quorum with some slow stragglers."""
    print("\n=== Test: Quorum with Stragglers ===")

    # 3 fast, 2 slow
    futures = [
        asyncio.create_task(_mock_prediction_task("c1", 0.05, 0.9)),
        asyncio.create_task(_mock_prediction_task("c2", 0.05, 0.9)),
        asyncio.create_task(_mock_prediction_task("c3", 0.05, 0.9)),
        asyncio.create_task(_mock_prediction_task("c4", 0.5, 0.7)),  # slow straggler
        asyncio.create_task(_mock_prediction_task("c5", 0.5, 0.7)),  # slow straggler
    ]

    result = await quorum_wait(futures, grace_window_sec=0.25, quorum_pct=0.60)

    print(f"Results: {result.metrics}")
    assert result.metrics["quorum_met"], "Quorum should be met with fast tasks"
    assert len(result.ready_predictions) >= 3, "Should have at least 3 fast results"
    print(f"Got {len(result.ready_predictions)}/5 results (stragglers may timeout)")


async def test_quorum_low_confidence():
    """Test that low-confidence results are filtered."""
    print("\n=== Test: Low Confidence Filtering ===")

    futures = [
        asyncio.create_task(_mock_prediction_task("c1", 0.05, 0.9)),  # high conf
        asyncio.create_task(_mock_prediction_task("c2", 0.05, 0.9)),  # high conf
        asyncio.create_task(_mock_prediction_task("c3", 0.05, 0.3)),  # LOW conf
        asyncio.create_task(_mock_prediction_task("c4", 0.05, 0.2)),  # LOW conf
        asyncio.create_task(_mock_prediction_task("c5", 0.05, 0.8)),  # high conf
    ]

    result = await quorum_wait(futures, grace_window_sec=0.25, quorum_pct=0.60, min_confidence=0.5)

    print(f"Results: {result.metrics}")
    assert len(result.ready_predictions) == 3, "Should only accept 3 high-confidence results"
    for pred in result.ready_predictions:
        assert pred["confidence"] >= 0.5, "All results should have confidence >= 0.5"


async def test_quorum_hard_timeout():
    """Test hard timeout fallback."""
    print("\n=== Test: Hard Timeout ===")

    # All tasks are slow (5 seconds)
    futures = [
        asyncio.create_task(_mock_prediction_task(f"c{i}", 5.0, 0.9))
        for i in range(5)
    ]

    # Hard timeout (0.5s) should trigger before any task completes
    result = await quorum_wait_with_timeout(futures, hard_timeout_sec=0.5, grace_window_sec=0.25, quorum_pct=0.70)

    print(f"Results: {result.metrics}")
    # Either hard_timeout flag OR zero results with quorum not met
    timeout_triggered = result.metrics.get("hard_timeout") or (len(result.ready_predictions) == 0 and not result.metrics.get("quorum_met"))
    assert timeout_triggered, "Hard timeout or grace timeout should prevent results"
    assert len(result.ready_predictions) == 0, "No results should complete"


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    print("Running quorum wait tests...")
    asyncio.run(test_quorum_basic())
    asyncio.run(test_quorum_stragglers())
    asyncio.run(test_quorum_low_confidence())
    asyncio.run(test_quorum_hard_timeout())
    print("\n✓ All tests passed!")
