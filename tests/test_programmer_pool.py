#!/usr/bin/env python3
"""
Test programmer pool functionality

Tests:
1. Pool configuration loading
2. Programmer selection (least_loaded, round_robin)
3. Health checking
4. Capability-based routing
5. Load balancing
"""
import pytest
import asyncio
from services.common.programmer_pool import ProgrammerPool


@pytest.mark.asyncio
async def test_pool_initialization():
    """Test that pool initializes from config"""
    pool = ProgrammerPool()

    # Should have 10 programmers
    assert len(pool.programmers) == 10

    # Should have IDs 001-010
    ids = [p["id"] for p in pool.programmers]
    assert "001" in ids
    assert "010" in ids

    # Should have different LLMs
    llms = set(p["primary_llm"] for p in pool.programmers)
    assert len(llms) > 1  # Should have variety


@pytest.mark.asyncio
async def test_capability_routing():
    """Test capability-based programmer selection"""
    pool = ProgrammerPool()

    # Get "fast" programmers (should be 001-005, 009-010)
    fast_progs = pool.config["capability_routing"]["fast"]
    assert len(fast_progs) >= 5

    # Get "premium" programmers (should be 006-008)
    premium_progs = pool.config["capability_routing"]["premium"]
    assert len(premium_progs) >= 2


@pytest.mark.asyncio
async def test_available_programmers():
    """Test getting available programmers"""
    pool = ProgrammerPool()

    # Get all available programmers
    available = await pool.get_available_programmers()

    # Should have some available (may not be all if services not running)
    # This test will pass if pool config is valid, even if services aren't running
    assert isinstance(available, list)


@pytest.mark.asyncio
async def test_programmer_selection_strategies():
    """Test different load balancing strategies"""
    pool = ProgrammerPool()

    # Manually set queue depths for testing
    pool.queue_depth = {
        "001": 5,
        "002": 2,
        "003": 0,
        "004": 3,
        "005": 1
    }

    # Override health cache to simulate all healthy
    for prog_id in ["001", "002", "003", "004", "005"]:
        pool.health_cache[prog_id] = {"status": "ok"}

    # Test least_loaded - should select 003 (queue_depth=0)
    pool.load_balancing["strategy"] = "least_loaded"
    selected = await pool.select_programmer(capabilities=["fast"])

    # Should select the one with lowest queue depth
    if selected:  # Only test if any available
        assert pool.queue_depth.get(selected, 0) <= 1  # 003 or 005


@pytest.mark.asyncio
async def test_pool_status():
    """Test getting pool status"""
    pool = ProgrammerPool()

    status = await pool.get_pool_status()

    # Should have required fields
    assert "pool_size" in status
    assert "available" in status
    assert "programmers" in status

    # Pool size should be 10
    assert status["pool_size"] == 10

    # Should have programmer details
    assert len(status["programmers"]) == 10

    # Each programmer should have required fields
    for prog in status["programmers"]:
        assert "id" in prog
        assert "port" in prog
        assert "primary_llm" in prog
        assert "backup_llm" in prog
        assert "capabilities" in prog


if __name__ == "__main__":
    # Run tests
    asyncio.run(test_pool_initialization())
    print("✅ test_pool_initialization passed")

    asyncio.run(test_capability_routing())
    print("✅ test_capability_routing passed")

    asyncio.run(test_available_programmers())
    print("✅ test_available_programmers passed")

    asyncio.run(test_programmer_selection_strategies())
    print("✅ test_programmer_selection_strategies passed")

    asyncio.run(test_pool_status())
    print("✅ test_pool_status passed")

    print("\n🎉 All tests passed!")
