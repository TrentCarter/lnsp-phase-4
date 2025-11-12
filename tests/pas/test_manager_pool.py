#!/usr/bin/env python3
"""
Tests for Manager Pool

Tests lifecycle management, state transitions, allocation,
and thread-safety of the Manager Pool singleton.
"""
import pytest
import time
from unittest.mock import Mock, patch

from services.common.manager_pool.manager_pool import (
    ManagerPool,
    ManagerState,
    ManagerInfo,
    get_manager_pool,
)


class TestManagerPoolSingleton:
    """Test singleton behavior of Manager Pool"""

    def test_singleton_pattern(self):
        """Manager Pool should return same instance"""
        pool1 = get_manager_pool()
        pool2 = get_manager_pool()
        assert pool1 is pool2

    def test_fresh_pool_is_empty(self):
        """New pool should start empty"""
        pool = ManagerPool()
        assert pool.get_all_managers() == []


class TestManagerRegistration:
    """Test Manager registration and deregistration"""

    def setup_method(self):
        """Create fresh pool for each test"""
        self.pool = ManagerPool()

    def test_register_manager(self):
        """Should register a new Manager"""
        manager_id = "mgr-code-001"
        lane = "Code"
        llm_model = "qwen2.5-coder:7b"

        self.pool.register_manager(
            manager_id=manager_id,
            lane=lane,
            llm_model=llm_model,
            parent_director="Dir-Code",
        )

        info = self.pool.get_manager_info(manager_id)
        assert info is not None
        assert info.manager_id == manager_id
        assert info.lane == lane
        assert info.llm_model == llm_model
        assert info.state == ManagerState.CREATED

    def test_register_duplicate_manager_fails(self):
        """Registering same Manager ID twice should fail"""
        manager_id = "mgr-code-001"
        self.pool.register_manager(
            manager_id=manager_id,
            lane="Code",
            llm_model="qwen2.5-coder:7b",
        )

        with pytest.raises(ValueError, match="already registered"):
            self.pool.register_manager(
                manager_id=manager_id,
                lane="Code",
                llm_model="qwen2.5-coder:7b",
            )

    def test_deregister_manager(self):
        """Should deregister a Manager"""
        manager_id = "mgr-code-001"
        self.pool.register_manager(
            manager_id=manager_id,
            lane="Code",
            llm_model="qwen2.5-coder:7b",
        )

        self.pool.deregister_manager(manager_id)
        assert self.pool.get_manager_info(manager_id) is None

    def test_deregister_nonexistent_manager(self):
        """Deregistering nonexistent Manager should not error"""
        # Should not raise
        self.pool.deregister_manager("nonexistent-id")


class TestManagerStateTransitions:
    """Test Manager state lifecycle"""

    def setup_method(self):
        """Create pool and register a Manager"""
        self.pool = ManagerPool()
        self.manager_id = "mgr-code-001"
        self.pool.register_manager(
            manager_id=self.manager_id,
            lane="Code",
            llm_model="qwen2.5-coder:7b",
        )

    def test_initial_state_is_created(self):
        """New Manager should start in CREATED state"""
        info = self.pool.get_manager_info(self.manager_id)
        assert info.state == ManagerState.CREATED

    def test_transition_created_to_idle(self):
        """Should transition from CREATED to IDLE"""
        self.pool.set_manager_state(self.manager_id, ManagerState.IDLE)
        info = self.pool.get_manager_info(self.manager_id)
        assert info.state == ManagerState.IDLE

    def test_transition_idle_to_busy(self):
        """Should transition from IDLE to BUSY"""
        self.pool.set_manager_state(self.manager_id, ManagerState.IDLE)
        self.pool.set_manager_state(self.manager_id, ManagerState.BUSY)
        info = self.pool.get_manager_info(self.manager_id)
        assert info.state == ManagerState.BUSY

    def test_transition_busy_to_idle(self):
        """Should transition from BUSY back to IDLE"""
        self.pool.set_manager_state(self.manager_id, ManagerState.IDLE)
        self.pool.set_manager_state(self.manager_id, ManagerState.BUSY)
        self.pool.set_manager_state(self.manager_id, ManagerState.IDLE)
        info = self.pool.get_manager_info(self.manager_id)
        assert info.state == ManagerState.IDLE

    def test_transition_to_failed(self):
        """Should transition to FAILED on error"""
        self.pool.set_manager_state(self.manager_id, ManagerState.FAILED)
        info = self.pool.get_manager_info(self.manager_id)
        assert info.state == ManagerState.FAILED

    def test_transition_to_terminated(self):
        """Should transition to TERMINATED on shutdown"""
        self.pool.set_manager_state(self.manager_id, ManagerState.TERMINATED)
        info = self.pool.get_manager_info(self.manager_id)
        assert info.state == ManagerState.TERMINATED


class TestManagerAllocation:
    """Test Manager allocation to Directors"""

    def setup_method(self):
        """Create pool and register multiple Managers"""
        self.pool = ManagerPool()
        # Register 3 Code lane Managers
        for i in range(3):
            manager_id = f"mgr-code-{i:03d}"
            self.pool.register_manager(
                manager_id=manager_id,
                lane="Code",
                llm_model="qwen2.5-coder:7b",
            )
            # Set to IDLE to make available
            self.pool.set_manager_state(manager_id, ManagerState.IDLE)

        # Register 2 Models lane Managers
        for i in range(2):
            manager_id = f"mgr-models-{i:03d}"
            self.pool.register_manager(
                manager_id=manager_id,
                lane="Models",
                llm_model="deepseek-r1:7b",
            )
            self.pool.set_manager_state(manager_id, ManagerState.IDLE)

    def test_allocate_idle_manager(self):
        """Should allocate an IDLE Manager"""
        manager_id = self.pool.allocate_manager(lane="Code", job_card_id="jc-001")
        assert manager_id is not None
        assert manager_id.startswith("mgr-code-")

        info = self.pool.get_manager_info(manager_id)
        assert info.state == ManagerState.BUSY
        assert info.current_job_card_id == "jc-001"
        assert info.last_allocated_at is not None

    def test_allocate_specific_lane(self):
        """Should allocate Manager from requested lane"""
        manager_id = self.pool.allocate_manager(lane="Models", job_card_id="jc-002")
        assert manager_id is not None
        assert manager_id.startswith("mgr-models-")

        info = self.pool.get_manager_info(manager_id)
        assert info.lane == "Models"

    def test_allocate_when_no_idle_managers(self):
        """Should return None when no IDLE Managers available"""
        # Allocate all Code Managers
        for i in range(3):
            self.pool.allocate_manager(lane="Code", job_card_id=f"jc-{i:03d}")

        # Try to allocate another - should return None
        manager_id = self.pool.allocate_manager(lane="Code", job_card_id="jc-999")
        assert manager_id is None

    def test_release_manager(self):
        """Should release Manager back to IDLE"""
        manager_id = self.pool.allocate_manager(lane="Code", job_card_id="jc-001")
        assert manager_id is not None

        self.pool.release_manager(manager_id)

        info = self.pool.get_manager_info(manager_id)
        assert info.state == ManagerState.IDLE
        assert info.current_job_card_id is None

    def test_reuse_released_manager(self):
        """Released Manager should be available for reallocation"""
        # Allocate and release
        manager_id_1 = self.pool.allocate_manager(lane="Code", job_card_id="jc-001")
        self.pool.release_manager(manager_id_1)

        # Reallocate - should get same Manager
        manager_id_2 = self.pool.allocate_manager(lane="Code", job_card_id="jc-002")
        # Note: May or may not be same Manager depending on allocation strategy
        assert manager_id_2 is not None


class TestManagerQueries:
    """Test Manager query operations"""

    def setup_method(self):
        """Create pool and register Managers in various states"""
        self.pool = ManagerPool()

        # IDLE Managers
        for i in range(2):
            manager_id = f"mgr-code-idle-{i:03d}"
            self.pool.register_manager(
                manager_id=manager_id,
                lane="Code",
                llm_model="qwen2.5-coder:7b",
            )
            self.pool.set_manager_state(manager_id, ManagerState.IDLE)

        # BUSY Managers
        for i in range(3):
            manager_id = f"mgr-code-busy-{i:03d}"
            self.pool.register_manager(
                manager_id=manager_id,
                lane="Code",
                llm_model="qwen2.5-coder:7b",
            )
            self.pool.set_manager_state(manager_id, ManagerState.BUSY)

        # FAILED Manager
        manager_id = "mgr-code-failed-001"
        self.pool.register_manager(
            manager_id=manager_id,
            lane="Code",
            llm_model="qwen2.5-coder:7b",
        )
        self.pool.set_manager_state(manager_id, ManagerState.FAILED)

    def test_get_all_managers(self):
        """Should return all Managers"""
        managers = self.pool.get_all_managers()
        assert len(managers) == 6  # 2 IDLE + 3 BUSY + 1 FAILED

    def test_get_managers_by_state(self):
        """Should filter Managers by state"""
        idle = self.pool.get_managers_by_state(ManagerState.IDLE)
        busy = self.pool.get_managers_by_state(ManagerState.BUSY)
        failed = self.pool.get_managers_by_state(ManagerState.FAILED)

        assert len(idle) == 2
        assert len(busy) == 3
        assert len(failed) == 1

    def test_get_managers_by_lane(self):
        """Should filter Managers by lane"""
        code_managers = self.pool.get_managers_by_lane("Code")
        assert len(code_managers) == 6

        models_managers = self.pool.get_managers_by_lane("Models")
        assert len(models_managers) == 0

    def test_get_idle_count(self):
        """Should count IDLE Managers"""
        count = self.pool.get_idle_count(lane="Code")
        assert count == 2

    def test_get_busy_count(self):
        """Should count BUSY Managers"""
        count = self.pool.get_busy_count(lane="Code")
        assert count == 3


class TestThreadSafety:
    """Test thread-safe operations"""

    def setup_method(self):
        """Create pool"""
        self.pool = ManagerPool()

    def test_concurrent_registration(self):
        """Multiple threads should safely register Managers"""
        import threading

        def register_managers(start_idx):
            for i in range(10):
                manager_id = f"mgr-thread-{start_idx}-{i:03d}"
                self.pool.register_manager(
                    manager_id=manager_id,
                    lane="Code",
                    llm_model="qwen2.5-coder:7b",
                )

        threads = []
        for i in range(5):
            t = threading.Thread(target=register_managers, args=(i,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        # Should have 50 Managers (5 threads * 10 each)
        managers = self.pool.get_all_managers()
        assert len(managers) == 50

    def test_concurrent_allocation(self):
        """Multiple threads should safely allocate Managers"""
        import threading

        # Register 20 IDLE Managers
        for i in range(20):
            manager_id = f"mgr-code-{i:03d}"
            self.pool.register_manager(
                manager_id=manager_id,
                lane="Code",
                llm_model="qwen2.5-coder:7b",
            )
            self.pool.set_manager_state(manager_id, ManagerState.IDLE)

        allocated = []
        lock = threading.Lock()

        def allocate_managers():
            for i in range(5):
                manager_id = self.pool.allocate_manager(
                    lane="Code", job_card_id=f"jc-{i:03d}"
                )
                if manager_id:
                    with lock:
                        allocated.append(manager_id)

        threads = []
        for i in range(4):
            t = threading.Thread(target=allocate_managers)
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        # Should have allocated 20 unique Managers (4 threads * 5 each)
        assert len(allocated) == 20
        assert len(set(allocated)) == 20  # All unique


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
