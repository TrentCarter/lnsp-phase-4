#!/usr/bin/env python3
"""
Tests for Manager Factory

Tests Manager creation, configuration, and integration
with Manager Pool and Heartbeat Monitor.
"""
import pytest
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from services.common.manager_pool.manager_factory import ManagerFactory
from services.common.manager_pool.manager_pool import (
    ManagerPool,
    ManagerState,
    get_manager_pool,
)
from services.common.heartbeat import AgentState


class TestManagerFactoryCreation:
    """Test Manager creation via factory"""

    def setup_method(self):
        """Create fresh factory and pool for each test"""
        # Reset singleton
        self.pool = ManagerPool()
        # Patch get_manager_pool to return our pool
        self.pool_patcher = patch(
            "services.common.manager_pool.manager_factory.get_manager_pool",
            return_value=self.pool,
        )
        self.pool_patcher.start()

        # Mock heartbeat monitor
        self.mock_monitor = Mock()
        self.monitor_patcher = patch(
            "services.common.manager_pool.manager_factory.get_monitor",
            return_value=self.mock_monitor,
        )
        self.monitor_patcher.start()

        self.factory = ManagerFactory()

    def teardown_method(self):
        """Stop patches"""
        self.pool_patcher.stop()
        self.monitor_patcher.stop()

    def test_create_manager_basic(self):
        """Should create a Manager with basic config"""
        manager_id = self.factory.create_manager(
            lane="Code",
            director="Dir-Code",
            job_card_id="jc-001",
            llm_model="qwen2.5-coder:7b",
        )

        assert manager_id is not None
        assert manager_id.startswith("mgr-code-")

        # Check registered in pool
        info = self.pool.get_manager_info(manager_id)
        assert info is not None
        assert info.lane == "Code"
        assert info.llm_model == "qwen2.5-coder:7b"
        assert info.parent_director == "Dir-Code"
        assert info.current_job_card_id == "jc-001"

    def test_create_manager_auto_id(self):
        """Should auto-generate Manager IDs"""
        mgr1 = self.factory.create_manager(
            lane="Code", director="Dir-Code", job_card_id="jc-001"
        )
        mgr2 = self.factory.create_manager(
            lane="Code", director="Dir-Code", job_card_id="jc-002"
        )

        assert mgr1 != mgr2
        assert mgr1.startswith("mgr-code-")
        assert mgr2.startswith("mgr-code-")

    def test_create_managers_different_lanes(self):
        """Should create Managers for different lanes"""
        lanes = ["Code", "Models", "Data", "DevSecOps", "Docs"]

        manager_ids = []
        for lane in lanes:
            mgr_id = self.factory.create_manager(
                lane=lane,
                director=f"Dir-{lane}",
                job_card_id=f"jc-{lane}",
            )
            manager_ids.append(mgr_id)

        # Check all unique
        assert len(set(manager_ids)) == 5

        # Check lane prefixes
        assert manager_ids[0].startswith("mgr-code-")
        assert manager_ids[1].startswith("mgr-models-")
        assert manager_ids[2].startswith("mgr-data-")
        assert manager_ids[3].startswith("mgr-devsecops-")
        assert manager_ids[4].startswith("mgr-docs-")

    def test_register_with_heartbeat_monitor(self):
        """Should register Manager with Heartbeat Monitor"""
        manager_id = self.factory.create_manager(
            lane="Code",
            director="Dir-Code",
            job_card_id="jc-001",
        )

        # Check heartbeat registration was called
        self.mock_monitor.register.assert_called_once()
        call_args = self.mock_monitor.register.call_args
        assert call_args[1]["agent_id"] == manager_id
        assert call_args[1]["agent_type"] == "Manager"
        assert call_args[1]["state"] == AgentState.IDLE


class TestManagerConfiguration:
    """Test Manager configuration options"""

    def setup_method(self):
        """Create factory with mocks"""
        self.pool = ManagerPool()
        self.pool_patcher = patch(
            "services.common.manager_pool.manager_factory.get_manager_pool",
            return_value=self.pool,
        )
        self.pool_patcher.start()

        self.mock_monitor = Mock()
        self.monitor_patcher = patch(
            "services.common.manager_pool.manager_factory.get_monitor",
            return_value=self.mock_monitor,
        )
        self.monitor_patcher.start()

        self.factory = ManagerFactory()

    def teardown_method(self):
        """Stop patches"""
        self.pool_patcher.stop()
        self.monitor_patcher.stop()

    def test_llm_model_configuration(self):
        """Should configure LLM model"""
        models = [
            "qwen2.5-coder:7b",
            "deepseek-r1:7b",
            "claude-sonnet-4.5",
            "gemini-2.5-flash",
        ]

        for model in models:
            mgr_id = self.factory.create_manager(
                lane="Code",
                director="Dir-Code",
                job_card_id=f"jc-{model}",
                llm_model=model,
            )

            info = self.pool.get_manager_info(mgr_id)
            assert info.llm_model == model

    def test_metadata_passthrough(self):
        """Should pass through custom metadata"""
        metadata = {
            "max_programmers": 5,
            "timeout": 300,
            "priority": "high",
        }

        mgr_id = self.factory.create_manager(
            lane="Code",
            director="Dir-Code",
            job_card_id="jc-001",
            metadata=metadata,
        )

        info = self.pool.get_manager_info(mgr_id)
        assert info.metadata == metadata

    def test_default_llm_model_per_lane(self):
        """Should use default LLM model if not specified"""
        mgr_id = self.factory.create_manager(
            lane="Code",
            director="Dir-Code",
            job_card_id="jc-001",
            # No llm_model specified
        )

        info = self.pool.get_manager_info(mgr_id)
        # Should have some default model (implementation-dependent)
        assert info.llm_model is not None


class TestManagerTermination:
    """Test Manager lifecycle termination"""

    def setup_method(self):
        """Create factory with mocks"""
        self.pool = ManagerPool()
        self.pool_patcher = patch(
            "services.common.manager_pool.manager_factory.get_manager_pool",
            return_value=self.pool,
        )
        self.pool_patcher.start()

        self.mock_monitor = Mock()
        self.monitor_patcher = patch(
            "services.common.manager_pool.manager_factory.get_monitor",
            return_value=self.mock_monitor,
        )
        self.monitor_patcher.start()

        self.factory = ManagerFactory()

    def teardown_method(self):
        """Stop patches"""
        self.pool_patcher.stop()
        self.monitor_patcher.stop()

    def test_terminate_manager(self):
        """Should terminate Manager and cleanup resources"""
        mgr_id = self.factory.create_manager(
            lane="Code",
            director="Dir-Code",
            job_card_id="jc-001",
        )

        self.factory.terminate_manager(mgr_id)

        info = self.pool.get_manager_info(mgr_id)
        assert info.state == ManagerState.TERMINATED

        # Should deregister from heartbeat
        self.mock_monitor.deregister.assert_called_once_with(mgr_id)

    def test_terminate_nonexistent_manager(self):
        """Terminating nonexistent Manager should not error"""
        # Should not raise
        self.factory.terminate_manager("nonexistent-id")


class TestManagerPoolIntegration:
    """Test Factory integration with Manager Pool"""

    def setup_method(self):
        """Create factory with real pool"""
        self.pool = ManagerPool()
        self.pool_patcher = patch(
            "services.common.manager_pool.manager_factory.get_manager_pool",
            return_value=self.pool,
        )
        self.pool_patcher.start()

        self.mock_monitor = Mock()
        self.monitor_patcher = patch(
            "services.common.manager_pool.manager_factory.get_monitor",
            return_value=self.mock_monitor,
        )
        self.monitor_patcher.start()

        self.factory = ManagerFactory()

    def teardown_method(self):
        """Stop patches"""
        self.pool_patcher.stop()
        self.monitor_patcher.stop()

    def test_created_manager_appears_in_pool(self):
        """Manager created by Factory should appear in Pool queries"""
        mgr_id = self.factory.create_manager(
            lane="Code",
            director="Dir-Code",
            job_card_id="jc-001",
        )

        all_managers = self.pool.get_all_managers()
        assert len(all_managers) == 1
        assert all_managers[0].manager_id == mgr_id

    def test_multiple_managers_in_pool(self):
        """Multiple Managers should coexist in Pool"""
        managers = []
        for i in range(5):
            mgr_id = self.factory.create_manager(
                lane="Code",
                director="Dir-Code",
                job_card_id=f"jc-{i:03d}",
            )
            managers.append(mgr_id)

        all_managers = self.pool.get_all_managers()
        assert len(all_managers) == 5

        manager_ids = [m.manager_id for m in all_managers]
        for mgr_id in managers:
            assert mgr_id in manager_ids


class TestErrorHandling:
    """Test error handling in Manager Factory"""

    def setup_method(self):
        """Create factory with mocks"""
        self.pool = ManagerPool()
        self.pool_patcher = patch(
            "services.common.manager_pool.manager_factory.get_manager_pool",
            return_value=self.pool,
        )
        self.pool_patcher.start()

        self.mock_monitor = Mock()
        self.monitor_patcher = patch(
            "services.common.manager_pool.manager_factory.get_monitor",
            return_value=self.mock_monitor,
        )
        self.monitor_patcher.start()

        self.factory = ManagerFactory()

    def teardown_method(self):
        """Stop patches"""
        self.pool_patcher.stop()
        self.monitor_patcher.stop()

    def test_invalid_lane(self):
        """Should handle invalid lane gracefully"""
        with pytest.raises(ValueError, match="Invalid lane"):
            self.factory.create_manager(
                lane="InvalidLane",
                director="Dir-Invalid",
                job_card_id="jc-001",
            )

    def test_missing_required_fields(self):
        """Should validate required fields"""
        with pytest.raises(ValueError):
            self.factory.create_manager(
                lane="Code",
                # Missing director and job_card_id
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
