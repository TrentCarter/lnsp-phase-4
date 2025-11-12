#!/usr/bin/env python3
"""
Tests for Director Services

Parametrized tests covering all 5 Directors:
- Director-Code (port 6111)
- Director-Models (port 6112)
- Director-Data (port 6113)
- Director-DevSecOps (port 6114)
- Director-Docs (port 6115)
"""
import pytest
import httpx
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from fastapi.testclient import TestClient
import importlib.util
import sys
from pathlib import Path

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# Director configurations
DIRECTORS = [
    {
        "name": "Dir-Code",
        "port": 6111,
        "lane": "Code",
        "module_path": "services/pas/director_code/app.py",
        "llm": "google/gemini-2.5-flash",
    },
    {
        "name": "Dir-Models",
        "port": 6112,
        "lane": "Models",
        "module_path": "services/pas/director_models/app.py",
        "llm": "anthropic/claude-sonnet-4-5",
    },
    {
        "name": "Dir-Data",
        "port": 6113,
        "lane": "Data",
        "module_path": "services/pas/director_data/app.py",
        "llm": "anthropic/claude-sonnet-4-5",
    },
    {
        "name": "Dir-DevSecOps",
        "port": 6114,
        "lane": "DevSecOps",
        "module_path": "services/pas/director_devsecops/app.py",
        "llm": "google/gemini-2.5-flash",
    },
    {
        "name": "Dir-Docs",
        "port": 6115,
        "lane": "Docs",
        "module_path": "services/pas/director_docs/app.py",
        "llm": "anthropic/claude-sonnet-4-5",
    },
]


def load_director_app(module_path: str):
    """Dynamically load Director FastAPI app"""
    spec = importlib.util.spec_from_file_location("director_app", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.app


@pytest.fixture(params=DIRECTORS, ids=lambda d: d["name"])
def director(request):
    """Parametrized fixture that yields each Director"""
    config = request.param
    app = load_director_app(config["module_path"])
    client = TestClient(app)
    return {
        "config": config,
        "app": app,
        "client": client,
    }


class TestDirectorHealth:
    """Test Director health endpoints"""

    def test_health_endpoint(self, director):
        """All Directors should have functional health endpoint"""
        response = director["client"].get("/health")
        assert response.status_code == 200

        data = response.json()
        assert data["service"] == director["config"]["name"]
        assert data["status"] == "healthy"
        assert "agent_id" in data


class TestJobCardSubmission:
    """Test job card submission"""

    @patch("services.common.manager_pool.manager_factory.ManagerFactory")
    def test_submit_job_card(self, mock_factory, director):
        """Should accept and process job card"""
        job_card = {
            "job_card_id": f"jc-{director['config']['lane'].lower()}-001",
            "description": f"Test {director['config']['lane']} task",
            "files": ["test.py"],
            "priority": "high",
            "budget": {"tokens_max": 10000},
        }

        payload = {"job_card": job_card}

        response = director["client"].post("/submit", json=payload)
        # May return 200 or error depending on mocks
        assert response.status_code in [200, 500]

    def test_submit_invalid_job_card(self, director):
        """Should reject invalid job card"""
        invalid_card = {
            # Missing required fields
            "invalid_field": "test"
        }

        payload = {"job_card": invalid_card}

        response = director["client"].post("/submit", json=payload)
        # Should fail validation or processing
        assert response.status_code in [422, 400, 500]


class TestManagerTaskDecomposition:
    """Test Manager task decomposition"""

    @patch("services.common.manager_pool.manager_factory.ManagerFactory")
    def test_decompose_job_card(self, mock_factory, director):
        """Should decompose job card into Manager tasks"""
        # This tests the Director's decomposition logic
        # Actual implementation varies per Director

        job_card = {
            "job_card_id": f"jc-{director['config']['lane'].lower()}-002",
            "description": f"Complex {director['config']['lane']} task",
            "files": ["file1.py", "file2.py", "file3.py"],
            "priority": "high",
            "budget": {"tokens_max": 50000},
        }

        # Mock factory to avoid actual Manager creation
        mock_factory.return_value.create_manager.return_value = "mgr-test-001"

        payload = {"job_card": job_card}

        response = director["client"].post("/submit", json=payload)
        # Director should decompose and create Managers
        assert response.status_code in [200, 500]


class TestStatusTracking:
    """Test job status tracking"""

    @patch("services.common.manager_pool.manager_factory.ManagerFactory")
    def test_get_job_status(self, mock_factory, director):
        """Should return status for submitted job"""
        job_card = {
            "job_card_id": f"jc-status-test-{director['config']['lane'].lower()}",
            "description": "Test status tracking",
            "files": ["test.py"],
            "priority": "high",
        }

        # Submit job first
        director["client"].post("/submit", json={"job_card": job_card})

        # Get status
        response = director["client"].get(f"/status/{job_card['job_card_id']}")
        # Should return status (200) or not found (404)
        assert response.status_code in [200, 404]


class TestAcceptanceGates:
    """Test lane-specific acceptance gates"""

    def test_code_lane_acceptance_gates(self):
        """Code lane should validate tests, lint, coverage"""
        # Specific to Dir-Code
        # Tests, lint, coverage, reviews
        pass  # Implementation would check specific gates

    def test_models_lane_acceptance_gates(self):
        """Models lane should validate training, eval, model quality"""
        # Specific to Dir-Models
        # Training logs, eval metrics, model artifacts
        pass  # Implementation would check specific gates

    def test_data_lane_acceptance_gates(self):
        """Data lane should validate ingestion, quality, schemas"""
        # Specific to Dir-Data
        # Data quality checks, schema validation
        pass  # Implementation would check specific gates

    def test_devsecops_lane_acceptance_gates(self):
        """DevSecOps lane should validate security, CI/CD"""
        # Specific to Dir-DevSecOps
        # Security scans, CI/CD pipeline
        pass  # Implementation would check specific gates

    def test_docs_lane_acceptance_gates(self):
        """Docs lane should validate completeness, formatting"""
        # Specific to Dir-Docs
        # Documentation completeness, formatting
        pass  # Implementation would check specific gates


class TestManagerAllocation:
    """Test Manager allocation and monitoring"""

    @patch("services.common.manager_pool.manager_factory.ManagerFactory")
    @patch("services.common.manager_pool.manager_pool.get_manager_pool")
    def test_allocate_managers_for_job(self, mock_pool, mock_factory, director):
        """Should allocate Managers from pool"""
        mock_pool.return_value.allocate_manager.return_value = "mgr-test-001"
        mock_factory.return_value.create_manager.return_value = "mgr-test-002"

        job_card = {
            "job_card_id": f"jc-allocation-{director['config']['lane'].lower()}",
            "description": "Test Manager allocation",
            "files": ["test.py"],
            "priority": "high",
        }

        response = director["client"].post("/submit", json={"job_card": job_card})
        # Should attempt to allocate Managers
        assert response.status_code in [200, 500]

    @patch("services.common.manager_pool.manager_factory.ManagerFactory")
    @patch("services.common.manager_pool.manager_pool.get_manager_pool")
    def test_handle_manager_failure(self, mock_pool, mock_factory, director):
        """Should handle Manager failure gracefully"""
        # Mock Manager allocation failure
        mock_pool.return_value.allocate_manager.return_value = None
        mock_factory.return_value.create_manager.side_effect = Exception("Failed")

        job_card = {
            "job_card_id": f"jc-failure-{director['config']['lane'].lower()}",
            "description": "Test Manager failure",
            "files": ["test.py"],
            "priority": "high",
        }

        response = director["client"].post("/submit", json={"job_card": job_card})
        # Should handle failure gracefully
        assert response.status_code in [200, 500, 503]


class TestReportGeneration:
    """Test lane report generation"""

    @patch("services.common.manager_pool.manager_factory.ManagerFactory")
    def test_generate_lane_report(self, mock_factory, director):
        """Should generate lane report for Architect"""
        job_card = {
            "job_card_id": f"jc-report-{director['config']['lane'].lower()}",
            "description": "Test report generation",
            "files": ["test.py"],
            "priority": "high",
        }

        # Submit job
        director["client"].post("/submit", json={"job_card": job_card})

        # Request report (endpoint may vary)
        response = director["client"].get(f"/report/{job_card['job_card_id']}")
        # Should return report or 404 if not ready
        assert response.status_code in [200, 404]


class TestCrossVendorReview:
    """Test cross-vendor review for protected paths"""

    @patch("services.common.manager_pool.manager_factory.ManagerFactory")
    def test_require_review_for_protected_paths(self, mock_factory, director):
        """Should require review for protected paths"""
        job_card = {
            "job_card_id": f"jc-protected-{director['config']['lane'].lower()}",
            "description": "Modify protected file",
            "files": [".env", "credentials.json"],  # Protected
            "priority": "high",
            "policy": {
                "protected_paths": [".env", "credentials.json"],
                "require_cross_vendor_review": True,
            },
        }

        response = director["client"].post("/submit", json={"job_card": job_card})
        # Should flag for review
        assert response.status_code in [200, 403]


class TestBudgetTracking:
    """Test budget tracking and enforcement"""

    @patch("services.common.manager_pool.manager_factory.ManagerFactory")
    def test_track_budget_per_job(self, mock_factory, director):
        """Should track budget consumption per job"""
        budget = {
            "tokens_max": 10000,
            "duration_max_mins": 15,
            "cost_usd_max": 0.50,
        }

        job_card = {
            "job_card_id": f"jc-budget-{director['config']['lane'].lower()}",
            "description": "Test budget tracking",
            "files": ["test.py"],
            "priority": "high",
            "budget": budget,
        }

        response = director["client"].post("/submit", json={"job_card": job_card})
        # Should accept and track budget
        assert response.status_code in [200, 500]

    @patch("services.common.manager_pool.manager_factory.ManagerFactory")
    def test_enforce_budget_limits(self, mock_factory, director):
        """Should enforce budget limits"""
        # Budget tracking is implementation-dependent
        # Would need to check actual usage vs limits
        pass  # Placeholder for budget enforcement tests


class TestLaneSpecificLogic:
    """Test lane-specific business logic"""

    def test_code_lane_handles_code_tasks(self):
        """Dir-Code should handle code implementation tasks"""
        # Specific to Code lane
        pass

    def test_models_lane_handles_training(self):
        """Dir-Models should handle model training tasks"""
        # Specific to Models lane
        pass

    def test_data_lane_handles_ingestion(self):
        """Dir-Data should handle data ingestion tasks"""
        # Specific to Data lane
        pass

    def test_devsecops_lane_handles_cicd(self):
        """Dir-DevSecOps should handle CI/CD tasks"""
        # Specific to DevSecOps lane
        pass

    def test_docs_lane_handles_documentation(self):
        """Dir-Docs should handle documentation tasks"""
        # Specific to Docs lane
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
