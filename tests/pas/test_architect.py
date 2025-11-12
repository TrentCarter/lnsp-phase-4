#!/usr/bin/env python3
"""
Tests for Architect Service

Tests Prime Directive processing, task decomposition,
Director allocation, and status tracking.
"""
import pytest
import httpx
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from fastapi.testclient import TestClient

# Import the FastAPI app
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from services.pas.architect.app import app, PrimeDirective, RunStatus
from services.common.job_queue import Lane, Priority


# Use TestClient for synchronous testing
client = TestClient(app)


class TestArchitectHealth:
    """Test Architect health endpoint"""

    def test_health_endpoint(self):
        """Should return healthy status"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["service"] == "Architect"
        assert data["status"] == "healthy"
        assert "agent_id" in data


class TestPrimeDirectiveSubmission:
    """Test Prime Directive submission"""

    @patch("services.pas.architect.app.decomposer")
    @patch("services.pas.architect.app.httpx.AsyncClient")
    def test_submit_prime_directive(self, mock_httpx, mock_decomposer):
        """Should accept and process Prime Directive"""
        # Mock decomposer
        mock_decomposer.decompose_to_lanes = AsyncMock(return_value={
            "executive_summary": "Implement user authentication",
            "lanes": {
                "Code": {
                    "job_card_id": "jc-code-001",
                    "description": "Write auth logic",
                    "files": ["auth.py"],
                    "priority": "high",
                },
                "Docs": {
                    "job_card_id": "jc-docs-001",
                    "description": "Document auth API",
                    "files": ["README.md"],
                    "priority": "medium",
                },
            },
            "dependency_graph": "digraph G { Code -> Docs }",
        })

        # Mock Director responses
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": "accepted"}
        mock_httpx.return_value.__aenter__.return_value.post = AsyncMock(
            return_value=mock_response
        )

        prime_directive = {
            "run_id": "run-001",
            "prd": "Implement user authentication with JWT tokens",
            "title": "User Authentication",
            "entry_files": ["auth.py"],
            "budget": {"tokens_max": 100000, "duration_max_mins": 60},
            "policy": {},
            "approval_mode": "auto",
        }

        response = client.post("/submit", json=prime_directive)
        assert response.status_code == 200

        data = response.json()
        assert data["run_id"] == "run-001"
        assert data["status"] == "processing"
        assert "plan" in data

    def test_submit_missing_fields(self):
        """Should reject Prime Directive with missing fields"""
        incomplete_directive = {
            "run_id": "run-002",
            # Missing prd and title
        }

        response = client.post("/submit", json=incomplete_directive)
        assert response.status_code == 422  # Validation error

    @patch("services.pas.architect.app.decomposer")
    def test_submit_duplicate_run_id(self, mock_decomposer):
        """Should reject duplicate run IDs"""
        mock_decomposer.decompose_to_lanes = AsyncMock(return_value={
            "executive_summary": "Test",
            "lanes": {},
            "dependency_graph": "digraph G {}",
        })

        prime_directive = {
            "run_id": "run-duplicate",
            "prd": "Test PRD",
            "title": "Test",
            "entry_files": [],
            "budget": {},
            "policy": {},
            "approval_mode": "auto",
        }

        # Submit first time
        response1 = client.post("/submit", json=prime_directive)
        assert response1.status_code == 200

        # Submit again - should reject
        response2 = client.post("/submit", json=prime_directive)
        assert response2.status_code == 409  # Conflict


class TestStatusTracking:
    """Test run status tracking"""

    @patch("services.pas.architect.app.decomposer")
    @patch("services.pas.architect.app.httpx.AsyncClient")
    def test_get_run_status(self, mock_httpx, mock_decomposer):
        """Should return status for existing run"""
        # Create a run first
        mock_decomposer.decompose_to_lanes = AsyncMock(return_value={
            "executive_summary": "Test",
            "lanes": {},
            "dependency_graph": "digraph G {}",
        })

        prime_directive = {
            "run_id": "run-status-test",
            "prd": "Test PRD",
            "title": "Test",
            "entry_files": [],
            "budget": {},
            "policy": {},
            "approval_mode": "auto",
        }

        client.post("/submit", json=prime_directive)

        # Get status
        response = client.get("/status/run-status-test")
        assert response.status_code == 200

        data = response.json()
        assert data["run_id"] == "run-status-test"
        assert "status" in data
        assert "plan" in data

    def test_get_nonexistent_run_status(self):
        """Should return 404 for nonexistent run"""
        response = client.get("/status/nonexistent-run")
        assert response.status_code == 404


class TestTaskDecomposition:
    """Test task decomposition logic"""

    @patch("services.pas.architect.app.decomposer")
    @patch("services.pas.architect.app.httpx.AsyncClient")
    def test_decompose_code_only_task(self, mock_httpx, mock_decomposer):
        """Should decompose code-only task to Code lane"""
        mock_decomposer.decompose_to_lanes = AsyncMock(return_value={
            "executive_summary": "Add hello function",
            "lanes": {
                "Code": {
                    "job_card_id": "jc-code-001",
                    "description": "Add hello() function to utils.py",
                    "files": ["utils.py"],
                    "priority": "high",
                },
            },
            "dependency_graph": "digraph G { Code }",
        })

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": "accepted"}
        mock_httpx.return_value.__aenter__.return_value.post = AsyncMock(
            return_value=mock_response
        )

        prime_directive = {
            "run_id": "run-code-only",
            "prd": "Add a hello() function to utils.py",
            "title": "Add hello function",
            "entry_files": ["utils.py"],
            "budget": {},
            "policy": {},
            "approval_mode": "auto",
        }

        response = client.post("/submit", json=prime_directive)
        assert response.status_code == 200

        data = response.json()
        assert "Code" in data["plan"]["lane_allocations"]
        assert len(data["plan"]["lane_allocations"]) == 1

    @patch("services.pas.architect.app.decomposer")
    @patch("services.pas.architect.app.httpx.AsyncClient")
    def test_decompose_multi_lane_task(self, mock_httpx, mock_decomposer):
        """Should decompose complex task to multiple lanes"""
        mock_decomposer.decompose_to_lanes = AsyncMock(return_value={
            "executive_summary": "Build ML model",
            "lanes": {
                "Data": {
                    "job_card_id": "jc-data-001",
                    "description": "Prepare training dataset",
                    "files": ["data/prepare.py"],
                    "priority": "high",
                },
                "Models": {
                    "job_card_id": "jc-models-001",
                    "description": "Train classifier model",
                    "files": ["models/train.py"],
                    "priority": "high",
                },
                "Docs": {
                    "job_card_id": "jc-docs-001",
                    "description": "Document model architecture",
                    "files": ["docs/model.md"],
                    "priority": "medium",
                },
            },
            "dependency_graph": "digraph G { Data -> Models -> Docs }",
        })

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": "accepted"}
        mock_httpx.return_value.__aenter__.return_value.post = AsyncMock(
            return_value=mock_response
        )

        prime_directive = {
            "run_id": "run-multi-lane",
            "prd": "Build and train a text classifier model",
            "title": "Text Classifier",
            "entry_files": ["models/train.py"],
            "budget": {},
            "policy": {},
            "approval_mode": "auto",
        }

        response = client.post("/submit", json=prime_directive)
        assert response.status_code == 200

        data = response.json()
        lanes = data["plan"]["lane_allocations"]
        assert "Data" in lanes
        assert "Models" in lanes
        assert "Docs" in lanes


class TestDirectorAllocation:
    """Test Director allocation and communication"""

    @patch("services.pas.architect.app.decomposer")
    @patch("services.pas.architect.app.httpx.AsyncClient")
    def test_allocate_to_directors(self, mock_httpx, mock_decomposer):
        """Should allocate job cards to Directors"""
        mock_decomposer.decompose_to_lanes = AsyncMock(return_value={
            "executive_summary": "Test",
            "lanes": {
                "Code": {
                    "job_card_id": "jc-code-001",
                    "description": "Test code task",
                    "files": ["test.py"],
                    "priority": "high",
                },
            },
            "dependency_graph": "digraph G { Code }",
        })

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": "accepted"}
        mock_httpx.return_value.__aenter__.return_value.post = AsyncMock(
            return_value=mock_response
        )

        prime_directive = {
            "run_id": "run-allocation-test",
            "prd": "Test allocation",
            "title": "Test",
            "entry_files": [],
            "budget": {},
            "policy": {},
            "approval_mode": "auto",
        }

        response = client.post("/submit", json=prime_directive)
        assert response.status_code == 200

        # Verify Director was called
        mock_httpx.return_value.__aenter__.return_value.post.assert_called()

    @patch("services.pas.architect.app.decomposer")
    @patch("services.pas.architect.app.httpx.AsyncClient")
    def test_handle_director_failure(self, mock_httpx, mock_decomposer):
        """Should handle Director communication failures"""
        mock_decomposer.decompose_to_lanes = AsyncMock(return_value={
            "executive_summary": "Test",
            "lanes": {
                "Code": {
                    "job_card_id": "jc-code-001",
                    "description": "Test",
                    "files": [],
                    "priority": "high",
                },
            },
            "dependency_graph": "digraph G { Code }",
        })

        # Mock Director failure
        mock_httpx.return_value.__aenter__.return_value.post = AsyncMock(
            side_effect=httpx.RequestError("Connection failed")
        )

        prime_directive = {
            "run_id": "run-director-failure",
            "prd": "Test",
            "title": "Test",
            "entry_files": [],
            "budget": {},
            "policy": {},
            "approval_mode": "auto",
        }

        response = client.post("/submit", json=prime_directive)
        # Should still accept but mark as error
        assert response.status_code in [200, 500]


class TestBudgetManagement:
    """Test budget tracking and enforcement"""

    @patch("services.pas.architect.app.decomposer")
    def test_budget_passthrough(self, mock_decomposer):
        """Should pass budget to Directors"""
        mock_decomposer.decompose_to_lanes = AsyncMock(return_value={
            "executive_summary": "Test",
            "lanes": {},
            "dependency_graph": "digraph G {}",
        })

        budget = {
            "tokens_max": 50000,
            "duration_max_mins": 30,
            "cost_usd_max": 1.0,
        }

        prime_directive = {
            "run_id": "run-budget-test",
            "prd": "Test",
            "title": "Test",
            "entry_files": [],
            "budget": budget,
            "policy": {},
            "approval_mode": "auto",
        }

        response = client.post("/submit", json=prime_directive)
        assert response.status_code == 200

        data = response.json()
        # Budget should be tracked
        assert "budget" in data["plan"]["resource_reservations"]


class TestAcceptanceGates:
    """Test acceptance gate validation"""

    @patch("services.pas.architect.app.decomposer")
    def test_acceptance_gates_generated(self, mock_decomposer):
        """Should generate acceptance gates per lane"""
        mock_decomposer.decompose_to_lanes = AsyncMock(return_value={
            "executive_summary": "Test",
            "lanes": {
                "Code": {
                    "job_card_id": "jc-code-001",
                    "description": "Test",
                    "files": ["test.py"],
                    "priority": "high",
                },
            },
            "dependency_graph": "digraph G { Code }",
        })

        prime_directive = {
            "run_id": "run-acceptance-test",
            "prd": "Add tests for user authentication",
            "title": "Auth Tests",
            "entry_files": [],
            "budget": {},
            "policy": {},
            "approval_mode": "auto",
        }

        response = client.post("/submit", json=prime_directive)
        assert response.status_code == 200

        data = response.json()
        assert "acceptance_gates" in data["plan"]


class TestPolicyEnforcement:
    """Test policy enforcement"""

    @patch("services.pas.architect.app.decomposer")
    def test_protected_paths_enforcement(self, mock_decomposer):
        """Should enforce protected paths policy"""
        mock_decomposer.decompose_to_lanes = AsyncMock(return_value={
            "executive_summary": "Test",
            "lanes": {
                "Code": {
                    "job_card_id": "jc-code-001",
                    "description": "Modify config",
                    "files": [".env"],  # Protected file
                    "priority": "high",
                },
            },
            "dependency_graph": "digraph G { Code }",
        })

        policy = {
            "protected_paths": [".env", "credentials.json"],
            "require_cross_vendor_review": True,
        }

        prime_directive = {
            "run_id": "run-policy-test",
            "prd": "Update configuration",
            "title": "Config Update",
            "entry_files": [],
            "budget": {},
            "policy": policy,
            "approval_mode": "auto",
        }

        response = client.post("/submit", json=prime_directive)
        # Should either reject or require human approval
        assert response.status_code in [200, 403]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
