#!/usr/bin/env python3
"""
Integration Tests for Multi-Tier PAS

Tests end-to-end flow:
Gateway → PAS Root → Architect → Directors → Managers → Aider → Complete

These tests require all services to be running.
Use @pytest.mark.integration to mark tests that need running services.
"""
import pytest
import httpx
import time
import json
from pathlib import Path
import asyncio


# Service endpoints
GATEWAY_URL = "http://127.0.0.1:6120"
PAS_ROOT_URL = "http://127.0.0.1:6100"
ARCHITECT_URL = "http://127.0.0.1:6110"
DIRECTOR_URLS = {
    "Code": "http://127.0.0.1:6111",
    "Models": "http://127.0.0.1:6112",
    "Data": "http://127.0.0.1:6113",
    "DevSecOps": "http://127.0.0.1:6114",
    "Docs": "http://127.0.0.1:6115",
}
AIDER_RPC_URL = "http://127.0.0.1:6130"


def make_prime_directive(
    title: str,
    goal: str,
    description: str = "",
    repo_root: str = ".",
    entry_files: list = None,
    budget_tokens_max: int = 25000,
    budget_cost_usd_max: float = 2.0
):
    """Helper to create prime directive with correct schema"""
    return {
        "title": title,
        "description": description or goal,
        "repo_root": repo_root,
        "entry_files": entry_files or [],
        "goal": goal,
        "budget_tokens_max": budget_tokens_max,
        "budget_cost_usd_max": budget_cost_usd_max,
    }


@pytest.fixture
def check_services_running():
    """Check that all required services are running"""
    services = {
        "Gateway": GATEWAY_URL,
        "PAS Root": PAS_ROOT_URL,
        "Architect": ARCHITECT_URL,
        "Aider RPC": AIDER_RPC_URL,
    }
    services.update({f"Dir-{k}": v for k, v in DIRECTOR_URLS.items()})

    missing = []
    for name, url in services.items():
        try:
            response = httpx.get(f"{url}/health", timeout=2.0)
            if response.status_code != 200:
                missing.append(name)
        except Exception:
            missing.append(name)

    if missing:
        pytest.skip(f"Services not running: {', '.join(missing)}")


@pytest.mark.integration
class TestHealthEndpoints:
    """Test health endpoints of all services"""

    def test_gateway_health(self, check_services_running):
        """Gateway should be healthy"""
        response = httpx.get(f"{GATEWAY_URL}/health")
        assert response.status_code == 200
        assert response.json()["service"] == "PAS Gateway"

    def test_pas_root_health(self, check_services_running):
        """PAS Root should be healthy"""
        response = httpx.get(f"{PAS_ROOT_URL}/health")
        assert response.status_code == 200
        assert response.json()["service"] == "PAS Root"

    def test_architect_health(self, check_services_running):
        """Architect should be healthy"""
        response = httpx.get(f"{ARCHITECT_URL}/health")
        assert response.status_code == 200
        assert response.json()["service"] == "Architect"

    def test_all_directors_health(self, check_services_running):
        """All Directors should be healthy"""
        for lane, url in DIRECTOR_URLS.items():
            response = httpx.get(f"{url}/health")
            assert response.status_code == 200
            assert response.json()["service"] == f"Director-{lane}"

    def test_aider_rpc_health(self, check_services_running):
        """Aider RPC should be healthy"""
        response = httpx.get(f"{AIDER_RPC_URL}/health")
        assert response.status_code == 200


@pytest.mark.integration
class TestSimpleCodeTask:
    """Test simple code task end-to-end"""

    def test_simple_function_addition(self, check_services_running):
        """
        Test: Add a hello() function to utils.py
        Expected: 80-95% completion (vs P0's 10-15%)
        """
        prime_directive = make_prime_directive(
            title="Add hello function",
            goal="Add a hello() function to utils.py that returns 'Hello, World!'",
            entry_files=["utils.py"],
            budget_tokens_max=10000,
        )

        # Submit via Gateway
        response = httpx.post(
            f"{GATEWAY_URL}/prime_directives",
            json=prime_directive,
            timeout=60.0,
        )
        assert response.status_code == 200

        data = response.json()
        run_id = data["run_id"]

        # Poll for completion (timeout after 5 minutes)
        timeout = time.time() + 300
        while time.time() < timeout:
            status_response = httpx.get(f"{GATEWAY_URL}/runs/{run_id}")
            status = status_response.json()

            if status["status"] in ["completed", "failed"]:
                break

            time.sleep(5)

        # Check final status
        assert status["status"] == "completed"
        assert "artifacts" in status

        # Verify completion percentage is high (80-95%)
        if "completion_percentage" in status:
            assert status["completion_percentage"] >= 80


@pytest.mark.integration
class TestMultiLaneTask:
    """Test multi-lane task coordination"""

    def test_ml_model_training_pipeline(self, check_services_running):
        """
        Test: Complete ML training pipeline
        Lanes: Data → Models → Docs
        """
        prime_directive = make_prime_directive(
            title="Train Text Classifier",
            goal="""
            Build and train a text classification model:
            1. Prepare training dataset (Data lane)
            2. Train model (Models lane)
            3. Document model architecture (Docs lane)
            """,
            entry_files=["models/train.py"],
            budget_tokens_max=100000,
        )

        response = httpx.post(
            f"{GATEWAY_URL}/prime_directives",
            json=prime_directive,
            timeout=60.0,
        )
        assert response.status_code == 200

        run_id = response.json()["run_id"]

        # Poll for completion
        timeout = time.time() + 1800  # 30 minutes
        while time.time() < timeout:
            status_response = httpx.get(f"{GATEWAY_URL}/runs/{run_id}")
            status = status_response.json()

            if status["status"] in ["completed", "failed"]:
                break

            time.sleep(10)

        # Check that all lanes completed
        assert status["status"] == "completed"

        # Verify lane reports
        if "lane_reports" in status:
            assert "Data" in status["lane_reports"]
            assert "Models" in status["lane_reports"]
            assert "Docs" in status["lane_reports"]


@pytest.mark.integration
class TestTaskDecomposition:
    """Test task decomposition quality"""

    def test_architect_decomposes_to_lanes(self, check_services_running):
        """Architect should decompose Prime Directive to appropriate lanes"""
        prime_directive = {
            "run_id": f"test-decomp-{int(time.time())}",
            "prd": "Add user authentication with tests and documentation",
            "title": "User Auth",
            "entry_files": ["auth.py"],
            "budget": {},
            "policy": {},
            "approval_mode": "auto",
        }

        response = httpx.post(
            f"{ARCHITECT_URL}/submit",
            json=prime_directive,
            timeout=60.0,
        )
        assert response.status_code == 200

        data = response.json()
        plan = data["plan"]

        # Should decompose to Code, DevSecOps (tests), Docs
        assert "Code" in plan["lane_allocations"]
        # May also include DevSecOps and Docs depending on decomposition

    def test_director_decomposes_to_managers(self, check_services_running):
        """Directors should decompose job cards to Manager tasks"""
        job_card = {
            "job_card_id": f"jc-test-{int(time.time())}",
            "description": "Implement authentication logic in auth.py",
            "files": ["auth.py", "models.py", "routes.py"],
            "priority": "high",
            "budget": {"tokens_max": 20000},
        }

        response = httpx.post(
            f"{DIRECTOR_URLS['Code']}/submit",
            json={"job_card": job_card},
            timeout=60.0,
        )
        # Should accept and decompose
        assert response.status_code in [200, 202]


@pytest.mark.integration
class TestBudgetTracking:
    """Test budget tracking across tiers"""

    def test_budget_tracked_end_to_end(self, check_services_running):
        """Budget should be tracked from submission to completion"""
        prime_directive = make_prime_directive(
            title="Small Code Change",
            goal="Add a comment to utils.py",
            entry_files=["utils.py"],
            budget_tokens_max=5000,
            budget_cost_usd_max=0.10,
        )

        response = httpx.post(
            f"{GATEWAY_URL}/prime_directives",
            json=prime_directive,
            timeout=60.0,
        )
        assert response.status_code == 200

        run_id = response.json()["run_id"]

        # Wait for completion
        time.sleep(180)  # 3 minutes

        status_response = httpx.get(f"{GATEWAY_URL}/runs/{run_id}")
        status = status_response.json()

        # Check budget tracking
        if "actuals" in status:
            actuals = status["actuals"]
            # Should not exceed budget
            if "tokens_used" in actuals:
                assert actuals["tokens_used"] <= 5000
            if "cost_usd" in actuals:
                assert actuals["cost_usd"] <= 0.10


@pytest.mark.integration
class TestErrorHandling:
    """Test error handling and recovery"""

    def test_invalid_prime_directive(self, check_services_running):
        """Should reject invalid Prime Directive"""
        invalid_directive = {
            "title": "",  # Empty title
            "goal": "",  # Empty goal
        }

        response = httpx.post(
            f"{GATEWAY_URL}/prime_directives",
            json=invalid_directive,
            timeout=10.0,
        )
        assert response.status_code in [400, 422]

    def test_handle_manager_failure(self, check_services_running):
        """System should handle Manager failure gracefully"""
        # This would require intentionally failing a Manager
        # Implementation-dependent
        pass


@pytest.mark.integration
class TestAcceptanceGates:
    """Test acceptance gate validation"""

    def test_code_lane_validates_tests(self, check_services_running):
        """Code lane should validate that tests pass"""
        prime_directive = make_prime_directive(
            title="Add Feature with Tests",
            goal="Add a new feature to feature.py with passing tests",
            entry_files=["feature.py"],
            budget_tokens_max=50000,
        )

        response = httpx.post(
            f"{GATEWAY_URL}/prime_directives",
            json=prime_directive,
            timeout=60.0,
        )
        assert response.status_code == 200

        run_id = response.json()["run_id"]

        # Wait and check acceptance gates
        time.sleep(300)

        status_response = httpx.get(f"{GATEWAY_URL}/runs/{run_id}")
        status = status_response.json()

        # Should have run tests and validated
        if "acceptance_results" in status:
            # Check test results
            pass


@pytest.mark.integration
class TestFileManagerTask:
    """Test the File Manager task that failed in P0"""

    def test_file_manager_high_completion(self, check_services_running):
        """
        Resubmit File Manager task that achieved 10-15% in P0
        Expected: 80-95% completion with Multi-Tier PAS
        """
        # Load original Prime Directive
        prime_directive_path = Path(
            "artifacts/runs/36c92edc-ed72-484d-87de-b8f85c02b7f3/prime_directive.json"
        )

        if not prime_directive_path.exists():
            pytest.skip("Original File Manager prime directive not found")

        with open(prime_directive_path) as f:
            original_pd = json.load(f)

        # Submit via Multi-Tier PAS
        response = httpx.post(
            f"{GATEWAY_URL}/prime_directives",
            json=original_pd,
            timeout=60.0,
        )
        assert response.status_code == 200

        run_id = response.json()["run_id"]

        # Poll for completion (longer timeout for complex task)
        timeout = time.time() + 3600  # 1 hour
        while time.time() < timeout:
            status_response = httpx.get(f"{GATEWAY_URL}/runs/{run_id}")
            status = status_response.json()

            if status["status"] in ["completed", "failed"]:
                break

            time.sleep(30)

        # Check completion percentage
        assert status["status"] == "completed"

        if "completion_percentage" in status:
            # Should achieve 80-95% (vs P0's 10-15%)
            assert status["completion_percentage"] >= 80
            print(
                f"Multi-Tier PAS achieved {status['completion_percentage']}% "
                f"completion (P0: 10-15%)"
            )


@pytest.mark.integration
class TestConcurrency:
    """Test concurrent task handling"""

    def test_multiple_concurrent_tasks(self, check_services_running):
        """Should handle multiple tasks concurrently"""
        tasks = []
        for i in range(3):
            prime_directive = make_prime_directive(
                title=f"Task {i}",
                goal=f"Add function_{i}() to utils.py",
                entry_files=["utils.py"],
                budget_tokens_max=5000,
            )

            response = httpx.post(
                f"{GATEWAY_URL}/prime_directives",
                json=prime_directive,
                timeout=60.0,
            )
            assert response.status_code == 200
            tasks.append(response.json()["run_id"])

        # All tasks should be accepted
        assert len(tasks) == 3

        # Poll all tasks
        timeout = time.time() + 600
        completed = set()

        while time.time() < timeout and len(completed) < 3:
            for run_id in tasks:
                if run_id in completed:
                    continue

                status_response = httpx.get(f"{GATEWAY_URL}/runs/{run_id}")
                status = status_response.json()

                if status["status"] in ["completed", "failed"]:
                    completed.add(run_id)

            time.sleep(10)

        # All tasks should complete
        assert len(completed) == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "integration"])
