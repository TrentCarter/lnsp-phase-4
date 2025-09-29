import os
import pytest

# Skip entire module if neo4j not available
neo4j = pytest.importorskip("neo4j")

@pytest.fixture(scope="module", autouse=True)
def _enable_graph_flag():
    """Enable GraphRAG for this test module."""
    old_value = os.environ.get("LNSP_GRAPHRAG_ENABLED")
    os.environ["LNSP_GRAPHRAG_ENABLED"] = "1"
    yield
    if old_value is None:
        os.environ.pop("LNSP_GRAPHRAG_ENABLED", None)
    else:
        os.environ["LNSP_GRAPHRAG_ENABLED"] = old_value

@pytest.fixture
def client():
    """Create FastAPI test client."""
    from fastapi.testclient import TestClient
    from src.api.retrieve import app
    return TestClient(app)

def test_graph_health(client):
    """Test graph health endpoint."""
    r = client.get("/graph/health")
    assert r.status_code in (200, 501)  # ok or disabled
    if r.status_code == 200:
        j = r.json()
        assert "concepts" in j and "edges" in j
        assert isinstance(j["concepts"], int)
        assert isinstance(j["edges"], int)

def test_graph_search_minimal(client):
    """Test graph search endpoint with basic query."""
    # Test with text query
    r = client.post("/graph/search", json={"q": "photosynthesis", "top_k": 3})
    assert r.status_code in (200, 501)
    if r.status_code == 200:
        result = r.json()
        assert isinstance(result, list)
        # If we have results, check structure
        if result:
            assert "cpe_id" in result[0]
            assert "concept_text" in result[0]

def test_graph_search_by_seed_ids(client):
    """Test graph search with seed IDs."""
    r = client.post("/graph/search", json={
        "seed_ids": ["test-id-1", "test-id-2"],
        "top_k": 5
    })
    assert r.status_code in (200, 501)
    if r.status_code == 200:
        result = r.json()
        assert isinstance(result, list)

def test_graph_hop_minimal(client):
    """Test graph hop expansion endpoint."""
    r = client.post("/graph/hop", json={
        "node_id": "cpe:nonexistent",
        "max_hops": 1,
        "top_k": 3
    })
    assert r.status_code in (200, 404, 501)  # may not exist yet
    if r.status_code == 200:
        result = r.json()
        assert isinstance(result, list)

def test_graph_search_with_lane_filter(client):
    """Test graph search with lane filtering."""
    r = client.post("/graph/search", json={
        "q": "test",
        "lane": 0,
        "top_k": 5
    })
    assert r.status_code in (200, 501)
    if r.status_code == 200:
        result = r.json()
        assert isinstance(result, list)
        # If we have results, check lane filtering
        if result:
            for item in result:
                if "lane_index" in item:
                    assert item["lane_index"] == 0

def test_graph_disabled_returns_501():
    """Test that endpoints return 501 when GraphRAG is disabled."""
    # Temporarily disable GraphRAG
    old_value = os.environ.get("LNSP_GRAPHRAG_ENABLED")
    os.environ["LNSP_GRAPHRAG_ENABLED"] = "0"

    try:
        from fastapi.testclient import TestClient
        from src.api.retrieve import app
        client = TestClient(app)

        r = client.get("/graph/health")
        assert r.status_code == 501

        r = client.post("/graph/search", json={"q": "test"})
        assert r.status_code == 501

        r = client.post("/graph/hop", json={"node_id": "test"})
        assert r.status_code == 501

    finally:
        # Restore original value
        if old_value is None:
            os.environ.pop("LNSP_GRAPHRAG_ENABLED", None)
        else:
            os.environ["LNSP_GRAPHRAG_ENABLED"] = old_value