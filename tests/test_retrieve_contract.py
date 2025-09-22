import json, os, pytest, requests

API = os.getenv("API_URL", "http://localhost:8080/search")

@pytest.mark.contract
def test_search_contract():
    payload = {"q": "Define FactoidWiki", "lane": "L1_FACTOID", "top_k": 5}
    r = requests.post(API, json=payload, timeout=10)
    assert r.status_code == 200
    js = r.json()
    assert isinstance(js.get("items"), list)
    assert all("id" in it for it in js["items"])
    # optional but nice to have
    assert js.get("lane") in {"L1_FACTOID", "L2_GRAPH", "L3_SYNTH"}
    assert js.get("mode") in {"DENSE", "GRAPH", "HYBRID"}

@pytest.mark.contract
def test_missing_query_field():
    """Test that missing 'q' field returns 422."""
    payload = {"lane": "L1_FACTOID", "top_k": 5}
    r = requests.post(API, json=payload, timeout=10)
    assert r.status_code == 422
    js = r.json()
    assert "detail" in js

@pytest.mark.contract
def test_invalid_lane():
    """Test that invalid lane returns 422."""
    payload = {"q": "test query", "lane": "INVALID_LANE", "top_k": 5}
    r = requests.post(API, json=payload, timeout=10)
    assert r.status_code == 422
    js = r.json()
    assert "detail" in js

@pytest.mark.contract
def test_query_too_long():
    """Test that overly long query (>512 chars) returns 422."""
    long_query = "x" * 600  # Exceeds 512 char limit
    payload = {"q": long_query, "lane": "L1_FACTOID", "top_k": 5}
    r = requests.post(API, json=payload, timeout=10)
    assert r.status_code == 422
    js = r.json()
    assert "detail" in js

@pytest.mark.contract
def test_empty_query():
    """Test that empty query returns 422."""
    payload = {"q": "", "lane": "L1_FACTOID", "top_k": 5}
    r = requests.post(API, json=payload, timeout=10)
    assert r.status_code == 422
    js = r.json()
    assert "detail" in js

@pytest.mark.contract
def test_invalid_top_k_values():
    """Test that invalid top_k values return 422."""
    # top_k too small
    payload = {"q": "test", "lane": "L1_FACTOID", "top_k": 0}
    r = requests.post(API, json=payload, timeout=10)
    assert r.status_code == 422

    # top_k too large
    payload = {"q": "test", "lane": "L1_FACTOID", "top_k": 101}
    r = requests.post(API, json=payload, timeout=10)
    assert r.status_code == 422

@pytest.mark.contract
def test_trace_id_echo():
    """Test that trace_id is echoed in response when provided."""
    trace_id = "test-trace-123"
    payload = {"q": "Define FactoidWiki", "lane": "L1_FACTOID", "top_k": 5}
    headers = {"x-trace-id": trace_id}
    r = requests.post(API, json=payload, headers=headers, timeout=10)
    assert r.status_code == 200
    js = r.json()
    assert js.get("trace_id") == trace_id

@pytest.mark.contract
def test_trace_id_generated():
    """Test that trace_id is generated when not provided."""
    payload = {"q": "Define FactoidWiki", "lane": "L1_FACTOID", "top_k": 5}
    r = requests.post(API, json=payload, timeout=10)
    assert r.status_code == 200
    js = r.json()
    assert js.get("trace_id") is not None
    assert len(js.get("trace_id", "")) > 0