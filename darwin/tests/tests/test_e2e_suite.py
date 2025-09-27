import time
from typing import Dict, Any, List

from fastapi.testclient import TestClient


# 1) STARTUP AND CORE ENDPOINTS

def test_startup_memory_system_initialized(client: TestClient):
    """
    Validate that the FastAPI lifespan initialized the Integrated Memory System
    and stored it on app.state.
    """
    assert hasattr(client.app.state, "memory_system")
    assert client.app.state.memory_system is not None


def test_core_health_endpoints(client: TestClient):
    """
    Validate both /health and /healthz endpoints.
    These may be served by different routers (core vs. root alias),
    so accept both legacy "ok" and new "healthy" payloads.
    """
    for path in ("/health", "/healthz"):
        resp = client.get(path)
        assert resp.status_code == 200, f"{path} should be 200"
        data = resp.json()
        assert isinstance(data, dict)
        assert data.get("status") in ("ok", "healthy"), f"Unexpected status for {path}: {data}"


def test_core_info_endpoint(client: TestClient):
    """
    Validate /info returns expected structure with basic fields.
    """
    resp = client.get("/info")
    assert resp.status_code == 200
    data = resp.json()
    # Basic schema checks
    for key in ("name", "version", "environment", "started_at", "uptime_seconds", "python_version", "platform", "capabilities"):
        assert key in data, f"Missing key in /info response: {key}"
    assert isinstance(data["capabilities"], dict)
    # Ensure values have expected types
    assert isinstance(data["name"], str)
    assert isinstance(data["version"], str)
    assert isinstance(data["uptime_seconds"], (int, float))


# 2) NOTEBOOKS AND DATA ENDPOINTS (safe I/O only, no external dependencies)

def test_notebooks_list(client: TestClient):
    """
    Validate /notebooks returns a predictable structure:
    { "notebooks": [...], "count": int }
    """
    resp = client.get("/notebooks")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, dict)
    assert "count" in data
    assert isinstance(data["count"], int)
    # notebooks may be an array (can be empty)
    assert "notebooks" in data


def test_data_ag5_datasets(client: TestClient):
    """
    Validate /data/ag5/datasets returns a list and count.
    """
    resp = client.get("/data/ag5/datasets")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, dict)
    assert "datasets" in data and isinstance(data["datasets"], list)
    assert "count" in data and isinstance(data["count"], int)


def test_data_ag5_search(client: TestClient):
    """
    Validate /data/ag5/search?q=... returns expected structure.
    """
    q = "test"
    resp = client.get(f"/data/ag5/search?q={q}")
    assert resp.status_code == 200
    data = resp.json()
    # Accept flexible schema but ensure basics
    assert isinstance(data, dict)
    assert "count" in data
    assert "results" in data
    assert "query" in data


def test_data_helio_summaries(client: TestClient):
    """
    Validate /data/helio/summaries?limit=1 returns a list (can be empty).
    """
    resp = client.get("/data/helio/summaries?limit=1")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)
    # If elements are present, ensure dict-like structure
    if data:
        assert isinstance(data[0], dict)


# 3) SECURITY (BÁSICA): endpoints que exigem autenticação devem negar acesso sem credenciais

def test_processing_requires_auth(client: TestClient):
    """
    /processing endpoints depend on authentication (verify_token).
    Without Authorization header, expect 401/403.
    """
    # POST /processing/jobs without Authorization
    payload: Dict[str, Any] = {
        "task_name": "noop",
        "args": [],
        "kwargs": {},
        "priority": "normal"
    }
    r1 = client.post("/processing/jobs", json=payload)
    assert r1.status_code in (401, 403), f"Expected 401/403, got {r1.status_code}"

    # GET /processing/jobs/{job_id} without Authorization
    r2 = client.get("/processing/jobs/invalid-id")
    assert r2.status_code in (401, 403), f"Expected 401/403, got {r2.status_code}"


# 4) PERFORMANCE (SMOKE): latência média/percentil em /healthz sob baixa carga

def test_performance_health_smoke(client: TestClient):
    """
    Smoke performance check for /healthz:
    - Make a small number of sequential requests
    - Assert average and worst-case latency are within generous bounds
    This is not a load test; just a quick regression check.
    """
    n = 10
    latencies_ms: List[float] = []
    for _ in range(n):
        t0 = time.perf_counter()
        resp = client.get("/healthz")
        t1 = time.perf_counter()
        assert resp.status_code == 200
        latencies_ms.append((t1 - t0) * 1000.0)

    avg = sum(latencies_ms) / len(latencies_ms)
    worst = max(latencies_ms)

    # Generous thresholds to avoid flakiness in CI environments:
    assert avg < 1000.0, f"Average /healthz latency too high: {avg:.2f} ms"
    assert worst < 2000.0, f"Worst-case /healthz latency too high: {worst:.2f} ms"