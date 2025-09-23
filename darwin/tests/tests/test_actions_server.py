import os
import yaml
import pytest
from fastapi.testclient import TestClient

from src.kec_biomat_api.main import app


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.setenv("KEC_API_KEY", "test-key")
    with TestClient(app) as test_client:
        yield test_client


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "uptime_seconds" in data


def test_compute_requires_api_key(client):
    # Missing header should be rejected when KEC_API_KEY is set
    response = client.post("/kec/compute", json={"graph_id": "demo"})
    assert response.status_code == 401


def test_compute_returns_metrics(client):
    response = client.post(
        "/kec/compute",
        json={"graph_id": "demo-001", "sigma_q": True},
        headers={"X-API-Key": "test-key"},
    )
    assert response.status_code == 200
    payload = response.json()
    assert set(payload.keys()) == {"H_spectral", "k_forman_mean", "sigma", "swp"}


def test_job_status_response(client):
    response = client.get("/jobs/job-123", headers={"X-API-Key": "test-key"})
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == "job-123"
    assert data["status"] in {"queued", "running", "done", "error"}


def test_openapi_yaml_endpoint(client, tmp_path, monkeypatch):
    # Ensure endpoint serves bundled openapi.yaml by pointing env to a temp file
    temp_spec = tmp_path / "openapi.yaml"
    temp_spec.write_text("openapi: 3.1.0\ninfo:\n  title: Test\n", encoding="utf-8")
    monkeypatch.setenv("KEC_OPENAPI_PATH", str(temp_spec))

    response = client.get("/openapi.yaml")
    assert response.status_code == 200
    doc = yaml.safe_load(response.text)
    assert doc["openapi"].startswith("3.1")


def test_default_openapi_yaml(client, monkeypatch):
    # Remove override and ensure repository spec is served (should include server URL)
    monkeypatch.delenv("KEC_OPENAPI_PATH", raising=False)
    response = client.get("/openapi.yaml")
    assert response.status_code == 200
    doc = yaml.safe_load(response.text)
    assert doc["openapi"].startswith("3.1")
    assert doc["servers"][0]["url"] == "https://api.agourakis.med.br"
