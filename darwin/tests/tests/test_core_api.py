from fastapi.testclient import TestClient


def test_health_check(client: TestClient):
    """
    Tests the /healthz endpoint to ensure the API is running.

    Accept both legacy "ok" and new "healthy" payloads depending on router precedence.
    """
    response = client.get("/healthz")
    assert response.status_code == 200
    json_response = response.json()
    assert json_response.get("status") in ("ok", "healthy")
    # Non-breaking optional fields (namespace for legacy route, or version/env for core router)
    assert isinstance(json_response, dict)
