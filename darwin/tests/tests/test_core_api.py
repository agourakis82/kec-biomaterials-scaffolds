from fastapi.testclient import TestClient


def test_health_check(client: TestClient):
    """
    Tests the /healthz endpoint to ensure the API is running.
    """
    response = client.get("/healthz")
    assert response.status_code == 200
    json_response = response.json()
    assert json_response["status"] == "ok"
    assert "namespace" in json_response
