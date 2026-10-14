from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health():
    assert client.get("/health").json() == {"status": "ok", "trace_version": "1.0.0"}


def test_validated_request_to_trace():
    response = client.post("/api/neuron", json={"x": [2, 1.5], "w": [.4, .7], "b": .1, "activation": "relu"})
    assert response.status_code == 200
    assert response.json()["steps"][1]["values"]["a"] == 1.95


def test_shape_error_is_explained_not_server_error():
    response = client.post("/api/math", json={"operation": "dot", "left": [1, 2], "right": [3]})
    assert response.status_code == 422
    assert "equal one-dimensional shapes" in response.json()["detail"]

