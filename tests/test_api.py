from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


def make_client(monkeypatch, artifact_dir):
    monkeypatch.setenv("KOSPI_API_ARTIFACT_DIR", str(artifact_dir))
    from app.config import get_settings
    get_settings.cache_clear()
    import app.main as main_module
    main_module.app.router.lifespan_context = main_module.lifespan
    return TestClient(main_module.app)


def test_health_and_latest(monkeypatch, artifact_dir):
    with make_client(monkeypatch, artifact_dir) as client:
        assert client.get("/health/ready").status_code == 200
        response = client.get("/api/v1/predictions/latest?limit=3")
        assert response.status_code == 200
        assert response.json()["total"] == 10
        assert len(response.json()["items"]) == 3


def test_single_feature_inference(monkeypatch, artifact_dir):
    with make_client(monkeypatch, artifact_dir) as client:
        info = client.get("/api/v1/model/info").json()
        payload = {
            "code": "5930",
            "close": 100000,
            "features": {feature: 0.5 for feature in info["features"]},
        }
        response = client.post("/api/v1/predict", json=payload)
        assert response.status_code == 200
        body = response.json()
        assert body["code"] == "005930"
        assert body["feature_coverage"] == 1.0
        assert body["predicted_excess_log_return_60d"] == pytest.approx(0.024)


def test_rejects_unknown_feature(monkeypatch, artifact_dir):
    with make_client(monkeypatch, artifact_dir) as client:
        info = client.get("/api/v1/model/info").json()
        features = {feature: 0.5 for feature in info["features"]}
        features["unexpected"] = 1.0
        response = client.post("/api/v1/predict", json={"code": "005930", "features": features})
        assert response.status_code == 422
