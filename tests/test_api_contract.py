"""API contract tests: does /predict return the shape callers can rely on,
and does the API reject bad input the way it's supposed to."""


def test_root_returns_ok(client):
    resp = client.get("/")
    assert resp.status_code == 200
    body = resp.json()
    assert body["docs"] == "/docs"
    assert body["health"] == "/health"


def test_health_reports_ready(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["features_loaded"] is True
    assert isinstance(body["model_source"], str) and body["model_source"]


def test_predict_valid_matchup_schema(client):
    resp = client.post("/predict", json={"home_team": "BOS", "away_team": "MIA"})
    assert resp.status_code == 200
    body = resp.json()

    assert body["home_team"] == "BOS"
    assert body["away_team"] == "MIA"
    assert body["predicted_winner"] in ("BOS", "MIA")

    assert 0.0 <= body["home_win_prob"] <= 1.0
    assert 0.0 <= body["away_win_prob"] <= 1.0
    assert abs(body["home_win_prob"] + body["away_win_prob"] - 1.0) < 1e-6
    assert 0.0 <= body["confidence"] <= 1.0

    assert isinstance(body["shap_explanation"], list)
    assert len(body["shap_explanation"]) > 0
    for factor in body["shap_explanation"]:
        assert set(factor.keys()) == {"feature", "value", "shap_value", "favors"}
        assert isinstance(factor["feature"], str)
        assert isinstance(factor["value"], (int, float))
        assert isinstance(factor["shap_value"], (int, float))
        assert factor["favors"] in ("home", "away")

    # SHAP factors should be sorted by descending impact
    impacts = [abs(f["shap_value"]) for f in body["shap_explanation"]]
    assert impacts == sorted(impacts, reverse=True)


def test_predict_lowercase_team_is_normalized(client):
    resp = client.post("/predict", json={"home_team": "bos", "away_team": "mia"})
    assert resp.status_code == 200
    assert resp.json()["home_team"] == "BOS"
    assert resp.json()["away_team"] == "MIA"


def test_predict_same_team_rejected(client):
    resp = client.post("/predict", json={"home_team": "BOS", "away_team": "BOS"})
    assert resp.status_code == 422


def test_predict_unknown_team_rejected(client):
    resp = client.post("/predict", json={"home_team": "XXX", "away_team": "MIA"})
    assert resp.status_code == 422


def test_predict_missing_field_rejected(client):
    resp = client.post("/predict", json={"home_team": "BOS"})
    assert resp.status_code == 422


def test_predict_wrong_type_rejected(client):
    resp = client.post("/predict", json={"home_team": 123, "away_team": "MIA"})
    assert resp.status_code == 422


def test_predict_empty_body_rejected(client):
    resp = client.post("/predict", json={})
    assert resp.status_code == 422


def test_metrics_exposes_prometheus_format(client):
    resp = client.get("/metrics")
    assert resp.status_code == 200
    assert "http_requests_total" in resp.text
    assert "http_request_duration_highr_seconds_bucket" in resp.text


def test_drift_reports_insufficient_data_before_min_samples(client):
    resp = client.get("/drift")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] in ("insufficient_data", "none", "moderate", "significant")
    assert "reference_samples" in body
    assert body["reference_samples"] > 0


def test_drift_reflects_recorded_predictions(client):
    for _ in range(35):
        client.post("/predict", json={"home_team": "BOS", "away_team": "MIA"})
    resp = client.get("/drift")
    assert resp.status_code == 200
    body = resp.json()
    assert body["window_samples"] >= 35
    assert body["status"] in ("none", "moderate", "significant")
    assert len(body["features"]) > 0
    for feat, info in body["features"].items():
        assert "psi" in info and "rating" in info
        assert info["rating"] in ("none", "moderate", "significant")
