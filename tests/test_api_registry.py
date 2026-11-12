import copy
import json
import shutil

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from shipping.api import app, example_quote
from shipping.modeling import chronological_split
from shipping.registry import (
    ApprovalError,
    ModelUnavailable,
    load_approved,
    promote,
    read_registry,
    rollback,
)
from shipping.storage import load_history


def test_api_valid_and_offline_preprocessing_consistency(project):
    quote = example_quote()
    model, _, entry = load_approved()
    offline = model.predict_proba(pd.DataFrame([quote]))[0, 1]
    with TestClient(app) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/ready").json()["model_version"] == entry["version"]
        assert client.get("/model-info").json()["evaluation"]["version"] == entry["version"]
        response = client.post("/predict", json=quote)
        assert response.status_code == 200
        assert response.json()["acceptance_probability"] == pytest.approx(offline, abs=1e-8)
        quote.pop("quoted_price")
        result = client.post("/recommend", json=dict(quote, maximum_price=30))
        assert result.status_code == 200
        assert 14 <= result.json()["recommended_price"] <= 30


@pytest.mark.parametrize("change", [
    {"length_in": -1}, {"weight_lbs": 0}, {"shipping_zone": 9},
    {"shipping_zone": 4.5}, {"service_level": "teleport"},
    {"quote_timestamp": "bad-date"}, {"quote_timestamp": "2025-01-01T00:00:00"},
    {"accepted_quote": 1}, {"estimated_shipping_cost": "NaN"},
])
def test_api_rejects_invalid_requests(project, change):
    with TestClient(app) as client:
        response = client.post("/predict", json={**example_quote(), **change})
        assert response.status_code == 422


def test_api_rejects_infeasible_price_range(project):
    quote = example_quote()
    quote.pop("quoted_price")
    with TestClient(app) as client:
        assert client.post("/recommend", json=dict(quote, maximum_price=10)).status_code == 422


def test_missing_model_readiness_and_inference_503(tmp_path, monkeypatch):
    monkeypatch.setenv("UPS_PRICING_HOME", str(tmp_path))
    with TestClient(app) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/ready").status_code == 503
        assert client.get("/model-info").status_code == 503
        assert client.post("/predict", json=example_quote()).status_code == 503
        quote = example_quote()
        quote.pop("quoted_price")
        assert client.post("/recommend", json=quote).status_code == 503


def test_failed_approval_keeps_previous_model(project):
    _, report, _ = load_approved()
    before = (project.artifacts / "approved.json").read_bytes()
    broken = copy.deepcopy(report)
    broken["validation"][report["selected_model"]]["log_loss"] = 2.0
    with pytest.raises(ApprovalError):
        promote(broken, chronological_split(load_history())["validation"])
    assert (project.artifacts / "approved.json").read_bytes() == before
    load_approved()


def test_incomplete_model_cannot_replace_current(project):
    _, report, _ = load_approved()
    before = read_registry()
    incomplete = copy.deepcopy(report)
    incomplete["version"] = "model-0000000000000000"
    with pytest.raises(FileNotFoundError):
        promote(incomplete, chronological_split(load_history())["validation"])
    assert read_registry() == before


def test_atomic_promotion_rollback_and_corruption_fallback(project):
    # Clone trusted test artifacts to exercise two complete registry versions, without retuning.
    _, report, first = load_approved()
    second_report = copy.deepcopy(report)
    second_report["version"] = "model-abcdef0123456789"
    directory = project.artifacts / "models" / second_report["version"]
    shutil.copytree(project.artifacts / "models" / first["version"], directory)
    (directory / "evaluation.json").write_text(json.dumps(second_report))
    second = promote(second_report, chronological_split(load_history())["validation"])
    assert read_registry()["previous"] == first
    assert rollback()["version"] == first["version"]
    assert read_registry()["previous"] == second
    # Current artifact corruption falls back to the other complete known-good version.
    current_path = project.artifacts / "models" / first["version"] / f'{first["model_name"]}.joblib'
    current_path.write_bytes(b"incomplete")
    assert load_approved()[2]["version"] == second["version"]
    (directory / f'{second["model_name"]}.joblib').write_bytes(b"also-corrupt")
    with pytest.raises(ModelUnavailable):
        load_approved()
