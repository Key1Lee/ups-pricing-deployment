"""Real HTTP requests against uvicorn or Docker, not an in-process TestClient."""

import time

import httpx

from shipping.api import PredictionResponse, RecommendationResponse, example_quote


def smoke(url="http://127.0.0.1:8000"):
    with httpx.Client(base_url=url, timeout=15) as client:
        for attempt in range(60):
            try:
                if client.get("/ready").status_code == 200:
                    break
            except httpx.TransportError:
                pass
            time.sleep(0.25)
        else:
            raise RuntimeError(f"API did not become ready at {url}")
        statuses = {}
        for endpoint in ["/health", "/ready", "/model-info"]:
            response = client.get(endpoint)
            response.raise_for_status()
            statuses[endpoint] = response.status_code
        model_info = response.json()
        quote = example_quote()
        response = client.post("/predict", json=quote)
        response.raise_for_status()
        prediction = PredictionResponse.model_validate(response.json())
        statuses["/predict"] = response.status_code
        quote.pop("quoted_price")
        response = client.post("/recommend", json={**quote, "maximum_price": 30})
        response.raise_for_status()
        result = RecommendationResponse.model_validate(response.json())
        statuses["/recommend"] = response.status_code
        assert quote["estimated_shipping_cost"] + 2 <= result.recommended_price <= 30
        assert abs(result.expected_contribution - result.predicted_acceptance_probability * (
            result.recommended_price - quote["estimated_shipping_cost"])) < 1e-6
        assert result.model_version == prediction.model_version == model_info["model_version"]
        invalid = client.post("/recommend", json={**quote, "length_in": -1})
        assert invalid.status_code == 422
        statuses["invalid_recommendation"] = invalid.status_code
        return {"base_url": url, "statuses": statuses, "prediction": prediction.model_dump(),
                "recommendation": result.model_dump(), "passed": True}
