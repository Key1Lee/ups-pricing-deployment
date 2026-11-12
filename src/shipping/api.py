"""FastAPI serves the approved sklearn pipeline; it never trains or promotes models."""

import logging
import time
from contextlib import asynccontextmanager
from typing import Literal

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from pydantic import AwareDatetime, BaseModel, ConfigDict, Field

from shipping.config import paths
from shipping.pricing import (
    PriceRules,
    PricingError,
    UnreliablePriceCurve,
    checked_probabilities,
    recommend,
)
from shipping.registry import ModelUnavailable, load_approved
from shipping.storage import init_db, record_api_event
from shipping.validation import DataQualityError

logger = logging.getLogger(__name__)


class PackageQuote(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    quote_id: str = Field(min_length=1, max_length=100, pattern=r"^\S+$")
    quote_timestamp: AwareDatetime
    shipping_zone: int = Field(ge=2, le=8, strict=True)
    service_level: Literal["ground", "two_day", "next_day"]
    customer_segment: Literal["occasional", "small_business", "enterprise"]
    weight_lbs: float = Field(gt=0, le=1000)
    length_in: float = Field(gt=0, le=200)
    width_in: float = Field(gt=0, le=200)
    height_in: float = Field(gt=0, le=200)
    fuel_surcharge_pct: float = Field(ge=0, le=1)
    estimated_shipping_cost: float = Field(gt=0, le=10_000)


class PredictionRequest(PackageQuote):
    quoted_price: float = Field(gt=0, le=50_000)


class RecommendationRequest(PackageQuote):
    minimum_margin: float = Field(default=2.0, gt=0)
    minimum_price: float = Field(default=0.01, gt=0)
    maximum_price: float = Field(default=500.0, gt=0, le=50_000)
    price_step: float = Field(default=0.50, ge=0.01)


class PredictionResponse(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)
    acceptance_probability: float = Field(ge=0, le=1)
    model_version: str


class RecommendationResponse(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)
    quote_id: str
    recommended_price: float = Field(gt=0)
    predicted_acceptance_probability: float = Field(ge=0, le=1)
    expected_contribution: float = Field(ge=0)
    model_version: str


class ModelCache:
    def __init__(self):
        self.key = None
        self.value = None

    def get(self):
        try:
            root = paths().root
            key = (root, (paths().artifacts / "approved.json").read_bytes())
        except OSError as exc:
            raise ModelUnavailable("No approved model pointer") from exc
        if key != self.key:
            value = load_approved()  # Only swap cached state after a successful load.
            self.value, self.key = value, key
        return self.value


cache = ModelCache()


@asynccontextmanager
async def lifespan(app):
    init_db()
    try:
        cache.get()
    except ModelUnavailable:
        logger.warning("API started without an approved model; readiness returns 503")
    yield


app = FastAPI(title="Synthetic Shipping Pricing", version="0.1.0", lifespan=lifespan)


@app.middleware("http")
async def request_metrics(request: Request, call_next):
    started = time.perf_counter()
    status = 500
    try:
        response = await call_next(request)
        status = response.status_code
        return response
    finally:
        try:
            record_api_event(request.url.path, status, (time.perf_counter() - started) * 1000,
                             getattr(request.state, "model_version", None))
        except Exception:
            logger.exception("Unable to persist API event")
        if status >= 400:
            logger.warning("API request failed: endpoint=%s status=%s", request.url.path, status)


def approved(request):
    try:
        model, report, entry = cache.get()
        request.state.model_version = entry["version"]
        return model, report, entry
    except ModelUnavailable as exc:
        raise HTTPException(503, "No usable approved model") from exc


@app.get("/health")
def health():
    return {"status": "alive"}


@app.get("/ready")
def ready(request: Request):
    _, _, entry = approved(request)
    return {"status": "ready", "model_version": entry["version"]}


@app.get("/model-info")
def model_info(request: Request):
    _, report, entry = approved(request)
    return {"model_version": entry["version"], "model_name": entry["model_name"],
            "approved_at": entry["approved_at"], "approval_checks": entry["approval_checks"],
            "evaluation": report}


@app.post("/predict", response_model=PredictionResponse)
def predict(body: PredictionRequest, request: Request):
    model, _, entry = approved(request)
    try:
        probability = checked_probabilities(model, pd.DataFrame([body.model_dump(mode="json")]))[0]
    except UnreliablePriceCurve as exc:
        raise HTTPException(503, str(exc)) from exc
    except DataQualityError as exc:
        raise HTTPException(422, str(exc)) from exc
    return {"acceptance_probability": float(probability), "model_version": entry["version"]}


@app.post("/recommend", response_model=RecommendationResponse)
def recommendation(body: RecommendationRequest, request: Request):
    model, _, entry = approved(request)
    quote = body.model_dump(mode="json", exclude={"minimum_margin", "minimum_price", "maximum_price", "price_step"})
    quote["quoted_price"] = body.estimated_shipping_cost * 1.5
    rules = PriceRules(minimum_margin=body.minimum_margin, minimum_price=body.minimum_price,
                       maximum_price=body.maximum_price, step=body.price_step)
    try:
        result, _ = recommend(model, quote, entry["version"], rules)
        return result
    except UnreliablePriceCurve as exc:
        raise HTTPException(503, str(exc)) from exc
    except PricingError as exc:
        raise HTTPException(422, str(exc)) from exc


def example_quote():
    return {
        "quote_id": "interview-example", "quote_timestamp": "2025-08-29T12:00:00+00:00",
        "shipping_zone": 4, "service_level": "ground", "customer_segment": "small_business",
        "weight_lbs": 5.0, "length_in": 10.0, "width_in": 8.0, "height_in": 6.0,
        "fuel_surcharge_pct": 0.12, "estimated_shipping_cost": 12.0, "quoted_price": 18.0,
    }
