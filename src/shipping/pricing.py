"""A small enumerated price search, with an explicit refusal path for unsafe curves."""

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from shipping.validation import validate_quotes


class PricingError(ValueError):
    pass


class UnreliablePriceCurve(PricingError):
    pass


@dataclass(frozen=True)
class PriceRules:
    minimum_margin: float = 2.0  # Dollar contribution margin before acceptance probability.
    minimum_price: float = 0.01
    maximum_price: float = 500.0
    step: float = 0.50
    minimum_markup: float = 1.12  # Stay within the randomized historical pricing support.
    maximum_markup: float = 2.60


def checked_probabilities(model, frame):
    values = np.asarray(model.predict_proba(frame))[:, 1]
    if not np.isfinite(values).all() or not ((values >= 0) & (values <= 1)).all():
        raise UnreliablePriceCurve("Nonfinite or out-of-range acceptance predictions")
    return values


def candidate_table(model, quote, rules=PriceRules()):
    # The provided offered price is irrelevant for a search; each candidate replaces it.
    raw = validate_quotes(pd.DataFrame([quote]), labeled=False).iloc[0].to_dict()
    values = vars(rules).values()
    if not all(math.isfinite(x) for x in values):
        raise PricingError("Price rules must be finite")
    if rules.minimum_margin <= 0 or rules.step < 0.01 or rules.minimum_price <= 0:
        raise PricingError("Margin and price must be positive; step must be at least one cent")
    if rules.minimum_markup < 1 or rules.maximum_markup < rules.minimum_markup:
        raise PricingError("Invalid markup bounds")
    cost = float(raw["estimated_shipping_cost"])
    lower = math.ceil(max(cost + rules.minimum_margin, rules.minimum_price,
                          cost * rules.minimum_markup) * 100 - 1e-8)
    upper = math.floor(min(rules.maximum_price, cost * rules.maximum_markup) * 100 + 1e-8)
    step = max(1, round(rules.step * 100))
    if lower > upper:
        raise PricingError("No feasible price satisfies margin, maximum, and historical support")
    if (upper - lower) // step > 5000:
        raise PricingError("Too many candidates; increase price step")
    cents = np.unique(np.append(np.arange(lower, upper + 1, step), upper))
    prices = cents / 100.0
    # Unique IDs preserve the same input validation used during training and prediction.
    frame = pd.DataFrame([dict(raw, quote_id=f"candidate-{i}", quoted_price=float(price))
                          for i, price in enumerate(prices)])
    probability = checked_probabilities(model, frame)
    # Price AND markup change together. A constraint on price alone would be insufficient.
    if len(prices) < 2 or np.any(np.diff(probability) > 1e-6):
        raise UnreliablePriceCurve("Acceptance increases with price, or curve cannot be checked")
    if np.ptp(probability) < 1e-5:
        raise UnreliablePriceCurve("Price response is flat over the allowed range")
    margin = prices - cost
    return pd.DataFrame({
        "candidate_price": prices, "predicted_probability": probability,
        "margin": margin, "expected_contribution": probability * margin,
    })


def recommend(model, quote, version, rules=PriceRules()):
    table = candidate_table(model, quote, rules)
    # Prices are ascending, so ties deterministically choose the lower price.
    best = table.loc[table.expected_contribution.idxmax()]
    return {
        "quote_id": quote["quote_id"], "recommended_price": float(best.candidate_price),
        "predicted_acceptance_probability": float(best.predicted_probability),
        "expected_contribution": float(best.expected_contribution),
        "model_version": version,
    }, table
