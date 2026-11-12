"""Deterministic fictional quotes. No UPS data, connections, or personal information."""

from datetime import date

import numpy as np
import pandas as pd

from shipping.config import DEMO_DAY, HISTORY_SEED, SEED, SEGMENTS, SERVICES, paths
from shipping.validation import HISTORY_COLUMNS, QUOTE_COLUMNS, validate_quotes


def _quote_inputs(n, start, days, rng, prefix):
    # Stratified days guarantee date coverage, while times and packages remain random.
    offsets = np.arange(n) % days
    rng.shuffle(offsets)
    timestamps = pd.Timestamp(start, tz="UTC") + pd.to_timedelta(offsets, unit="D")
    timestamps += pd.to_timedelta(rng.integers(0, 86400, n), unit="s")
    zone = rng.integers(2, 9, n)
    service = rng.choice(SERVICES, n, p=[0.60, 0.26, 0.14])
    segment = rng.choice(SEGMENTS, n, p=[0.35, 0.45, 0.20])
    weight = np.round(np.clip(rng.lognormal(1.5, 0.8, n), 0.2, 75), 2)
    length = np.round(rng.uniform(4, 30, n), 2)
    width = np.round(rng.uniform(3, 22, n), 2)
    height = np.round(rng.uniform(2, 18, n), 2)
    fuel = np.round(0.12 + 0.025 * np.sin(offsets / 25), 4)
    billable_weight = np.maximum(weight, length * width * height / 139)
    speed_multiplier = pd.Series(service).map({"ground": 1, "two_day": 1.55, "next_day": 2.3})
    cost = np.round((3.4 + 0.42 * zone + 0.31 * billable_weight) * speed_multiplier * (1 + fuel), 2)
    # Independent random markups provide overlapping historical price variation.
    price = np.round(cost * rng.uniform(1.12, 2.60, n), 2)
    return pd.DataFrame({
        "quote_id": [f"{prefix}-{i:06d}" for i in range(n)],
        "quote_timestamp": timestamps, "shipping_zone": zone,
        "service_level": service, "customer_segment": segment, "weight_lbs": weight,
        "length_in": length, "width_in": width, "height_in": height,
        "fuel_surcharge_pct": fuel, "estimated_shipping_cost": cost, "quoted_price": price,
    })[QUOTE_COLUMNS]


def _simulated_acceptance_probability(frame, rng):
    """Simulation oracle; deliberately kept out of features, fitting, and serving."""
    markup = frame.quoted_price / frame.estimated_shipping_cost
    day = (frame.quote_timestamp - pd.Timestamp("2025-01-01", tz="UTC")).dt.days
    # This hidden simulator is used ONLY to draw historical outcomes, never as a feature.
    sensitivity = frame.customer_segment.map({"occasional": 4.0, "small_business": 5.5, "enterprise": 7.0})
    speed = frame.service_level.map({"ground": 0.0, "two_day": 0.6, "next_day": 1.2})
    logit = (
        1.9 - sensitivity * (markup - 1.35) + speed
        + 0.5 * (frame.shipping_zone >= 6)
        + 0.7 * ((frame.service_level == "next_day") & (frame.shipping_zone >= 6))
        - 0.5 * (frame.weight_lbs > 12) + 0.35 * np.sin(day / 25)
        + 0.25 * (frame.quote_timestamp.dt.dayofweek < 5) + rng.normal(0, 0.25, len(frame))
    )
    return 1 / (1 + np.exp(-logit))


def generate_history(n=10_000, days=240, seed=HISTORY_SEED):
    if n < days or days < 180:
        raise ValueError("History needs at least 180 days and one quote per day")
    rng = np.random.default_rng(seed)
    frame = _quote_inputs(n, "2025-01-01", days, rng, "hist")
    probability = _simulated_acceptance_probability(frame, rng)
    frame["accepted_quote"] = rng.binomial(1, probability)
    frame["outcome_timestamp"] = frame.quote_timestamp + pd.to_timedelta(
        rng.integers(1, 73, n), unit="h"
    )
    return validate_quotes(frame[HISTORY_COLUMNS], labeled=True).sort_values("quote_timestamp")


def generate_daily(batch_date=DEMO_DAY, n=200, seed=SEED):
    day = date.fromisoformat(batch_date)
    rng = np.random.default_rng(seed + day.toordinal())
    return validate_quotes(
        _quote_inputs(n, batch_date, 1, rng, f"daily-{batch_date}"), labeled=False
    )


def bootstrap(batch_date=DEMO_DAY):
    from shipping.storage import save_quotes

    p = paths()
    historical = generate_history()
    daily = generate_daily(batch_date)
    historical.to_csv(p.data / "historical_quotes.csv", index=False)
    daily.to_csv(p.data / f"quotes-{batch_date}.csv", index=False)
    save_quotes(historical, labeled=True)
    save_quotes(daily, labeled=False, batch_date=batch_date)
    return {"historical_quotes": len(historical), "daily_quotes": len(daily), "days": 240}
