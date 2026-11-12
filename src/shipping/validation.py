"""Reject bad input at the boundary, before fitting a model or saving a batch."""

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype

from shipping.config import SEGMENTS, SERVICES

NUMERIC = [
    "shipping_zone", "weight_lbs", "length_in", "width_in", "height_in",
    "fuel_surcharge_pct", "estimated_shipping_cost", "quoted_price",
]
QUOTE_COLUMNS = [
    "quote_id", "quote_timestamp", "shipping_zone", "service_level", "customer_segment",
    "weight_lbs", "length_in", "width_in", "height_in", "fuel_surcharge_pct",
    "estimated_shipping_cost", "quoted_price",
]
HISTORY_COLUMNS = QUOTE_COLUMNS + ["accepted_quote", "outcome_timestamp"]


class DataQualityError(ValueError):
    pass


def validate_quotes(frame, *, labeled):
    """Return a copy with UTC timestamps; never silently coerce numeric strings."""
    required = HISTORY_COLUMNS if labeled else QUOTE_COLUMNS
    missing = set(required) - set(frame.columns)
    if missing or frame.empty:
        raise DataQualityError(f"Missing columns {sorted(missing)} or empty batch")
    if not labeled and {"accepted_quote", "outcome_timestamp"} & set(frame.columns):
        raise DataQualityError("Daily scoring input must have no outcome or label columns")
    result = frame.copy()
    if result[required].isna().any().any():
        raise DataQualityError("Required values cannot be null")
    ids = result.quote_id
    if not ids.map(lambda x: isinstance(x, str) and bool(x.strip())).all():
        raise DataQualityError("quote_id must be a nonempty string")
    if ids.duplicated().any():
        raise DataQualityError("Duplicate quote IDs")
    for name in NUMERIC + (["accepted_quote"] if labeled else []):
        if not is_numeric_dtype(result[name]) or is_bool_dtype(result[name]):
            raise DataQualityError(f"{name} must have numeric dtype")
        if not np.isfinite(result[name]).all():
            raise DataQualityError(f"{name} must be finite")
    positive = [x for x in NUMERIC if x not in ("shipping_zone", "fuel_surcharge_pct")]
    if (result[positive] <= 0).any().any():
        raise DataQualityError("Weights, dimensions, costs, and prices must be positive")
    if not result.shipping_zone.isin(range(2, 9)).all():
        raise DataQualityError("Shipping zones must be integers from 2 through 8")
    if not result.service_level.isin(SERVICES).all():
        raise DataQualityError("Unknown service level")
    if not result.customer_segment.isin(SEGMENTS).all():
        raise DataQualityError("Unknown customer segment")
    if not result.fuel_surcharge_pct.between(0, 1).all():
        raise DataQualityError("Fuel surcharge is a fraction in [0, 1]")
    for name in ["quote_timestamp"] + (["outcome_timestamp"] if labeled else []):
        if is_numeric_dtype(result[name]):
            raise DataQualityError(f"Invalid {name}: expected a timestamp, not an epoch number")
        try:
            result[name] = pd.to_datetime(result[name], utc=True, errors="raise", format="mixed")
        except (ValueError, TypeError) as exc:
            raise DataQualityError(f"Invalid {name}") from exc
        if result[name].isna().any():
            raise DataQualityError(f"Invalid {name}")
    if labeled:
        if not result.accepted_quote.isin([0, 1]).all():
            raise DataQualityError("accepted_quote must be 0 or 1")
        if (result.outcome_timestamp < result.quote_timestamp).any():
            raise DataQualityError("An outcome cannot precede its quote")
    return result


def labels_available(frame, cutoff):
    """A training row exists only after its outcome has actually been observed."""
    valid = validate_quotes(frame, labeled=True)
    return valid.loc[valid.outcome_timestamp < pd.Timestamp(cutoff)].copy()
