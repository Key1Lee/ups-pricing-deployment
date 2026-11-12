"""The serialized model calls this SAME quote-time transformation offline and online."""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from shipping.validation import QUOTE_COLUMNS, validate_quotes

NUMERIC_FEATURES = [
    "weight_lbs", "shipping_zone", "package_volume", "fuel_surcharge_pct",
    "estimated_shipping_cost", "quoted_price", "markup_ratio", "day_of_week", "month",
]
CATEGORICAL_FEATURES = ["service_level", "customer_segment"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def quote_features(frame):
    # Whitelisting excludes labels, outcome times, and every simulator-only variable.
    raw = validate_quotes(frame[QUOTE_COLUMNS], labeled=False)
    result = raw.copy()
    result["package_volume"] = raw.length_in * raw.width_in * raw.height_in
    result["markup_ratio"] = raw.quoted_price / raw.estimated_shipping_cost
    result["day_of_week"] = raw.quote_timestamp.dt.dayofweek
    result["month"] = raw.quote_timestamp.dt.month
    return result[FEATURES]


def make_preprocessor():
    # fit() is called on training rows only; the resulting encoders travel with the model.
    return Pipeline([
        ("features", FunctionTransformer(quote_features)),
        ("columns", ColumnTransformer([
            ("numeric", StandardScaler(), NUMERIC_FEATURES),
            ("category", OneHotEncoder(handle_unknown="error", sparse_output=False), CATEGORICAL_FEATURES),
        ], remainder="drop")),
    ])


def one_quote_frame(quote):
    return pd.DataFrame([quote])
