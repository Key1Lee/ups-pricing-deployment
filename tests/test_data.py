import numpy as np
import pandas as pd
import pytest

from shipping.data import bootstrap, generate_daily, generate_history
from shipping.storage import counts, load_history
from shipping.validation import HISTORY_COLUMNS, DataQualityError, validate_quotes


@pytest.fixture(scope="module")
def history():
    return generate_history()


def test_generator_schema_and_determinism(history):
    assert list(history.columns) == HISTORY_COLUMNS
    assert len(history) == history.quote_id.nunique() == 10_000
    assert history.quote_timestamp.dt.date.nunique() == 240
    assert set(history.accepted_quote) == {0, 1}
    assert all(pd.api.types.is_numeric_dtype(history[x]) for x in ["weight_lbs", "quoted_price"])
    pd.testing.assert_frame_equal(history, generate_history())


@pytest.mark.parametrize("column,value", [
    ("quote_id", "duplicate"), ("length_in", -1), ("weight_lbs", 0),
    ("quoted_price", np.inf), ("accepted_quote", 2), ("shipping_zone", 1),
    ("shipping_zone", 2.5), ("service_level", "teleport"),
    ("quote_timestamp", "bad-date"), ("quote_timestamp", 5), ("fuel_surcharge_pct", -0.1),
    ("estimated_shipping_cost", "12.00"), ("width_in", np.nan),
])
def test_invalid_data_rejected(history, column, value):
    frame = history.head(3).copy()
    frame[column] = value
    with pytest.raises(DataQualityError):
        validate_quotes(frame, labeled=True)


def test_missing_column_rejected(history):
    with pytest.raises(DataQualityError):
        validate_quotes(history.drop(columns="height_in"), labeled=True)


def test_daily_is_unlabeled():
    daily = generate_daily()
    assert "accepted_quote" not in daily
    assert "outcome_timestamp" not in daily
    with pytest.raises(DataQualityError):
        validate_quotes(daily, labeled=True)
    with pytest.raises(DataQualityError):
        validate_quotes(daily.assign(accepted_quote=0), labeled=False)


def test_sqlite_bootstrap_idempotent(tmp_path, monkeypatch):
    monkeypatch.setenv("UPS_PRICING_HOME", str(tmp_path))
    bootstrap()
    bootstrap()
    assert counts()["historical_quotes"] == 10_000
    assert counts()["daily_quotes"] == 200
    assert len(validate_quotes(load_history(), labeled=True)) == 10_000
