"""Optional delayed synthetic outcomes exercise monitoring, never model retraining."""

import numpy as np
import pandas as pd

from shipping.config import paths
from shipping.data import _simulated_acceptance_probability
from shipping.storage import connect, load_daily, utc_now
from shipping.validation import DataQualityError, validate_quotes


def ingest_outcomes(frame, as_of):
    required = ["quote_id", "accepted_quote", "outcome_timestamp", "offered_price"]
    if set(required) - set(frame) or frame.empty or frame[required].isna().any().any():
        raise DataQualityError("Outcome batch missing required values")
    if frame.quote_id.duplicated().any() or not frame.accepted_quote.isin([0, 1]).all():
        raise DataQualityError("Duplicate IDs or invalid acceptance label")
    valid = frame.copy()
    valid["outcome_timestamp"] = pd.to_datetime(valid.outcome_timestamp, utc=True, errors="raise")
    if (valid.outcome_timestamp > pd.Timestamp(as_of)).any():
        raise DataQualityError("Cannot observe future outcomes")
    with connect() as con:
        known = pd.read_sql_query("""SELECT r.quote_id, r.recommended_price, q.quote_timestamp
            FROM recommendations r JOIN daily_quotes q USING(quote_id)""", con)
        merged = valid.merge(known, on="quote_id", how="left", validate="one_to_one")
        if merged.recommended_price.isna().any():
            raise DataQualityError("Outcome must match a saved recommendation")
        if not np.allclose(merged.offered_price, merged.recommended_price, rtol=0, atol=0.005):
            raise DataQualityError("Outcome was observed at a different offered price")
        if (merged.outcome_timestamp < pd.to_datetime(merged.quote_timestamp, utc=True)).any():
            raise DataQualityError("Outcome precedes quote")
        con.executemany("INSERT INTO observed_outcomes VALUES(?,?,?) ON CONFLICT(quote_id) DO NOTHING", [
            (x.quote_id, int(x.accepted_quote), x.outcome_timestamp.isoformat())
            for x in valid.itertuples(index=False)
        ])
    return len(valid)


def simulate_outcomes(batch_date):
    with connect() as con:
        recs = pd.read_sql_query("SELECT quote_id,recommended_price FROM recommendations WHERE batch_date=? ORDER BY quote_id",
                                 con, params=[batch_date])
    frame = validate_quotes(load_daily(batch_date), labeled=False).merge(recs, on="quote_id", validate="one_to_one")
    if frame.empty:
        raise DataQualityError("Score this batch before simulating outcomes")
    frame["quoted_price"] = frame.recommended_price
    rng = np.random.default_rng(123)
    frame["accepted_quote"] = rng.binomial(1, _simulated_acceptance_probability(frame, rng))
    frame["outcome_timestamp"] = frame.quote_timestamp + pd.Timedelta(np.timedelta64(2, "D"))
    observed = frame[["quote_id", "accepted_quote", "outcome_timestamp"]].assign(offered_price=frame.quoted_price)
    observed.to_csv(paths().data / f"synthetic-outcomes-{batch_date}.csv", index=False)
    count = ingest_outcomes(observed, utc_now())
    return {"synthetic_outcomes_ingested": count, "label_source": "hidden synthetic simulator, not model predictions",
            "training_tables_updated": False}
