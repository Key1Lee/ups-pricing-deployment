"""Distribution drift is a warning signal; unlabeled predictions are not accuracy."""

import json

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, log_loss

from shipping.config import paths
from shipping.features import CATEGORICAL_FEATURES, NUMERIC_FEATURES, quote_features
from shipping.storage import connect, save_json, utc_now


def distribution_drift(reference, recent):
    output = {}
    for column in NUMERIC_FEATURES:
        # KS statistic: largest difference between two empirical cumulative distributions.
        statistic = float(ks_2samp(reference[column], recent[column]).statistic)
        output[column] = {"statistic": "KS", "value": statistic, "warning": statistic > 0.15}
    for column in CATEGORICAL_FEATURES:
        a = reference[column].value_counts(normalize=True)
        b = recent[column].value_counts(normalize=True)
        keys = a.index.union(b.index)
        distance = float(np.abs(a.reindex(keys, fill_value=0) - b.reindex(keys, fill_value=0)).sum() / 2)
        output[column] = {"statistic": "total_variation", "value": distance, "warning": distance > 0.10}
    return output


def monitoring_report(batch_date, run_id, frame, entry, as_of=None):
    now = utc_now() if as_of is None else pd.Timestamp(as_of).isoformat()
    directory = paths().artifacts / "models" / entry["version"]
    reference = pd.read_csv(directory / "reference_features.csv")
    with connect() as con:
        recommendations = pd.read_sql_query("SELECT * FROM recommendations WHERE batch_date=?", con, params=[batch_date])
        observed = pd.read_sql_query("""SELECT r.*, o.accepted_quote FROM recommendations r
            JOIN observed_outcomes o USING(quote_id) WHERE r.batch_date=? AND o.outcome_timestamp <= ?""",
            con, params=[batch_date, now])
        api = pd.read_sql_query("SELECT * FROM api_events", con)
        failed_runs = con.execute("SELECT COUNT(*) FROM runs WHERE status='FAILURE'").fetchone()[0]
    labels = None
    if len(observed):
        probability = observed.predicted_acceptance_probability.to_numpy()
        actual, predicted = calibration_curve(observed.accepted_quote, probability, n_bins=5)
        labels = {
            "labeled_quotes": len(observed), "label_source": "synthetic_simulator",
            "observed_acceptance_rate": float(observed.accepted_quote.mean()),
            "brier_score": float(brier_score_loss(observed.accepted_quote, probability)),
            "log_loss": float(log_loss(observed.accepted_quote, probability, labels=[0, 1])),
            "calibration": {"mean_predicted": predicted.tolist(), "fraction_accepted": actual.tolist()},
            "simulated_realized_contribution": float((observed.accepted_quote * (
                observed.recommended_price - observed.estimated_shipping_cost)).sum()),
        }
    report = {
        "run_id": run_id, "batch_date": batch_date, "created_at": now,
        "new_quotes_processed": len(frame), "successful_recommendations": len(recommendations),
        "model_version_loaded": entry["version"],
        "persisted_model_versions": sorted(recommendations.model_version.unique().tolist()),
        "average_batch_inference_ms": float(recommendations.latency_ms.mean()),
        "predicted_expected_contribution_total": float(recommendations.expected_contribution.sum()),
        "api_scope": "all locally recorded requests since database creation",
        "failed_requests": int((api.status_code >= 400).sum()),
        "api_request_count": len(api),
        "average_api_latency_ms": float(api.latency_ms.mean()) if len(api) else None,
        "failed_pipeline_runs_total": failed_runs,
        "drift": distribution_drift(reference, quote_features(frame)),
        "drift_note": "Heuristic warnings, not significance tests; month/day shifts may be expected seasonality",
        "outcomes_as_of": now, "observed_outcomes": labels,
        "experimentally_verified_incremental_profit": None,
        "interpretation": "Synthetic predictive estimates; no real-world causal uplift is measured",
    }
    with connect() as con:
        con.execute("INSERT OR REPLACE INTO monitoring VALUES(?,?,?)", (run_id, now, json.dumps(report, allow_nan=False)))
    save_json(paths().artifacts / f"monitoring-{batch_date}.json", report)
    return report
