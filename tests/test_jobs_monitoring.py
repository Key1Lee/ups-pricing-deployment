import json

import pandas as pd
import pytest
from dagster import DagsterInstance, build_schedule_context

from shipping.config import DEMO_DAY
from shipping.definitions import daily_schedule, defs
from shipping.monitoring import monitoring_report
from shipping.outcomes import ingest_outcomes, simulate_outcomes
from shipping.registry import load_approved
from shipping.storage import connect, counts, load_daily
from shipping.validation import DataQualityError


def run_job(name):
    with DagsterInstance.ephemeral() as instance:
        return defs.resolve_job_def(name).execute_in_process(instance=instance, raise_on_error=False)


def test_weekly_job_and_asset_check(project):
    result = run_job("weekly_training_job")
    assert result.success
    checks = result.get_asset_check_evaluations()
    assert checks and all(check.passed for check in checks)
    with connect() as con:
        row = con.execute("SELECT status,record_count,duration_seconds FROM runs WHERE run_id=?", (result.run_id,)).fetchone()
    assert row[0] == "SUCCESS" and row[1] == 3000 and row[2] > 0


def test_daily_jobs_idempotent_persist_expected_count_and_do_not_retrain(project, monkeypatch):
    import shipping.definitions as definitions

    monkeypatch.setattr(definitions, "train_models", lambda *args: pytest.fail("Daily job retrained"))
    first = run_job("daily_scoring_job")
    second = run_job("daily_scoring_job")
    assert first.success and second.success
    assert counts()["recommendations"] == 200
    assert counts()["historical_quotes"] == 3000
    with connect() as con:
        rows = con.execute("SELECT status,record_count FROM runs").fetchall()
        duplicates = con.execute("SELECT quote_id FROM recommendations GROUP BY quote_id HAVING COUNT(*)>1").fetchall()
    assert rows == [("SUCCESS", 200), ("SUCCESS", 200)]
    assert duplicates == []
    report = json.loads((project.artifacts / f"monitoring-{DEMO_DAY}.json").read_text())
    assert report["new_quotes_processed"] == report["successful_recommendations"] == 200
    assert report["observed_outcomes"] is None  # Unlabeled predictions cannot be called accuracy.
    assert "weight_lbs" in report["drift"]


def test_corrupt_batch_blocks_downstream_assets(project):
    file = project.data / f"quotes-{DEMO_DAY}.csv"
    frame = pd.read_csv(file)
    frame.loc[0, "length_in"] = -5
    frame.to_csv(file, index=False)
    result = run_job("daily_scoring_job")
    assert not result.success
    evaluations = result.get_asset_check_evaluations()
    assert len(evaluations) == 1 and not evaluations[0].passed
    materialized = {event.asset_key.to_user_string() for event in result.get_asset_materialization_events()}
    assert "price_recommendations" not in materialized
    assert "approved_model" not in materialized
    assert counts()["recommendations"] == 0
    with connect() as con:
        assert con.execute("SELECT status FROM runs WHERE run_id=?", (result.run_id,)).fetchone()[0] == "FAILURE"


def test_delayed_labels_enable_calibration_without_becoming_training_labels(project):
    assert run_job("daily_scoring_job").success
    result = simulate_outcomes(DEMO_DAY)
    assert result["synthetic_outcomes_ingested"] == 200
    assert counts()["historical_quotes"] == 3000
    _, _, entry = load_approved()
    report = monitoring_report(DEMO_DAY, "observed-test", load_daily(DEMO_DAY), entry)
    labels = report["observed_outcomes"]
    assert labels["labeled_quotes"] == 200
    assert 0 <= labels["observed_acceptance_rate"] <= 1
    assert 0 <= labels["brier_score"] <= 1
    assert labels["calibration"]["mean_predicted"]
    assert report["experimentally_verified_incremental_profit"] is None
    before_labels = monitoring_report(DEMO_DAY, "before-outcomes", load_daily(DEMO_DAY), entry,
                                     as_of="2025-08-30T00:00:00+00:00")
    assert before_labels["observed_outcomes"] is None
    observed = pd.read_csv(project.data / f"synthetic-outcomes-{DEMO_DAY}.csv")
    with pytest.raises(DataQualityError, match="future"):
        ingest_outcomes(observed, "2025-08-29T00:00:00+00:00")
    with pytest.raises(DataQualityError, match="different offered price"):
        ingest_outcomes(observed.assign(offered_price=999), "2025-09-05T00:00:00+00:00")


def test_schedule_date_and_two_job_definitions():
    timestamp = pd.Timestamp("2025-09-10T06:00:00Z").to_pydatetime()
    with build_schedule_context(scheduled_execution_time=timestamp) as context:
        request = daily_schedule(context)
    assert request.run_config["ops"]["new_shipping_quotes"]["config"]["batch_date"] == "2025-09-10"
    assert defs.resolve_job_def("weekly_training_job").name == "weekly_training_job"
    assert defs.resolve_job_def("daily_scoring_job").name == "daily_scoring_job"
