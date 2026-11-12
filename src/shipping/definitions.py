"""Two asset jobs: training can promote; daily scoring only reads approved artifacts."""

import pandas as pd
from dagster import (
    AssetCheckResult,
    AssetExecutionContext,
    AssetSelection,
    Config,
    Definitions,
    RunRequest,
    ScheduleDefinition,
    asset,
    asset_check,
    define_asset_job,
    failure_hook,
    schedule,
)

from shipping.config import DEMO_DAY, paths
from shipping.data import generate_daily
from shipping.features import quote_features
from shipping.modeling import chronological_split, evaluate_models, train_models
from shipping.monitoring import monitoring_report
from shipping.registry import approval_criteria, load_approved, promote
from shipping.storage import (
    finish_run,
    load_history,
    save_quotes,
    save_recommendations,
    score_batch,
    start_run,
)
from shipping.validation import validate_quotes


class BatchConfig(Config):
    batch_date: str = DEMO_DAY


@failure_hook
def record_failure(context):
    # Keep errors concise and avoid logging full request bodies or row contents.
    finish_run(context.run_id, "FAILURE", error=f"Failed step: {context.op.name}")


@asset(group_name="training")
def historical_data(context: AssetExecutionContext):
    start_run(context.run.run_id, "weekly_training_job")
    frame = load_history()
    context.add_output_metadata({"records": len(frame)})
    return frame


@asset_check(asset=historical_data, blocking=True)
def historical_contract(historical_data):
    try:
        validate_quotes(historical_data, labeled=True)
        return AssetCheckResult(passed=True)
    except ValueError as exc:
        return AssetCheckResult(passed=False, metadata={"reason": str(exc)})


@asset(group_name="training")
def validated_history(historical_data):
    return validate_quotes(historical_data, labeled=True)


@asset(group_name="training")
def training_features(context: AssetExecutionContext, validated_history):
    splits = chronological_split(validated_history)
    engineered = quote_features(splits["train"])
    context.add_output_metadata({"training_rows": len(engineered), "features": list(engineered.columns)})
    engineered.to_csv(paths().artifacts / "training_features.csv", index=False)
    # Raw quote fields travel to the serialized sklearn pipeline, which owns transformation.
    return {"raw": validated_history, "splits": splits}


@asset(group_name="training")
def trained_models(training_features):
    return train_models(training_features["splits"], training_features["raw"])


@asset(group_name="training")
def model_evaluation(trained_models, training_features):
    return evaluate_models(trained_models, training_features["splits"])


@asset(group_name="training")
def model_approval(context: AssetExecutionContext, model_evaluation):
    checks = approval_criteria(model_evaluation)
    context.add_output_metadata({"checks": checks, "selected": model_evaluation["selected_model"]})
    return model_evaluation


@asset(group_name="training")
def versioned_model_artifact(context: AssetExecutionContext, model_approval, training_features):
    entry = promote(model_approval, training_features["splits"]["validation"])
    finish_run(context.run.run_id, "SUCCESS", len(training_features["raw"]))
    context.add_output_metadata({"version": entry["version"], "model": entry["model_name"]})
    return entry


@asset(group_name="scoring")
def new_shipping_quotes(context: AssetExecutionContext, config: BatchConfig):
    start_run(context.run.run_id, "daily_scoring_job")
    pd.Timestamp(config.batch_date)  # Fail on a malformed date before building a path.
    file = paths().data / f"quotes-{config.batch_date}.csv"
    if not file.exists():
        generate_daily(config.batch_date).to_csv(file, index=False)
    frame = pd.read_csv(file)
    context.add_output_metadata({"batch_date": config.batch_date, "records": len(frame)})
    return {"frame": frame, "batch_date": config.batch_date}


@asset_check(asset=new_shipping_quotes, blocking=True)
def daily_contract(new_shipping_quotes):
    try:
        validate_quotes(new_shipping_quotes["frame"], labeled=False)
        return AssetCheckResult(passed=True)
    except ValueError as exc:
        return AssetCheckResult(passed=False, metadata={"reason": str(exc)})


@asset(group_name="scoring")
def validated_daily(new_shipping_quotes):
    frame = validate_quotes(new_shipping_quotes["frame"], labeled=False)
    save_quotes(frame, labeled=False, batch_date=new_shipping_quotes["batch_date"])
    return dict(new_shipping_quotes, frame=frame)


@asset(group_name="scoring")
def daily_features(context: AssetExecutionContext, validated_daily):
    engineered = quote_features(validated_daily["frame"])
    context.add_output_metadata({"records": len(engineered), "features": list(engineered.columns)})
    return validated_daily


@asset(group_name="scoring", deps=[daily_features, versioned_model_artifact])
def approved_model():
    model, report, entry = load_approved()
    return {"model": model, "report": report, "entry": entry}


@asset(group_name="scoring")
def price_recommendations(daily_features, approved_model):
    rows = score_batch(approved_model["model"], approved_model["entry"], daily_features["frame"])
    return {"rows": rows, "batch_date": daily_features["batch_date"]}


@asset(group_name="scoring")
def persisted_recommendations(context: AssetExecutionContext, price_recommendations):
    total = save_recommendations(price_recommendations["rows"], price_recommendations["batch_date"])
    context.add_output_metadata({"persisted_records": total})
    return total


@asset(group_name="scoring")
def daily_monitoring(context: AssetExecutionContext, persisted_recommendations, daily_features, approved_model):
    report = monitoring_report(daily_features["batch_date"], context.run.run_id, daily_features["frame"], approved_model["entry"])
    if persisted_recommendations != len(daily_features["frame"]):
        raise ValueError("Persisted count differs from expected daily batch count")
    finish_run(context.run.run_id, "SUCCESS", persisted_recommendations)
    return report


weekly_training_job = define_asset_job(
    "weekly_training_job", selection=AssetSelection.groups("training"), hooks={record_failure}
)
daily_scoring_job = define_asset_job(
    "daily_scoring_job", selection=AssetSelection.groups("scoring"), hooks={record_failure}
)
weekly_schedule = ScheduleDefinition(
    name="weekly_training_schedule", job=weekly_training_job, cron_schedule="0 3 * * 0",
    execution_timezone="UTC",
)


@schedule(job=daily_scoring_job, cron_schedule="0 6 * * *", execution_timezone="UTC")
def daily_schedule(context):
    day = context.scheduled_execution_time.date().isoformat()
    return RunRequest(run_key=day, run_config={"ops": {"new_shipping_quotes": {"config": {"batch_date": day}}}})


defs = Definitions(
    assets=[historical_data, validated_history, training_features, trained_models, model_evaluation,
            model_approval, versioned_model_artifact, new_shipping_quotes, validated_daily,
            daily_features, approved_model, price_recommendations, persisted_recommendations, daily_monitoring],
    asset_checks=[historical_contract, daily_contract],
    jobs=[weekly_training_job, daily_scoring_job], schedules=[weekly_schedule, daily_schedule],
)
