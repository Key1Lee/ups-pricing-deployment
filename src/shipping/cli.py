"""Small commands that wrap real Dagster jobs and the reusable Python components."""

import argparse
import json
import os
import uuid

from shipping.config import DEMO_DAY, paths
from shipping.data import bootstrap
from shipping.storage import counts, finish_run, save_json


def execute_job(name, batch_date=DEMO_DAY):
    from dagster import DagsterInstance

    from shipping.definitions import defs

    home = paths().root / ".dagster"
    home.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("DAGSTER_HOME", str(home))
    configuration = home / "dagster.yaml"
    if not configuration.exists():
        configuration.write_text("telemetry:\n  enabled: false\n")
    run_id = str(uuid.uuid4())
    config = {"ops": {"new_shipping_quotes": {"config": {"batch_date": batch_date}}}} if name == "daily_scoring_job" else {}
    with DagsterInstance.get() as instance:
        result = defs.resolve_job_def(name).execute_in_process(
            instance=instance, run_id=run_id, run_config=config, raise_on_error=False,
        )
    if not result.success:
        finish_run(run_id, "FAILURE", error="Dagster job failed; inspect persisted event log")
        raise RuntimeError(f"{name} failed: {run_id}")
    return {"job": name, "run_id": run_id, "success": result.success}


def show():
    from shipping.api import example_quote
    from shipping.pricing import recommend
    from shipping.registry import load_approved

    model, report, entry = load_approved()
    result, table = recommend(model, example_quote(), entry["version"])
    summary = {
        "model_version": entry["version"], "selected_model": entry["model_name"],
        "validation": {k: {m: v[m] for m in ["roc_auc", "log_loss", "brier_score", "median_inference_ms"]}
                       for k, v in report["validation"].items()},
        "test": {k: {m: v[m] for m in ["roc_auc", "log_loss", "brier_score", "median_inference_ms"]}
                 for k, v in report["test"].items()},
        "recommendation": result, "persisted_counts": counts(),
    }
    print(table.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(json.dumps(summary, indent=2))
    save_json(paths().artifacts / "demo_results.json", summary)
    save_json(paths().artifacts / "example_recommendation.json", result)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["bootstrap", "train", "score", "demo", "show", "smoke", "rollback", "simulate-outcomes", "monitor"])
    parser.add_argument("--batch-date", default=DEMO_DAY)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    args = parser.parse_args()
    if args.command == "bootstrap":
        print(json.dumps(bootstrap(args.batch_date), indent=2))
    elif args.command in ("train", "score"):
        name = "weekly_training_job" if args.command == "train" else "daily_scoring_job"
        print(json.dumps(execute_job(name, args.batch_date), indent=2))
    elif args.command == "demo":
        print(json.dumps(bootstrap(args.batch_date), indent=2))
        training = execute_job("weekly_training_job")
        scoring = execute_job("daily_scoring_job", args.batch_date)
        save_json(paths().artifacts / "dagster_runs.json", [training, scoring])
        show()
    elif args.command == "show":
        show()
    elif args.command == "smoke":
        from shipping.smoke import smoke

        result = smoke(args.url)
        print(json.dumps(result, indent=2))
        save_json(paths().artifacts / "http_smoke.json", result)
    elif args.command == "rollback":
        from shipping.registry import rollback

        print(json.dumps(rollback(), indent=2))
    elif args.command == "simulate-outcomes":
        from shipping.outcomes import simulate_outcomes

        print(json.dumps(simulate_outcomes(args.batch_date), indent=2))
    elif args.command == "monitor":
        from shipping.monitoring import monitoring_report
        from shipping.registry import load_approved
        from shipping.storage import load_daily

        _, _, entry = load_approved()
        result = monitoring_report(args.batch_date, str(uuid.uuid4()), load_daily(args.batch_date), entry)
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
