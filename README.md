# Synthetic shipping price optimization

A runnable local Python project for learning production ML and preparing for a
Fetcherr Data Science Engineer III interview. All data and outcomes are fictional.
This is **not a UPS integration or deployment**.

The model estimates acceptance at an offered price. A bounded search maximizes
`P(accept) × (price − estimated cost)`. This is predicted contribution, not proof
of incremental profit. Start with [INTERVIEW_GUIDE.md](INTERVIEW_GUIDE.md) for the
business explanation and five-minute walkthrough, or
[DELIVERY_REPORT.md](DELIVERY_REPORT.md) for executed results.

The final historical snapshot uses fixed seed 2026; model fitting and daily quote
generation use seed 42. Seed-42 development history and its run results were
preserved separately before the final snapshot was trained. The final design was
recorded in `evidence/frozen_final_design.json` before opening the new final test.

## Run locally

Prerequisites: `uv`, `make`, and internet access for the first dependency install.
`uv` installs Python 3.12 if needed. Dependencies are frozen in `uv.lock`.
Docker Desktop is needed only for the container commands. macOS and Linux are the
supported local platforms (`fcntl` supplies the promotion lock).

```bash
cd /Users/key/worktrees/_AI-OS/topaz-garnet/_AI-OS/projects/ups-pricing-deployment
make setup
make demo
make lint
make test
```

`make demo` bootstraps 10,000 labeled quotes over 240 days and 200 unlabeled daily
quotes, then **executes the actual weekly and daily Dagster jobs**. It prints both
models' metrics, a candidate price table, the example recommendation, and database
counts. A repeated identical experiment reads its saved evaluation rather than
reopening the final test. No notebook, credentials, cloud service, or real shipping
system is needed.

For the individual stages:

```bash
make bootstrap
make train
make score
make show
```

Start the API in one terminal, then run HTTP smoke tests in another:

```bash
make serve                  # http://127.0.0.1:8000
# Another terminal, in this project:
make smoke
```

Interactive API documentation: <http://127.0.0.1:8000/docs>.

```bash
make dagster                # http://127.0.0.1:3000; webserver AND daemon
make docker-up              # Requires make demo or make bootstrap && make train first
make docker-smoke           # HTTP checks and a real container restart test, port 8001
make docker-down            # Keeps host data and model artifacts
```

Stop `make serve` and `make dagster` with Ctrl-C. Schedules are **defined but stopped
by default**; enable them in the Dagster UI when desired. Weekly training is Sunday
03:00 UTC; daily scoring is 06:00 UTC. The daily schedule uses its scheduled date.
Manual commands use the reproducible example batch date `2025-08-29`:

```bash
make score BATCH_DATE=2025-08-30
make serve API_PORT=8010
make smoke API_PORT=8010
make docker-up DOCKER_PORT=8011
make docker-smoke DOCKER_PORT=8011
```

An existing daily CSV is read and validated, not regenerated to conceal bad input.
A missing daily CSV is generated synthetically. The historical job reads the
historical SQLite table; it does not ingest daily data or retrain on recommendations.
Weekly scheduling alone does not create fresh historical observations.

## What lives where

```text
ups-pricing-deployment/
├── src/shipping/
│   ├── config.py           # Storage root, seeds, valid categories
│   ├── data.py             # Synthetic inputs and hidden outcome simulator
│   ├── validation.py       # Hard data-quality boundaries
│   ├── features.py         # Shared quote-time features and sklearn preprocessing
│   ├── modeling.py         # Time splits, baseline, constrained XGBoost, evaluation
│   ├── pricing.py          # Cent-rounded bounded candidate search and refusal rules
│   ├── registry.py         # Approval, atomic promotion, checksum, rollback
│   ├── storage.py          # SQLite tables, transactional saves, run metadata
│   ├── definitions.py      # Dagster assets, blocking checks, jobs, schedules
│   ├── monitoring.py       # Drift, latency, failures, delayed-label calibration
│   ├── outcomes.py         # Optional synthetic delayed outcomes, never training data
│   ├── api.py              # Pydantic contracts and FastAPI endpoints
│   ├── smoke.py            # Real HTTP smoke checks
│   └── cli.py              # Executable command entry point
├── tests/                  # Isolated unit and integration tests
├── .github/workflows/ci.yml
├── Dockerfile
├── compose.yaml
├── Makefile
├── pyproject.toml
├── uv.lock
├── ARCHITECTURE.md
├── INTERVIEW_GUIDE.md
├── DELIVERY_REPORT.md
├── evidence/               # Recorded execution evidence
├── data/                   # Generated CSVs and shipping.sqlite (ignored by Git)
├── artifacts/              # Generated models, metrics, charts, reports (ignored)
└── .dagster/               # Persistent local Dagster run/event storage (ignored)
```

The default storage root is this project. Set `UPS_PRICING_HOME` to an absolute
directory to isolate another complete experiment. Docker mounts `./artifacts`
read-only and `./data` read-write into `/app/state`; the image contains code and
dependencies, while the mounted artifact is the **real trained model**. A missing
or unusable model makes `/ready`, `/model-info`, `/predict`, and `/recommend` return
503. `/health` still returns 200. Input validation or an infeasible range returns
422; an unreliable model price curve returns 503.

SQLite tables are `historical_quotes`, `daily_quotes`, `recommendations`, `runs`,
`monitoring`, `api_events`, and `observed_outcomes`. `recommendations.quote_id` is a
primary key. A retry retains the original saved price and model version; it never
silently rewrites a quote after a model promotion. Use a new batch date for new
quotes. Do not edit an already-scored batch to simulate a new batch.

## Outputs and checks

- `artifacts/model_comparison.json`: validation and final test metrics, date cutoffs,
  exclusions, latency, price bands, selection rule, dependencies, top ten features.
- `artifacts/calibration.png`: final test probability reliability for both models.
- `artifacts/feature_importance.png`: top ten predictive features; **not causal evidence**.
- `artifacts/models/<version>/`: both fitted pipelines, reference features, probe,
  and immutable evaluation report.
- `artifacts/approved.json`: one atomic pointer containing current and previous versions.
- `artifacts/demo_results.json`, `example_recommendation.json`, `dagster_runs.json`.
- `artifacts/monitoring-<date>.json`: recent batch compared with training reference.

Promotion requires validation ROC-AUC ≥ 0.75, log loss ≤ 0.60, Brier score ≤ 0.22,
and all 50 validation price curves passing sanity checks. Among eligible models,
the lowest validation log loss wins. A later candidate must also avoid regressing
more than 0.005 in log loss against the incumbent on the same validation data.
These are transparent demonstration thresholds, not Fetcherr or UPS standards.
An incomplete or failed candidate leaves the approved pointer intact.

```bash
make rollback               # Requires a previous successful approval
make simulate-outcomes      # Draw delayed outcomes from the HIDDEN simulator
make monitor                # Now also reports observed calibration and simulated profit
```

Before synthetic outcomes arrive, observed acceptance/calibration are `null`.
Afterward, they refer to the saved **recommended offered prices**, not the original
daily placeholder prices. Future observations and outcomes at a different offered
price are rejected. These outcomes stay outside the historical training table.
Experimentally verified incremental profit remains `null` throughout this demo.

`make test` uses a smaller, separate seeded experiment in temporary directories.
The live Docker test is explicitly skipped in the normal suite; `make docker-smoke`
executes it. `make lint` runs Ruff. The CI workflow performs lint, tests, demo,
real HTTP smoke, Docker build, and Docker restart checks on push and pull request
**when this standalone directory is the root of its own GitHub repository**.
GitHub ignores nested `.github/workflows` in the enclosing AI-OS repository; no
shared workflow was changed. The remote GitHub workflow has not been run here.
