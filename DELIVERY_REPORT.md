# Executed delivery report

Recorded 2026-11-11T21:00:38.974191+00:00. This is a local synthetic learning deployment,
not a real UPS deployment. No sibling repositories or notebooks were modified.

## A. Project directory tree

```text
ups-pricing-deployment/
├── src/shipping/
│   ├── __init__.py
│   ├── api.py
│   ├── cli.py
│   ├── config.py
│   ├── data.py
│   ├── definitions.py
│   ├── features.py
│   ├── modeling.py
│   ├── monitoring.py
│   ├── outcomes.py
│   ├── pricing.py
│   ├── registry.py
│   ├── smoke.py
│   ├── storage.py
│   └── validation.py
├── tests/
│   ├── conftest.py
│   ├── test_api_registry.py
│   ├── test_data.py
│   ├── test_docker.py
│   ├── test_jobs_monitoring.py
│   ├── test_modeling.py
│   └── test_pricing.py
├── .github/workflows/ci.yml
├── .dockerignore
├── .gitignore
├── .python-version
├── Dockerfile
├── compose.yaml
├── Makefile
├── pyproject.toml
├── uv.lock
├── README.md
├── ARCHITECTURE.md
├── INTERVIEW_GUIDE.md
├── DELIVERY_REPORT.md
├── evidence/                     # Executed logs, frozen design, JSON results
│   └── development-runtime/      # Preserved earlier data/models; not final evidence
├── data/
│   ├── historical_quotes.csv
│   ├── quotes-2025-08-29.csv
│   ├── synthetic-outcomes-2025-08-29.csv
│   └── shipping.sqlite
├── artifacts/
│   ├── approved.json
│   ├── model_comparison.json
│   ├── calibration.png
│   ├── feature_importance.png
│   ├── training_features.csv
│   ├── example_recommendation.json
│   ├── demo_results.json
│   ├── dagster_runs.json
│   ├── monitoring-2025-08-29.json
│   └── models/model-75c4c0d2390c0016/
│       ├── logistic_regression.joblib
│       ├── xgboost.joblib
│       ├── evaluation.json
│       ├── reference_features.csv
│       └── probe.json
├── .venv/                        # Installed, ignored
└── .dagster/                     # Persistent UI/event/run storage, ignored
```

## B. Exact installation and execution commands

`uv` and Make are prerequisites. The executed setup installed a project-local
Python 3.12 environment from `uv.lock`. Actual core versions:
`dagster 1.13.29`, `xgboost 3.4.1`,
`scikit-learn 1.9.1`.

```bash
cd /Users/key/worktrees/_AI-OS/topaz-garnet/_AI-OS/projects/ups-pricing-deployment
make setup
make bootstrap
make train
make score
make lint
make test
make demo
```

`make demo` also executes bootstrap, training and scoring. The above individual
commands remain available for learning each stage. Repeated training reuses the
same evaluated version; repeated scoring preserves one recommendation per quote.

```bash
# Terminal A
make serve
# Terminal B
make smoke
# Terminal C
make dagster
# Terminal B, with Docker Desktop running
make docker-up
make docker-smoke
make docker-down
```

Additional executed monitoring commands:

```bash
make simulate-outcomes
make monitor
make show
```

The services were left running for inspection at local API
<http://127.0.0.1:8000/docs>, Dagster <http://127.0.0.1:3000>, and container API
<http://127.0.0.1:8001/docs>. The API and Dagster use foreground Make commands;
stop the corresponding processes before launching another copy on the same port.
Docker can be stopped with `make docker-down` without deleting persistent data.

## C. Actual test results and failures

Executed normal final suite: **55 passed, 1 skipped, 1 warning in 12.43 seconds**.
The skip explicitly requires a live container. Executed full suite with
`RUN_DOCKER_TESTS=1 make test`: **56 passed, 0 skipped, 0 failed, 1 warning in
12.92 seconds**. The final fresh model was additionally checked by
`make docker-smoke`: **1 passed in 2.71 seconds**, including container restart.
Ruff's final result: **All checks passed**.

The warning is Starlette's deprecation notice for its HTTPX TestClient adapter.
Tests work; the real-network smoke tests also passed. Dagster's working `dagster
 dev` command emits a supersession notice recommending `dg dev`; no definition
or execution failures remain. Deprecated asset-context accesses were replaced
with the installed version's `context.run.run_id` API.

No pytest invocation failed. Earlier development commands did fail, and those
failures were fixed rather than omitted:

| Failure observed | Cause and correction | Executed evidence |
| --- | --- | --- |
| Curve checks all failed; printing raised `KeyError: top_features` | Labeled history rows were passed into the unlabeled quote validator. Project quote-time columns explicitly; reject all-invalid candidates before opening the test. | `evidence/development-training-failure.log`; two regression tests now pass |
| Daily Dagster job failed with `table recommendations has 9 columns but 10 values were supplied` | Incorrect INSERT placeholder count. Specify the nine columns and nine values explicitly. | `evidence/development-scoring-failure.log`; successful jobs and transactional/idempotency tests |
| Docker could not connect to daemon | Docker Desktop was stopped. Started installed Docker Desktop and verified server version. | Subsequent successful builds, HTTP checks, restart test |
| Docker built but startup failed: `all predefined address pools have been fully subnetted` | Compose attempted a new project network. Single-service API now reuses built-in `bridge`; no unrelated networks were deleted. | `evidence/development-docker-network-failure.log`; final container healthy |
| An ad hoc final fingerprint assertion initially failed | The diagnostic compared SQLite timestamp strings with the training asset's normalized UTC timestamps. Repeating the check after the same validation/normalization matched the approved version exactly; no model or source change was needed. | Corrected final delivery check passed; frozen source checksums also matched |

Initial lint also corrected four import/unused-import issues. An initial 16-test
Phase 1 run passed before later components were added. See actual logs in
`evidence/pytest.log`, `pytest-all-with-docker.log`, and `http-docker-and-restart.log`.

The tests cover all 17 requested cases, including a real Docker model read/restart,
late-label exclusion, encoder fitting on training rows, shared offline/API
probabilities, blocking corrupt data checks, failed promotion preserving the pointer,
rollback/corruption fallback, and successful persisted counts.

## D. Actual baseline versus XGBoost metrics

The development snapshot used historical seed 42. Its data, models, and run records
were preserved before the final experiment. Before training the fresh seed-2026
snapshot, modeling settings, selection, approval thresholds, dependencies, and
source checksums were frozen in `evidence/frozen_final_design.json`. These sources
were unchanged at evidence capture. The final holdout was evaluated once for each
fixed model; subsequent identical training reused its cached report. No settings
were changed in response to the final results below. Test fixtures train on a
separate seed-7, 3,000-row experiment.

All windows are UTC, half-open intervals:

| Split | Quote dates | Eligible rows | Late labels excluded |
| --- | --- | ---: | ---: |
| Training | 2025-01-01 to before 2025-06-18 | 6,991 | 57 |
| Validation | 2025-06-18 to before 2025-07-24 | 1,417 | 59 |
| Final test | 2025-07-24 to before 2025-08-29 | 1,476 | 0; evaluated after maturation |

Final evaluation observation cutoff: `2025-08-31 13:27:42+00:00`.
The 116 training/validation exclusions remain in historical storage, but did not
enter those model-development splits.

| Split | Model | ROC-AUC ↑ | Log loss ↓ | Brier ↓ | Median inference ms |
| --- | --- | ---: | ---: | ---: | ---: |
| validation | logistic_regression | 0.900921 | 0.398938 | 0.127764 | 3.771 |
| validation | xgboost | 0.901110 | 0.398895 | 0.127832 | 3.706 |
| test | logistic_regression | 0.908289 | 0.384035 | 0.121421 | 3.665 |
| test | xgboost | 0.906289 | 0.388007 | 0.122590 | 3.676 |

XGBoost won the predeclared **validation log-loss** rule by only about 0.000043;
this tiny difference is not evidence of practical superiority. Both models passed
all 50 validation price-curve checks. Logistic regression was slightly better on
all three final-test predictive metrics. We report that result and keep the
validation-selected XGBoost, rather than changing selection after seeing the test.
Neither model was refit on validation or test labels.

Latency is a local median over 15 warmed single-quote predictions, including
feature processing, not a production service-level guarantee. Acceptance rates by
price band and calibration points for both splits/models are in
[evidence/model_comparison.json](evidence/model_comparison.json). Both generated
PNG plots were opened and visually inspected for readability.

Ten highest XGBoost gain importances:

| Rank | Transformed feature | Gain importance |
| --- | --- | ---: |
| 1 | `numeric__markup_ratio` | 0.500317 |
| 2 | `category__customer_segment_occasional` | 0.100177 |
| 3 | `category__customer_segment_enterprise` | 0.085834 |
| 4 | `category__service_level_ground` | 0.063488 |
| 5 | `category__service_level_next_day` | 0.047212 |
| 6 | `numeric__shipping_zone` | 0.037892 |
| 7 | `numeric__estimated_shipping_cost` | 0.036534 |
| 8 | `numeric__fuel_surcharge_pct` | 0.034667 |
| 9 | `numeric__weight_lbs` | 0.024215 |
| 10 | `numeric__month` | 0.022509 |

These are predictive model usage statistics, not causal effects. Importance can
move between correlated features, including offered price and markup.

## E. Example recommendation JSON

Input: `interview-example`, ground service, small business segment, zone 4,
5 lb, 10 × 8 × 6 inches, fuel 0.12, estimated cost **$12.00**.
The original offered price is $18 for `/predict`; recommendation searches new
candidate prices. The smoke test caps the allowed price at $30.

```json
{
  "quote_id": "interview-example",
  "recommended_price": 19.5,
  "predicted_acceptance_probability": 0.6693926453590393,
  "expected_contribution": 5.020444840192795,
  "model_version": "model-75c4c0d2390c0016"
}
```

Check: `0.6693926453590393 × (19.50 − 12.00) = 5.020444840192795`.
The complete readable candidate table is in `evidence/demo.log` and
`evidence/final-results.log`. This is the best candidate on the configured grid,
not an analytically proven continuous global optimum or a causal profit estimate.

## F. Actual persisted record counts

SQLite snapshot at `2026-11-11T21:00:38.974191+00:00`:

| Table | Rows |
| --- | ---: |
| `historical_quotes` | 10,000 |
| `daily_quotes` | 200 |
| `recommendations` | 200 |
| `runs` | 4 |
| `monitoring` | 3 |
| `api_events` | 44 |
| `observed_outcomes` | 200 |

Duplicate recommendation IDs: **0**.
The second scoring run still left exactly 200 recommendation rows, all tied to
`model-75c4c0d2390c0016`. All final experiment pipeline runs succeeded. Request events
include expected invalid smoke-test requests and Docker health probes; that count
continues increasing while the healthy container is running.

`observed_outcomes` is a separate synthetic monitoring table. Those 200 labels
were not added to `historical_quotes`, and no retraining happened after their
arrival. SQLite's explicit schemas are recorded in
[evidence/execution_summary.json](evidence/execution_summary.json).

## G. Dagster job execution evidence

| Job | Actual run ID | Status | Records | Asset workflow seconds |
| --- | --- | --- | ---: | ---: |
| weekly_training_job | `9f0007b4-1ceb-4753-bcc8-a161a616db87` | SUCCESS | 10000 | 1.834 |
| daily_scoring_job | `c2f67897-c492-4c1f-82c3-ac5e4bf2017f` | SUCCESS | 200 | 1.502 |
| weekly_training_job | `a33fffc0-8075-4c84-bfa5-cacaff0b1486` | SUCCESS | 10000 | 0.546 |
| daily_scoring_job | `87690c7e-f846-4ded-8891-4213806aac5c` | SUCCESS | 200 | 1.536 |

Training record counts mean validated historical input rows, not fitted rows.
Training used only the 6,991 eligible training examples. The second training run
read the same saved experiment evaluation, preserving the final holdout.

The running Dagster webserver returned HTTP **200** to a GraphQL query listing
both named jobs, both schedules, and **14 assets**. The daemon startup is recorded
in `evidence/dagster-startup.log`; the graph inventory is in
[evidence/dagster-ui.json](evidence/dagster-ui.json). The diagram and component
boundaries are in [ARCHITECTURE.md](ARCHITECTURE.md).
Schedules are defined but remain stopped by default. No claim is made that a
future Sunday/daily cron occurrence was observed.

## H. Actual HTTP smoke evidence

The same checks ran using real HTTP against both `127.0.0.1:8000` (uvicorn) and
`127.0.0.1:8001` (Docker), serving `model-75c4c0d2390c0016`:

| Request | Local HTTP | Docker HTTP |
| --- | ---: | ---: |
| GET /health | 200 | 200 |
| GET /ready | 200 | 200 |
| GET /model-info | 200 | 200 |
| POST /predict, valid quote | 200 | 200 |
| POST /recommend, valid quote | 200 | 200 |
| POST /recommend, negative dimension | 422 | 422 |

Both returned acceptance probability **0.7646671533584595** at the offered $18,
and the $19.50 recommendation above. The smoke checks validate response models,
version consistency, probability bounds, expected-contribution arithmetic, margin,
and maximum price. Missing-model 503 behavior was executed in isolated API tests.
See `evidence/http-local.log` and `evidence/http-docker-and-restart.log`.

## I. Docker build, startup, restart evidence

Docker server **29.8.2**, image architecture
**arm64**, container state **running**,
health **healthy**. Final image ID:

```text
sha256:d0657e1da554d728ec117a14872a392f831f70ddb94468105d177151744ebb5c
```

The trained artifacts mount at `/app/state/artifacts` read-only; the SQLite/data
mount at `/app/state/data` is read-write. The image contains the installed locked
application, not an untrained model placeholder. `/ready` passed after startup and
after a real `docker compose restart api`; the served model version remained the
approved version. Build/startup evidence is in
`evidence/docker-build-and-startup.log`, including `Container ... Healthy`.
Only the native arm64 Docker build was executed here; other architectures were
not tested.

## J. Monitoring, limitations, and unverified items

The unlabeled daily run wrote no accuracy metrics. After drawing fresh synthetic
outcomes from the hidden simulator (not from model predictions), monitoring found:

- 200 observed outcomes; acceptance **58%**.
- Observed Brier score **0.236922**, log loss
  **0.669389**, with calibration bins saved to JSON.
- Total predicted expected contribution **$1,534.88**.
- Simulated realized contribution **$1,378.64**.
- Average recorded per-quote batch recommendation latency **5.889 ms**.
- Drift warnings for fuel surcharge, weekday, and month against the broad historical
  reference. A single day has a concentrated calendar/fuel distribution; these
  heuristic warnings require interpretation, not automatic retraining.
- Experimentally verified incremental profit remains **null**.

The optimized-price population differs from historical random offers. Its observed
calibration shows overprediction in this small synthetic sample, especially the
higher-probability bin. Do not market historical predictive metrics as proof that
the optimization policy is calibrated or improves real profit. No calibration
adjustment or model reselection was made after inspecting these final outcomes.

Other explicit limits:

- Synthetic single-quote objective with estimated costs, not actual cost accounting,
  capacity optimization, competitor modeling, or customer lifetime value.
- Search uses cent-rounded discrete steps and synthetic markup support 1.12–2.60;
  XGBoost's piecewise-constant responses and prediction errors can affect the chosen
  grid price. Sanity checks reduce obvious failures, not all model risk.
- No real UPS connections, real customer data, cloud deployment, GCP/BigQuery
  implementation, authorized online experiment, or real-world uplift evidence.
- No million-quote benchmark, sustained load test, multi-host orchestration,
  authentication, paging integration, or Windows support was executed.
- The fresh final experiment has one approved version, so its `previous` pointer
  is null until another distinct version passes promotion. Rollback and corrupt
  artifact fallback were executed against two complete trusted test artifacts.
  The prior development runtime remains preserved separately.
- Model files use trusted local joblib/pickle and matching locked dependencies.
  This is a POSIX local registry, not a distributed registry or disaster-recovery
  system. Abrupt process termination can leave an application run marked RUNNING;
  use Dagster's event log to diagnose it.
- GitHub Actions YAML was created for this standalone project as a repository root.
  GitHub ignores a nested workflow in AI-OS. Remote CI has not been triggered or
  claimed successful; local equivalents and Docker integration were executed.
- The TestClient and `dagster dev` notices described above remain non-failing
  dependency/CLI migration notices. Future dependency upgrades need testing.

Start learning with [INTERVIEW_GUIDE.md](INTERVIEW_GUIDE.md), including all 15
requested explanations and a five-minute verbal walkthrough.
