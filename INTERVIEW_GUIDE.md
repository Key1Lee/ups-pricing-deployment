# Interview guide

This project practices the end-to-end ownership described in the
[Fetcherr Data Science Engineer III posting](https://www.fetcherr.io/careers/position-fc_472):
features, validation, production Python, pricing, orchestration, and monitoring.
Describe it as a **synthetic local learning system**, not production work experience
or a real UPS deployment.

## 1. What business problem are we solving?

A customer needs a shipping quote. A low price may win many customers but leave
little contribution. A high price leaves more contribution per sale but may lose
the customer. We compare feasible prices using
`predicted probability of acceptance × (price − estimated shipping cost)`.
We maximize expected contribution for one quote, subject to simple business rules.
We do not model capacity, competitive reactions, lifetime value, or total company profit.

## 2. What is one row in our dataset?

One row is one quote offered to one fictional customer segment at one time.
It describes the package, shipping service and zone, fuel surcharge, estimated
cost, offered price, and eventual acceptance. `quote_id` identifies that quote.
`outcome_timestamp` tells us when its answer became observable. It is metadata,
not a model input. A daily scoring row has only quote-time inputs.

## 3. What is our prediction target?

`accepted_quote` is 1 for accepted and 0 for rejected. The output is a probability
between 0 and 1. A predicted 0.7 means that across many comparable predictions near
0.7, we would like roughly 70% to accept. It is not a promise about one customer.
The target is acceptance, not shipping cost or a recommended price.

## 4. Why is XGBoost supervised learning?

It learns from examples containing inputs and a known target. XGBoost combines
small decision trees, each improving the errors left by the existing trees.
It can learn nonlinear effects and interactions, such as urgent service affecting
price sensitivity differently by segment. We use binary classification with
probability output. It is neither reinforcement learning nor an agent that learns
automatically from its own recommendations.

## 5. Why must we split data by time?

Deployment predicts future quotes using information available earlier. A random
split can mix future patterns into training and make performance look too good.
We split 240 distinct dates into 168 training days, 36 validation days, and 36 final
test days. Training ends at June 18, validation ends at July 24, and the final test
ends before August 29, 2025. All boundaries are UTC and non-overlapping.

Quote time is not enough: a June 17 quote answered on June 19 is unavailable at
the June 18 training cutoff. We exclude such late labels. Validation also has an
observation cutoff before test begins. Final evaluation waits until test labels
mature. Only training rows fit scalers, category encoders, and classifiers.

## 6. Why does the model need a baseline?

Logistic regression is a simpler reference. It tells us whether a more complex
model earns its extra maintenance cost. Both models see the same features and
chronological data. Validation log loss chooses the eligible model; we do not
assume XGBoost wins. The final test reports the result after selection is frozen.
There is no test-driven parameter search.

## 7. How is probability different from profit?

Probability measures the chance of winning a sale. Margin is price minus estimated
cost. Their product is predicted expected contribution. For a $12 cost, 90%
acceptance at $16 gives $3.60 expected contribution; 65% at $20 gives $5.20.
These are illustrations, not measured UPS results.

Three statements must stay separate:

| Quantity | Meaning in this project |
| --- | --- |
| Predicted expected contribution | Probability estimate multiplied by estimated margin |
| Simulated realized contribution | A newly drawn synthetic outcome multiplied by margin; it can differ from expectation |
| Experimentally verified incremental profit | A treatment-versus-control effect from a credible experiment; never measured here |

## 8. What does Dagster actually execute?

It calls the Python functions behind assets in dependency order. Assets represent
useful outputs such as validated history, an evaluated model, or saved daily
recommendations. Dagster tracks execution, checks, logs, materializations, and
failures. The weekly job validates history, prepares features, fits both models,
evaluates, checks approval, and publishes the versioned model. The daily job
validates new quotes, loads the approved model, searches prices, saves results,
and writes monitoring. The daily graph has no training function.

Both jobs run manually with Make commands. Schedules define when to launch them;
the daemon must be running and schedules enabled. `make dagster` starts the local
webserver and daemon. See the actual graph in the UI or [ARCHITECTURE.md](ARCHITECTURE.md).
The implementation uses documented
[Dagster asset jobs](https://docs.dagster.io/guides/build/jobs/asset-jobs) and
[schedules](https://docs.dagster.io/guides/automate/schedules), tested against the
locked installed version rather than relying only on examples.

## 9. What does FastAPI do that Dagster does not?

FastAPI handles an individual HTTP request now, validates JSON, and returns a
prediction or recommendation. Dagster coordinates multi-step batch workflows.
The API remains alive between requests and never retrains. It loads the approved
fitted pipeline, including exactly the preprocessing used offline.

`/health` checks the process; `/ready` checks model availability. `/model-info`
explains what version and evaluation are served. `/predict` scores an offered
price. `/recommend` runs the guarded candidate search. Invalid requests get 422;
missing usable models or unreliable prediction curves get 503.

## 10. How is a model promoted and rolled back?

Training writes versioned files without changing what the API serves. Approval
uses validation performance and price-curve checks, then compares with an existing
incumbent if present. A candidate must deserialize and predict successfully before
publication. Its checksum identifies the exact model file. One atomic file rename
updates current and previous pointers together; a reader never sees half a pointer.

`make rollback` switches to the previous approved version after checking it loads.
The API notices a pointer change on the next request. Old files stay on disk.
An unapproved or incomplete candidate cannot replace a working model. If the
current artifact is corrupt, loading can use the previous known-good artifact.
This is a small local registry, not a managed model-registry service.

## 11. What happens if a daily pipeline fails?

Corrupt input fails a blocking asset check before feature generation, model loading,
or recommendation. An invalid response curve fails scoring before its save
transaction. Dagster marks the run failed; SQLite records a run ID, time, duration,
and concise failure. The approved model remains unchanged. Fix the input or code
and rerun the same batch. Quote primary keys prevent duplicate recommendations.
Already-saved quotes retain their original model versions after a later promotion.

There is no automatic paging or retry storm. The UI and logs are the local
operator tools. A failed monitoring step after a successful save can be retried
without duplicating quotes. Killing the entire process can leave a run marked
RUNNING in the small application run table; Dagster's own event log is the source
for diagnosing such an interruption.

## 12. How do we know our predictions are reliable?

We need several kinds of evidence:

| Check | Plain-English interpretation |
| --- | --- |
| ROC-AUC, higher is better | Does the model rank accepted quotes above rejected ones? It does not establish calibration. |
| Log loss, lower is better | How much does the model penalize wrong probability predictions, especially confident wrong answers? |
| Brier score, lower is better | Average squared difference between a probability and its observed 0/1 result |
| Calibration plot | Among predictions near 60%, do about 60% accept? The diagonal is ideal. Small bins are noisy. |
| Acceptance by price band | Where do observed and predicted rates differ across offered dollar ranges? Package/service mix also varies by band. |
| Inference latency | How long prediction takes, including shared feature processing; local timings are not a throughput guarantee |
| Drift statistics | Has the incoming feature distribution changed relative to training? A warning asks for investigation. |

Data-quality tests, temporal leakage tests, API consistency checks, and business
guardrails complement metrics. We check that increasing a candidate price does
not increase acceptance, predictions remain finite, and the price response is
not flat. XGBoost has a verified negative monotonic constraint on **both** price
and markup; see the official
[monotonic constraints documentation](https://xgboost.readthedocs.io/en/stable/tutorials/monotonic.html).
Those constraints encode a modeling assumption; they do not discover a causal law.

Feature importance says which transformed columns the fitted model used heavily.
Correlated features can share or displace importance, and gain is not a business
effect size. It is **not causal evidence**. Follow the ten-feature output in
`artifacts/model_comparison.json` and its plot.

Daily unlabeled quotes can support drift and latency monitoring, not accuracy.
Optional simulated delayed labels enable observed acceptance, Brier score,
log loss, and calibration against the probabilities saved at offer time. Outcomes
at different offered prices are rejected. They never become historical training
labels automatically.

## 13. How would this system scale to a million quotes?

First measure where time is spent. One XGBoost prediction over all candidates for
one quote is already vectorized. Next, generate candidates for chunks of quotes
and predict those chunks together, rather than doing one model call per quote.
Use explicit batch IDs, immutable input snapshots, database uniqueness, and
bounded transactions. Store larger datasets in a warehouse or columnar files;
SQLite's single-writer pattern becomes a limit. Partition Dagster work by date,
parallelize independent chunks within capacity, and load the model once per worker.

Profile API latency and concurrent demand before adding replicas or changing
serving architecture. Add alerting, data lineage, model compatibility checks,
and load tests based on actual service-level requirements. A million-quote
benchmark and multi-host operations have not been implemented here.

## 14. What would change with GCP and BigQuery?

BigQuery could hold historical quotes, outcomes, feature snapshots, and monitoring
tables with date partitions and quote/batch keys. SQL would select a reproducible
training snapshot while enforcing event and label-observation cutoffs. Cloud
Storage could hold versioned model files and evaluation metadata. A managed
registry or a storage operation with a generation precondition could replace the
local POSIX lock and atomic rename.

A deployed orchestration service would execute scheduled jobs, and a container
service could serve the API. We'd add service identities, least-privilege access,
managed secrets, deployment controls, logging, and cost monitoring. BigQuery is
useful for analytical batches; it would not automatically become a low-latency
per-request feature store. These are design options, not GCP features tested by
this project.

## 15. Why does a historical backtest not prove causal uplift?

A predictive model asks who accepted at observed prices. A pricing intervention
asks what the same customer would have done at another price. In real historical
data, prices can depend on urgency, negotiation, competition, or hidden willingness
to pay. Those factors can confound the comparison. Good AUC and monotonic
constraints do not remove that problem.

Our synthetic historical price variation is randomized and has known support,
which makes the teaching example easier. Even perfect recovery of its simulator
would not validate real customers. To claim real incremental profit, design an
authorized randomized experiment with a control policy, guardrails, sufficient
sample size, reliable cost/outcome measurement, and uncertainty estimates. Never
call simulated backtest contribution real-world uplift.

## Five-minute verbal walkthrough

**0:00–1:00 — Business and data.** “I built a local synthetic shipping pricing
system. It answers two questions: what is the chance a customer accepts this price,
and which allowed price gives the highest predicted expected contribution? One
row is one offered shipping quote, not one shipment or customer lifetime. The
generator creates 10,000 quotes over 240 days with packages, zones, services,
segments, estimated costs, independently varied markups, and noisy acceptance.
Daily quotes have no labels. No real UPS data or systems are involved.”

**1:00–2:00 — Features and evaluation.** “I use only quote-time information:
weight, zone, service, segment, package volume, fuel, estimated cost, offered price,
markup, weekday, and month. A whitelist excludes outcomes and simulator variables.
I split by dates to imitate future deployment, and exclude labels that arrived
after the relevant cutoff. Scalers and one-hot encoders fit on training rows only.
I compare logistic regression with XGBoost using validation log loss, AUC, Brier
score, and price-response checks. I freeze selection before reporting the final
test and cache an evaluated experiment so retries do not reuse the test for tuning.”

**2:00–3:00 — Decision layer.** “Acceptance alone is not the business objective.
For each feasible candidate price I rebuild price-dependent features, predict
acceptance, and multiply by price minus estimated cost. I choose the maximum on
a simple grid with cent rounding, minimum margin, maximum price, and historical
markup support. XGBoost constrains both price and markup downward, verified in
working tests. I still refuse rising, flat, or nonfinite response curves. This
adds a clear failure path instead of hiding unreliable recommendations.”

**3:00–4:00 — Execution and serving.** “Dagster has separate training and daily
scoring asset graphs. The weekly job can publish a model only after explicit
validation gates and an incumbent comparison. It writes complete versioned
artifacts, verifies their checksum and a prediction probe, then atomically updates
the current and previous pointers. Daily scoring loads that model, never retrains,
and saves one recommendation per quote transactionally. FastAPI handles individual
requests with the same fitted feature pipeline. Readiness fails with 503 if no
approved model can load. Docker serves the actual host-trained artifact with
persistent model and SQLite mounts.”

**4:00–5:00 — Evidence and limits.** “I executed pytest, both Dagster jobs, real
HTTP requests, and the Docker restart test. The delivery report contains actual
metrics, run IDs, failures fixed, and persisted counts. Monitoring tracks volume,
versions, failures, latency, and simple numeric/categorical drift. Before labels
arrive, it makes no accuracy claim; delayed synthetic outcomes enable calibration
and simulated realized contribution. Feature importance explains model usage,
not causal effects. This is a local learning deployment, not real UPS production
experience. Real profit uplift would need an authorized experiment, and scale
would need measured batching, warehouse storage, deployment controls, and load
tests.”
