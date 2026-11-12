"""Temporal evaluation: validation selects the model; the final test is opened once."""

import hashlib
import importlib.metadata
import json
import time
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from shipping.config import SEED, paths
from shipping.features import make_preprocessor, quote_features
from shipping.pricing import PriceRules, candidate_table, checked_probabilities
from shipping.storage import save_json
from shipping.validation import QUOTE_COLUMNS, labels_available, validate_quotes

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

MODEL_CONFIG = {"seed": SEED, "xgb_trees": 180, "xgb_depth": 3, "learning_rate": 0.045}


def chronological_split(frame):
    valid = validate_quotes(frame, labeled=True).sort_values("quote_timestamp")
    dates = valid.quote_timestamp.dt.normalize().sort_values().unique()
    if len(dates) < 180:
        raise ValueError("At least 180 distinct quote dates are required")
    train_end = pd.Timestamp(dates[int(len(dates) * 0.70)])
    validation_end = pd.Timestamp(dates[int(len(dates) * 0.85)])
    test_end = pd.Timestamp(dates[-1]) + pd.Timedelta(np.timedelta64(1, "D"))
    # Validation outcomes must be available before the final test period begins.
    train_raw = valid.loc[valid.quote_timestamp < train_end]
    validation_raw = valid.loc[(valid.quote_timestamp >= train_end) & (valid.quote_timestamp < validation_end)]
    test = valid.loc[valid.quote_timestamp >= validation_end].copy()
    train = labels_available(train_raw, train_end)
    validation = labels_available(validation_raw, validation_end)
    # Final evaluation waits for outcomes to mature. Its cutoff is recorded explicitly.
    evaluation_as_of = valid.outcome_timestamp.max() + pd.Timedelta(np.timedelta64(1, "s"))
    test = labels_available(test, evaluation_as_of)
    for split in (train, validation, test):
        if split.accepted_quote.nunique() != 2:
            raise ValueError("Each split needs both outcomes")
    metadata = {
        "train": {"start": str(dates[0]), "end_exclusive": str(train_end), "rows": len(train),
                  "late_labels_excluded": len(train_raw) - len(train)},
        "validation": {"start": str(train_end), "end_exclusive": str(validation_end),
                       "rows": len(validation), "late_labels_excluded": len(validation_raw) - len(validation)},
        "test": {"start": str(validation_end), "end_exclusive": str(test_end), "rows": len(test)},
        "final_evaluation_as_of": str(evaluation_as_of),
    }
    return {"train": train, "validation": validation, "test": test, "metadata": metadata}


def versions():
    return {name: importlib.metadata.version(name) for name in
            ["scikit-learn", "xgboost", "pandas", "numpy", "dagster"]}


def experiment_version(frame):
    # Identical data, code, dependencies, and settings reuse their evaluated artifact.
    digest = hashlib.sha256(frame.to_csv(index=False).encode())
    digest.update(json.dumps({"config": MODEL_CONFIG, "versions": versions()}, sort_keys=True).encode())
    for name in ["modeling.py", "features.py", "validation.py", "pricing.py"]:
        digest.update(Path(__file__).with_name(name).read_bytes())
    return "model-" + digest.hexdigest()[:16]


def train_models(splits, raw_frame):
    version = experiment_version(raw_frame)
    directory = paths().artifacts / "models" / version
    cached = directory / "evaluation.json"
    if cached.exists():
        return {"version": version, "cached": True, "report": json.loads(cached.read_text())}
    models = {}
    train = splits["train"]
    for name in ["logistic_regression", "xgboost"]:
        preprocessing = make_preprocessor()
        transformed = preprocessing.fit_transform(train)
        feature_names = preprocessing.named_steps["columns"].get_feature_names_out()
        if name == "logistic_regression":
            classifier = LogisticRegression(max_iter=1500, C=1.0, random_state=SEED)
        else:
            # StandardScaler preserves order. Constrain BOTH changing price features.
            constraints = tuple(-1 if x in ("numeric__quoted_price", "numeric__markup_ratio") else 0
                                for x in feature_names)
            classifier = XGBClassifier(
                n_estimators=MODEL_CONFIG["xgb_trees"], max_depth=MODEL_CONFIG["xgb_depth"],
                learning_rate=MODEL_CONFIG["learning_rate"], tree_method="hist", max_bin=512,
                min_child_weight=15, reg_lambda=5, subsample=1, colsample_bytree=1,
                monotone_constraints=constraints, eval_metric="logloss", random_state=SEED, n_jobs=1,
            )
        classifier.fit(transformed, train.accepted_quote)
        models[name] = Pipeline([("preprocessing", preprocessing), ("classifier", classifier)])
    return {"version": version, "cached": False, "models": models}


def metrics(model, frame):
    probability = checked_probabilities(model, frame)
    # Warm up first; report median single-quote latency including feature transformation.
    model.predict_proba(frame.head(1))
    timings = []
    for _ in range(15):
        start = time.perf_counter()
        model.predict_proba(frame.head(1))
        timings.append((time.perf_counter() - start) * 1000)
    observed, predicted = calibration_curve(frame.accepted_quote, probability, n_bins=10)
    bands = pd.DataFrame({"quoted_price": frame.quoted_price, "label": frame.accepted_quote,
                          "probability": probability})
    bands["price_band"] = pd.cut(bands.quoted_price, [0, 15, 25, 40, 60, 100, np.inf])
    grouped = bands.groupby("price_band", observed=True).agg(
        count=("label", "size"), acceptance_rate=("label", "mean"),
        mean_predicted_probability=("probability", "mean"),
    ).reset_index()
    grouped["price_band"] = grouped.price_band.astype(str)
    return {
        "roc_auc": float(roc_auc_score(frame.accepted_quote, probability)),
        "log_loss": float(log_loss(frame.accepted_quote, probability)),
        "brier_score": float(brier_score_loss(frame.accepted_quote, probability)),
        "median_inference_ms": float(np.median(timings)),
        "acceptance_rate": float(frame.accepted_quote.mean()), "rows": len(frame),
        "calibration": {"mean_predicted": predicted.tolist(), "fraction_accepted": observed.tolist()},
        "acceptance_by_price_band": grouped.to_dict(orient="records"),
    }


def curve_check(model, frame):
    failures = []
    for _, row in frame[QUOTE_COLUMNS].head(50).iterrows():
        try:
            candidate_table(model, row.to_dict(), PriceRules())
        except ValueError as exc:
            failures.append(str(exc))
    return {"checked": min(50, len(frame)), "failures": len(failures), "examples": failures[:3]}


def evaluate_models(bundle, splits):
    if bundle["cached"]:
        report = bundle["report"]
        save_json(paths().artifacts / "model_comparison.json", report)
        return report
    models = bundle["models"]
    validation = {name: metrics(model, splits["validation"]) for name, model in models.items()}
    curves = {name: curve_check(model, splits["validation"]) for name, model in models.items()}
    eligible = [name for name in models if curves[name]["failures"] == 0]
    if not eligible:
        raise ValueError(f"No valid price-response model; final test remains unopened: {curves}")
    selected = min(eligible, key=lambda name: validation[name]["log_loss"])
    # Selection is now frozen. The held-out test is evaluated once for each fixed model.
    test = {name: metrics(model, splits["test"]) for name, model in models.items()}
    report = {
        "version": bundle["version"], "selected_model": selected,
        "selection_rule": "Lowest validation log loss among models passing all validation price-curve checks",
        "validation": validation, "test": test, "price_curve_checks": curves,
        "splits": splits["metadata"], "dependencies": versions(), "config": MODEL_CONFIG,
        "test_policy": "Reported once per immutable data/code/config version; never used for selection or approval",
        "causal_warning": "Predictive synthetic evaluation does not establish real-world incremental profit",
    }
    directory = paths().artifacts / "models" / bundle["version"]
    directory.mkdir(parents=True, exist_ok=True)
    for name, model in models.items():
        joblib.dump(model, directory / f"{name}.joblib")
    if selected:
        model = models[selected]
        names = model.named_steps["preprocessing"].named_steps["columns"].get_feature_names_out()
        classifier = model.named_steps["classifier"]
        importance = (classifier.feature_importances_ if selected == "xgboost"
                      else np.abs(classifier.coef_[0]))
        ranked = sorted(zip(names, importance.tolist()), key=lambda x: x[1], reverse=True)[:10]
        report["top_features"] = [{"feature": x, "importance": y} for x, y in ranked]
        report["importance_warning"] = "Gain importance or absolute standardized coefficients are predictive, not causal"
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.barh([x[0] for x in ranked][::-1], [x[1] for x in ranked][::-1])
        ax.set_title(f"{selected}: predictive feature importance (not causal)")
        fig.tight_layout()
        fig.savefig(paths().artifacts / "feature_importance.png", dpi=140)
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    for name in models:
        cal = test[name]["calibration"]
        ax.plot(cal["mean_predicted"], cal["fraction_accepted"], "o-", label=name)
    ax.set(xlabel="Predicted acceptance probability", ylabel="Observed acceptance fraction",
           title="Final test calibration (synthetic data)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(paths().artifacts / "calibration.png", dpi=140)
    plt.close(fig)
    # Save reference quote-time features for later drift reports; no labels are needed.
    quote_features(splits["train"]).to_csv(directory / "reference_features.csv", index=False)
    splits["train"].head(1).drop(columns=["accepted_quote", "outcome_timestamp"]).to_json(
        directory / "probe.json", orient="records", date_format="iso"
    )
    save_json(directory / "evaluation.json", report)
    save_json(paths().artifacts / "model_comparison.json", report)
    return report
