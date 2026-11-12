import json

import joblib
import numpy as np
import pandas as pd
import pytest

from shipping.features import FEATURES, quote_features
from shipping.modeling import chronological_split, curve_check, evaluate_models, train_models
from shipping.registry import load_approved
from shipping.storage import load_history
from shipping.validation import QUOTE_COLUMNS, DataQualityError


def test_date_splits_and_label_cutoffs(project):
    split = chronological_split(load_history())
    train, validation, test = (split[k] for k in ["train", "validation", "test"])
    assert train.quote_timestamp.max().normalize() < validation.quote_timestamp.min().normalize()
    assert validation.quote_timestamp.max().normalize() < test.quote_timestamp.min().normalize()
    assert train.outcome_timestamp.max() < pd.Timestamp(split["metadata"]["train"]["end_exclusive"])
    assert validation.outcome_timestamp.max() < pd.Timestamp(split["metadata"]["validation"]["end_exclusive"])
    assert not set(train.quote_id) & set(test.quote_id)
    assert split["metadata"]["train"]["late_labels_excluded"] > 0


def test_future_label_cannot_enter_training(project):
    raw = load_history()
    original = chronological_split(raw)
    quote_id = original["train"].iloc[0].quote_id
    cutoff = pd.Timestamp(original["metadata"]["train"]["end_exclusive"])
    raw.loc[raw.quote_id == quote_id, "outcome_timestamp"] = (cutoff + pd.Timedelta(np.timedelta64(10, "D"))).isoformat()
    split = chronological_split(raw)
    assert quote_id not in set(split["train"].quote_id)


def test_no_label_or_latent_features(project):
    raw = load_history().head(10)
    features = quote_features(raw.assign(hidden_acceptance_probability=0.999))
    assert list(features) == FEATURES
    changed = raw.copy()
    changed["accepted_quote"] = 1 - changed.accepted_quote
    changed["outcome_timestamp"] = "2099-01-01"
    pd.testing.assert_frame_equal(features, quote_features(changed))


@pytest.mark.parametrize("name", ["logistic_regression", "xgboost"])
def test_models_probability_and_encoder_training_only(project, name):
    _, report, _ = load_approved()
    model = joblib.load(project.artifacts / "models" / report["version"] / f"{name}.joblib")
    split = chronological_split(load_history())
    probability = model.predict_proba(split["test"])[:, 1]
    assert np.isfinite(probability).all()
    assert ((0 <= probability) & (probability <= 1)).all()
    scaler = model.named_steps["preprocessing"].named_steps["columns"].named_transformers_["numeric"]
    assert scaler.n_samples_seen_ == len(split["train"])
    assert curve_check(model, split["validation"])["failures"] == 0


def test_xgboost_constraints_match_transformed_columns(project):
    _, report, _ = load_approved()
    model = joblib.load(project.artifacts / "models" / report["version"] / "xgboost.joblib")
    names = model.named_steps["preprocessing"].named_steps["columns"].get_feature_names_out()
    constraints = model.named_steps["classifier"].get_params()["monotone_constraints"]
    mapped = dict(zip(names, constraints, strict=True))
    assert mapped["numeric__quoted_price"] == mapped["numeric__markup_ratio"] == -1


def test_cached_training_does_not_reopen_test(project, monkeypatch):
    import shipping.modeling as modeling

    raw = load_history()
    split = chronological_split(raw)
    monkeypatch.setattr(modeling, "metrics", lambda *args: pytest.fail("Cached experiment reopened evaluation"))
    bundle = train_models(split, raw)
    assert bundle["cached"]
    report = evaluate_models(bundle, split)
    assert report["selected_model"] == min(report["validation"], key=lambda x: report["validation"][x]["log_loss"])
    assert json.loads((project.artifacts / "model_comparison.json").read_text())["version"] == report["version"]


def test_unlabeled_daily_cannot_be_split_for_training(project):
    from shipping.data import generate_daily

    with pytest.raises(DataQualityError):
        chronological_split(generate_daily())


def test_price_curve_check_uses_quote_time_columns(project):
    model, _, _ = load_approved()
    labeled = load_history()
    assert "accepted_quote" in labeled
    assert set(QUOTE_COLUMNS).issubset(labeled)
    assert curve_check(model, labeled)["failures"] == 0


def test_bad_validation_curves_stop_before_final_test(project, monkeypatch):
    import shipping.modeling as modeling

    model, report, _ = load_approved()
    splits = chronological_split(load_history())
    metric_calls = []

    def validation_only_metrics(model, frame):
        assert frame is splits["validation"], "Final test opened despite failed validation curves"
        metric_calls.append(True)
        return {"log_loss": 0.4}

    monkeypatch.setattr(modeling, "metrics", validation_only_metrics)
    monkeypatch.setattr(modeling, "curve_check", lambda *args: {"checked": 50, "failures": 50})
    with pytest.raises(ValueError, match="final test remains unopened"):
        evaluate_models({"cached": False, "version": report["version"],
                         "models": {"baseline": model, "candidate": model}}, splits)
    assert len(metric_calls) == 2
