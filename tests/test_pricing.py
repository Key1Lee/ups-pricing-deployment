import numpy as np
import pytest

from shipping.api import example_quote
from shipping.pricing import (
    PriceRules,
    PricingError,
    UnreliablePriceCurve,
    candidate_table,
    recommend,
)
from shipping.registry import load_approved


def test_optimal_price_and_margin_guardrails(project):
    model, _, entry = load_approved()
    rules = PriceRules(minimum_margin=3.0, maximum_price=30, step=0.25)
    result, table = recommend(model, example_quote(), entry["version"], rules)
    assert 15 <= result["recommended_price"] <= 30
    assert np.diff(table.predicted_probability).max() <= 1e-6
    assert result["expected_contribution"] == table.expected_contribution.max()
    assert np.allclose(table.expected_contribution, table.predicted_probability * table.margin)
    assert np.allclose(table.candidate_price * 100, np.round(table.candidate_price * 100))


@pytest.mark.parametrize("rules", [PriceRules(maximum_price=10), PriceRules(step=-1), PriceRules(maximum_price=np.inf)])
def test_infeasible_rules_rejected(project, rules):
    model, _, _ = load_approved()
    with pytest.raises(PricingError):
        candidate_table(model, example_quote(), rules)


class BrokenCurve:
    def __init__(self, mode):
        self.mode = mode

    def predict_proba(self, frame):
        values = np.linspace(0.1, 0.9, len(frame))
        if self.mode == "flat":
            values[:] = 0.5
        if self.mode == "nan":
            values[:] = np.nan
        if self.mode == "outside":
            values[:] = 1.1
        return np.column_stack([1 - values, values])


@pytest.mark.parametrize("mode", ["increasing", "flat", "nan", "outside"])
def test_unreliable_curves_refuse_optimization(mode):
    with pytest.raises(UnreliablePriceCurve):
        candidate_table(BrokenCurve(mode), example_quote())
