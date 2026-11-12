import os
import shutil

import pytest

from shipping.config import DEMO_DAY, Paths
from shipping.data import generate_daily, generate_history
from shipping.modeling import chronological_split, evaluate_models, train_models
from shipping.registry import promote
from shipping.storage import load_history, save_quotes


@pytest.fixture(scope="session")
def trained_project(tmp_path_factory):
    """A separate small synthetic experiment; tests never mutate the demo's model."""
    root = tmp_path_factory.mktemp("trained-project")
    previous = os.environ.get("UPS_PRICING_HOME")
    os.environ["UPS_PRICING_HOME"] = str(root)
    try:
        p = Paths(root).ensure()
        history = generate_history(n=3000, days=180, seed=7)
        daily = generate_daily()
        save_quotes(history, labeled=True)
        save_quotes(daily, labeled=False, batch_date=DEMO_DAY)
        daily.to_csv(p.data / f"quotes-{DEMO_DAY}.csv", index=False)
        raw = load_history()
        split = chronological_split(raw)
        report = evaluate_models(train_models(split, raw), split)
        promote(report, split["validation"])
        return p
    finally:
        if previous is None:
            os.environ.pop("UPS_PRICING_HOME", None)
        else:
            os.environ["UPS_PRICING_HOME"] = previous


@pytest.fixture
def project(trained_project, tmp_path, monkeypatch):
    for directory in ["data", "artifacts"]:
        shutil.copytree(trained_project.root / directory, tmp_path / directory)
    monkeypatch.setenv("UPS_PRICING_HOME", str(tmp_path))
    return Paths(tmp_path)
