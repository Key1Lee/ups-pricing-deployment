"""Immutable local model versions and one atomic pointer to current/previous versions."""

import fcntl
import hashlib
import json
import logging
import os
import re
import tempfile
from contextlib import contextmanager

import joblib
import pandas as pd

from shipping.config import paths
from shipping.modeling import metrics
from shipping.pricing import checked_probabilities
from shipping.storage import utc_now

logger = logging.getLogger(__name__)


class ModelUnavailable(RuntimeError):
    pass


class ApprovalError(ValueError):
    pass


def atomic_json(path, value):
    """Same-filesystem rename prevents readers from observing half-written JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=".pending-")
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(value, handle, indent=2, allow_nan=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


@contextmanager
def registry_lock():
    # Serialize promotions and rollback; API readers only need the atomic pointer.
    with (paths().artifacts / ".registry.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def read_registry():
    try:
        value = json.loads((paths().artifacts / "approved.json").read_text())
        if not value.get("current"):
            raise ValueError("No current model")
        return value
    except (OSError, ValueError, TypeError) as exc:
        raise ModelUnavailable("No readable approved model registry") from exc


def load_entry(entry):
    try:
        version = entry["version"]
        name = entry["model_name"]
        if not re.fullmatch(r"model-[a-f0-9]{16}", version) or name not in ("xgboost", "logistic_regression"):
            raise ValueError("Invalid registry entry")
        directory = paths().artifacts / "models" / version
        model_path = directory / f"{name}.joblib"
        if hashlib.sha256(model_path.read_bytes()).hexdigest() != entry["sha256"]:
            raise ValueError("Artifact checksum mismatch")
        # Pickle is only safe for trusted artifacts that this local project generated.
        model = joblib.load(model_path)
        report = json.loads((directory / "evaluation.json").read_text())
        if report["version"] != version or report["selected_model"] != name:
            raise ValueError("Artifact metadata does not match its registry entry")
        probe = pd.read_json(directory / "probe.json")
        checked_probabilities(model, probe)
        return model, report
    except Exception as exc:
        raise ModelUnavailable("Approved artifact is incomplete, corrupt, or incompatible") from exc


def load_approved():
    registry = read_registry()
    try:
        model, report = load_entry(registry["current"])
        return model, report, registry["current"]
    except ModelUnavailable:
        if registry.get("previous"):
            logger.error("Current artifact failed; loading previous approved model")
            model, report = load_entry(registry["previous"])
            return model, report, registry["previous"]
        raise


def approval_criteria(report):
    name = report["selected_model"]
    if name is None:
        raise ApprovalError("No model passed price-response checks")
    score = report["validation"][name]
    checks = {
        "validation_auc_at_least_0.75": score["roc_auc"] >= 0.75,
        "validation_log_loss_at_most_0.60": score["log_loss"] <= 0.60,
        "validation_brier_at_most_0.22": score["brier_score"] <= 0.22,
        "validation_price_curves_pass": report["price_curve_checks"][name]["failures"] == 0,
    }
    if not all(checks.values()):
        raise ApprovalError(f"Model approval failed: {checks}")
    return checks


def promote(report, validation):
    checks = approval_criteria(report)  # Failed criteria never change an existing pointer.
    version, name = report["version"], report["selected_model"]
    model_path = paths().artifacts / "models" / version / f"{name}.joblib"
    entry = {"version": version, "model_name": name,
             "sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
             "approved_at": utc_now(), "approval_checks": checks}
    load_entry(entry)  # Full deserialization and prediction must work before promotion.
    with registry_lock():
        try:
            old = read_registry()
        except ModelUnavailable:
            old = {}
        if old.get("current", {}).get("version") == version:
            return old["current"]
        incumbent = None
        if old:
            incumbent_model, incumbent_report, incumbent = load_approved()
            if pd.Timestamp(incumbent_report["splits"]["train"]["end_exclusive"]) > pd.Timestamp(
                report["splits"]["train"]["end_exclusive"]
            ):
                raise ApprovalError("Refusing promotion against a model trained on later dates")
            incumbent_loss = metrics(incumbent_model, validation)["log_loss"]
            if report["validation"][name]["log_loss"] > incumbent_loss + 0.005:
                raise ApprovalError("Candidate validation log loss regressed against the incumbent")
        atomic_json(paths().artifacts / "approved.json", {"current": entry, "previous": incumbent})
    return entry


def rollback():
    with registry_lock():
        state = read_registry()
        previous = state.get("previous")
        if previous is None:
            raise ModelUnavailable("No prior approved model exists to roll back to")
        load_entry(previous)
        atomic_json(paths().artifacts / "approved.json", {"current": previous, "previous": state["current"]})
        return previous
