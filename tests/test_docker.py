import os
import subprocess

import httpx
import pytest

from shipping.registry import load_approved
from shipping.smoke import smoke


@pytest.mark.docker
@pytest.mark.skipif(os.environ.get("RUN_DOCKER_TESTS") != "1", reason="Run make docker-up && make docker-smoke to test a real container")
def test_docker_api_reads_persisted_approved_model():
    url = os.environ.get("DOCKER_API_URL", "http://127.0.0.1:8001")
    _, _, entry = load_approved()
    assert smoke(url)["recommendation"]["model_version"] == entry["version"]
    # Restart the actual container and confirm the mounted artifact survives.
    subprocess.run(["docker", "compose", "restart", "api"], check=True, capture_output=True)
    assert smoke(url)["passed"]
    info = httpx.get(url + "/model-info").json()
    assert info["model_version"] == entry["version"]
