import os

import pytest
from fastapi.testclient import TestClient

# Tests exercise the local-joblib-file path deterministically — no dependency
# on a running MLflow server. api/registry.py already falls back to local
# files if MLFLOW_TRACKING_URI is unset, so this just makes sure a
# developer's or CI runner's environment doesn't accidentally point at one.
os.environ.pop("MLFLOW_TRACKING_URI", None)

from api.main import app  # noqa: E402  (import must follow the env cleanup above)


@pytest.fixture(scope="session")
def client():
    with TestClient(app) as c:
        yield c
