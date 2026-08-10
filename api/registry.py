"""
Loads model + SHAP explainer + feature list from the MLflow Model Registry's
"production" alias, if MLFLOW_TRACKING_URI is configured and reachable.

The registered model (see train.py) is logged under a run that also carries
the SHAP explainer and feature_cols.txt as plain artifacts on that same run,
so resolving the "production" alias to a run_id gives us everything the API
needs, all versioned together.
"""
import os

import joblib

REGISTERED_MODEL_NAME = "nba-predictor"
ALIAS = "production"


def load_production_model():
    """Returns (model, explainer, feature_cols, source_label) or None if
    MLFLOW_TRACKING_URI isn't set / the registry can't be reached / no
    version has been promoted to the "production" alias yet."""
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        return None

    import mlflow
    import mlflow.sklearn
    from mlflow import MlflowClient

    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()
    version = client.get_model_version_by_alias(REGISTERED_MODEL_NAME, ALIAS)

    model = mlflow.sklearn.load_model(f"models:/{REGISTERED_MODEL_NAME}@{ALIAS}")

    local_dir = mlflow.artifacts.download_artifacts(run_id=version.run_id)
    explainer = joblib.load(os.path.join(local_dir, "explainer", "shap_explainer.joblib"))
    with open(os.path.join(local_dir, "feature_cols.txt")) as f:
        feature_cols = [line.strip() for line in f if line.strip()]

    source = f"mlflow:{REGISTERED_MODEL_NAME}@{ALIAS} (v{version.version}, run {version.run_id[:8]})"
    return model, explainer, feature_cols, source
