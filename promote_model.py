"""
promote_model.py
-----------------
Promotes a specific version of the "nba-predictor" model in the MLflow Model
Registry to the "production" alias. api/main.py resolves that alias at
startup ("models:/nba-predictor@production") — this is the switch that
decides which trained model the API actually serves.

Every `python train.py` run registers a new, unpromoted model version; none
of them are served until explicitly promoted here. Review a run's metrics
first (MLflow UI: `mlflow ui --backend-store-uri file:./mlruns`, or
`--list` below) before promoting.

Usage:
    python promote_model.py --list        # show all versions + current production
    python promote_model.py 4             # promote version 4 to production
"""

import argparse
import os
import sys

import mlflow
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

REGISTERED_MODEL_NAME = "nba-predictor"
ALIAS = "production"


def list_versions(client: MlflowClient):
    versions = client.search_model_versions(f"name='{REGISTERED_MODEL_NAME}'")
    versions = sorted(versions, key=lambda v: int(v.version), reverse=True)
    if not versions:
        print(f"No versions registered yet for '{REGISTERED_MODEL_NAME}'. Run train.py first.")
        return

    try:
        current_prod_version = client.get_model_version_by_alias(REGISTERED_MODEL_NAME, ALIAS).version
    except MlflowException:
        current_prod_version = None

    print(f"{'Version':<10}{'CV AUC':<10}{'CV Acc':<10}{'Run ID':<35}")
    for v in versions:
        run = client.get_run(v.run_id)
        auc = run.data.metrics.get("production_cv_auc_mean")
        acc = run.data.metrics.get("production_cv_accuracy_mean")
        marker = "  <-- current production" if v.version == current_prod_version else ""
        auc_s = f"{auc:.3f}" if auc is not None else "n/a"
        acc_s = f"{acc:.3f}" if acc is not None else "n/a"
        print(f"{v.version:<10}{auc_s:<10}{acc_s:<10}{v.run_id:<35}{marker}")


def promote(client: MlflowClient, version: str):
    client.set_registered_model_alias(REGISTERED_MODEL_NAME, ALIAS, version)
    print(f"Promoted version {version} of '{REGISTERED_MODEL_NAME}' to '{ALIAS}'.")
    print("Restart (or redeploy) the API for it to pick up the new production model.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("version", nargs="?", help="Model version number to promote to production")
    parser.add_argument("--list", action="store_true", help="List all registered versions and exit")
    args = parser.parse_args()

    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "file:./mlruns"))
    client = MlflowClient()

    if args.list or not args.version:
        list_versions(client)
        sys.exit(0)

    promote(client, args.version)
