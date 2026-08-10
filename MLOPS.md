# MLOps Infrastructure

This document covers the production infrastructure built around the model —
serving, containerization, experiment tracking, and (as later stages land)
CI/CD, deployment, and monitoring. See [README.md](README.md) for the model
and data pipeline itself.

## Serving API

`api/main.py` is a FastAPI service wrapping the trained model:

- `POST /predict` — body `{"home_team": "BOS", "away_team": "MIA"}`, validated
  with Pydantic (team abbreviations checked against the real NBA team list,
  home/away must differ). Returns win probabilities plus a full SHAP
  breakdown (every feature, sorted by |impact|, tagged `favors: home|away`).
- `GET /health` — reports whether the model/explainer/features loaded, and
  where the model came from (`model_source`: either a local file path or the
  MLflow registry version/run it resolved).
- Every prediction is logged as one structured JSON line (timestamp, latency,
  a hash of the input, and the output) via `api/logging_conf.py`.
- Model/explainer/feature list are loaded **once** at startup (`lifespan`),
  not per-request — `predict.py`'s `predict_game()` accepts them as optional
  arguments for this reason; the CLI (`python predict.py --home X --away Y`)
  still loads fresh each run.

Run locally: `uvicorn api.main:app --reload`

## Containerization

`Dockerfile` is a multi-stage build: a `builder` stage (with `build-essential`)
installs pinned dependencies from `requirements-api.txt` into a venv; the
`runtime` stage (`python:3.11-slim`, no compilers) copies just that venv plus
the app code and runs as a non-root user. `requirements-api.txt` is
deliberately smaller than `requirements.txt` — no `nba_api`, `Flask`, or
`matplotlib`, since those are only needed for data fetching, the legacy Flask
UI, and training-time plots, not for serving.

`docker-compose.yml` runs the API alongside its supporting services (MLflow
now; Prometheus/Grafana once Stage 6 lands):

```bash
docker compose up --build
curl http://localhost:8000/health
```

## Experiment Tracking & Model Registry

`train.py` logs every run to MLflow: hyperparameters (`XGB_PARAMS`,
calibration settings), CV metrics for all five model variants it evaluates
(raw XGBoost, sigmoid/isotonic-calibrated, logistic regression, ensemble —
accuracy/AUC/log loss/Brier), feature importances (`feature_importance.json`),
the SHAP summary plot, and the production model artifact itself. Where it
writes to depends on `MLFLOW_TRACKING_URI`:

- **Unset** (default, e.g. running `train.py` on your laptop): logs to a local
  `file:./mlruns` store. No server needed. Browse it with:
  ```bash
  mlflow ui --backend-store-uri file:./mlruns
  ```
- **Set** (e.g. `http://mlflow:5000` inside `docker-compose`): logs to that
  tracking server instead.

Each run also **registers** a new version of the `nba-predictor` model in the
MLflow Model Registry. Registering ≠ serving — a fresh version sits unpromoted
until you explicitly point the `production` alias at it. This is the gate
that stops a training run from silently changing what's live.

### Promoting a model to production

```bash
# 1. See what's registered, and what's currently live
python promote_model.py --list

# 2. Review the candidate version's metrics (via --list, or the MLflow UI)
#    before promoting — there's no automatic quality gate here.

# 3. Point the "production" alias at the version you want served
python promote_model.py 4

# 4. Restart (or redeploy) the API so it re-resolves the alias
```

### How the API resolves "production"

At startup, `api/registry.py` checks `MLFLOW_TRACKING_URI`:

- If set, it resolves `models:/nba-predictor@production` via the MLflow
  Model Registry, loads that model version, and downloads the SHAP explainer
  + `feature_cols.txt` from the **same run** (they're logged as plain
  artifacts alongside the registered model, so one alias resolution gets you
  everything the API needs, all versioned together — not three separately
  drifting files).
- If it's unset, or the registry lookup fails for any reason (server
  unreachable, no version promoted yet), the API logs a warning and falls
  back to the local `nba_model.joblib` / `shap_explainer.joblib` /
  `feature_cols.txt` files instead of refusing to start. This trades strict
  "always serve exactly what's tagged production" for availability — a
  registry hiccup degrades to last-known-good rather than an outage. Check
  `GET /health`'s `model_source` field to see which one actually happened.

Verified locally: trained a model, registered it (v1), promoted it to
`production`, and confirmed via `/health` that the API resolved and served
that exact registry version (not the local files) — then deleted the alias
and confirmed it fell back to local files with a logged warning, and
re-promoted it afterward.

## CI/CD

`.github/workflows/ci.yml` runs on every push/PR to `main`:

1. **`test`** — installs `requirements-api.txt` + `requirements-dev.txt`
   (i.e. exactly what the API needs at runtime, plus pytest/httpx — not the
   full dev environment) and runs `pytest`. See `tests/` — API contract
   tests (`test_api_contract.py`: response schema, probability bounds,
   validation rejects bad teams/types/missing fields) and model sanity tests
   (`test_model_sanity.py`: probabilities always in [0,1] and sum to 1 across
   several matchups, unknown teams raise, SHAP output always matches the
   feature count).
2. **`build-and-push`** (only on push to `main`, and only if `test` passed) —
   builds the Docker image, actually **runs the built container** and hits
   `/health` and `/predict` over HTTP (this is the first point in the whole
   pipeline that runs the real container — local dev on this machine had no
   Docker daemon, so this closes that gap), and only then pushes to
   `ghcr.io/<owner>/<repo>` tagged `latest` and the commit SHA. No extra
   registry credentials needed — it uses the built-in `GITHUB_TOKEN`.

**Note:** `nba_model.joblib`, `shap_explainer.joblib`, and `features.csv` were
previously gitignored, which meant a fresh `git clone` had no model to test
or containerize at all — `.gitignore` was fixed to track them (they're small:
~3.7MB combined) since CI needs something deterministic to build against
without calling the live NBA API on every push. **These still need to be
committed** — I didn't commit anything in this session; run:
```bash
git add nba_model.joblib shap_explainer.joblib features.csv .gitignore
git commit -m "Track serving artifacts needed by Docker/CI"
```
before pushing, or CI's Docker build will fail with a missing-file error.

## Deployment

Deployed on **Render** (free web service tier) — see the platform comparison
and rationale in the chat where this stage was built. `render.yaml` is a
Blueprint: point Render at the repo and it builds `Dockerfile` directly,
health-checks `/health`, and redeploys automatically on every push to `main`.

`MLFLOW_TRACKING_URI` is intentionally left unset in the deployed environment
— the API serves from the `nba_model.joblib` / `shap_explainer.joblib` files
baked into the image (see `api/registry.py`'s fallback behavior above) rather
than depending on a second always-on MLflow service. Promoting a new model
means: retrain, `promote_model.py` locally, copy the new artifacts over the
committed ones, commit, push — no MLflow server needs to be kept running in
production for this project's scale.

**Known tradeoff:** Render's free tier sleeps a service after 15 minutes of
no traffic; the next request pays a ~30-60s cold-start penalty. This matters
for Stage 6's latency numbers — they'll be reported both cold and warm,
since only warm numbers are representative of sustained load.

### Incident: cross-Python-version pickle failure on first deploy

The first real deploy came up "live" but degraded — `/health` returned
`model_loaded: false`. The structured logging built in Stage 1 pointed
straight at it: `"error": "code() argument 13 must be str, not int"`.

Root cause: `shap.TreeExplainer` (shap 0.44.1) embeds a numba-JIT-compiled
closure in its pickled state (`shap_explainer.joblib`). numba serializes
that closure's Python code object directly, and `types.CodeType`'s
constructor signature changed between Python 3.8 (used to train/pickle the
original artifacts) and 3.11 (what the Docker image runs) — reconstructing
the code object on unpickle threw. `nba_model.joblib` alone loaded fine
under 3.11; the explainer was the only affected artifact, confirmed by
loading each file in isolation with a full traceback.

Fix: retrained under Python 3.11 (matching the serving image exactly) and
re-pickled both artifacts. This also required rebuilding the local dev
`venv/` from Python 3.8.8 to 3.11 — pickle compatibility isn't guaranteed in
either direction across Python minor versions, so training and serving now
have to use the same one. `numba`/`llvmlite` are pinned explicitly in
`requirements.txt`/`requirements-api.txt` for the same reason: shap's own
dependency spec doesn't pin them, and letting pip pick "latest compatible"
independently for training vs. serving reopens the exact same failure mode.

A second gotcha surfaced while fixing the first: pip's resolver silently
upgrades already-installed packages to satisfy a new package's dependency
range if they aren't re-pinned in that same install command (e.g.
`pip install mlflow` alone quietly bumped `pandas` past its pin). All
pickle-sensitive packages (`numpy`, `pandas`, `scikit-learn`, `xgboost`,
`shap`, `numba`, `llvmlite`, `joblib`) now need to be installed together in
one command — installing them piecemeal across separate commands can drift
versions apart without any visible error until something tries to unpickle
across the mismatch.

Verified the fix with the exact `requirements-api.txt` pin set, isolated
from every other dependency, in a clean Python 3.11 venv: both artifacts
load, `/health` reports `model_loaded: true`, and `/predict` returns the
correct response including the full SHAP breakdown.
