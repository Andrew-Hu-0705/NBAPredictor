# Resume-Ready Metrics

Every number here is either measured directly (with the exact command used
to produce it) or explicitly marked as **not measured** — do not upgrade a
"not measured" line into a resume claim without actually running the
measurement first.

## Model performance

Source: `mlflow` run `f96151293ec348868ae9e0832a14d568`, promoted to the
`production` alias (`python promote_model.py --list`). 5-fold
`TimeSeriesSplit` cross-validation over 3,551 games (2022-23 through
2024-25 seasons, after dropping early-season rows with unpopulated rolling
stats).

| Metric | Value |
|---|---|
| CV accuracy (production model: XGBoost + sigmoid calibration) | **64.8% ± 2.7%** |
| CV AUC | **0.697 ± 0.043** |
| CV log loss | 0.628 |
| Baseline (always pick home team) | 55.7% |
| Improvement over baseline | **+9.1 points** |

Resume-safe phrasing: *"Achieved 64.8% cross-validated accuracy (0.70 AUC)
predicting NBA game outcomes via 5-fold time-series CV over 3,551 games — a
9-point improvement over the always-favor-home-team baseline."*

## Load test — live deployment

Source: `locust --headless -u 20 -r 5 -t 90s --host https://nba-predictor-api-zxwo.onrender.com --csv=results -f locustfile.py`,
run 2026-08-10 against the live Render deployment (free tier, single shared
vCPU instance). Traffic: 90% `/predict` (random valid team matchups), 10%
`/health`.

| Metric | Value |
|---|---|
| Total requests | 899 (824 `/predict`, 75 `/health`) |
| Failures | **0 (0.00% error rate)** |
| Sustained throughput | **~10.0 req/s** aggregate (9.2 req/s `/predict`) |
| `/predict` p50 | 740 ms |
| `/predict` p95 | 1,500 ms |
| `/predict` p99 | 1,900 ms |
| `/predict` max | 2,262 ms |

Resume-safe phrasing: *"Load-tested the deployed API at 20 concurrent
users, sustaining ~10 req/s with a 0% error rate; p95 latency 1.5s on a
free-tier single-vCPU instance."*

**Context that belongs in an interview, not necessarily the bullet itself:**
this includes real network latency from a local test client to Render's
servers, and Render's free tier is a shared/throttled vCPU — not a
production-grade instance. The same test run locally (API and load
generator both on localhost, eliminating network + removing the free-tier
CPU ceiling) hit p50=9ms / p95=60ms / p99=70ms at ~7.7 req/s with 10 users —
i.e., the model inference + SHAP computation itself is fast; the gap is
infra, not the code. Both numbers are real measurements of different
things — don't blend them into one claim.

## Engineering / test coverage

- **25/25 automated tests passing** (API contract tests, model sanity
  tests, drift-detection unit tests), run in CI on every push.
- CI/CD pipeline (GitHub Actions): automated test run → Docker build →
  **the built container is actually started and hit over HTTP** (`/health`,
  `/predict`) before the image is pushed to a container registry — not just
  "the Dockerfile has the right lines."
- Data drift detection (PSI-based) verified against real production traffic
  from the load test above: correctly flagged the load test's synthetic,
  uniformly-random team-matchup traffic as statistically different from the
  training distribution (`HOME_REST_DAYS` PSI 4.6, `HOME_MISSING_ROTATION_MIN`
  PSI 0.99) — expected and correct, since repeatedly querying random team
  pairs doesn't resemble a real season's game-by-game distribution. This is
  the drift monitor doing its job, not a data quality problem with the model.

## Explicitly NOT measured — do not put these on a resume

- **Uptime.** The service was deployed today. "99.9% uptime" or any uptime
  percentage is not something we have data for yet — it would be fabricated.
  If you want a real number: let it run for a few days to a week, then check
  Render's dashboard (it shows deploy/restart history) or set up a free
  external monitor (e.g. UptimeRobot) and pull the actual percentage from
  there.
- **Requests/day or "at scale" claims.** We measured req/s under a
  synthetic 90-second load test, not sustained real-world traffic. Don't
  extrapolate to a daily/monthly number.
- **Cost savings, revenue impact, or business metrics.** This is a
  portfolio project with no real users — don't invent business impact.

## Suggested resume bullets (copy/adapt, not verbatim-required)

- Built and deployed a production ML serving system (FastAPI + XGBoost +
  SHAP) with request validation, structured logging, containerization, and
  CI/CD; achieved 64.8% cross-validated accuracy (0.70 AUC) on NBA game
  outcome prediction, a 9-point lift over baseline.
- Implemented MLOps infrastructure end-to-end: MLflow experiment tracking
  and model registry with an explicit promotion workflow, Prometheus/Grafana
  monitoring, PSI-based data drift detection, and a GitHub Actions pipeline
  that builds, smoke-tests, and pushes a Docker image on every passing
  commit.
- Load-tested the deployed API, sustaining ~10 req/s with a 0% error rate
  (p95 latency 1.5s) against a live free-tier deployment; 25 automated
  tests (contract, model-sanity, drift) run in CI on every push.
