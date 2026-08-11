"""
Load test for the NBA predictor API. /predict is the endpoint that matters —
/health is included at low weight since it's the cheap path most real
traffic wouldn't hit repeatedly.

Usage:
    # Interactive web UI at http://localhost:8089
    locust --host http://localhost:8000

    # Headless, e.g. 20 concurrent users, spawn 5/sec, for 60 seconds,
    # writing summary stats (including p50/p95/p99) to results_*.csv
    locust --headless -u 20 -r 5 -t 60s --host http://localhost:8000 --csv=results

    # Same, against the live deployment
    locust --headless -u 20 -r 5 -t 60s --host https://nba-predictor-api-zxwo.onrender.com --csv=results
"""
import random

from locust import HttpUser, task, between

from teams import NBA_TEAMS


class PredictorUser(HttpUser):
    wait_time = between(0.5, 2.0)

    @task(9)
    def predict(self):
        home, away = random.sample(NBA_TEAMS, 2)
        with self.client.post(
            "/predict",
            json={"home_team": home, "away_team": away},
            name="/predict",
            catch_response=True,
        ) as resp:
            if resp.status_code != 200:
                resp.failure(f"status {resp.status_code}: {resp.text[:200]}")

    @task(1)
    def health(self):
        self.client.get("/health", name="/health")
