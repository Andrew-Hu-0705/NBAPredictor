"""
Rolling data drift check: compares the feature distributions of recent
/predict requests against the training-data reference computed by train.py
(see compute_drift_reference there). No database — just an in-memory ring
buffer of the last N requests, which is all a single-instance portfolio
deployment needs. Restarting the process resets the window; that's fine,
this is a "has anything looked weird lately" signal, not an audit log.

Population Stability Index (PSI) is the metric: bucket both the reference
and current windows into the same bins, compare bin proportions. Standard
thresholds (widely used in industry, not something we derived): PSI < 0.1 no
meaningful drift, 0.1-0.25 moderate, > 0.25 significant.
"""
import math
from collections import deque

MIN_SAMPLES = 30
PSI_MODERATE = 0.1
PSI_SIGNIFICANT = 0.25


class DriftMonitor:
    def __init__(self, reference: dict, window_size: int = 500):
        self.feature_cols = reference["feature_cols"]
        self.bins = reference["bins"]
        self.reference_proportions = reference["reference_proportions"]
        self.n_reference_samples = reference["n_reference_samples"]
        self.window = deque(maxlen=window_size)

    def record(self, feature_values):
        self.window.append(list(feature_values))

    def _psi_for_feature(self, col: str, idx: int, current_values: list) -> float:
        edges = self.bins.get(col)
        ref_props = self.reference_proportions.get(col)
        if edges is None:
            return 0.0

        counts = [0] * (len(edges) - 1)
        for v in current_values:
            x = v[idx]
            for b in range(len(edges) - 1):
                if edges[b] <= x < edges[b + 1] or (b == len(edges) - 2 and x == edges[-1]):
                    counts[b] += 1
                    break

        n = sum(counts)
        if n == 0:
            return 0.0

        eps = 1e-6
        psi = 0.0
        for actual_count, ref_p in zip(counts, ref_props):
            actual_p = actual_count / n
            psi += (actual_p - ref_p) * math.log((actual_p + eps) / (ref_p + eps))
        return psi

    def report(self) -> dict:
        n = len(self.window)
        if n < MIN_SAMPLES:
            return {
                "status": "insufficient_data",
                "window_samples": n,
                "min_samples_required": MIN_SAMPLES,
                "reference_samples": self.n_reference_samples,
                "features": {},
            }

        current_values = list(self.window)
        features = {}
        worst = "none"
        for idx, col in enumerate(self.feature_cols):
            if col not in self.bins:
                continue
            psi = round(self._psi_for_feature(col, idx, current_values), 4)
            if psi >= PSI_SIGNIFICANT:
                rating = "significant"
            elif psi >= PSI_MODERATE:
                rating = "moderate"
            else:
                rating = "none"
            features[col] = {"psi": psi, "rating": rating}
            if rating == "significant":
                worst = "significant"
            elif rating == "moderate" and worst == "none":
                worst = "moderate"

        return {
            "status": worst,
            "window_samples": n,
            "min_samples_required": MIN_SAMPLES,
            "reference_samples": self.n_reference_samples,
            "features": features,
        }
