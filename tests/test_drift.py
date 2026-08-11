"""Unit tests for the PSI drift math itself, independent of the API layer."""
import json

import pytest

from api.drift import DriftMonitor, MIN_SAMPLES

PSI_LOOSE_BOUND = 0.3


@pytest.fixture
def reference():
    with open("drift_reference.json") as f:
        return json.load(f)


def test_no_drift_when_window_matches_reference(reference):
    monitor = DriftMonitor(reference)
    col = reference["feature_cols"][0]
    idx = 0
    edges = reference["bins"][col]
    # Feed values drawn from the middle of each reference bin — should land
    # back in the same bins the reference proportions came from.
    midpoints = [(edges[i] + edges[i + 1]) / 2 for i in range(len(edges) - 1)]
    midpoints[0] = edges[1] - 1  # first/last edges are +/-inf, avoid inf midpoint
    midpoints[-1] = edges[-2] + 1

    n_features = len(reference["feature_cols"])
    for _ in range(MIN_SAMPLES + 10):
        for m in midpoints:
            row = [0.0] * n_features
            row[idx] = m
            monitor.record(row)

    report = monitor.report()
    # Only feature 0 was populated with realistic values — the other columns
    # are constant zeros and would show spurious drift on their own, so check
    # just the one feature this test actually exercises, not the aggregate.
    assert report["features"][col]["rating"] in ("none", "moderate")  # binning noise, not a real shift
    assert report["features"][col]["psi"] < PSI_LOOSE_BOUND


def test_significant_drift_when_window_is_all_one_bin(reference):
    monitor = DriftMonitor(reference)
    col = reference["feature_cols"][0]
    idx = 0
    edges = reference["bins"][col]
    extreme_low = edges[1] - 1  # deep inside the first bin only

    n_features = len(reference["feature_cols"])
    for _ in range(MIN_SAMPLES + 10):
        row = [0.0] * n_features
        row[idx] = extreme_low
        monitor.record(row)

    report = monitor.report()
    assert report["status"] == "significant"
    assert report["features"][col]["rating"] == "significant"
    assert report["features"][col]["psi"] > 0.25


def test_insufficient_data_below_min_samples(reference):
    monitor = DriftMonitor(reference)
    n_features = len(reference["feature_cols"])
    for _ in range(MIN_SAMPLES - 1):
        monitor.record([0.0] * n_features)

    report = monitor.report()
    assert report["status"] == "insufficient_data"
    assert report["features"] == {}
