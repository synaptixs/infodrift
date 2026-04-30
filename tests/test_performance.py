import numpy as np
import pandas as pd
import pytest

from drift_monitor import brier_score, ece, per_class_prf, tpr_fpr, ttd_percentiles
from drift_monitor.core.performance import RecallDriftTracker


def test_brier_perfect_calibration():
    y = np.array([0, 1, 0, 1])
    p = np.array([0.0, 1.0, 0.0, 1.0])
    assert brier_score(y, p) == 0.0


def test_brier_worst_case():
    y = np.array([1, 1])
    p = np.array([0.0, 0.0])
    assert brier_score(y, p) == 1.0


def test_ece_well_calibrated():
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, 5000)
    y = (rng.uniform(0, 1, 5000) < p).astype(int)
    res = ece(y, p, n_bins=10)
    assert res.ece < 0.05


def test_ece_miscalibrated_overconfident():
    # Predicts 0.9 but events occur 0.5 of the time -> gap ~0.4 in that bin.
    n = 1000
    p = np.full(n, 0.9)
    y = np.zeros(n, dtype=int)
    y[: n // 2] = 1
    res = ece(y, p, n_bins=10)
    assert res.ece > 0.03  # averaged across all bins


def test_per_class_prf_basic():
    y_true = np.array(["a", "a", "b", "b", "c"])
    y_pred = np.array(["a", "b", "b", "b", "a"])
    out = {r.cls: r for r in per_class_prf(y_true, y_pred)}
    assert out["a"].precision == pytest.approx(0.5)
    assert out["a"].recall == pytest.approx(0.5)
    assert out["b"].recall == pytest.approx(1.0)
    assert out["c"].recall == 0.0


def test_tpr_fpr_basic():
    y = np.array([1, 1, 0, 0])
    s = np.array([0.9, 0.4, 0.3, 0.8])
    r = tpr_fpr(y, s, tau=0.5)
    assert r.tpr == pytest.approx(0.5)  # one of two positives caught
    assert r.fpr == pytest.approx(0.5)  # one of two negatives flagged


def test_ttd_percentiles_seconds():
    df = pd.DataFrame(
        {
            "onset_ts": [0, 0, 0, 0, 0],
            "alert_ts": [10, 50, 100, 200, 600],
        }
    )
    r = ttd_percentiles(df)
    assert r["p50"] == pytest.approx(100.0)
    assert r["p90"] == pytest.approx(440.0)
    assert r["n"] == 5


def test_ttd_percentiles_floors_negative():
    df = pd.DataFrame({"onset_ts": [10], "alert_ts": [5]})
    r = ttd_percentiles(df)
    assert r["p50"] == 0.0


def test_recall_drift_alerts_on_drop():
    tracker = RecallDriftTracker(drop_threshold=0.20, lookback=7)
    high = per_class_prf(np.array([1, 1, 1, 0, 0]), np.array([1, 1, 1, 0, 0]), classes=[0, 1])
    out1 = tracker.update("ent", high)
    assert all(not r.alert for r in out1)
    low = per_class_prf(np.array([1, 1, 1, 1, 0]), np.array([0, 0, 0, 1, 0]), classes=[0, 1])
    out2 = tracker.update("ent", low)
    by_cls = {r.cls: r for r in out2}
    assert by_cls["1"].alert is True
