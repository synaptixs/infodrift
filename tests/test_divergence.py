import numpy as np
import pandas as pd
import pytest

from drift_monitor import DivergenceMonitor


def test_js_zero_on_identical_distribution():
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"x": rng.normal(0, 1, 4000)})
    m = DivergenceMonitor(rolling_window=10)
    m.fit(df, ["x"], "ent")
    [r] = m.compute(df, ["x"], "ent")
    assert r.js_score < 0.01
    assert r.kl_score < 0.05


def test_js_detects_bimodal_shift_kde():
    rng = np.random.default_rng(2)
    train = pd.DataFrame({"x": rng.normal(25, 5, 5000)})
    drifted = pd.DataFrame(
        {"x": np.concatenate([rng.normal(25, 5, 3500), rng.normal(80, 10, 1500)])}
    )
    m = DivergenceMonitor(rolling_window=10, alert_sigma=3.0)
    m.fit(train, ["x"], "ent")
    # Warm up rolling history with stable batches.
    for _ in range(8):
        stable = pd.DataFrame({"x": rng.normal(25, 5, 500)})
        m.compute(stable, ["x"], "ent")
    [r] = m.compute(drifted, ["x"], "ent")
    assert r.js_score > 0.05
    assert r.alert is True
    assert r.z_score > 3.0


def test_js_warmup_does_not_alert():
    rng = np.random.default_rng(3)
    df = pd.DataFrame({"x": rng.normal(0, 1, 1000)})
    m = DivergenceMonitor(rolling_window=10)
    m.fit(df, ["x"], "ent")
    [r] = m.compute(df, ["x"], "ent")
    assert r.alert is False
    assert "warming up" in r.reason


def test_js_categorical():
    train = pd.DataFrame({"c": ["a"] * 800 + ["b"] * 200})
    drifted = pd.DataFrame({"c": ["a"] * 200 + ["b"] * 800})
    m = DivergenceMonitor(rolling_window=10)
    m.fit(train, ["c"], "ent")
    [r] = m.compute(drifted, ["c"], "ent")
    assert r.js_score > 0.05


def test_invalid_bandwidth_raises():
    with pytest.raises(ValueError):
        DivergenceMonitor(kde_bandwidth="silverman")


def test_js_bounded():
    # JS is bounded above by ln(2). Check we never exceed it materially.
    rng = np.random.default_rng(4)
    train = pd.DataFrame({"x": rng.normal(0, 1, 2000)})
    drifted = pd.DataFrame({"x": rng.normal(50, 1, 2000)})
    m = DivergenceMonitor(rolling_window=10)
    m.fit(train, ["x"], "ent")
    [r] = m.compute(drifted, ["x"], "ent")
    assert r.js_score <= np.log(2) + 1e-6
