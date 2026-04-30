import numpy as np
import pandas as pd
import pytest

from drift_monitor import PSIMonitor


@pytest.fixture
def rng():
    return np.random.default_rng(0)


def test_psi_zero_on_identical_distribution(rng):
    df = pd.DataFrame({"x": rng.normal(0, 1, 5000)})
    m = PSIMonitor(n_bins=10)
    m.fit(df, ["x"], "ent::A::I")
    [r] = m.compute(df, ["x"], "ent::A::I")
    assert r.psi_score < 0.01
    assert r.drift_level == "stable"


def test_psi_detects_bimodal_shift(rng):
    train = pd.DataFrame({"x": rng.normal(25, 5, 5000)})
    prod = pd.DataFrame(
        {
            "x": np.concatenate(
                [rng.normal(25, 5, 3500), rng.normal(80, 10, 1500)]
            )
        }
    )
    m = PSIMonitor(n_bins=10)
    m.fit(train, ["x"], "AMF_v2::RegionA::N11")
    [r] = m.compute(prod, ["x"], "AMF_v2::RegionA::N11")
    assert r.psi_score > 0.25
    assert r.drift_level == "material"


def test_psi_categorical_with_unseen_category(rng):
    train = pd.DataFrame({"sbi": rng.choice(["200", "400", "500"], 5000, p=[0.9, 0.07, 0.03])})
    prod = pd.DataFrame({"sbi": rng.choice(["200", "400", "500", "503"], 5000, p=[0.7, 0.07, 0.2, 0.03])})
    m = PSIMonitor(n_bins=10)
    m.fit(train, ["sbi"], "ent")
    [r] = m.compute(prod, ["sbi"], "ent")
    assert r.psi_score > 0.10
    # OTHER bin captures unseen "503"
    assert "__OTHER__" in r.bin_details["bin"].tolist()


def test_psi_unregistered_entity_raises():
    m = PSIMonitor()
    with pytest.raises(KeyError):
        m.compute(pd.DataFrame({"x": [1, 2, 3]}), ["x"], "missing")


def test_psi_handles_constant_feature():
    train = pd.DataFrame({"x": np.zeros(100)})
    m = PSIMonitor(n_bins=10)
    m.fit(train, ["x"], "ent")
    [r] = m.compute(train, ["x"], "ent")
    assert np.isfinite(r.psi_score)
