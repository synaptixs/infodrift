import json

import numpy as np
import pandas as pd
import pytest

from drift_monitor import DriftOrchestrator, HealthReporter


@pytest.fixture
def slo():
    return {
        "psi": {"warning": 0.10, "critical": 0.25},
        "ttd_seconds": {"warning": 120, "critical": 300},
        "fpr": {"warning": 0.08, "critical": 0.15},
        "ece": {"warning": 0.05, "critical": 0.10},
    }


def _make_train(rng):
    return pd.DataFrame(
        {
            "n11_latency_ms": rng.normal(25, 5, 5000),
            "sbi_code": rng.choice(["200", "400", "500"], 5000, p=[0.9, 0.07, 0.03]),
        }
    )


def _make_drifted(rng):
    return pd.DataFrame(
        {
            "n11_latency_ms": np.concatenate(
                [rng.normal(25, 5, 3500), rng.normal(80, 10, 1500)]
            ),
            "sbi_code": rng.choice(["200", "400", "500"], 5000, p=[0.7, 0.07, 0.23]),
        }
    )


def test_orchestrator_end_to_end_drifted_features(slo):
    rng = np.random.default_rng(0)
    train = _make_train(rng)
    prod = _make_drifted(rng)
    orch = DriftOrchestrator(slo_config=slo, divergence_window=5, log_window=5)
    orch.register_entity(
        "AMF_v2::RegionA::N11",
        train,
        numeric_features=["n11_latency_ms"],
        categorical_features=["sbi_code"],
        train_logs=["INFO ok"] * 200 + ["WARN slow"] * 50,
        model_version="v2",
    )
    alerts = orch.run(
        "AMF_v2::RegionA::N11",
        prod_features_df=prod,
        prod_logs=["INFO ok"] * 100 + ["CRIT new failure mode"] * 50,
        window_id="2025-W30",
    )
    metrics = {a.metric_type for a in alerts}
    assert "psi" in metrics
    assert "log_templates_new" in metrics
    crits = [a for a in alerts if a.severity == "critical"]
    assert any(a.metric_type == "psi" for a in crits)


def test_orchestrator_full_signal_run(slo):
    rng = np.random.default_rng(1)
    train = _make_train(rng)
    prod = _make_drifted(rng)
    orch = DriftOrchestrator(slo_config=slo, divergence_window=5, anomaly_threshold=0.5)
    orch.register_entity(
        "ent::A::I",
        train,
        numeric_features=["n11_latency_ms"],
        categorical_features=["sbi_code"],
        model_version="v1",
    )
    n = 500
    y_true = rng.integers(0, 3, n)
    y_pred = y_true.copy()
    flip = rng.choice(n, size=50, replace=False)
    y_pred[flip] = (y_pred[flip] + 1) % 3
    y_bin = rng.integers(0, 2, n)
    y_prob = np.clip(y_bin + rng.normal(0, 0.2, n), 0, 1)
    anom_y = rng.integers(0, 2, n)
    anom_s = np.clip(anom_y * 0.7 + rng.normal(0, 0.3, n), 0, 1)
    inc = pd.DataFrame({"onset_ts": np.zeros(20), "alert_ts": rng.uniform(50, 600, 20)})
    alerts = orch.run(
        "ent::A::I",
        prod_features_df=prod,
        y_true_clf=y_true,
        y_pred_clf=y_pred,
        y_true_calib=y_bin,
        y_prob_calib=y_prob,
        y_true_anom=anom_y,
        y_score_anom=anom_s,
        incidents_df=inc,
        window_id="W1",
    )
    report = orch.latest_report("ent::A::I")
    assert report.brier is not None
    assert report.ece is not None
    assert report.tpr_fpr is not None
    assert report.ttd is not None and report.ttd["n"] == 20
    assert isinstance(alerts, list)


def test_health_reporter_json_export(slo):
    rng = np.random.default_rng(2)
    train = _make_train(rng)
    orch = DriftOrchestrator(slo_config=slo)
    orch.register_entity(
        "ent::A::I",
        train,
        numeric_features=["n11_latency_ms"],
        categorical_features=["sbi_code"],
        model_version="v1",
    )
    orch.run("ent::A::I", prod_features_df=train, window_id="W1")
    reporter = HealthReporter(orch)
    js = reporter.full_report(as_json=True)
    payload = json.loads(js)
    assert "ent::A::I" in payload["entities"]


def test_canary_comparison(slo):
    rng = np.random.default_rng(3)
    train = _make_train(rng)
    prod = _make_drifted(rng)
    orch = DriftOrchestrator(slo_config=slo)
    for ent, model in [("AMF_v2::RegionA::N11", "v2"), ("AMF_v3::RegionA::N11", "v3")]:
        orch.register_entity(
            ent,
            train,
            numeric_features=["n11_latency_ms"],
            categorical_features=["sbi_code"],
            model_version=model,
        )
        orch.run(
            ent,
            prod_features_df=prod,
            y_true_calib=np.array([0, 1, 0, 1]),
            y_prob_calib=np.array([0.1, 0.9, 0.2, 0.8]),
            window_id="W1",
        )
    reporter = HealthReporter(orch)
    df = reporter.canary_comparison("AMF_v2::RegionA::N11", "AMF_v3::RegionA::N11")
    assert list(df.index) == ["baseline", "canary"]
    assert "brier_score" in df.columns
    assert "max_psi" in df.columns


def test_unregistered_entity_raises(slo):
    orch = DriftOrchestrator(slo_config=slo)
    with pytest.raises(KeyError):
        orch.run("missing", prod_features_df=pd.DataFrame({"x": [1]}))
