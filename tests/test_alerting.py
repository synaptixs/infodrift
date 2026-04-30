from drift_monitor import AlertEngine


def test_alert_critical_breach():
    eng = AlertEngine({"psi": {"warning": 0.10, "critical": 0.25}})
    a = eng.evaluate_threshold("psi", 0.30, "ent", feature="x")
    assert a is not None
    assert a.severity == "critical"
    assert a.threshold == 0.25
    assert a.feature == "x"
    assert "Investigate" in a.recommendation


def test_alert_warning_breach():
    eng = AlertEngine({"psi": {"warning": 0.10, "critical": 0.25}})
    a = eng.evaluate_threshold("psi", 0.15, "ent")
    assert a is not None
    assert a.severity == "warning"


def test_alert_no_breach():
    eng = AlertEngine({"psi": {"warning": 0.10, "critical": 0.25}})
    assert eng.evaluate_threshold("psi", 0.05, "ent") is None


def test_alert_higher_is_better_metric():
    # tpr breach when LOWER than threshold
    eng = AlertEngine({"tpr": {"warning": 0.9, "critical": 0.7}})
    assert eng.evaluate_threshold("tpr", 0.6, "ent", higher_is_worse=False).severity == "critical"
    assert eng.evaluate_threshold("tpr", 0.85, "ent", higher_is_worse=False).severity == "warning"
    assert eng.evaluate_threshold("tpr", 0.95, "ent", higher_is_worse=False) is None


def test_alert_no_config_metric():
    eng = AlertEngine({})
    assert eng.evaluate_threshold("ttd_seconds", 999, "ent") is None


def test_boolean_alert_carries_governance():
    eng = AlertEngine({})
    a = eng.boolean_alert(
        "js_zscore",
        "ent",
        observed=4.5,
        feature="x",
        message="z=4.5",
        model_version="v2",
        baseline_id="b1",
    )
    assert a.severity == "warning"
    assert a.model_version == "v2"
    assert a.baseline_id == "b1"
    d = a.to_dict()
    assert d["metric_type"] == "js_zscore"
    assert d["timestamp"]
