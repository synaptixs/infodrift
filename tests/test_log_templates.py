from drift_monitor import LogTemplateMonitor
from drift_monitor.core.log_templates import normalize_log


def test_normalize_replaces_variables():
    line = "2024-01-01T00:01:00Z INFO N11 SUPI 001-01-123456789 latency=22ms ip=192.168.1.10 0xdeadbeef"
    norm = normalize_log(line)
    assert "<TS>" in norm
    assert "<NUM>" in norm
    assert "<IP>" in norm
    assert "<HEX>" in norm
    # SUPI is normalized via numeric replacement -> <NUM>-<NUM>-<NUM>
    assert "001-01-123456789" not in norm


def _make_logs(n_inc, n_warn, n_err):
    return (
        ["2024-01-01T00:01:00Z INFO AMF N11 session setup latency=22ms"] * n_inc
        + ["2024-01-01T00:01:01Z WARN AMF N11 timeout retrying"] * n_warn
        + ["2024-01-01T00:01:02Z ERROR AMF N11 reject cause=roaming_not_allowed"] * n_err
    )


def test_log_monitor_detects_new_template_unconditionally():
    base = _make_logs(500, 100, 50)
    prod = _make_logs(300, 50, 25) + [
        "2024-01-15T00:01:00Z CRIT UPF N3 tunnel DROPPED rate_limit threshold=100Mbps"
    ] * 200
    m = LogTemplateMonitor(use_drain=False, rolling_window=5)
    m.fit(base, "ent")
    res = m.compute(prod, "ent")
    assert res.alert is True
    assert res.new_templates  # at least one
    assert res.js_score > 0


def test_log_monitor_stable_when_distribution_unchanged():
    logs = _make_logs(500, 100, 50)
    m = LogTemplateMonitor(use_drain=False, rolling_window=5)
    m.fit(logs, "ent")
    res = m.compute(logs, "ent")
    assert res.js_score < 0.01
    assert not res.new_templates
    # During warmup, alert should be False (no z-score, no new templates).
    assert res.alert is False


def test_log_monitor_top_shifted_ordering():
    base = _make_logs(500, 100, 50)
    prod = _make_logs(50, 500, 100)
    m = LogTemplateMonitor(use_drain=False, rolling_window=5)
    m.fit(base, "ent")
    res = m.compute(prod, "ent")
    assert res.top_shifted.iloc[0]["abs_delta"] >= res.top_shifted.iloc[1]["abs_delta"]
