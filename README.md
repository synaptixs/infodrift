# Information Drift — Production ML Monitoring for RCA

A reference implementation of the framework described in the white paper
**"Monitoring Information Drift in Production Predictive AI Systems"**
(Rohilla, v2.0, March 2026).

The package detects and diagnoses the silent decay of deployed predictive AI systems
by combining evidence-distribution monitoring, predictive-reliability tracking, and
operational-effectiveness measurement — all scoped per **ontology entity** so drift
can be localized to the exact deployment unit responsible.

---

## Why this exists

Predictive models do not fail loudly. They drift. Feature distributions shift,
log composition changes after a rollout, calibration erodes, and time-to-detect
slowly grows — all while traditional software alarms remain silent. This project
turns the white-paper framework into runnable code for root-cause analysis (RCA)
of model degradation in production.

## The four dimensions of information drift

1. **Feature distribution shift** — `P(X)` changes (covariate shift).
2. **Concept drift** — `P(Y|X)` changes; the labelling rules themselves shift.
3. **Log / event composition shift** — the mix of observable events changes.
4. **Calibration erosion** — confidence scores no longer match observed frequencies.

## Three monitoring layers

| Layer | Question | Core metrics |
|------|---------|--------------|
| **L1 — Evidence Distribution Health** | Is the evidence changing? | PSI, KL/JS divergence (with KDE), log template frequency shift |
| **L2 — Predictive Reliability** | Is the model still reliable? | Precision / Recall / F1 per class, Brier score, ECE, recall-drift alerting |
| **L3 — Operational Effectiveness** | Is the system operationally useful? | TPR / FPR, Time-to-Detect (TTD) at p50 / p90 / p99 |
| **Cross-cutting** | Where is it happening? | Segment-aware metrics keyed by ontology entity |

## Ontology entity scoping

Every metric is computed within the scope of a structured entity key:

```
entity_key = "<Component>_<Version>::<Region>::<Interface>"

# Examples
"AMF_v2::RegionA::N11"
"FraudDetector_v5::APAC::CardTransactions"
"PredictiveMaintenance_v2::PlantB::PressureSensors"
```

This pattern answers the operationally critical RCA question:
*which component, in which region, at which version, is causing the drift?*

---

## Project layout

```
drift_monitor/
  core/
    psi.py            # Population Stability Index monitor
    divergence.py     # KL / JS divergence monitor (with optional KDE)
    log_templates.py  # Log template frequency-shift monitor
    performance.py    # Precision/Recall/F1, Brier, ECE, TPR/FPR, TTD
  monitoring/
    orchestrator.py   # Central DriftOrchestrator (per-entity registration + run)
    alerting.py       # SLO-based AlertEngine
  reporting/
    report.py         # HealthReporter, canary comparisons, JSON export
tests/                # Automated test suite (target: 55 checks per white paper)
```

## Installation

### From PyPI (recommended)

```bash
pip install drift_monitor                 # core
pip install "drift_monitor[logs]"         # + drain3 log template miner
pip install "drift_monitor[dev]"          # + pytest, build, twine (contributors)
```

Requirements: Python 3.10+. Core dependencies (`numpy`, `pandas`, `scipy`,
`scikit-learn`) are pulled in automatically.

### From source

```bash
git clone https://github.com/nrohilla-fibonacci/infodrift.git
cd infodrift
pip install -e ".[logs,dev]"
pytest                                    # 35 checks should pass
```

### Verify the install

```bash
python -c "import drift_monitor as dm; print(dm.__version__)"
```

For build/release tooling and the PyPI publish flow, see
[`docs/packaging.md`](docs/packaging.md). For an end-to-end runnable
example, see [`examples/quickstart.py`](examples/quickstart.py).

## Quick start

```python
from drift_monitor import DriftOrchestrator, HealthReporter

orch = DriftOrchestrator(slo_config={
    "psi":          {"warning": 0.10, "critical": 0.20},
    "ttd_seconds":  {"warning": 120,  "critical": 300},
    "fpr":          {"warning": 0.08, "critical": 0.15},
})

orch.register_entity(
    entity_key="AMF_v2::RegionA::N11",
    train_features_df=train_df,
    numeric_features=["n11_latency_ms"],
    categorical_features=["sbi_code"],
    train_logs=baseline_logs,
)

alerts = orch.run(
    entity_key="AMF_v2::RegionA::N11",
    prod_features_df=prod_df,
    prod_logs=prod_logs,
    y_true_clf=y_true, y_pred_clf=y_pred,
    y_true_calib=y_bin, y_prob_calib=y_prob,
    incidents_df=inc_df,
    window_id="2025-W30",
)

HealthReporter(orch).print_summary()
```

## Metric selection cheat sheet

| Metric | Best for | Caution |
|--------|---------|---------|
| **PSI** | Fast screening of binned drift | Binning-sensitive; 0.10/0.25 are credit-risk heuristics, not laws |
| **KL divergence** | Directional shape diagnosis | Asymmetric, unbounded — use diagnostically, not for alerting |
| **JS divergence** | Dashboards & alerting | Symmetric, bounded `[0, ln 2]` — preferred alert metric |
| **Rolling z-score** | Per-feature adaptive threshold | Needs ≥ 5 (ideally 15–30) stable batches to warm up |
| **Template frequency shift** | Logs / event mix | A shift can mean parser change, not behavior change |
| **Brier score** | Probability calibration | Lower is better; `[0, 1]`; lags with delayed labels |
| **ECE** | Confidence reliability | Bin-count sensitive; keep bins consistent across windows |
| **TPR / FPR** | Anomaly detector quality | Acceptable FPR is domain-specific |
| **Time-to-Detect** | Operational usefulness | Validity depends on trustworthy onset labels |

## The composited decision surface

| Observed pattern | Diagnostic interpretation |
|------------------|---------------------------|
| PSI spike; recall stable | Distribution shifted but model adapted — monitor, check schema |
| JS spike; PSI stable | Subtle multimodal / categorical shift — early warning |
| New log templates; metrics stable | Code or parser change upstream — may precede degradation |
| Recall drop; PSI stable | Concept drift or label-quality issue |
| All three layers fire | Material drift — escalate to model-governance review |
| TTD increasing; metrics stable | Operational latency growing — check feature staleness, routing |
| ECE up; Brier stable | Miscalibration — try Platt / temperature scaling first |

## Limitations

- Thresholds are local heuristics, not universal laws — calibrate against incident history.
- Labels can lag; treat L1 signals as leading indicators when L2 is silent.
- Template drift can reflect parser changes, not real behavior.
- Drift ≠ model failure; investigate before retraining.
- Rolling baselines need warm-up before z-scores are meaningful.

## Roadmap

- Multivariate drift via MMD (Gretton et al., 2012)
- Online / streaming detection via ADWIN (Bifet & Gavaldà, 2007)
- Graph-structured ontology for causal alert propagation
- Federated monitoring for privacy-sensitive deployments
- Automated remediation triggers (Platt scaling, retraining)
- Conformal-prediction-based label-free uncertainty drift signal

## Reference

Rohilla, N. (2026). *Monitoring Information Drift in Production Predictive AI Systems —
A Unified Framework for Evidence Drift, Log Composition Shift, Calibration Monitoring,
Operational Timeliness, and Governance.* Engineering White Paper, v2.0.

Foundational citations: Sculley et al. 2015; Breck et al. 2017; Gama et al. 2014;
Guo et al. 2017; Sambasivan et al. 2021; Lin 1991; Kullback & Leibler 1951;
Anderson 2007; He et al. 2017; Brier 1950.
