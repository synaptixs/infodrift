# Quickstart — Integrating `drift_monitor`

A 15-minute guide for any team adopting `drift_monitor` in an existing
predictive AI project. By the end you will have:

1. The package installed.
2. One **ontology entity** registered against your training data.
3. One production batch monitored end-to-end.
4. Alerts wired into your existing notification channel.

If you want a runnable script first, look at
[`examples/quickstart.py`](examples/quickstart.py) — this guide is
the *integration* version of the same flow, with hooks for your real
data.

---

## 0. Mental model (1 minute)

`drift_monitor` answers three questions per deployed model:

| Question | Layer | Metrics |
|---|---|---|
| Is the evidence changing? | L1 | PSI, KL/JS divergence, log template shift |
| Is the model still reliable? | L2 | Per-class P/R/F1, Brier, ECE |
| Is the system operationally useful? | L3 | TPR/FPR, Time-to-Detect |

Every metric is computed within one **entity scope**:

```
entity_key = "<Component>_<Version>::<Region>::<Interface>"
# e.g.  "FraudDetector_v5::APAC::CardTransactions"
```

Metrics across entities are **never** aggregated — that scoping is what
makes root-cause analysis possible. Pick your entity key shape before
writing code; changing it later means recomputing baselines.

---

## 1. Install

```bash
pip install -e ".[logs]"   # from a clone, or:
# pip install drift_monitor[logs]   # once published
```

Requirements: Python 3.10+. The `[logs]` extra pulls in `drain3` for
template mining; the regex normaliser is the fallback if drain3 is
unavailable.

---

## 2. Decide your entity key shape

Pick three to four dimensions that bound how a failure could be
localised in your org. Common patterns:

| Domain | Example |
|---|---|
| Telecom / Network | `AMF_v2::RegionA::N11` |
| Fraud / Financial | `FraudDetector_v5::APAC::CardTransactions` |
| Healthcare | `DiagnosticModel_v3::ICU::LabFeatures` |
| Industrial IoT | `PredictiveMaintenance_v2::PlantB::PressureSensors` |
| Cybersecurity | `ThreatClassifier_v4::TenantA::EndpointEvents` |

Rule of thumb: **include the model version**. When metrics shift, the
first question is always "did we just deploy?" — embedding the version
in the key answers it without a join.

---

## 3. Register an entity from your training data

You need three things per entity:

- A pandas DataFrame of training features (the **baseline reference window**).
- The lists of which columns are numeric vs categorical.
- Optionally: a list of training-window log lines if you have logs.

```python
from drift_monitor import DriftOrchestrator

orch = DriftOrchestrator(slo_config={
    "psi":         {"warning": 0.10, "critical": 0.25},
    "ttd_seconds": {"warning": 120,  "critical": 300},
    "fpr":         {"warning": 0.08, "critical": 0.15},
    "ece":         {"warning": 0.05, "critical": 0.10},
})

orch.register_entity(
    entity_key="FraudDetector_v5::APAC::CardTransactions",
    train_features_df=train_df,                     # your training DataFrame
    numeric_features=["txn_amount_usd", "device_age_days"],
    categorical_features=["merchant_category", "channel"],
    train_logs=baseline_log_lines,                  # optional list[str]
    model_version="v5",
    baseline_id="train-2026Q1",
)
```

Do this **once per entity**, when the model is trained or a baseline
window is deliberately refreshed (white paper §9.1). The orchestrator
caches the baseline distributions; subsequent `run` calls reuse them.

> **Note**: if your process exits between scheduled runs, you must
> persist orchestrator state — see
> [`docs/deployment.md`](docs/deployment.md#6-state-persistence).

---

## 4. Run on a production batch

Pull together whatever signals you have for the current window. Every
input is optional; the orchestrator skips the layer if its inputs are
missing.

```python
alerts = orch.run(
    entity_key="FraudDetector_v5::APAC::CardTransactions",
    prod_features_df=prod_df,                # required for L1 (features)
    prod_logs=prod_log_lines,                # optional, L1 (logs)
    y_true_clf=y_true,                       # optional, L2 (P/R/F1, recall drift)
    y_pred_clf=y_pred,
    y_true_calib=y_bin,                      # optional, L2 (Brier, ECE)
    y_prob_calib=y_prob,
    y_true_anom=anom_y,                      # optional, L3 (TPR/FPR)
    y_score_anom=anom_score,
    anomaly_threshold=0.5,
    incidents_df=incidents_df,               # optional, L3 (TTD)
    window_id="2026-W17",
    verbose=True,
)
```

Returns a `list[Alert]`. The full structured `RunReport` is retrievable
via `orch.latest_report(entity_key)`.

### What each input expects

| Argument | Shape | Notes |
|---|---|---|
| `prod_features_df` | `pd.DataFrame` with the same columns as `train_features_df` | Order doesn't matter; missing columns raise. |
| `prod_logs` | `list[str]` | Raw log lines. Normalised internally. |
| `y_true_clf` / `y_pred_clf` | `np.ndarray` of class labels | Any hashable label type; classes auto-detected. |
| `y_true_calib` / `y_prob_calib` | `np.ndarray` binary 0/1 and float in `[0,1]` | Used for Brier + ECE. |
| `y_true_anom` / `y_score_anom` | `np.ndarray` binary 0/1 and float score | Threshold via `anomaly_threshold`. |
| `incidents_df` | `pd.DataFrame` with columns `onset_ts`, `alert_ts` | Numeric seconds or datetime-like. |

---

## 5. Wire alerts into your notification channel

Each `Alert` carries severity, metric type, observed value, threshold,
recommendation, and full governance metadata
(timestamp, model version, baseline ID, entity key). Convert and ship:

```python
def post_alerts(alerts):
    for a in alerts:
        if a.severity == "critical":
            page(a.to_dict())          # PagerDuty / Opsgenie / etc.
        else:
            slack_post(a.to_dict())    # warnings to Slack
```

The framework **does not** call out to any external service. That
isolation is intentional — you wire the transport, you keep the
secrets.

---

## 6. Schedule it

For batch use cases (most predictive AI), one run per entity per hour
is a sensible default:

- **VM**: `systemd` timer — see [`docs/deployment.md`](docs/deployment.md#43-systemd-timer-recommended-over-cron).
- **Kubernetes**: a `CronJob` running the same image.
- **Cloud-managed**: scheduled Fargate task / Cloud Run job / Container
  Apps job — see deployment guide.

Tighten the schedule (15 min, 5 min) only after you understand alert
volume at hourly cadence. Faster cadence ≠ better detection if humans
cannot triage.

---

## 7. Scale to multiple entities

Two adoption patterns work, in this order:

1. **Start with the highest-stakes entity.** One model, one region.
   Run it for two weeks. Calibrate SLO thresholds against your real
   incident history.
2. **Replicate horizontally.** Each new entity is just another
   `register_entity` + scheduled `run`. Above ~50 entities, switch
   from one-job-per-entity to a worker-pool pattern (deployment guide
   §5.4).

Per-entity SLO overrides are easy: pass a different `slo_config` to a
second `DriftOrchestrator` instance, or wrap your call site so the
config table is keyed by `entity_key`.

---

## 8. Calibrate your SLOs

The defaults in this guide are starting points, not laws. The white
paper is explicit (§10.1) that PSI thresholds 0.10 / 0.25 are credit-
risk heuristics, not universal statistics. Process to calibrate:

1. **Run for ≥2 weeks** in shadow mode (alerts logged, not paged).
2. **Replay your last quarter's incidents** through the orchestrator
   using historical features + logs. Note what triggered.
3. **Tune per-metric thresholds** so true incidents fire and routine
   environmental change does not. Document the chosen thresholds and
   the incidents they were tuned against.
4. **Re-review monthly** (white paper §9.5).

---

## 9. Reading common alert patterns

The white paper provides a diagnostic decision surface (§3.12). Most
useful patterns to memorise:

| Signal pattern | Likely cause | First action |
|---|---|---|
| PSI spike, recall stable | Feature dist changed but model adapted | Check schema/rollout; do not retrain yet |
| JS spike, PSI stable | Subtle multimodal change | Increase monitoring frequency |
| New log templates, no metric change | Code or parser change upstream | Validate parser; expect KPIs to follow |
| Recall drop, PSI stable | Concept drift or label quality | Investigate labels before retraining |
| All three layers fire | Material, broad drift | Escalate to model governance review |
| TTD growing, accuracy stable | Operational latency | Check feature staleness, alert routing |
| ECE up, Brier flat | Miscalibration on specific bins | Platt / temperature scaling, not retrain |

---

## 10. Common integration gotchas

| Gotcha | Symptom | Fix |
|---|---|---|
| Entity key changes between `register_entity` and `run` | `KeyError: entity ... not registered` | Centralise the key in a constant; never construct it ad-hoc. |
| Numeric feature passed as categorical | PSI suspiciously high / unstable | Check `numeric_features` vs `categorical_features` lists. |
| Production columns are a superset of training | Extra columns are silently ignored, but `KeyError` on missing | Match your training DataFrame exactly; do not add or drop columns mid-flight. |
| Rolling z-score never alerts | History resets every run | Persist orchestrator state between runs (deployment guide §6). |
| `incidents_df` timestamps mixed types | TTD comes out as 0 or NaN | Use either all numeric seconds or all `datetime`-like; do not mix. |
| Categorical column has NaN | `KeyError` during fit | Drop or impute NaN before passing to `register_entity`. |

---

## 11. Where to go next

- **Operationalisation**: [`docs/deployment.md`](docs/deployment.md) for
  VM / cloud deployment, state persistence, CI/CD.
- **Contribution**: [`CONTRIBUTING.md`](CONTRIBUTING.md) — even
  just adding a new monitor follows the same pattern as the existing
  `core/` modules.
- **The framework itself**: [`docs/MonitoringDrift.pdf`](docs/MonitoringDrift.pdf)
  is the authoritative reference. Read sections 3 (math), 9
  (governance), and 10 (limitations) before tuning thresholds in
  production.
