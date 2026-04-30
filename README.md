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

## Metric selection — what each one measures, when to use it, and the formula

The framework is layered intentionally: no single metric answers all three
monitoring questions. Each entry below covers **what** the metric measures,
**when** you should reach for it, the **formula** as it is implemented in
this package, and the practical **caution** the white paper attaches to it.
Section references map back to [`docs/MonitoringDrift.pdf`](docs/MonitoringDrift.pdf).

### PSI — Population Stability Index (§3.1)

**What it measures.** How much a single feature's binned distribution has
shifted between training and production. PSI is mathematically equivalent
to the symmetric KL divergence over a shared histogram.

**When to use it.** Fast first-pass screening of every numeric and
categorical feature. Cheap to compute and easy to interpret.

**Formula.** For B bins with quantile edges fixed at training time,

```
PSI = Σ_{i=1..B} (p_curr_i - p_base_i) · ln(p_curr_i / p_base_i)
```

where `p_base_i` is the fraction of training rows in bin `i` and
`p_curr_i` the fraction of production rows. Both are smoothed with
Laplace ε = 1e-6 to avoid `log(0)`. Standard interpretation:
`< 0.10` stable · `0.10–0.25` moderate · `≥ 0.25` material.

**Caution.** Binning-sensitive: a multimodal shift inside an existing
bin will not move PSI. The `0.10 / 0.25` thresholds originate in credit-
risk modelling (Anderson, 2007) — treat them as starting heuristics and
recalibrate against your incident history.

### KL divergence — Kullback–Leibler (§3.2)

**What it measures.** The expected information gain when revising
beliefs from baseline `Q` to current `P`. Equivalently, the extra bits
needed to encode samples from `P` using a code optimised for `Q`.

**When to use it.** Diagnostic deep-dive after a JS or PSI alert —
KL's asymmetry tells you which direction the shift went.

**Formula.** Discrete:

```
KL(P || Q) = Σ_x P(x) · ln(P(x) / Q(x))
```

Continuous (via KDE):

```
KL(P || Q) = ∫ p(x) · ln(p(x) / q(x)) dx
```

**Caution.** Asymmetric (`KL(P||Q) ≠ KL(Q||P)`) and unbounded — diverges
to ∞ on non-overlapping support. Use diagnostically, **not** as an
alerting metric.

### JS divergence — Jensen–Shannon (§3.3)

**What it measures.** A symmetric, bounded version of KL built from the
mixture distribution `M = (P + Q) / 2`. The square root `√JS` is a
proper metric (Jensen–Shannon distance).

**When to use it.** Default alerting metric for feature drift and log
template shift. Its bounded range makes rolling thresholds stable.

**Formula.**

```
M = (P + Q) / 2
JS(P || Q) = ½ · KL(P || M)  +  ½ · KL(Q || M)
JS ∈ [0, ln 2]   (nats)   or   [0, 1]   (bits)
```

**Caution.** Still requires baseline windows and local validation.
Faster computation than KL, but does not replace PSI for cheap
screening (Lin, 1991).

### KDE — Kernel Density Estimation (§3.5)

**What it measures.** A smoothed estimate of the continuous density
underlying the samples — needed before computing KL/JS on a continuous
feature, since raw histograms miss intra-bin shape changes.

**When to use it.** Whenever a feature is continuous and you care about
detecting multimodal shifts that PSI misses.

**Formula.** Gaussian kernel with Scott's bandwidth (Scott, 1992):

```
ĥ(x) = (1 / n·h) · Σ_{i=1..n} K[ (x - x_i) / h ]
K(u) = (1 / √(2π)) · exp(-u² / 2)
h    = n^(-1/5) · σ̂          (Scott's rule, AMISE-optimal)
```

Densities are evaluated at G = 200 fixed grid points spanning
`[min_train, max_train]` (frozen at fit time), then renormalised to a
discrete probability vector for JS. This package additionally adds an
out-of-grid bucket so production samples beyond the training range are
counted as drift mass.

**Caution.** Bandwidth choice trades bias against variance; Scott's
rule is a sensible default but not optimal for heavy-tailed data.

### Rolling z-score alert threshold (§3.6)

**What it measures.** How unusual today's JS is, expressed in standard
deviations of recent JS history — a self-calibrating alarm threshold.

**When to use it.** Always, in front of JS-based alerting. Fixed
thresholds on JS over-alert on naturally volatile features and miss
shifts on stable ones.

**Formula.**

```
z_t = (JS_t - µ_{t-1}) / σ_{t-1}
µ_{t-1}, σ_{t-1} = rolling mean and std over the last W batches
Alert when z_t > k     (defaults: W = 30, k = 3)
```

A 3-sigma threshold corresponds to ≈ 0.27 % FPR under normality, in
line with industrial Statistical Process Control practice.

**Caution.** Needs ≥ 5 stable batches to warm up; 15–30 is more
reliable. During warm-up the framework suppresses z-score alerts —
fall back to absolute JS thresholds for a brand-new entity.

### Log template frequency shift (§3.7)

**What it measures.** Whether the **mix** of system events has changed,
even when no individual KPI has moved yet. Often precedes metric
degradation by hours after a software rollout.

**When to use it.** Anywhere logs are richer than features — telecom,
SRE, security, industrial. Covers parser-level and behaviour-level
change.

**Formula.** After normalising raw lines to templates `t_1..t_V`:

```
p_i      = count(t_i) / Σ_j count(t_j)            (per-template frequency)
p_new    = 1 - Σ_{i ∈ T_base} p_curr_i            (mass on unseen templates)
JS_log   = JS(p_curr || p_base)                    (rolling z-score alerted)
PSI_log  = Σ_i (p_curr_i - p_base_i) · ln(p_curr_i / p_base_i)
```

The appearance of new templates (`p_new > ε`) is treated as an
**unconditional** alert regardless of the aggregate JS value.

**Caution.** A template shift can mean a parser change, not a behaviour
change — always validate the parser before attributing template drift
to model-relevant evidence.

### Brier score — calibration (§3.8)

**What it measures.** The mean squared error between predicted
probabilities and observed binary outcomes. A *proper scoring rule*: it
is minimised in expectation only by reporting the true probability.

**When to use it.** Whenever a model outputs probabilities and you have
labels. Catches both miscalibration and irreducible uncertainty.

**Formula.**

```
BS = (1/n) · Σ_{i=1..n} (ŝ_i - y_i)²       BS ∈ [0, 1]   (lower is better)
```

**Caution.** Requires ground truth. Lags when labels arrive with delay
— pair with leading L1 indicators in delayed-label settings.

### Expected Calibration Error — ECE (§3.8)

**What it measures.** The average gap between predicted confidence and
observed accuracy across probability bins — a direct test of whether a
"0.8" really happens 80 % of the time.

**When to use it.** Any system that routes work, suppresses alerts, or
automates action based on probability thresholds.

**Formula.**

```
ECE = (1/B) · Σ_{k=1..B} |frac_pos_k - mean_pred_k|
frac_pos_k  = fraction of positives in bin k    (observed frequency)
mean_pred_k = mean predicted probability in bin k
```

The reliability curve `frac_pos vs mean_pred` is the visual companion;
ideal calibration sits on the 45° diagonal.

**Caution.** Bin-count sensitive — keep `B` consistent across
evaluation windows or trends become unreadable.

### Per-class Precision / Recall / F1 (§3.9)

**What it measures.** Class-localised classification quality. A global
accuracy drop is hard to act on; a recall drop on one class in one
region directly points to the affected component and failure mode.

**When to use it.** Any multiclass classifier with reliable labels —
this is the primary signal for **concept drift**.

**Formula.** Treating each class `k` as one-vs-rest:

```
Precision_k = TP_k / (TP_k + FP_k)
Recall_k    = TP_k / (TP_k + FN_k)
F1_k        = 2 · Precision_k · Recall_k / (Precision_k + Recall_k)
```

Recall-drift alert:

```
Alert if  Recall_k(t - L)  -  Recall_k(t)  >  δ_recall
Defaults: δ_recall = 0.20,  L = 7 batches
```

**Caution.** Labels can lag. When they do, escalate based on L1
signals; do not let silent L2 dashboards imply model health.

### TPR / FPR — anomaly detector quality (§3.10)

**What it measures.** At decision threshold τ, the fraction of true
anomalies caught (TPR) and the fraction of normals falsely flagged
(FPR). Directly tied to alert burden vs missed-incident risk.

**When to use it.** Anomaly / alerting systems where threshold tuning
controls operator fatigue.

**Formula.**

```
TPR(τ) = TP / (TP + FN)        (sensitivity)
FPR(τ) = FP / (FP + TN)        (1 − specificity)
AUC    = P(ŝ_+ > ŝ_-)         (ranking quality, threshold-free)
```

Choose τ to satisfy `FPR ≤ α_SLO` while maximising TPR.

**Caution.** Acceptable FPR is domain-specific. Alert when FPR exceeds
your SLO budget — high FPR causes alert fatigue and silently degrades
operator trust.

### Time-to-Detect — TTD (§3.11)

**What it measures.** How long after an incident's true onset the
detector actually fires, summarised at p50 / p90 / p99. Operational
usefulness, not statistical quality.

**When to use it.** Always when you have onset labels — TTD growth is
the main signal that is invisible to accuracy metrics.

**Formula.**

```
TTD_i  = max(alert_time_i − onset_time_i, 0)        (per incident, seconds)
TTD_50 = median(TTD_i)
TTD_90 = 90th percentile
TTD_99 = 99th percentile
Alert if TTD_50 > τ_SLO       (default τ_SLO = 300 s)
```

**Caution.** Validity depends on trustworthy onset definitions. A
detector that fires after the incident window has effectively closed
can still score well retrospectively yet fail operationally.

### Quick reference table

| Metric | Question | Range | Higher = | Direct alert? |
|---|---|---|---|---|
| PSI | Has the binned distribution shifted? | `[0, ∞)` | more drift | yes (thresholded) |
| KL | How much info changed, directional? | `[0, ∞)` | more drift | no (diagnostic) |
| JS | How much info changed, symmetric? | `[0, ln 2]` | more drift | yes (rolling z-score) |
| Template freq shift | Has the event mix changed? | JS scale | more drift | yes (z-score + new-template) |
| Brier | How calibrated are probabilities? | `[0, 1]` | worse | yes (thresholded) |
| ECE | Do confidences match frequencies? | `[0, 1]` | worse | yes (thresholded) |
| Per-class recall | Are we catching this class? | `[0, 1]` | better | yes (drop-vs-lookback) |
| TPR / FPR | Detector sensitivity vs noise | `[0, 1]` | TPR ↑ better, FPR ↓ better | yes (FPR thresholded) |
| TTD percentiles | How fast do alerts arrive? | `[0, ∞)` seconds | worse | yes (p50 thresholded) |

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
