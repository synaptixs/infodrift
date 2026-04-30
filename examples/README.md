# Examples

Self-contained runtimes that exercise the full `drift_monitor` framework on
synthetic data. They run offline and are safe to commit.

## quickstart.py

End-to-end demo mirroring white-paper §4.5:

1. Train a toy logistic-regression model on a baseline distribution
   (`n11_latency_ms`, `sbi_code`).
2. Register the production entity `AMF_v2::RegionA::N11` with the
   `DriftOrchestrator`, using the same baseline as the monitoring
   reference window and a representative log corpus.
3. Warm up the rolling z-score baselines with **30 stable batches** so
   alerting is calibrated.
4. Inject one **drifted batch** that simultaneously exhibits feature
   shift (new latency mode + 5xx surge), new log templates, recall
   regression from concept drift, and lagging incident detection (TTD
   exceeds the 300s SLO).
5. Train a challenger model `v3` with a corrected decision boundary
   and produce a champion/challenger comparison table.
6. Write a governance JSON report to `examples/health_report.json`.

### Run

From the repo root, with the package installed in editable mode:

```bash
pip install -e ".[dev,logs]"
python examples/quickstart.py
```

Expected runtime: a few seconds. Expected output: structured per-batch
alerts (PSI critical on `n11_latency_ms`, JS z-score alert, new log
templates, recall drift on the incident class, TTD breach), a canary
table comparing `v2` vs `v3`, and a JSON snapshot under
`examples/health_report.json`.

The artefact is git-ignored — re-run the script any time to regenerate.

## Adapting to a real model

Replace these three pieces with production equivalents and the rest of the
pipeline transfers unchanged:

| Demo helper | Real-world replacement |
|---|---|
| `make_features` | Your feature-store query for the entity's training and live windows. |
| `train_model` / `make_labels_and_predictions` | Your model's training pipeline and inference call (returning probabilities for calibration metrics). |
| `make_logs` / `make_incidents` | Your log shipping pipeline and incident store, scoped to the same `entity_key`. |

Keep the **entity key structure** (`<Component>_<Version>::<Region>::<Interface>`)
consistent across registration and every `run` call — that is what makes
per-segment root-cause analysis possible.
