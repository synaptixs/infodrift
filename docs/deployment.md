# Deployment Guide

How to deploy `drift_monitor` as a production monitoring service. Two
target environments are covered:

- **Local VM** — single Linux host, suitable for staging, on-prem
  deployments, or air-gapped environments.
- **Cloud** — AWS, GCP, or Azure using managed compute + object storage
  + a scheduler.

The same artefacts (Python wheel, container image, config files) work
in both. Pick the section that matches your target.

---

## 1. What you are deploying

`drift_monitor` is a **Python library**, not a long-running server.
A deployment wraps it in one of three shapes:

| Shape | When to use | What runs |
|---|---|---|
| **Batch job** | You score model batches on a schedule (hourly, daily). | A scheduled invocation of `DriftOrchestrator.run` per entity. |
| **Streaming consumer** | You consume Kafka / Pub/Sub / Kinesis events in micro-batches. | A long-running worker that buffers events into windows and calls `run` per window. |
| **Sidecar** | The model serves online and you want monitoring close to it. | A second container in the same pod/VM that reads features+predictions from a queue and calls `run`. |

The reference implementation provided in this repository is a **batch
job**. The other shapes wrap the same orchestrator.

---

## 2. Pre-requisites

Identical for both targets.

- **Python**: 3.10+.
- **CPU/RAM**: a single batch run for one entity over 1M rows fits
  comfortably in 2 vCPU / 4 GiB. Scale linearly per entity. PSI and
  divergence are O(n_features × n_rows); KDE is O(n_grid × n_rows).
- **Disk**: 90 days of metric history at 1 batch/hour ≈ 60 MB per
  entity (governance §9.6 recommendation: 12 months).
- **Outbound network**: only required if you publish alerts to PagerDuty
  / Slack / a metrics backend. The framework itself does not call out.
- **Secrets**: none required by `drift_monitor`. Anything you wire in
  (e.g. SMTP for alerts, Slack webhooks, S3 credentials) follows your
  organisation's secret-management practice.

> **Do not commit the white paper PDF or any production data into the
> deployed image.** `docs/MonitoringDrift.pdf` is for engineering
> reference only — the deployed image only needs the package + your
> entity configs.

---

## 3. Build artefacts

Build once, deploy anywhere.

### 3.1 Python wheel

```bash
python -m pip install --upgrade build
python -m build  # produces dist/drift_monitor-<version>-py3-none-any.whl
```

### 3.2 Container image

A minimal `Dockerfile` (place at the repo root):

```dockerfile
FROM python:3.12-slim AS base
WORKDIR /app

# System deps for numpy/scipy wheels (already binary on Linux x86_64).
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY drift_monitor ./drift_monitor
COPY examples ./examples
RUN pip install --no-cache-dir -e ".[logs]"

# Default entrypoint runs whatever job script you mount/COPY in.
ENTRYPOINT ["python", "-m"]
CMD ["examples.quickstart"]
```

Build:

```bash
docker build -t drift-monitor:0.1.0 .
```

A multi-stage build (separating wheel build from runtime) shaves ~150 MB
but is optional. The image size with the above is ~280 MB.

---

## 4. Local VM deployment

Target: one Linux VM with cron / systemd. No container orchestrator.

### 4.1 Install

```bash
sudo useradd --system --create-home --shell /usr/sbin/nologin driftmon
sudo -u driftmon python3 -m venv /opt/driftmon/venv
sudo -u driftmon /opt/driftmon/venv/bin/pip install /path/to/drift_monitor-0.1.0-py3-none-any.whl
```

Lay out the working directory:

```
/opt/driftmon/
├── venv/                    # Python venv
├── config/
│   └── entities.yaml        # entity_key -> baseline path mapping
├── baselines/
│   └── AMF_v2_RegionA_N11/  # parquet files for train_features_df
├── runs/                    # JSON reports, one per (entity, window)
└── logs/                    # stdout/stderr from the runner
```

### 4.2 Job script

Drop a small runner under `/opt/driftmon/run_batch.py` that loads the
entity config, fetches the production batch from your data source,
invokes `DriftOrchestrator.run`, and writes a `RunReport` JSON to
`/opt/driftmon/runs/`. Use `examples/quickstart.py` as the template
and replace `make_features` / `make_logs` with your real loaders.

### 4.3 systemd timer (recommended over cron)

`/etc/systemd/system/driftmon.service`:

```ini
[Unit]
Description=drift_monitor batch run
After=network-online.target
Wants=network-online.target

[Service]
Type=oneshot
User=driftmon
WorkingDirectory=/opt/driftmon
ExecStart=/opt/driftmon/venv/bin/python /opt/driftmon/run_batch.py
StandardOutput=append:/opt/driftmon/logs/driftmon.log
StandardError=append:/opt/driftmon/logs/driftmon.err
```

`/etc/systemd/system/driftmon.timer`:

```ini
[Unit]
Description=Run drift_monitor every hour

[Timer]
OnCalendar=hourly
Persistent=true
RandomizedDelaySec=120

[Install]
WantedBy=timers.target
```

Activate:

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now driftmon.timer
systemctl list-timers driftmon.timer
```

### 4.4 Local observability

- **Logs**: rotate `/opt/driftmon/logs/*.log` with `logrotate` (daily,
  keep 30).
- **Metrics**: parse the JSON reports in `/opt/driftmon/runs/` with
  `node_exporter` textfile collector or push directly to your
  Prometheus pushgateway from `run_batch.py`.
- **Alerts**: from the `Alert` objects emitted by `orchestrator.run`,
  POST to PagerDuty / Slack inside `run_batch.py` (the framework does
  not do this for you).

### 4.5 Backup

Snapshot `/opt/driftmon/runs/` daily — that directory is your governance
record (white paper §9.6). Treat it the same as audit logs.

---

## 5. Cloud deployment

The same image runs unchanged. Pick the matching primitives in your
cloud:

| Concern | AWS | GCP | Azure |
|---|---|---|---|
| Container registry | ECR | Artifact Registry | ACR |
| Scheduled compute | EventBridge → ECS Fargate task / Lambda | Cloud Scheduler → Cloud Run job | Logic Apps → Container Apps job |
| Object store (baselines, reports) | S3 | GCS | Blob Storage |
| Secrets | Secrets Manager | Secret Manager | Key Vault |
| Metrics | CloudWatch | Cloud Monitoring | Azure Monitor |
| Alert routing | SNS / PagerDuty | Pub/Sub / PagerDuty | Action Groups / PagerDuty |

### 5.1 AWS reference (Fargate scheduled task)

1. **Push image** to ECR:
   ```bash
   aws ecr get-login-password --region us-east-1 | \
     docker login --username AWS --password-stdin <acct>.dkr.ecr.us-east-1.amazonaws.com
   docker tag drift-monitor:0.1.0 <acct>.dkr.ecr.us-east-1.amazonaws.com/drift-monitor:0.1.0
   docker push <acct>.dkr.ecr.us-east-1.amazonaws.com/drift-monitor:0.1.0
   ```
2. **Create an ECS task definition** (Fargate, 0.5 vCPU / 1 GiB) with:
   - the ECR image,
   - a task role allowing `s3:GetObject` on the baselines bucket and
     `s3:PutObject` on the reports bucket,
   - environment variables: `DRIFTMON_BASELINES_BUCKET`,
     `DRIFTMON_REPORTS_BUCKET`, `DRIFTMON_ENTITY_CONFIG`.
3. **Schedule** the task via EventBridge with a `rate(1 hour)` rule
   targeting the cluster.
4. **Alerts**: have `run_batch.py` publish high-severity alerts to an
   SNS topic; subscribe PagerDuty / email to the topic.

Total state lives in S3 — no database required.

### 5.2 GCP reference (Cloud Run job)

```bash
gcloud builds submit --tag gcr.io/$PROJECT/drift-monitor:0.1.0
gcloud run jobs create drift-monitor-batch \
    --image gcr.io/$PROJECT/drift-monitor:0.1.0 \
    --region us-central1 \
    --tasks 1 --max-retries 2 --task-timeout 30m \
    --service-account driftmon@$PROJECT.iam.gserviceaccount.com \
    --set-env-vars "DRIFTMON_REPORTS_BUCKET=gs://$PROJECT-drift-reports"
gcloud scheduler jobs create http drift-monitor-hourly \
    --schedule="0 * * * *" \
    --uri="https://run.googleapis.com/v1/.../jobs/drift-monitor-batch:run" \
    --oauth-service-account-email driftmon@$PROJECT.iam.gserviceaccount.com
```

### 5.3 Azure reference (Container Apps job)

Use a **Container Apps job** (not a regular Container App) so you do
not pay for idle compute. Trigger on a cron schedule via the job's
`triggerType: Schedule`.

### 5.4 Multi-entity scaling

If you monitor many entities, do **not** loop over them in a single
container. Two patterns work:

- **Fan-out scheduler**: one scheduled task per entity. Simple, isolated
  failure domains. Recommended below ~50 entities.
- **Worker pool**: a single scheduler enqueues one message per entity
  to SQS / Pub/Sub / Service Bus; a small fleet of workers pulls and
  runs `orchestrator.run`. Use this above ~50 entities.

Either way, **rolling histories must persist between runs** — store
the orchestrator's per-entity state (rolling JS history, recall history)
in object storage at the end of each run and reload it at the start of
the next. See §6.

---

## 6. State persistence

`DriftOrchestrator` keeps state in memory:

- per-feature rolling JS history (default 30 batches),
- per-class recall history (default 7 batches),
- per-entity baseline (registered once at training time).

For cron / scheduled-job deployments where the process exits between
runs, persist this state. Two easy options:

1. **Pickle the orchestrator** (simplest, fine for a single host):
   ```python
   import pickle
   with open(state_path, "wb") as f:
       pickle.dump(orch, f)
   ```
   On the next run, unpickle, call `orch.run(...)`, and re-pickle.
2. **Per-entity state files** (cleaner for cloud / multi-host):
   write `ent.history`, divergence rolling deques, and recall tracker
   deques to JSON / Parquet under
   `<reports_bucket>/<entity_key>/state.json`. Reload at the start of
   the next run, before calling `run`.

If state is missing on the first run after deployment, the framework
treats the entity as **warming up** — alerts based on rolling z-score
will be suppressed until 5 batches accumulate (§10.6 of the white
paper). This is the correct, conservative default; do not bypass it.

Baselines themselves (`PSIMonitor._baselines`,
`DivergenceMonitor._baselines`) are computed once from training data
and stored alongside the orchestrator state. Recompute them only after
intentional model retraining or baseline refresh (§9.1).

---

## 7. Configuration

Keep entity configuration outside the image. A minimal YAML schema:

```yaml
# config/entities.yaml
entities:
  - entity_key: "AMF_v2::RegionA::N11"
    model_version: "v2"
    baseline_id: "train-2026Q1"
    numeric_features: ["n11_latency_ms"]
    categorical_features: ["sbi_code"]
    baseline_path: "s3://drift-baselines/AMF_v2_RegionA_N11/train.parquet"
    log_baseline_path: "s3://drift-baselines/AMF_v2_RegionA_N11/logs.txt"

slo:
  psi:         {warning: 0.10, critical: 0.25}
  ttd_seconds: {warning: 120,  critical: 300}
  fpr:         {warning: 0.08, critical: 0.15}
  ece:         {warning: 0.05, critical: 0.10}
```

`run_batch.py` loads this file, calls `register_entity` per row on the
first run, and persists the resulting orchestrator state.

---

## 8. CI/CD

The repo's `.github/workflows/tests.yml` already gates `develop` and
`main` on the full pytest suite across Ubuntu+macOS × py3.10/3.11/3.12.
For deployment, add a second workflow that builds and pushes the
container image on tagged releases:

```yaml
# .github/workflows/release.yml (sketch)
on:
  push:
    tags: ["v*"]
jobs:
  release:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: docker/setup-buildx-action@v3
      - uses: docker/login-action@v3
        with:
          registry: <your-registry>
          username: ${{ secrets.REGISTRY_USER }}
          password: ${{ secrets.REGISTRY_TOKEN }}
      - uses: docker/build-push-action@v6
        with:
          context: .
          push: true
          tags: <your-registry>/drift-monitor:${{ github.ref_name }}
```

Pin `pyproject.toml`'s `version` to the same tag. Release notes should
reference the white-paper section any new metric is derived from.

---

## 9. Verification checklist

Before declaring a deployment "done", confirm each of:

- [ ] CI passes on the deployed commit.
- [ ] `pytest` runs green inside the deployed image
      (`docker run --rm --entrypoint pytest drift-monitor:<tag> -q`).
- [ ] `python examples/quickstart.py` runs to completion in the
      deployed environment and produces `examples/health_report.json`.
- [ ] At least one stable production batch has been processed and its
      report is visible in the configured reports bucket / directory.
- [ ] Alert routing has been tested end-to-end with a synthetic alert
      (e.g. a forced PSI threshold breach).
- [ ] Rolling state persists across two scheduled runs (warm-up
      counter advances rather than resetting).
- [ ] Backup / retention of the reports bucket meets the §9.6 minimum
      of 90 days.

If any item fails, treat the deployment as not yet complete.

---

## 10. Common operational pitfalls

| Pitfall | Symptom | Fix |
|---|---|---|
| State not persisted between runs | Every run reports "warming up"; rolling z-score never alerts | Persist orchestrator state to durable storage (§6) |
| Baselines drift silently | Old baselines never refreshed; PSI numbers become meaningless | Refresh baselines on a documented cadence (white paper §9.1); record `baseline_id` in every alert |
| Schedule too aggressive | Compute cost spikes; alerts arrive faster than humans can triage | Start at 1 batch/hour per entity; tighten only after alert volume is well-understood |
| Mixing entity scopes in one report | Per-segment metrics leak across entities, defeating root-cause analysis | One `entity_key` per `run` call; never aggregate metrics across keys (§3.3) |
| Production data in the image | Image leaks PII or restricted log lines | Build images from public source only; mount data at runtime via the cloud's secret/data primitives |
