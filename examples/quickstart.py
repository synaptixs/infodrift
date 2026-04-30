"""End-to-end runtime demo for the drift_monitor framework.

This script walks through the full lifecycle described in the white paper
(§4.5) using synthetic data so it runs offline:

    1. Train a tiny logistic-regression "model" on baseline data.
    2. Register an ontology entity with the DriftOrchestrator using the
       same baseline as the monitoring reference window.
    3. Stream 30 stable production batches to warm up the rolling z-score
       baselines for JS divergence and log-template monitoring.
    4. Inject a drifted batch (new latency mode, new SBI error mix, new
       log templates, recall regression on a class) and run the
       orchestrator.
    5. Print the structured health summary, export a JSON governance
       report, and run a champion/challenger canary comparison.

Run it from the repo root:

    python examples/quickstart.py

It writes one artefact:

    examples/health_report.json     - governance-style JSON snapshot
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from drift_monitor import DriftOrchestrator, HealthReporter

ARTIFACT_PATH = Path(__file__).parent / "health_report.json"


# ---------------------------------------------------------------------------
# Synthetic data generators
# ---------------------------------------------------------------------------


def make_features(n: int, *, drifted: bool, rng: np.random.Generator) -> pd.DataFrame:
    """Latency + SBI status code distribution. Drifted=True adds a slow
    second mode at ~80ms and inflates 5xx error rate.

    Even the stable baseline includes a small (~5%) tail at higher latency
    so both classes are represented during training.
    """
    if drifted:
        latency = np.concatenate(
            [
                rng.normal(25, 5, int(0.7 * n)),
                rng.normal(80, 10, n - int(0.7 * n)),
            ]
        )
        sbi = rng.choice(["200", "400", "500"], size=n, p=[0.7, 0.07, 0.23])
    else:
        n_tail = int(0.05 * n)
        latency = np.concatenate(
            [rng.normal(25, 5, n - n_tail), rng.normal(55, 6, n_tail)]
        )
        rng.shuffle(latency)
        sbi = rng.choice(["200", "400", "500"], size=n, p=[0.90, 0.07, 0.03])
    return pd.DataFrame({"n11_latency_ms": latency, "sbi_code": sbi})


def make_labels_and_predictions(
    df: pd.DataFrame,
    model: LogisticRegression,
    rng: np.random.Generator,
    *,
    concept_drift: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate ground-truth labels + the model's predictions for a batch.

    The "true" label is a deterministic function of the latency: high
    latency -> incident class 1. ``concept_drift=True`` shifts the
    decision boundary so the model's old rule no longer matches reality
    (this is what causes the recall drop)."""
    boundary = 60.0 if concept_drift else 50.0
    y_true = (df["n11_latency_ms"] > boundary).astype(int).to_numpy()
    X = df[["n11_latency_ms"]].to_numpy()
    y_prob = model.predict_proba(X)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)
    # Anomaly score = same probability for this toy demo.
    return y_true, y_pred, y_true.copy(), y_prob


def make_logs(n: int, *, drifted: bool) -> list[str]:
    """Realistic-shape log lines so the regex normalizer collapses them
    to a small template vocabulary."""
    base = (
        ["2024-01-01T00:01:00Z INFO AMF N11 session setup SUPI 001-01-123456789 latency=22ms"] * (n // 2)
        + ["2024-01-01T00:01:01Z WARN AMF N11 timeout SUPI 001-01-234567890 retrying"] * (n // 4)
        + ["2024-01-01T00:01:02Z ERROR AMF N11 reject SUPI 001-01-345678901 cause=roaming_not_allowed"] * (n // 4)
    )
    if drifted:
        base += [
            "2024-01-15T00:01:00Z CRIT UPF N3 tunnel TEID 0x12345678 DROPPED rate_limit threshold=100Mbps"
        ] * (n // 2)
    return base


def make_incidents(rng: np.random.Generator, *, lagging: bool) -> pd.DataFrame:
    n = 30
    onset = np.zeros(n)
    if lagging:
        alert = rng.uniform(200, 700, n)
    else:
        alert = rng.uniform(20, 100, n)
    return pd.DataFrame({"onset_ts": onset, "alert_ts": alert})


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def train_model(train_df: pd.DataFrame, boundary: float = 50.0) -> LogisticRegression:
    y = (train_df["n11_latency_ms"] > boundary).astype(int).to_numpy()
    X = train_df[["n11_latency_ms"]].to_numpy()
    model = LogisticRegression()
    model.fit(X, y)
    return model


def main() -> None:
    rng = np.random.default_rng(42)

    # 1. Build a baseline training set and train a toy model.
    train_df = make_features(8000, drifted=False, rng=rng)
    train_logs = make_logs(2000, drifted=False)
    model = train_model(train_df)
    print(f"Trained logistic regression: coef={model.coef_.ravel()[0]:.3f}, "
          f"intercept={model.intercept_[0]:.3f}")

    # 2. Configure the orchestrator and register the production entity.
    slo = {
        "psi":         {"warning": 0.10, "critical": 0.25},
        "ttd_seconds": {"warning": 120,  "critical": 300},
        "fpr":         {"warning": 0.08, "critical": 0.15},
        "ece":         {"warning": 0.05, "critical": 0.10},
        "brier":       {"warning": 0.15, "critical": 0.25},
    }
    orch = DriftOrchestrator(
        slo_config=slo,
        divergence_window=30,
        log_window=30,
    )
    entity_v2 = "AMF_v2::RegionA::N11"
    orch.register_entity(
        entity_v2,
        train_features_df=train_df,
        numeric_features=["n11_latency_ms"],
        categorical_features=["sbi_code"],
        train_logs=train_logs,
        model_version="v2",
        baseline_id="train-2026Q1",
    )

    # 3. Warm up the rolling baselines with 30 stable production batches.
    print("\nWarming up rolling baselines (30 stable batches)...")
    for i in range(30):
        prod = make_features(1500, drifted=False, rng=rng)
        prod_logs = make_logs(400, drifted=False)
        y_true, y_pred, y_bin, y_prob = make_labels_and_predictions(
            prod, model, rng, concept_drift=False
        )
        orch.run(
            entity_v2,
            prod_features_df=prod,
            prod_logs=prod_logs,
            y_true_clf=y_true,
            y_pred_clf=y_pred,
            y_true_calib=y_bin,
            y_prob_calib=y_prob,
            y_true_anom=y_true,
            y_score_anom=y_prob,
            anomaly_threshold=0.5,
            window_id=f"warmup-{i:02d}",
        )

    # 4. Drifted production batch — feature shift + new log templates +
    #    concept drift causing recall degradation + lagging incidents.
    print("\nRunning drifted production window...")
    drift_df = make_features(2000, drifted=True, rng=rng)
    drift_logs = make_logs(1200, drifted=True)
    y_true_d, y_pred_d, y_bin_d, y_prob_d = make_labels_and_predictions(
        drift_df, model, rng, concept_drift=True
    )
    incidents = make_incidents(rng, lagging=True)

    alerts = orch.run(
        entity_v2,
        prod_features_df=drift_df,
        prod_logs=drift_logs,
        y_true_clf=y_true_d,
        y_pred_clf=y_pred_d,
        y_true_calib=y_bin_d,
        y_prob_calib=y_prob_d,
        y_true_anom=y_true_d,
        y_score_anom=y_prob_d,
        anomaly_threshold=0.5,
        incidents_df=incidents,
        window_id="2025-W30",
        verbose=True,
    )
    print(f"\n{len(alerts)} alert(s) fired:")
    for a in alerts:
        feat = f" / {a.feature}" if a.feature else ""
        print(f"  [{a.severity:8s}] {a.metric_type:20s}{feat}  -> {a.message}")
        print(f"             action: {a.recommendation}")

    # 5. Champion/challenger canary — train a v3 model with a wider
    #    decision boundary that better matches the drifted reality.
    print("\nRegistering challenger v3 and running same drifted window...")
    challenger_train = make_features(8000, drifted=False, rng=rng)
    challenger = train_model(challenger_train, boundary=60.0)
    entity_v3 = "AMF_v3::RegionA::N11"
    orch.register_entity(
        entity_v3,
        train_features_df=challenger_train,
        numeric_features=["n11_latency_ms"],
        categorical_features=["sbi_code"],
        train_logs=train_logs,
        model_version="v3",
        baseline_id="train-2026Q2",
    )
    # Warm up shorter — just enough to clear the 5-batch minimum.
    for i in range(8):
        warm = make_features(1500, drifted=False, rng=rng)
        orch.run(entity_v3, prod_features_df=warm, window_id=f"v3-warmup-{i:02d}")

    yt_c, yp_c, _, ypr_c = make_labels_and_predictions(
        drift_df, challenger, rng, concept_drift=True
    )
    orch.run(
        entity_v3,
        prod_features_df=drift_df,
        prod_logs=drift_logs,
        y_true_clf=yt_c,
        y_pred_clf=yp_c,
        y_true_calib=y_bin_d,
        y_prob_calib=ypr_c,
        y_true_anom=yt_c,
        y_score_anom=ypr_c,
        anomaly_threshold=0.5,
        incidents_df=incidents,
        window_id="2025-W30",
    )

    reporter = HealthReporter(orch)
    print("\n=== Champion / challenger canary comparison ===")
    print(reporter.canary_comparison(entity_v2, entity_v3).to_string())

    # 6. Persist a governance-style JSON report.
    payload = reporter.full_report(as_json=True)
    ARTIFACT_PATH.write_text(payload)
    print(f"\nWrote governance report -> {ARTIFACT_PATH}")
    summary = json.loads(payload)
    print(f"Entities tracked: {list(summary['entities'])}")


if __name__ == "__main__":
    main()
