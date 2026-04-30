"""Central DriftOrchestrator — wires Layer 1/2/3 monitors per ontology entity.

Entity key convention (white paper §3.3):
    "<Component>_<Version>::<Region>::<Interface>"

The orchestrator owns:
  * one PSIMonitor + one DivergenceMonitor + one LogTemplateMonitor per
    ``register_entity`` call;
  * a per-entity RecallDriftTracker;
  * the AlertEngine and its SLO config;
  * a structured run-log keyed by ``(entity_key, window_id)`` for the
    HealthReporter to consume.

Inputs to ``run`` are all optional except features. Anything missing simply
skips the corresponding layer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from drift_monitor.core.divergence import DivergenceMonitor, DivergenceResult
from drift_monitor.core.log_templates import LogTemplateMonitor, LogTemplateResult
from drift_monitor.core.performance import (
    ClassPRF,
    ECEResult,
    RecallDriftResult,
    RecallDriftTracker,
    TPRFPR,
    brier_score,
    ece,
    per_class_prf,
    tpr_fpr,
    ttd_percentiles,
)
from drift_monitor.core.psi import PSIMonitor, PSIResult
from drift_monitor.monitoring.alerting import Alert, AlertEngine

EPS_NEW_TEMPLATE = 1e-9


@dataclass
class EntityState:
    entity_key: str
    numeric_features: list[str]
    categorical_features: list[str]
    psi: PSIMonitor
    divergence: DivergenceMonitor
    log_templates: LogTemplateMonitor | None
    recall_tracker: RecallDriftTracker
    model_version: str | None
    baseline_id: str
    history: dict[str, dict[str, Any]] = field(default_factory=dict)


@dataclass
class RunReport:
    entity_key: str
    window_id: str
    timestamp: str
    psi: list[PSIResult] = field(default_factory=list)
    divergence: list[DivergenceResult] = field(default_factory=list)
    log_templates: LogTemplateResult | None = None
    per_class_prf: list[ClassPRF] = field(default_factory=list)
    recall_drift: list[RecallDriftResult] = field(default_factory=list)
    ece: ECEResult | None = None
    brier: float | None = None
    tpr_fpr: TPRFPR | None = None
    ttd: dict[str, float] | None = None
    alerts: list[Alert] = field(default_factory=list)


class DriftOrchestrator:
    def __init__(
        self,
        slo_config: dict[str, dict[str, float]] | None = None,
        *,
        psi_bins: int = 10,
        divergence_window: int = 30,
        divergence_sigma: float = 3.0,
        log_window: int = 30,
        log_sigma: float = 3.0,
        ece_bins: int = 10,
        anomaly_threshold: float = 0.5,
        recall_drop: float = 0.20,
        recall_lookback: int = 7,
        use_drain: bool = True,
    ):
        self.slo_config = slo_config or {}
        self.alert_engine = AlertEngine(self.slo_config)
        self._cfg = {
            "psi_bins": psi_bins,
            "divergence_window": divergence_window,
            "divergence_sigma": divergence_sigma,
            "log_window": log_window,
            "log_sigma": log_sigma,
            "ece_bins": ece_bins,
            "anomaly_threshold": anomaly_threshold,
            "recall_drop": recall_drop,
            "recall_lookback": recall_lookback,
            "use_drain": use_drain,
        }
        self._entities: dict[str, EntityState] = {}

    def register_entity(
        self,
        entity_key: str,
        train_features_df: pd.DataFrame,
        numeric_features: list[str] | None = None,
        categorical_features: list[str] | None = None,
        train_logs: list[str] | None = None,
        *,
        model_version: str | None = None,
        baseline_id: str | None = None,
    ) -> None:
        numeric_features = list(numeric_features or [])
        categorical_features = list(categorical_features or [])
        all_features = numeric_features + categorical_features
        if not all_features:
            raise ValueError("must register at least one feature")

        psi = PSIMonitor(n_bins=self._cfg["psi_bins"])
        psi.fit(train_features_df, all_features, entity_key)

        div = DivergenceMonitor(
            use_kde=True,
            alert_sigma=self._cfg["divergence_sigma"],
            rolling_window=self._cfg["divergence_window"],
        )
        div.fit(train_features_df, all_features, entity_key)

        log_mon: LogTemplateMonitor | None = None
        if train_logs:
            log_mon = LogTemplateMonitor(
                alert_sigma=self._cfg["log_sigma"],
                rolling_window=self._cfg["log_window"],
                use_drain=self._cfg["use_drain"],
            )
            log_mon.fit(train_logs, entity_key)

        recall_tracker = RecallDriftTracker(
            drop_threshold=self._cfg["recall_drop"],
            lookback=self._cfg["recall_lookback"],
        )

        self._entities[entity_key] = EntityState(
            entity_key=entity_key,
            numeric_features=numeric_features,
            categorical_features=categorical_features,
            psi=psi,
            divergence=div,
            log_templates=log_mon,
            recall_tracker=recall_tracker,
            model_version=model_version,
            baseline_id=baseline_id or f"{entity_key}@register",
        )

    def get_entity(self, entity_key: str) -> EntityState:
        if entity_key not in self._entities:
            raise KeyError(f"entity {entity_key!r} not registered")
        return self._entities[entity_key]

    def run(
        self,
        entity_key: str,
        prod_features_df: pd.DataFrame | None = None,
        prod_logs: list[str] | None = None,
        *,
        y_true_clf: np.ndarray | None = None,
        y_pred_clf: np.ndarray | None = None,
        classes: list | None = None,
        y_true_calib: np.ndarray | None = None,
        y_prob_calib: np.ndarray | None = None,
        y_true_anom: np.ndarray | None = None,
        y_score_anom: np.ndarray | None = None,
        anomaly_threshold: float | None = None,
        incidents_df: pd.DataFrame | None = None,
        window_id: str | None = None,
        verbose: bool = False,
    ) -> list[Alert]:
        ent = self.get_entity(entity_key)
        window_id = window_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        report = RunReport(
            entity_key=entity_key,
            window_id=window_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        all_features = ent.numeric_features + ent.categorical_features

        # Layer 1 — PSI + JS divergence on features
        if prod_features_df is not None and not prod_features_df.empty:
            report.psi = ent.psi.compute(prod_features_df, all_features, entity_key)
            for r in report.psi:
                a = self.alert_engine.evaluate_threshold(
                    "psi",
                    r.psi_score,
                    entity_key,
                    feature=r.feature,
                    model_version=ent.model_version,
                    baseline_id=ent.baseline_id,
                    extra={"drift_level": r.drift_level},
                )
                if a:
                    report.alerts.append(a)

            report.divergence = ent.divergence.compute(
                prod_features_df, all_features, entity_key
            )
            for r in report.divergence:
                if r.alert:
                    report.alerts.append(
                        self.alert_engine.boolean_alert(
                            "js_zscore",
                            entity_key,
                            observed=r.z_score,
                            feature=r.feature,
                            message=(
                                f"JS divergence z={r.z_score:.2f}σ on {r.feature}"
                                f" (JS={r.js_score:.4f}, KL={r.kl_score:.4f})"
                            ),
                            severity="warning",
                            model_version=ent.model_version,
                            baseline_id=ent.baseline_id,
                            extra={"js": r.js_score, "kl": r.kl_score},
                        )
                    )

        # Layer 1 — Log templates
        if prod_logs and ent.log_templates is not None:
            log_res = ent.log_templates.compute(prod_logs, entity_key)
            report.log_templates = log_res
            if log_res.new_templates:
                report.alerts.append(
                    self.alert_engine.boolean_alert(
                        "log_templates_new",
                        entity_key,
                        observed=float(len(log_res.new_templates)),
                        message=(
                            f"{len(log_res.new_templates)} new log template(s) detected"
                        ),
                        severity="warning",
                        model_version=ent.model_version,
                        baseline_id=ent.baseline_id,
                        extra={"new_templates": log_res.new_templates[:10]},
                    )
                )
            if log_res.alert and not log_res.new_templates:
                report.alerts.append(
                    self.alert_engine.boolean_alert(
                        "log_templates_zscore",
                        entity_key,
                        observed=log_res.z_score,
                        message=(
                            f"Log template JS z={log_res.z_score:.2f}σ"
                            f" (JS={log_res.js_score:.4f}, PSI={log_res.psi_score:.4f})"
                        ),
                        severity="warning",
                        model_version=ent.model_version,
                        baseline_id=ent.baseline_id,
                    )
                )

        # Layer 2 — Classification quality
        if y_true_clf is not None and y_pred_clf is not None:
            report.per_class_prf = per_class_prf(y_true_clf, y_pred_clf, classes=classes)
            report.recall_drift = ent.recall_tracker.update(entity_key, report.per_class_prf)
            for rd in report.recall_drift:
                if rd.alert:
                    report.alerts.append(
                        self.alert_engine.boolean_alert(
                            "recall_drift",
                            entity_key,
                            observed=rd.drop,
                            feature=rd.cls,
                            message=(
                                f"Recall on class {rd.cls!r} dropped {rd.drop:.2f}"
                                f" (now {rd.current:.2f}, ref {rd.reference:.2f})"
                            ),
                            severity="critical",
                            model_version=ent.model_version,
                            baseline_id=ent.baseline_id,
                        )
                    )

        # Layer 2 — Calibration
        if y_true_calib is not None and y_prob_calib is not None:
            report.brier = brier_score(y_true_calib, y_prob_calib)
            report.ece = ece(y_true_calib, y_prob_calib, n_bins=self._cfg["ece_bins"])
            for metric, value in (("brier", report.brier), ("ece", report.ece.ece)):
                a = self.alert_engine.evaluate_threshold(
                    metric,
                    value,
                    entity_key,
                    model_version=ent.model_version,
                    baseline_id=ent.baseline_id,
                )
                if a:
                    report.alerts.append(a)

        # Layer 3 — TPR/FPR for anomaly detector
        if y_true_anom is not None and y_score_anom is not None:
            tau = anomaly_threshold if anomaly_threshold is not None else self._cfg["anomaly_threshold"]
            report.tpr_fpr = tpr_fpr(y_true_anom, y_score_anom, tau)
            a = self.alert_engine.evaluate_threshold(
                "fpr",
                report.tpr_fpr.fpr,
                entity_key,
                model_version=ent.model_version,
                baseline_id=ent.baseline_id,
                extra={"tpr": report.tpr_fpr.tpr, "tau": tau},
            )
            if a:
                report.alerts.append(a)
            a_tpr = self.alert_engine.evaluate_threshold(
                "tpr",
                report.tpr_fpr.tpr,
                entity_key,
                higher_is_worse=False,
                model_version=ent.model_version,
                baseline_id=ent.baseline_id,
                extra={"fpr": report.tpr_fpr.fpr, "tau": tau},
            )
            if a_tpr:
                report.alerts.append(a_tpr)

        # Layer 3 — Time-to-Detect
        if incidents_df is not None and not incidents_df.empty:
            report.ttd = ttd_percentiles(incidents_df)
            a = self.alert_engine.evaluate_threshold(
                "ttd_seconds",
                report.ttd["p50"],
                entity_key,
                model_version=ent.model_version,
                baseline_id=ent.baseline_id,
                extra={"p90": report.ttd["p90"], "p99": report.ttd["p99"]},
            )
            if a:
                report.alerts.append(a)

        ent.history[window_id] = {"report": report}

        if verbose:
            self._print_run(report)
        return report.alerts

    @staticmethod
    def _print_run(r: RunReport) -> None:
        print(f"[{r.timestamp}] {r.entity_key} window={r.window_id}")
        for p in r.psi:
            print(f"  PSI {p.feature}: {p.psi_score:.4f} [{p.drift_level}]")
        for d in r.divergence:
            flag = "ALERT" if d.alert else "ok"
            print(
                f"  JS  {d.feature}: js={d.js_score:.4f} kl={d.kl_score:.4f}"
                f" z={d.z_score:.2f} [{flag}]"
            )
        if r.log_templates is not None:
            lt = r.log_templates
            print(
                f"  LOG: js={lt.js_score:.4f} psi={lt.psi_score:.4f}"
                f" new={len(lt.new_templates)} alert={lt.alert}"
            )
        if r.brier is not None:
            print(f"  Brier={r.brier:.4f} ECE={r.ece.ece:.4f}")
        if r.tpr_fpr is not None:
            print(f"  TPR={r.tpr_fpr.tpr:.3f} FPR={r.tpr_fpr.fpr:.3f}")
        if r.ttd is not None:
            print(
                f"  TTD: p50={r.ttd['p50']:.1f}s p90={r.ttd['p90']:.1f}s"
                f" p99={r.ttd['p99']:.1f}s"
            )
        for a in r.alerts:
            print(f"  ALERT [{a.severity}] {a.metric_type}: {a.message}")

    def latest_report(self, entity_key: str) -> RunReport | None:
        ent = self.get_entity(entity_key)
        if not ent.history:
            return None
        # dicts preserve insertion order; the most recent run is the last key.
        last_window = next(reversed(ent.history))
        return ent.history[last_window]["report"]
