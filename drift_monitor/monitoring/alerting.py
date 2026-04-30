"""SLO-based AlertEngine.

White paper §9.4: each alert carries severity, metric type, feature, entity,
observed value, threshold, message, recommended action, plus governance
metadata (timestamp, model version, baseline id).

SLO config is a dict keyed by metric name with optional ``warning`` and
``critical`` thresholds:

    {"psi":          {"warning": 0.10, "critical": 0.25},
     "ttd_seconds":  {"warning": 120,  "critical": 300},
     "fpr":          {"warning": 0.08, "critical": 0.15},
     "ece":          {"warning": 0.05, "critical": 0.10}}

Some metrics fire on a boolean (e.g. JS z-score alert, recall drift, new log
templates) — those are evaluated by ``evaluate_boolean`` and always carry
"warning" severity unless the SLO config maps them to "critical".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

DEFAULT_RECOMMENDATIONS: dict[str, str] = {
    "psi": "Investigate schema, upstream pipeline, or deployment change.",
    "js_zscore": "Check for multimodal distribution shift; correlate with deployment events.",
    "log_templates_new": "Validate parser rules; check for software rollout or new failure modes.",
    "log_templates_zscore": "Investigate template-mix change; correlate with rollouts.",
    "recall_drift": "Label quality review; consider targeted retraining for the affected class.",
    "ece": "Apply Platt or temperature scaling; consider recalibration.",
    "brier": "Investigate calibration and label quality.",
    "fpr": "Threshold recalibration; check for benign traffic pattern changes.",
    "tpr": "Investigate detector sensitivity; review missed-incident cohort.",
    "ttd_seconds": "Investigate alert routing, feature staleness, detection window size.",
}


@dataclass
class Alert:
    severity: str  # "warning" | "critical"
    metric_type: str
    entity_key: str
    observed: float
    threshold: float | None
    message: str
    recommendation: str
    feature: str | None = None
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    model_version: str | None = None
    baseline_id: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "severity": self.severity,
            "metric_type": self.metric_type,
            "entity_key": self.entity_key,
            "feature": self.feature,
            "observed": self.observed,
            "threshold": self.threshold,
            "message": self.message,
            "recommendation": self.recommendation,
            "timestamp": self.timestamp,
            "model_version": self.model_version,
            "baseline_id": self.baseline_id,
            "extra": self.extra,
        }


class AlertEngine:
    def __init__(self, slo_config: dict[str, dict[str, float]] | None = None):
        self.slo_config = slo_config or {}

    def evaluate_threshold(
        self,
        metric_type: str,
        observed: float,
        entity_key: str,
        *,
        feature: str | None = None,
        model_version: str | None = None,
        baseline_id: str | None = None,
        higher_is_worse: bool = True,
        extra: dict[str, Any] | None = None,
    ) -> Alert | None:
        cfg = self.slo_config.get(metric_type)
        if not cfg:
            return None
        crit = cfg.get("critical")
        warn = cfg.get("warning")

        def _breach(thr: float | None) -> bool:
            if thr is None:
                return False
            return observed >= thr if higher_is_worse else observed <= thr

        if _breach(crit):
            severity, threshold = "critical", crit
        elif _breach(warn):
            severity, threshold = "warning", warn
        else:
            return None
        message = (
            f"{metric_type} {observed:.4g} breached {severity} threshold {threshold:.4g}"
            f" for {entity_key}" + (f" / {feature}" if feature else "")
        )
        return Alert(
            severity=severity,
            metric_type=metric_type,
            entity_key=entity_key,
            feature=feature,
            observed=float(observed),
            threshold=float(threshold),
            message=message,
            recommendation=DEFAULT_RECOMMENDATIONS.get(metric_type, "Investigate."),
            model_version=model_version,
            baseline_id=baseline_id,
            extra=extra or {},
        )

    def boolean_alert(
        self,
        metric_type: str,
        entity_key: str,
        *,
        observed: float,
        feature: str | None = None,
        message: str,
        severity: str = "warning",
        model_version: str | None = None,
        baseline_id: str | None = None,
        extra: dict[str, Any] | None = None,
    ) -> Alert:
        return Alert(
            severity=severity,
            metric_type=metric_type,
            entity_key=entity_key,
            feature=feature,
            observed=float(observed),
            threshold=None,
            message=message,
            recommendation=DEFAULT_RECOMMENDATIONS.get(metric_type, "Investigate."),
            model_version=model_version,
            baseline_id=baseline_id,
            extra=extra or {},
        )
