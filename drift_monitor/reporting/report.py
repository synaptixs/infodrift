"""HealthReporter — structured summaries, canary comparisons, JSON export.

White paper §4.5-4.6 and §9.6: timestamped, versioned reports keyed by entity
scope; champion/challenger comparison tables for canary rollouts.
"""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from typing import Any

import numpy as np
import pandas as pd

from drift_monitor.monitoring.orchestrator import DriftOrchestrator, RunReport


def _serializable(obj: Any) -> Any:
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if is_dataclass(obj):
        return {k: _serializable(v) for k, v in asdict(obj).items()}
    if isinstance(obj, dict):
        return {k: _serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_serializable(v) for v in obj]
    return obj


class HealthReporter:
    def __init__(self, orchestrator: DriftOrchestrator):
        self.orchestrator = orchestrator

    def print_summary(self) -> None:
        for entity_key, ent in self.orchestrator._entities.items():
            report = self.orchestrator.latest_report(entity_key)
            if report is None:
                print(f"{entity_key}: no runs yet")
                continue
            print(f"== {entity_key} (model={ent.model_version}) ==")
            print(f"  window: {report.window_id} @ {report.timestamp}")
            for p in report.psi:
                print(f"  PSI {p.feature}: {p.psi_score:.4f} [{p.drift_level}]")
            if report.brier is not None:
                print(f"  Brier={report.brier:.4f} ECE={report.ece.ece:.4f}")
            if report.tpr_fpr is not None:
                print(f"  TPR={report.tpr_fpr.tpr:.3f} FPR={report.tpr_fpr.fpr:.3f}")
            if report.ttd is not None:
                print(
                    f"  TTD p50/p90/p99 = {report.ttd['p50']:.1f}/"
                    f"{report.ttd['p90']:.1f}/{report.ttd['p99']:.1f}s"
                )
            if report.alerts:
                print(f"  alerts: {len(report.alerts)}")
                for a in report.alerts:
                    print(f"    [{a.severity}] {a.metric_type}: {a.message}")
            else:
                print("  alerts: none")

    def full_report(self, as_json: bool = False) -> dict | str:
        out: dict[str, Any] = {"entities": {}}
        for entity_key, ent in self.orchestrator._entities.items():
            report = self.orchestrator.latest_report(entity_key)
            if report is None:
                continue
            out["entities"][entity_key] = {
                "model_version": ent.model_version,
                "baseline_id": ent.baseline_id,
                "report": _serializable(report),
            }
        if as_json:
            return json.dumps(out, indent=2, default=str)
        return out

    def canary_comparison(
        self,
        baseline_entity: str,
        canary_entity: str,
    ) -> pd.DataFrame:
        b = self.orchestrator.latest_report(baseline_entity)
        c = self.orchestrator.latest_report(canary_entity)
        if b is None or c is None:
            raise ValueError("both entities must have at least one run")

        def row(r: RunReport) -> dict[str, float | None]:
            return {
                "brier_score": r.brier,
                "ece": r.ece.ece if r.ece is not None else None,
                "tpr": r.tpr_fpr.tpr if r.tpr_fpr is not None else None,
                "fpr": r.tpr_fpr.fpr if r.tpr_fpr is not None else None,
                "median_ttd_s": r.ttd["p50"] if r.ttd is not None else None,
                "max_psi": max((p.psi_score for p in r.psi), default=None),
                "alerts": len(r.alerts),
            }

        df = pd.DataFrame([row(b), row(c)], index=["baseline", "canary"])
        df.index.name = "cohort"
        return df
