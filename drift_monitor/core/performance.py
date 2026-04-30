"""Predictive reliability + operational metrics — Layers 2 & 3.

White paper §3.8-3.11:

    Brier score (proper scoring rule for binary forecasts)
    Expected Calibration Error (ECE) over B equal-width probability bins
    Reliability curve (frac_pos vs mean_pred per bin)
    Per-class Precision / Recall / F1 (one-vs-rest) with recall-drift alerting
    TPR / FPR for anomaly detectors at threshold τ
    Time-to-Detect percentiles  TTD_50 / TTD_90 / TTD_99
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

DEFAULT_RECALL_DROP = 0.20
DEFAULT_RECALL_LOOKBACK = 7


def brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    if y_true.shape != y_prob.shape:
        raise ValueError("y_true and y_prob must have the same shape")
    if y_true.size == 0:
        raise ValueError("empty inputs")
    return float(np.mean((y_prob - y_true) ** 2))


@dataclass
class ECEResult:
    ece: float
    reliability: pd.DataFrame


def ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> ECEResult:
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    if y_true.shape != y_prob.shape:
        raise ValueError("shape mismatch")
    if not (2 <= n_bins <= 100):
        raise ValueError("n_bins must be in [2, 100]")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    # Use the right-closed convention so probability == 1.0 lands in the last bin.
    idx = np.clip(np.digitize(y_prob, edges[1:-1], right=False), 0, n_bins - 1)
    rows = []
    gaps = []
    for k in range(n_bins):
        mask = idx == k
        if not mask.any():
            rows.append({"bin": k, "n": 0, "mean_pred": np.nan, "frac_pos": np.nan, "gap": 0.0})
            continue
        mean_pred = float(y_prob[mask].mean())
        frac_pos = float(y_true[mask].mean())
        gap = abs(frac_pos - mean_pred)
        rows.append(
            {"bin": k, "n": int(mask.sum()), "mean_pred": mean_pred, "frac_pos": frac_pos, "gap": gap}
        )
        gaps.append(gap)
    ece_value = float(np.mean(gaps)) if gaps else 0.0
    return ECEResult(ece=ece_value, reliability=pd.DataFrame(rows))


@dataclass
class ClassPRF:
    cls: str
    precision: float
    recall: float
    f1: float
    support: int


def per_class_prf(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    classes: list | None = None,
) -> list[ClassPRF]:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.shape != y_pred.shape:
        raise ValueError("shape mismatch")
    if classes is None:
        classes = sorted(set(y_true.tolist()) | set(y_pred.tolist()))
    out = []
    for c in classes:
        tp = int(((y_pred == c) & (y_true == c)).sum())
        fp = int(((y_pred == c) & (y_true != c)).sum())
        fn = int(((y_pred != c) & (y_true == c)).sum())
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        support = int((y_true == c).sum())
        out.append(ClassPRF(cls=str(c), precision=precision, recall=recall, f1=f1, support=support))
    return out


@dataclass
class TPRFPR:
    tpr: float
    fpr: float
    threshold: float


def tpr_fpr(y_true: np.ndarray, y_score: np.ndarray, tau: float) -> TPRFPR:
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    if y_true.shape != y_score.shape:
        raise ValueError("shape mismatch")
    pred_pos = y_score >= tau
    tp = int((pred_pos & (y_true == 1)).sum())
    fn = int((~pred_pos & (y_true == 1)).sum())
    fp = int((pred_pos & (y_true == 0)).sum())
    tn = int((~pred_pos & (y_true == 0)).sum())
    tpr = tp / (tp + fn) if (tp + fn) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    return TPRFPR(tpr=tpr, fpr=fpr, threshold=tau)


def ttd_percentiles(incidents_df: pd.DataFrame) -> dict[str, float]:
    """Time-to-Detect percentiles from per-incident (onset_ts, alert_ts) rows.

    Timestamps may be numeric (seconds) or datetime-like; output is in seconds,
    floored at zero per white-paper definition.
    """
    if incidents_df.empty:
        return {"p50": 0.0, "p90": 0.0, "p99": 0.0, "n": 0}
    onset = incidents_df["onset_ts"]
    alert = incidents_df["alert_ts"]
    if pd.api.types.is_numeric_dtype(onset) and pd.api.types.is_numeric_dtype(alert):
        deltas = (alert.to_numpy() - onset.to_numpy()).astype(float)
    else:
        onset = pd.to_datetime(onset)
        alert = pd.to_datetime(alert)
        deltas = (alert - onset).dt.total_seconds().to_numpy()
    deltas = np.maximum(deltas, 0.0)
    return {
        "p50": float(np.percentile(deltas, 50)),
        "p90": float(np.percentile(deltas, 90)),
        "p99": float(np.percentile(deltas, 99)),
        "n": int(deltas.size),
    }


@dataclass
class RecallDriftResult:
    cls: str
    current: float
    reference: float
    drop: float
    alert: bool


class RecallDriftTracker:
    """Per-entity, per-class recall history with drop-vs-lookback alerting."""

    def __init__(
        self,
        drop_threshold: float = DEFAULT_RECALL_DROP,
        lookback: int = DEFAULT_RECALL_LOOKBACK,
    ):
        self.drop_threshold = drop_threshold
        self.lookback = lookback
        self._history: dict[tuple[str, str], deque[float]] = {}

    def update(self, entity_key: str, prfs: list[ClassPRF]) -> list[RecallDriftResult]:
        out: list[RecallDriftResult] = []
        for prf in prfs:
            key = (entity_key, prf.cls)
            hist = self._history.setdefault(key, deque(maxlen=self.lookback + 1))
            if len(hist) >= 1:
                reference = hist[0]
                drop = reference - prf.recall
                alert = drop > self.drop_threshold
            else:
                reference = prf.recall
                drop = 0.0
                alert = False
            hist.append(prf.recall)
            out.append(
                RecallDriftResult(
                    cls=prf.cls,
                    current=prf.recall,
                    reference=reference,
                    drop=drop,
                    alert=alert,
                )
            )
        return out
