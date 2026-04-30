"""Predictive reliability + operational metrics — Layers 2 & 3.

See white paper §3.8-3.11:
- Brier score, Expected Calibration Error (ECE), reliability curve
- Per-class Precision / Recall / F1 with recall-drift alerting
- TPR / FPR for anomaly detectors at threshold tau
- Time-to-Detect (TTD) at p50 / p90 / p99
"""

# TODO: brier_score(y_true, y_prob)
# TODO: ece(y_true, y_prob, n_bins=10)
# TODO: per_class_prf(y_true, y_pred, classes)
# TODO: tpr_fpr(y_true, y_score, tau)
# TODO: ttd_percentiles(incidents_df) -> {"p50": ..., "p90": ..., "p99": ...}
