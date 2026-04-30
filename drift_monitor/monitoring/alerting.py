"""SLO-based AlertEngine.

See white paper §9.4: each alert carries severity, metric type, feature,
entity, observed value, threshold, message, and recommended action; logged
with timestamp and model version for governance.
"""


class AlertEngine:
    def __init__(self, slo_config: dict):
        raise NotImplementedError
