"""HealthReporter — structured summaries, canary comparisons, JSON export.

See white paper §4.5-4.6 and §9.6: timestamped, versioned reports keyed
by entity scope; supports champion/challenger comparison tables.
"""


class HealthReporter:
    def __init__(self, orchestrator):
        self.orchestrator = orchestrator

    def print_summary(self) -> None:
        raise NotImplementedError

    def full_report(self, as_json: bool = False):
        raise NotImplementedError

    def canary_comparison(self, baseline_entity: str, canary_entity: str):
        raise NotImplementedError
