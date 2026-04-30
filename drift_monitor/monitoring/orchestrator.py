"""Central DriftOrchestrator — wires Layer 1/2/3 monitors per ontology entity.

Entity key convention (white paper §3.3):
    "<Component>_<Version>::<Region>::<Interface>"
"""


class DriftOrchestrator:
    def __init__(self, slo_config: dict | None = None):
        raise NotImplementedError

    def register_entity(self, entity_key: str, **kwargs) -> None:
        raise NotImplementedError

    def run(self, entity_key: str, **kwargs) -> list:
        raise NotImplementedError
