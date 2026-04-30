"""Population Stability Index (PSI) monitor — Layer 1 evidence health.

White paper §3.1:

    PSI = Σ_i (p_curr_i - p_base_i) * ln(p_curr_i / p_base_i)

Quantile binning on training data (B = 10 by default; 10–20 typical), Laplace
smoothing eps = 1e-6 to prevent log(0). Categorical features use the observed
training categories as bins, with an explicit "__OTHER__" bin for unseen
categories in production.

Standard interpretation thresholds (Anderson, 2007):
    PSI < 0.10        -> stable
    0.10 <= PSI < 0.25 -> moderate
    PSI >= 0.25       -> material
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import pandas as pd

EPS = 1e-6
OTHER_BIN = "__OTHER__"


@dataclass
class PSIResult:
    feature: str
    entity_key: str
    psi_score: float
    drift_level: str
    bin_details: pd.DataFrame = field(repr=False)


def _drift_level(psi: float) -> str:
    if psi < 0.10:
        return "stable"
    if psi < 0.25:
        return "moderate"
    return "material"


class PSIMonitor:
    """Quantile-binned PSI per feature, scoped per ontology entity."""

    def __init__(self, n_bins: int = 10):
        if not (2 <= n_bins <= 100):
            raise ValueError("n_bins must be in [2, 100]")
        self.n_bins = n_bins
        # entity_key -> feature -> {"type": "numeric"|"categorical", "edges"|"categories": ..., "p_base": np.ndarray}
        self._baselines: dict[str, dict[str, dict]] = {}

    def fit(
        self,
        train_df: pd.DataFrame,
        features: Iterable[str],
        entity_key: str,
    ) -> None:
        ent = self._baselines.setdefault(entity_key, {})
        for feat in features:
            if feat not in train_df.columns:
                raise KeyError(f"feature {feat!r} missing from train_df")
            col = train_df[feat]
            if pd.api.types.is_numeric_dtype(col):
                ent[feat] = self._fit_numeric(col)
            else:
                ent[feat] = self._fit_categorical(col)

    def _fit_numeric(self, col: pd.Series) -> dict:
        x = col.dropna().to_numpy()
        if x.size < self.n_bins:
            raise ValueError(
                f"need at least n_bins={self.n_bins} non-null training samples"
            )
        qs = np.linspace(0.0, 1.0, self.n_bins + 1)
        edges = np.quantile(x, qs)
        # Collapse duplicate edges (constant or near-constant features) and
        # widen the outer edges so production tails fall inside.
        edges = np.unique(edges)
        if edges.size < 2:
            edges = np.array([edges[0] - 1.0, edges[0] + 1.0])
        edges = edges.astype(float)
        edges[0] = -np.inf
        edges[-1] = np.inf
        counts, _ = np.histogram(x, bins=edges)
        p_base = self._smooth(counts / counts.sum())
        return {"type": "numeric", "edges": edges, "p_base": p_base}

    def _fit_categorical(self, col: pd.Series) -> dict:
        s = col.dropna().astype(str)
        cats = sorted(s.unique().tolist())
        if not cats:
            raise ValueError("categorical feature has no non-null values")
        index = {c: i for i, c in enumerate(cats)}
        counts = np.zeros(len(cats) + 1, dtype=float)  # +1 for OTHER
        for c in s:
            counts[index[c]] += 1
        p_base = self._smooth(counts / counts.sum())
        return {
            "type": "categorical",
            "categories": cats,
            "index": index,
            "p_base": p_base,
        }

    @staticmethod
    def _smooth(p: np.ndarray) -> np.ndarray:
        p = np.clip(p, EPS, None)
        return p / p.sum()

    def compute(
        self,
        prod_df: pd.DataFrame,
        features: Iterable[str],
        entity_key: str,
    ) -> list[PSIResult]:
        if entity_key not in self._baselines:
            raise KeyError(f"entity {entity_key!r} not registered; call fit() first")
        ent = self._baselines[entity_key]
        out: list[PSIResult] = []
        for feat in features:
            if feat not in ent:
                raise KeyError(f"feature {feat!r} not fitted for {entity_key!r}")
            spec = ent[feat]
            if spec["type"] == "numeric":
                p_curr, labels = self._curr_numeric(prod_df[feat], spec)
            else:
                p_curr, labels = self._curr_categorical(prod_df[feat], spec)
            p_base = spec["p_base"]
            psi_per_bin = (p_curr - p_base) * np.log(p_curr / p_base)
            psi = float(psi_per_bin.sum())
            details = pd.DataFrame(
                {
                    "bin": labels,
                    "p_base": p_base,
                    "p_curr": p_curr,
                    "psi_contribution": psi_per_bin,
                }
            )
            out.append(
                PSIResult(
                    feature=feat,
                    entity_key=entity_key,
                    psi_score=psi,
                    drift_level=_drift_level(psi),
                    bin_details=details,
                )
            )
        return out

    def _curr_numeric(self, col: pd.Series, spec: dict) -> tuple[np.ndarray, list[str]]:
        x = col.dropna().to_numpy()
        edges = spec["edges"]
        counts, _ = np.histogram(x, bins=edges)
        if counts.sum() == 0:
            counts = np.zeros_like(counts, dtype=float)
        p = self._smooth(counts / max(counts.sum(), 1))
        labels = [f"[{edges[i]:.4g}, {edges[i+1]:.4g})" for i in range(len(edges) - 1)]
        return p, labels

    def _curr_categorical(
        self, col: pd.Series, spec: dict
    ) -> tuple[np.ndarray, list[str]]:
        s = col.dropna().astype(str)
        index = spec["index"]
        cats = spec["categories"]
        counts = np.zeros(len(cats) + 1, dtype=float)
        for c in s:
            i = index.get(c)
            if i is None:
                counts[-1] += 1
            else:
                counts[i] += 1
        p = self._smooth(counts / max(counts.sum(), 1))
        labels = list(cats) + [OTHER_BIN]
        return p, labels
