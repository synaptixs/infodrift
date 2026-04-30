"""KL / JS divergence monitor with optional KDE — Layer 1 evidence health.

White paper §3.2-3.6:

    KL(P || Q) = Σ_x P(x) ln[P(x) / Q(x)]
    JS(P || Q) = 0.5 KL(P||M) + 0.5 KL(Q||M),  M = (P+Q)/2,  bounded [0, ln2]

Continuous features are estimated with Gaussian KDE on a fixed grid of
G = 200 points spanning [min_train, max_train] (frozen at fit time so
batches are comparable). Bandwidth uses Scott's rule h = n^(-1/5) σ̂.

Categorical features use the observed training categories with an
"__OTHER__" bin for unseen production categories.

Alerting uses a rolling z-score over the last W batches of JS values; alert
fires when z > k (default k = 3, W = 30). At least 5 stable batches are
required before z-score alerting activates (white paper §10.6).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

EPS = 1e-12
GRID_POINTS = 200
MIN_WARMUP = 5


@dataclass
class DivergenceResult:
    feature: str
    entity_key: str
    js_score: float
    kl_score: float
    z_score: float
    alert: bool
    reason: str


def _scott_bandwidth(x: np.ndarray) -> float:
    n = x.size
    sigma = float(np.std(x, ddof=1)) if n > 1 else 1.0
    if sigma <= 0 or not np.isfinite(sigma):
        sigma = 1.0
    return max(n ** (-1 / 5) * sigma, 1e-6)


def _gaussian_kde(samples: np.ndarray, grid: np.ndarray, h: float) -> np.ndarray:
    # Evaluate Gaussian KDE on grid; return normalized discrete probability vector.
    if samples.size == 0:
        return np.full_like(grid, 1.0 / grid.size, dtype=float)
    diffs = (grid[:, None] - samples[None, :]) / h
    kernels = np.exp(-0.5 * diffs * diffs) / np.sqrt(2 * np.pi)
    density = kernels.sum(axis=1) / (samples.size * h)
    density = np.clip(density, EPS, None)
    return density / density.sum()


def _kl(p: np.ndarray, q: np.ndarray) -> float:
    p = np.clip(p, EPS, None)
    q = np.clip(q, EPS, None)
    return float(np.sum(p * np.log(p / q)))


def _js(p: np.ndarray, q: np.ndarray) -> float:
    m = 0.5 * (p + q)
    return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)


class DivergenceMonitor:
    """Per-feature KL/JS divergence with rolling z-score alerting.

    Parameters
    ----------
    use_kde : bool
        If True (default), continuous features are smoothed with Gaussian KDE
        on a fixed grid frozen at fit time. If False, a histogram with
        ``hist_bins`` bins (also frozen at fit time) is used.
    kde_bandwidth : str
        Only "scott" is supported (white paper §3.5).
    alert_sigma : float
        z-score threshold above the rolling mean to fire an alert.
    rolling_window : int
        Number of recent batches retained for the rolling mean / std.
    hist_bins : int
        Number of histogram bins used when ``use_kde=False`` and for
        categorical-style discrete inputs (ignored for true categoricals).
    """

    def __init__(
        self,
        use_kde: bool = True,
        kde_bandwidth: str = "scott",
        alert_sigma: float = 3.0,
        rolling_window: int = 30,
        hist_bins: int = 50,
    ):
        if kde_bandwidth != "scott":
            raise ValueError("only 'scott' bandwidth is supported")
        if rolling_window < MIN_WARMUP:
            raise ValueError(f"rolling_window must be >= {MIN_WARMUP}")
        self.use_kde = use_kde
        self.alert_sigma = alert_sigma
        self.rolling_window = rolling_window
        self.hist_bins = hist_bins
        self._baselines: dict[str, dict[str, dict]] = {}
        self._history: dict[tuple[str, str], deque[float]] = {}

    def fit(
        self,
        train_df: pd.DataFrame,
        features: Iterable[str],
        entity_key: str,
    ) -> None:
        ent = self._baselines.setdefault(entity_key, {})
        for feat in features:
            col = train_df[feat]
            if pd.api.types.is_numeric_dtype(col):
                ent[feat] = self._fit_numeric(col.dropna().to_numpy())
            else:
                ent[feat] = self._fit_categorical(col.dropna().astype(str))
            self._history[(entity_key, feat)] = deque(maxlen=self.rolling_window)

    def _fit_numeric(self, x: np.ndarray) -> dict:
        if x.size < 2:
            raise ValueError("need >= 2 numeric training samples")
        lo, hi = float(x.min()), float(x.max())
        if lo == hi:
            lo, hi = lo - 0.5, hi + 0.5
        if self.use_kde:
            grid = np.linspace(lo, hi, GRID_POINTS)
            h = _scott_bandwidth(x)
            base_kde = _gaussian_kde(x, grid, h)
            # Append an out-of-range bucket so production samples falling
            # outside [min_train, max_train] register as distributional mass.
            p_base = np.concatenate([base_kde, [EPS]])
            p_base = p_base / p_base.sum()
            return {"type": "numeric_kde", "grid": grid, "h": h, "lo": lo, "hi": hi, "p_base": p_base}
        edges = np.linspace(lo, hi, self.hist_bins + 1)
        edges[0], edges[-1] = -np.inf, np.inf
        counts, _ = np.histogram(x, bins=edges)
        p_base = np.clip(counts / counts.sum(), EPS, None)
        p_base = p_base / p_base.sum()
        return {"type": "numeric_hist", "edges": edges, "p_base": p_base}

    def _fit_categorical(self, s: pd.Series) -> dict:
        cats = sorted(s.unique().tolist())
        if not cats:
            raise ValueError("categorical feature has no values")
        index = {c: i for i, c in enumerate(cats)}
        counts = np.zeros(len(cats) + 1, dtype=float)
        for c in s:
            counts[index[c]] += 1
        p_base = np.clip(counts / counts.sum(), EPS, None)
        p_base = p_base / p_base.sum()
        return {"type": "categorical", "categories": cats, "index": index, "p_base": p_base}

    def compute(
        self,
        prod_df: pd.DataFrame,
        features: Iterable[str],
        entity_key: str,
    ) -> list[DivergenceResult]:
        if entity_key not in self._baselines:
            raise KeyError(f"entity {entity_key!r} not registered")
        ent = self._baselines[entity_key]
        out: list[DivergenceResult] = []
        for feat in features:
            spec = ent[feat]
            p_base = spec["p_base"]
            p_curr = self._estimate_current(prod_df[feat], spec)
            kl = _kl(p_curr, p_base)
            js = _js(p_curr, p_base)
            history = self._history[(entity_key, feat)]
            z, alert, reason = self._z_alert(js, history)
            history.append(js)
            out.append(
                DivergenceResult(
                    feature=feat,
                    entity_key=entity_key,
                    js_score=js,
                    kl_score=kl,
                    z_score=z,
                    alert=alert,
                    reason=reason,
                )
            )
        return out

    def _estimate_current(self, col: pd.Series, spec: dict) -> np.ndarray:
        if spec["type"] == "numeric_kde":
            x = col.dropna().to_numpy()
            if x.size == 0:
                p = np.full(spec["grid"].size + 1, 1.0 / (spec["grid"].size + 1))
                return p
            in_mask = (x >= spec["lo"]) & (x <= spec["hi"])
            in_x = x[in_mask]
            oor_frac = float((~in_mask).sum()) / x.size
            kde = _gaussian_kde(in_x, spec["grid"], spec["h"]) if in_x.size > 0 else np.full(spec["grid"].size, EPS)
            kde = kde * (1.0 - oor_frac)
            p = np.concatenate([kde, [max(oor_frac, EPS)]])
            return p / p.sum()
        if spec["type"] == "numeric_hist":
            x = col.dropna().to_numpy()
            counts, _ = np.histogram(x, bins=spec["edges"])
            total = max(counts.sum(), 1)
            p = np.clip(counts / total, EPS, None)
            return p / p.sum()
        # categorical
        s = col.dropna().astype(str)
        index = spec["index"]
        cats = spec["categories"]
        counts = np.zeros(len(cats) + 1, dtype=float)
        for c in s:
            i = index.get(c)
            counts[-1 if i is None else i] += 1
        total = max(counts.sum(), 1)
        p = np.clip(counts / total, EPS, None)
        return p / p.sum()

    def _z_alert(
        self, js: float, history: deque[float]
    ) -> tuple[float, bool, str]:
        n = len(history)
        if n < MIN_WARMUP:
            return 0.0, False, f"warming up ({n}/{MIN_WARMUP})"
        arr = np.fromiter(history, dtype=float)
        mu = float(arr.mean())
        sigma = float(arr.std(ddof=1)) if n >= 2 else 0.0
        if sigma <= 0:
            return 0.0, False, "zero variance in rolling window"
        z = (js - mu) / sigma
        return z, bool(z > self.alert_sigma), "z-score alert" if z > self.alert_sigma else "ok"
