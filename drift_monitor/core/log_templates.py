"""Log template frequency-shift monitor — Layer 1 evidence health.

White paper §3.7:

    p_i = count(t_i) / Σ_j count(t_j)         per-template frequency
    JS_log  = JS(p_curr || p_base)            with rolling z-score alerting
    PSI_log = Σ_i (p_curr_i - p_base_i) ln(p_curr_i / p_base_i)
    new templates -> unconditional alert (parser change OR new failure mode)

Normalization replaces variable tokens with typed placeholders so logically
identical events collapse to the same template:

    - timestamps  -> <TS>
    - IPv4/IPv6   -> <IP>
    - UUIDs       -> <UUID>
    - hex blobs   -> <HEX>
    - numerics    -> <NUM>

If ``drain3`` is installed, it is used for parsing instead. Pass
``use_drain=False`` to force the regex normalizer (used in tests for
determinism).
"""

from __future__ import annotations

import re
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import pandas as pd

from drift_monitor.core.divergence import MIN_WARMUP, _js
from drift_monitor.core.psi import EPS as PSI_EPS

_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z?"), "<TS>"),
    (re.compile(r"\b\d{4}-\d{2}-\d{2}\b"), "<TS>"),
    (re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"), "<IP>"),
    (re.compile(r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"), "<UUID>"),
    (re.compile(r"\b0x[0-9a-fA-F]+\b"), "<HEX>"),
    (re.compile(r"\b\d+(?:\.\d+)?(?:ms|s|Mbps|Gbps|MB|GB|KB|%)?\b"), "<NUM>"),
]


def normalize_log(line: str) -> str:
    out = line
    for pat, repl in _PATTERNS:
        out = pat.sub(repl, out)
    return " ".join(out.split())


@dataclass
class LogTemplateResult:
    entity_key: str
    js_score: float
    psi_score: float
    z_score: float
    alert: bool
    new_templates: list[str]
    top_shifted: pd.DataFrame = field(repr=False)
    reason: str = ""


class LogTemplateMonitor:
    def __init__(
        self,
        alert_sigma: float = 3.0,
        rolling_window: int = 30,
        use_drain: bool = True,
    ):
        if rolling_window < MIN_WARMUP:
            raise ValueError(f"rolling_window must be >= {MIN_WARMUP}")
        self.alert_sigma = alert_sigma
        self.rolling_window = rolling_window
        self._drain_miners: dict[str, "object"] = {}
        self._use_drain = use_drain and self._try_import_drain()
        self._baselines: dict[str, dict] = {}
        self._history: dict[str, deque[float]] = {}

    @staticmethod
    def _try_import_drain() -> bool:
        try:
            import drain3  # noqa: F401
            return True
        except Exception:
            return False

    def _parse(self, lines: Iterable[str], entity_key: str) -> list[str]:
        if self._use_drain:
            from drain3 import TemplateMiner
            miner = self._drain_miners.setdefault(entity_key, TemplateMiner())
            templates: list[str] = []
            for line in lines:
                res = miner.add_log_message(line)
                templates.append(res["template_mined"])
            return templates
        return [normalize_log(line) for line in lines]

    def fit(self, baseline_logs: Iterable[str], entity_key: str) -> None:
        templates = self._parse(baseline_logs, entity_key)
        counts = Counter(templates)
        if not counts:
            raise ValueError("no baseline log templates")
        vocab = sorted(counts)
        index = {t: i for i, t in enumerate(vocab)}
        total = sum(counts.values())
        p_base = np.array([counts[t] / total for t in vocab], dtype=float)
        p_base = np.clip(p_base, PSI_EPS, None)
        p_base = p_base / p_base.sum()
        self._baselines[entity_key] = {
            "vocab": vocab,
            "index": index,
            "p_base": p_base,
        }
        self._history[entity_key] = deque(maxlen=self.rolling_window)

    def compute(self, prod_logs: Iterable[str], entity_key: str) -> LogTemplateResult:
        if entity_key not in self._baselines:
            raise KeyError(f"entity {entity_key!r} not registered")
        spec = self._baselines[entity_key]
        templates = self._parse(prod_logs, entity_key)
        counts = Counter(templates)
        total = sum(counts.values()) or 1
        vocab = spec["vocab"]
        index = spec["index"]
        p_base = spec["p_base"]

        # Build current frequency vector aligned to the baseline vocabulary,
        # plus a single bucket for new (unseen) templates.
        curr = np.zeros(len(vocab) + 1, dtype=float)
        new_counts: Counter[str] = Counter()
        for t, c in counts.items():
            i = index.get(t)
            if i is None:
                new_counts[t] += c
                curr[-1] += c
            else:
                curr[i] += c
        p_curr = np.clip(curr / total, PSI_EPS, None)
        p_curr = p_curr / p_curr.sum()

        # Pad p_base with an EPS slot for the "new templates" bucket so the
        # vectors are the same length.
        p_base_padded = np.concatenate([p_base, [PSI_EPS]])
        p_base_padded = p_base_padded / p_base_padded.sum()

        js = _js(p_curr, p_base_padded)
        psi = float(np.sum((p_curr - p_base_padded) * np.log(p_curr / p_base_padded)))

        history = self._history[entity_key]
        n = len(history)
        if n < MIN_WARMUP:
            z = 0.0
            z_alert = False
            reason = f"warming up ({n}/{MIN_WARMUP})"
        else:
            arr = np.fromiter(history, dtype=float)
            sigma = float(arr.std(ddof=1)) if n >= 2 else 0.0
            mu = float(arr.mean())
            z = (js - mu) / sigma if sigma > 0 else 0.0
            z_alert = z > self.alert_sigma
            reason = "z-score alert" if z_alert else "ok"
        history.append(js)

        new_templates = sorted(new_counts, key=new_counts.get, reverse=True)
        unconditional = bool(new_templates)
        alert = z_alert or unconditional
        if unconditional and not z_alert:
            reason = "new templates detected"

        # Top-shifted by |Δp_i| over the baseline vocabulary (excluding the
        # synthetic "new templates" bucket).
        labels = list(vocab) + ["__NEW__"]
        delta = p_curr - p_base_padded
        top = pd.DataFrame(
            {
                "template": labels,
                "p_base": p_base_padded,
                "p_curr": p_curr,
                "delta": delta,
                "abs_delta": np.abs(delta),
            }
        ).sort_values("abs_delta", ascending=False).reset_index(drop=True)

        return LogTemplateResult(
            entity_key=entity_key,
            js_score=js,
            psi_score=psi,
            z_score=z,
            alert=alert,
            new_templates=new_templates,
            top_shifted=top,
            reason=reason,
        )
