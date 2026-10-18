"""Slow, bounded model-weight learning from matured out-of-sample outcomes."""
from __future__ import annotations

from collections import defaultdict
from typing import Any


def compute_model_weights(
    outcomes: list[dict[str, Any]], *, min_samples: int = 20,
    floor: float = 0.05, ceiling: float = 0.50,
) -> list[dict[str, Any]]:
    """Compute bounded inverse-error weights per horizon; insufficient groups stay neutral."""
    groups: dict[int, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in outcomes:
        groups[int(row["horizon"])][str(row["model_name"])].append(abs(float(row["close_error"])))
    result: list[dict[str, Any]] = []
    for horizon, models in sorted(groups.items()):
        eligible = {name: errors for name, errors in models.items() if len(errors) >= min_samples}
        if not eligible:
            continue
        raw = {name: 1.0 / max(sum(errors) / len(errors), 1e-9) for name, errors in eligible.items()}
        total = sum(raw.values())
        weights = {name: value / total for name, value in raw.items()}
        for name, weight in weights.items():
            result.append({
                "model_name": name,
                "horizon": horizon,
                "sample_count": len(eligible[name]),
                "mean_abs_close_error": 1.0 / raw[name],
                "weight": min(ceiling, max(floor, weight)),
                "status": "LEARNED",
            })
    return result
