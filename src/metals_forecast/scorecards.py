"""Independent model scorecards and empirical range calibration."""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Any


def build_scorecards(outcomes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, int, bool], list[dict[str, Any]]] = defaultdict(list)
    for row in outcomes:
        groups[(
            str(row["model_name"]), int(row["horizon"]),
            bool(row.get("orderflow_features_enabled", False)),
        )].append(row)
    result = []
    for (model_name, horizon, flow_enabled), rows in sorted(groups.items()):
        n = len(rows)
        def metrics(sample: list[dict[str, Any]]) -> dict[str, float]:
            count = len(sample)
            return {
                "mean_abs_close_error": sum(abs(float(r["close_error"])) for r in sample) / count,
                "range_coverage": sum(bool(r["range_contains_day"]) for r in sample) / count,
                "average_range_width": sum(float(r.get("predicted_range_width", 0.0)) for r in sample) / count,
            }
        all_metrics = metrics(rows)
        recent_20 = metrics(rows[-20:]) if len(rows) >= 20 else {"mean_abs_close_error": None, "range_coverage": None}
        recent_60 = metrics(rows[-60:]) if len(rows) >= 60 else {"mean_abs_close_error": None, "range_coverage": None}
        result.append({
            "model_name": model_name,
            "horizon": horizon,
            "data_mode": "flow_enhanced" if flow_enabled else "price_only",
            "matured_forecasts": n,
            "mean_abs_close_error": all_metrics["mean_abs_close_error"],
            "mean_abs_low_error": sum(abs(float(r["low_error"])) for r in rows) / n,
            "mean_abs_high_error": sum(abs(float(r["high_error"])) for r in rows) / n,
            "range_coverage": all_metrics["range_coverage"],
            "average_range_width": all_metrics["average_range_width"],
            "close_coverage": sum(bool(r.get("range_contains_close", False)) for r in rows) / n,
            "recent_20_abs_close_error": recent_20["mean_abs_close_error"],
            "recent_20_range_coverage": recent_20["range_coverage"],
            "recent_60_abs_close_error": recent_60["mean_abs_close_error"],
            "recent_60_range_coverage": recent_60["range_coverage"],
            "status": "INSUFFICIENT_DATA" if n < 30 else "EVALUATED",
        })
    return result


def empirical_range_calibration(
    outcomes: list[dict[str, Any]], *, model_name: str, horizon: int, minimum: int = 30,
) -> dict[str, float | int | str]:
    """Return a conservative calibration summary from matured out-of-sample errors."""
    rows = [r for r in outcomes if r.get("model_name") == model_name and int(r.get("horizon", -1)) == horizon]
    if len(rows) < minimum:
        return {"status": "INSUFFICIENT_DATA", "sample_count": len(rows)}
    low_errors = sorted(float(r["low_error"]) for r in rows)
    high_errors = sorted(float(r["high_error"]) for r in rows)
    low_index = max(0, math.floor(0.1 * (len(low_errors) - 1)))
    high_index = min(len(high_errors) - 1, math.ceil(0.9 * (len(high_errors) - 1)))
    return {
        "status": "CALIBRATED",
        "sample_count": len(rows),
        "low_error_p10": low_errors[low_index],
        "high_error_p90": high_errors[high_index],
    }
