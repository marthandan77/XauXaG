"""Empirical, out-of-sample calibration for individual forecast ranges."""
from __future__ import annotations

from typing import Any


def calibrate_range(
    forecast: dict[str, Any], outcomes: list[dict[str, Any]], *, minimum: int = 30,
) -> dict[str, Any]:
    """Adjust bounds only when enough same-model/same-horizon outcomes exist."""
    rows = [
        row for row in outcomes
        if row.get("model_name") == forecast.get("model_name")
        and int(row.get("horizon", -1)) == int(forecast.get("horizon", -2))
        and bool(row.get("orderflow_features_enabled", False))
        == bool(forecast.get("orderflow_features_enabled", False))
    ]
    result = dict(forecast)
    result["calibration_sample_count"] = len(rows)
    if len(rows) < minimum:
        result["calibration_status"] = "UNCALIBRATED"
        return result
    low_errors = sorted(float(row["low_error"]) for row in rows)
    high_errors = sorted(float(row["high_error"]) for row in rows)
    low_adjustment = low_errors[max(0, int(0.10 * (len(low_errors) - 1)))]
    high_adjustment = high_errors[min(len(high_errors) - 1, int(0.90 * (len(high_errors) - 1)))]
    # Actual - forecast error: lower bounds move down for negative errors;
    # upper bounds move up for positive errors.
    result["predicted_low"] = max(0.01, float(result["predicted_low"]) + min(low_adjustment, 0.0))
    result["predicted_high"] = max(result["predicted_low"], float(result["predicted_high"]) + max(high_adjustment, 0.0))
    result["calibration_status"] = "CALIBRATED"
    return result
