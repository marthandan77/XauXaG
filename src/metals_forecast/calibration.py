"""Empirical, out-of-sample calibration for individual forecast ranges."""
from __future__ import annotations

import math
from typing import Any


def _quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def calibrate_range(
    forecast: dict[str, Any], outcomes: list[dict[str, Any]], *, minimum: int = 30,
) -> dict[str, Any]:
    """Use matured residuals to correct close bias and target range coverage."""
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
        result["close_bias_correction"] = 0.0
        result["low_bound_expansion"] = 0.0
        result["high_bound_expansion"] = 0.0
        return result

    # Use recent realized forecasts so calibration can adapt as market behavior changes.
    recent = rows[-60:]
    close_errors = [float(row["close_error"]) for row in recent if "close_error" in row]
    low_errors = [float(row["low_error"]) for row in recent]
    high_errors = [float(row["high_error"]) for row in recent]
    close_bias_correction = -_quantile(close_errors, 0.5) if close_errors else 0.0

    # Errors are forecast minus actual. Positive lower-bound errors mean the
    # forecast low was too high; negative upper-bound errors mean its high was
    # too low. Expand only in the direction required for 90% one-sided coverage.
    low_expansion = max(_quantile(low_errors, 0.9), 0.0)
    high_expansion = max(-_quantile(high_errors, 0.1), 0.0)
    if "expected_close" in result:
        result["expected_close"] = max(0.01, float(result["expected_close"]) + close_bias_correction)
    result["predicted_low"] = max(
        0.01,
        float(result["predicted_low"]) + close_bias_correction - low_expansion,
    )
    result["predicted_high"] = max(
        result["predicted_low"],
        float(result["predicted_high"]) + close_bias_correction + high_expansion,
    )
    result["close_bias_correction"] = close_bias_correction
    result["low_bound_expansion"] = low_expansion
    result["high_bound_expansion"] = high_expansion
    result["calibration_status"] = "CALIBRATED"
    return result
