"""Individual forecasting models and scorecard primitives."""
from __future__ import annotations

from dataclasses import dataclass

import math
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True, slots=True)
class ModelForecast:
    model_name: str
    horizon: int
    target_session: str
    predicted_low: float
    expected_close: float
    predicted_high: float
    probability_up: float
    uncertainty: float
    orderflow_rows_used: int = 0
    orderflow_features_enabled: bool = False


def _model(name: str, quantile: float | None = None) -> Pipeline:
    if name == "ridge":
        estimator = Ridge(alpha=10.0)
        scale = True
    elif name == "hist_gradient_boosting":
        estimator = HistGradientBoostingRegressor(
            loss="quantile" if quantile is not None else "squared_error",
            quantile=quantile, max_iter=180, learning_rate=0.04,
            max_leaf_nodes=15, l2_regularization=2.0, random_state=42,
        )
        scale = False
    elif name == "random_forest":
        estimator = RandomForestRegressor(
            n_estimators=80, min_samples_leaf=5, max_features=0.8,
            random_state=42, n_jobs=1,
        )
        scale = False
    elif name == "extra_trees":
        estimator = ExtraTreesRegressor(
            n_estimators=80, min_samples_leaf=5, max_features=0.8,
            random_state=42, n_jobs=1,
        )
        scale = False
    else:
        raise ValueError(f"unknown model: {name}")
    steps: list[tuple[str, object]] = [("imputer", SimpleImputer(strategy="median", add_indicator=True))]
    if scale:
        steps.append(("scale", StandardScaler()))
    steps.append(("model", estimator))
    return Pipeline(steps)


def fit_individual_forecast(
    model_name: str, features: pd.DataFrame, targets: pd.DataFrame,
    *, reference_close: float, target_session: str, horizon: int, min_rows: int = 100,
) -> ModelForecast:
    joined = features.join(targets).dropna(subset=["forward_low_return", "forward_return", "forward_high_return"])
    if len(joined) < min_rows:
        raise ValueError(f"{model_name}: need {min_rows} rows, have {len(joined)}")
    orderflow_columns = [column for column in features.columns if column.startswith("flow_")]
    core_flow_columns = [
        column for column in ("flow_cvd_qty", "flow_dom_imbalance_1pct") if column in orderflow_columns
    ]
    orderflow_rows = (
        int(features.loc[joined.index, core_flow_columns].notna().all(axis=1).sum())
        if len(core_flow_columns) == 2 else 0
    )
    latest_has_orderflow = (
        bool(features.iloc[[-1]][core_flow_columns].notna().all(axis=1).iloc[0])
        if len(core_flow_columns) == 2 else False
    )
    # Use a clean same-session order-flow sample. Do not let thousands of older
    # price-only rows with imputed flow values dominate this model.
    orderflow_enabled = orderflow_rows >= max(min_rows, 100) and latest_has_orderflow
    if orderflow_enabled:
        flow_mask = features.loc[joined.index, core_flow_columns].notna().all(axis=1)
        training = joined.loc[flow_mask]
    else:
        training = joined
    usable_columns = [
        column for column in features.columns
        if training[column].notna().any() and (orderflow_enabled or not column.startswith("flow_"))
    ]
    x = training[usable_columns]
    latest = features.iloc[[-1]][usable_columns]
    models = {}
    close_residuals = None
    for key, target in (("low", "forward_low_return"), ("close", "forward_return"), ("high", "forward_high_return")):
        model = _model(model_name, 0.1 if key == "low" else 0.9 if key == "high" else None)
        model.fit(x, training[target])
        models[key] = float(model.predict(latest)[0])
        if key == "close":
            close_residuals = training[target].to_numpy(dtype=float) - model.predict(x)
    low = max(0.01, reference_close * (1 + models["low"]))
    close = max(0.01, reference_close * (1 + models["close"]))
    high = max(0.01, reference_close * (1 + models["high"]))
    low, high = min(low, high), max(low, high)
    residual_scale = max(float(np.nanstd(close_residuals)) if close_residuals is not None else 0.0, 1e-6)
    predicted_return = models["close"]
    probability_up = 0.5 * (1.0 + math.erf(predicted_return / (residual_scale * math.sqrt(2.0))))
    probability_up = float(np.clip(probability_up, 0.01, 0.99))
    uncertainty = float(np.clip(residual_scale + (high - low) / max(close, 1e-6), 0.0, 1.0))
    return ModelForecast(
        model_name, horizon, target_session, low, close, high, probability_up, uncertainty,
        orderflow_rows, orderflow_enabled,
    )


def score_forecast(
    forecast: dict, actual_low: float, actual_close: float, actual_high: float
) -> dict[str, float | bool | str]:
    """Score one matured forecast without changing the original forecast record."""
    predicted_low = float(forecast["predicted_low"])
    predicted_close = float(forecast["expected_close"])
    predicted_high = float(forecast["predicted_high"])
    low_error = predicted_low - actual_low
    close_error = predicted_close - actual_close
    high_error = predicted_high - actual_high
    result: dict[str, float | bool | str] = {
        "predicted_low": predicted_low,
        "predicted_close": predicted_close,
        "predicted_high": predicted_high,
        "actual_low": actual_low,
        "actual_close": actual_close,
        "actual_high": actual_high,
        "low_error": low_error,
        "close_error": close_error,
        "high_error": high_error,
        "predicted_range_width": predicted_high - predicted_low,
        "actual_range_width": actual_high - actual_low,
        "range_contains_close": bool(predicted_low <= actual_close <= predicted_high),
        "range_contains_day": bool(predicted_low <= actual_low and actual_high <= predicted_high),
        "close_diagnosis": "OVERPREDICTED" if close_error > 0 else "UNDERPREDICTED" if close_error < 0 else "EXACT",
    }
    reference = forecast.get("reference_close")
    if reference is not None:
        predicted_direction = predicted_close > float(reference)
        actual_direction = actual_close > float(reference)
        result["direction_correct"] = bool(predicted_direction == actual_direction)
    low_missed = low_error > 0
    high_missed = high_error < 0
    if low_missed and high_missed:
        result["range_miss_diagnosis"] = "TOO_NARROW_BOTH_SIDES"
    elif low_missed:
        result["range_miss_diagnosis"] = "LOWER_BOUND_TOO_HIGH"
    elif high_missed:
        result["range_miss_diagnosis"] = "UPPER_BOUND_TOO_LOW"
    else:
        result["range_miss_diagnosis"] = "RANGE_CONTAINED_DAY"
    return result


def walk_forward_evaluate(
    model_name: str, features: pd.DataFrame, targets: pd.DataFrame, *,
    horizon: int, min_train_rows: int = 100, step: int = 5,
) -> list[dict[str, float | str | int | bool]]:
    """Evaluate an individual model using expanding-window, future-blind forecasts."""
    joined = features.join(targets)
    rows: list[dict[str, float | str | int | bool]] = []
    for index in range(min_train_rows, len(joined), step):
        train_end = max(0, index - horizon)
        train_features = features.iloc[:train_end]
        train_targets = targets.iloc[:train_end]
        test = joined.iloc[index:index + step].dropna(
            subset=["forward_low_return", "forward_return", "forward_high_return"]
        )
        if test.empty:
            continue
        # Fit only on data strictly before the forecast block. The target rows at
        # the boundary are naturally excluded if their future horizon is incomplete.
        train = train_features.join(train_targets).dropna(
            subset=["forward_low_return", "forward_return", "forward_high_return"]
        )
        if len(train) < min_train_rows:
            continue
        usable = [column for column in features.columns if train[column].notna().any()]
        x = train[usable]
        models = {}
        for key, target in (("low", "forward_low_return"), ("close", "forward_return"), ("high", "forward_high_return")):
            model = _model(model_name, 0.1 if key == "low" else 0.9 if key == "high" else None)
            model.fit(x, train[target])
            models[key] = model
        predictions = {key: model.predict(features.loc[test.index, usable]) for key, model in models.items()}
        for position, timestamp in enumerate(test.index):
            reference = float(joined.loc[timestamp, "forward_return"])
            rows.append({
                "model_name": model_name,
                "horizon": horizon,
                "target_session": timestamp.isoformat(),
                "predicted_return": float(predictions["close"][position]),
                "actual_return": reference,
                "absolute_error": abs(float(predictions["close"][position]) - reference),
                "direction_correct": bool((predictions["close"][position] > 0) == (reference > 0)),
            })
    return rows
