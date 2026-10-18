"""Rolling five-session individual-model forecast engine."""
from __future__ import annotations

import hashlib
import os
from datetime import datetime, timezone
from typing import Iterable

import pandas as pd

from .features import build_features, build_forward_targets
from .calibration import calibrate_range
from .ledger import ForecastLedger
from .models import fit_individual_forecast
from .sessions import next_sessions


def generate_rolling_forecasts(
    prices: pd.DataFrame,
    *,
    context: pd.DataFrame | None = None,
    model_names: Iterable[str] | None = None,
    ledger: ForecastLedger | None = None,
    min_rows: int = 100,
    asset: str = "asset",
) -> list[dict]:
    """Generate one independent forecast per model and future session."""
    prices = prices.sort_index()
    if model_names is None:
        configured = os.getenv("FORECAST_MODELS", "ridge")
        model_names = tuple(name.strip() for name in configured.split(",") if name.strip())
    features = build_features(prices, context=context)
    latest_session = prices.index[-1]
    issued_at = datetime.now(timezone.utc).isoformat()
    output: list[dict] = []
    future_sessions = next_sessions(latest_session, 5)
    for horizon in (1, 2, 3, 4, 5):
        target_session = future_sessions[horizon - 1].date().isoformat()
        targets = build_forward_targets(prices, horizon)
        for model_name in model_names:
            forecast = fit_individual_forecast(
                model_name, features, targets, reference_close=float(prices.iloc[-1].close),
                target_session=target_session, horizon=horizon, min_rows=min_rows,
            )
            identity = (
                f"{latest_session.date()}|{target_session}|{model_name}|{horizon}|"
                f"flow:{int(forecast.orderflow_features_enabled)}"
            )
            record = {
                "forecast_id": asset.upper() + "-" + hashlib.sha256(identity.encode()).hexdigest()[:20],
                "issued_at": issued_at,
                "origin_session": latest_session.date().isoformat(),
                "target_session": forecast.target_session,
                "horizon": forecast.horizon,
                "model_name": forecast.model_name,
                "reference_close": float(prices.iloc[-1].close),
                "predicted_low": forecast.predicted_low,
                "expected_close": forecast.expected_close,
                "predicted_high": forecast.predicted_high,
                "probability_up": forecast.probability_up,
                "uncertainty": forecast.uncertainty,
                "orderflow_rows_used": forecast.orderflow_rows_used,
                "orderflow_features_enabled": forecast.orderflow_features_enabled,
                "status": "PROVISIONAL",
            }
            output.append(record)
            if ledger is not None:
                record = calibrate_range(record, ledger.outcomes())
                output[-1] = record
            if ledger is not None:
                ledger.append_forecast(record)
    return output
