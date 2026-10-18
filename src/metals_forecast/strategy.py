"""Rule-based, risk-defined pullback plans using PAXG proxy market data."""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd


def _model_bias(forecasts: list[dict] | None, atr: float) -> str:
    if not forecasts or atr <= 0:
        return "Unavailable"
    near_term = [row for row in forecasts if int(row.get("horizon", 99)) <= 3]
    if not near_term:
        return "Unavailable"
    changes = [
        float(row["expected_close"]) - float(row["reference_close"])
        for row in near_term
        if row.get("expected_close") is not None and row.get("reference_close") is not None
    ]
    if not changes:
        return "Unavailable"
    score = float(np.median(changes))
    if score > 0.1 * atr:
        return "Bullish"
    if score < -0.1 * atr:
        return "Bearish"
    return "Neutral"


def build_trade_plan(
    prices: pd.DataFrame,
    live_flow: dict | None,
    forecasts: list[dict] | None = None,
) -> dict:
    """Build a conditional EMA pullback plan with structural stops and 1R/2R targets."""
    empty = {
        "bias": "Neutral",
        "setup": "Wait — not enough completed price history",
        "current_price": None,
        "entry_low": None,
        "entry_high": None,
        "stop": None,
        "target_1": None,
        "target_2": None,
        "risk_per_unit": None,
        "atr": None,
        "model_bias": "Unavailable",
        "flow_state": "Unavailable",
        "reasons": ["At least 55 completed daily sessions are needed to assess the trend."],
    }
    if len(prices) < 55:
        return empty

    close = prices["close"].astype(float)
    high = prices["high"].astype(float)
    low = prices["low"].astype(float)
    ema20 = close.ewm(span=20, adjust=False).mean()
    ema50 = close.ewm(span=50, adjust=False).mean()
    true_range = pd.concat(
        [high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1
    ).max(axis=1)
    atr = float(true_range.rolling(14).mean().iloc[-1])
    last_close = float(close.iloc[-1])
    current = last_close
    if live_flow and live_flow.get("bids") and live_flow.get("asks"):
        best_bid = float(live_flow["bids"][0]["Price"])
        best_ask = float(live_flow["asks"][0]["Price"])
        if best_bid > 0 and best_ask >= best_bid:
            current = (best_bid + best_ask) / 2
    if not np.isfinite(atr) or atr <= 0:
        empty["current_price"] = current
        empty["setup"] = "Wait — volatility data is unavailable"
        empty["reasons"] = ["A valid 14-session ATR could not be calculated."]
        return empty

    fast = float(ema20.iloc[-1])
    slow = float(ema50.iloc[-1])
    slow_slope = slow - float(ema50.iloc[-6])
    if fast > slow and slow_slope > 0 and current > slow:
        bias = "Bullish"
        side = "long"
    elif fast < slow and slow_slope < 0 and current < slow:
        bias = "Bearish"
        side = "short"
    else:
        bias = "Neutral"
        side = "neutral"

    model_bias = _model_bias(forecasts, atr)
    flow_state = "Unavailable"
    flow_fresh = False
    flow_ratio = 0.0
    dom_imbalance = 0.0
    if live_flow:
        updated_at = live_flow.get("updated_at")
        if isinstance(updated_at, datetime):
            if updated_at.tzinfo is None:
                updated_at = updated_at.replace(tzinfo=timezone.utc)
            flow_fresh = (datetime.now(timezone.utc) - updated_at.astimezone(timezone.utc)).total_seconds() <= 120
        traded_qty = float(live_flow.get("buy_qty", 0)) + float(live_flow.get("sell_qty", 0))
        flow_ratio = float(live_flow.get("cvd_qty", 0)) / traded_qty if traded_qty else 0.0
        dom_imbalance = float(live_flow.get("depth_bands", {}).get("imbalance_1pct", 0))
        if not flow_fresh:
            flow_state = "Stale — refresh live flow"
        elif flow_ratio >= 0.05 and dom_imbalance >= 0.05:
            flow_state = "Buyers aligned"
        elif flow_ratio <= -0.05 and dom_imbalance <= -0.05:
            flow_state = "Sellers aligned"
        else:
            flow_state = "Mixed / no confirmation"

    plan = {
        "bias": bias,
        "setup": "Wait — no clear trend",
        "current_price": current,
        "entry_low": None,
        "entry_high": None,
        "stop": None,
        "target_1": None,
        "target_2": None,
        "risk_per_unit": None,
        "atr": atr,
        "model_bias": model_bias,
        "flow_state": flow_state,
        "reasons": [],
    }
    if side == "neutral":
        plan["reasons"] = [
            "The 20-session and 50-session trend filters do not agree.",
            "Wait for direction to become clearer before planning an entry.",
        ]
        return plan

    if side == "long":
        entry_low, entry_high = fast - 0.20 * atr, fast + 0.10 * atr
        stop = float(low.iloc[-5:].min()) - 0.15 * atr
        risk = fast - stop
        flow_aligned = flow_state == "Buyers aligned"
        model_conflict = model_bias == "Bearish"
        in_zone = entry_low <= current <= entry_high
        extended = current > entry_high + 0.25 * atr
        trend_reason = "20-session EMA is above a rising 50-session EMA."
        flow_reason = f"Live CVD ratio {flow_ratio:+.0%}; 1% DOM imbalance {dom_imbalance:+.0%}."
        zone_reason = "Price is inside the planned pullback zone." if in_zone else "The entry zone is a pullback toward the 20-session EMA."
        if model_conflict:
            setup = "Wait — ML direction conflicts with the trend"
        elif risk <= 0 or risk > 2.0 * atr:
            setup = "Wait — stop distance is too wide"
        elif extended:
            setup = "Wait — price is extended; do not chase"
        elif not in_zone:
            setup = "Wait for pullback into the entry zone"
        elif not flow_aligned:
            setup = "Wait for buyers to confirm in the zone"
        else:
            setup = "Conditional long — enter only while the zone holds"
        plan.update({
            "setup": setup,
            "entry_low": entry_low,
            "entry_high": entry_high,
            "stop": stop,
            "target_1": fast + risk,
            "target_2": fast + 2 * risk,
            "risk_per_unit": risk,
            "reasons": [trend_reason, zone_reason, flow_reason],
        })
    else:
        entry_low, entry_high = fast - 0.10 * atr, fast + 0.20 * atr
        stop = float(high.iloc[-5:].max()) + 0.15 * atr
        risk = stop - fast
        flow_aligned = flow_state == "Sellers aligned"
        model_conflict = model_bias == "Bullish"
        in_zone = entry_low <= current <= entry_high
        extended = current < entry_low - 0.25 * atr
        trend_reason = "20-session EMA is below a falling 50-session EMA."
        flow_reason = f"Live CVD ratio {flow_ratio:+.0%}; 1% DOM imbalance {dom_imbalance:+.0%}."
        zone_reason = "Price is inside the planned pullback zone." if in_zone else "The entry zone is a pullback toward the 20-session EMA."
        if model_conflict:
            setup = "Wait — ML direction conflicts with the trend"
        elif risk <= 0 or risk > 2.0 * atr:
            setup = "Wait — stop distance is too wide"
        elif extended:
            setup = "Wait — price is extended; do not chase"
        elif not in_zone:
            setup = "Wait for rally into the entry zone"
        elif not flow_aligned:
            setup = "Wait for sellers to confirm in the zone"
        else:
            setup = "Conditional short — enter only while the zone holds"
        plan.update({
            "setup": setup,
            "entry_low": entry_low,
            "entry_high": entry_high,
            "stop": stop,
            "target_1": fast - risk,
            "target_2": fast - 2 * risk,
            "risk_per_unit": risk,
            "reasons": [trend_reason, zone_reason, flow_reason],
        })
    if model_bias in {"Bullish", "Bearish"} and model_bias == bias:
        plan["reasons"].append(f"The short-horizon ML direction also leans {model_bias.lower()}.")
    elif model_bias == "Neutral":
        plan["reasons"].append("The short-horizon ML forecast is neutral; treat it as a caution.")
    return plan
