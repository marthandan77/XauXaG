"""Point-in-time features and rolling five-session targets."""
from __future__ import annotations

import numpy as np
import pandas as pd


def build_features(
    prices: pd.DataFrame,
    context: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build features using data available on each instrument session only."""
    df = prices.copy().sort_index()
    close, high, low, volume = df["close"], df["high"], df["low"], df["volume"]
    out = pd.DataFrame(index=df.index)
    for window in (1, 3, 5, 10, 20):
        out[f"return_{window}"] = close.pct_change(window)
    for window in (5, 10, 20):
        out[f"range_{window}"] = (high.rolling(window).max() - low.rolling(window).min()) / close
        out[f"realized_vol_{window}"] = close.pct_change().rolling(window).std() * np.sqrt(252)
    for window in (5, 10, 20, 50):
        ema = close.ewm(span=window, adjust=False).mean()
        out[f"price_vs_ema_{window}"] = close / ema - 1.0
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean().replace(0, np.nan)
    out["rsi_14"] = 100 - (100 / (1 + gain / loss))
    tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
    out["atr_14_pct"] = tr.rolling(14).mean() / close
    out["volume_ratio_20"] = volume / volume.rolling(20).mean()
    out["volume_zscore_20"] = (volume - volume.rolling(20).mean()) / volume.rolling(20).std()
    if context is not None:
        if "available_at" not in context.columns:
            raise ValueError("context must contain available_at")
        available = context.sort_values("available_at").copy()
        # Daily price rows represent completed sessions. Treat the row's data as
        # available at that session's end, so same-day context is usable without
        # letting future context leak into earlier rows.
        left = pd.DataFrame({"session_close": pd.DatetimeIndex(out.index) + pd.Timedelta(days=1)})
        left = left.sort_values("session_close")
        available["available_at"] = pd.to_datetime(available["available_at"], utc=True).dt.tz_convert(None)
        right = available.reset_index(drop=False)
        merged = pd.merge_asof(
            left, right, left_on="session_close", right_on="available_at", direction="backward"
        )
        merged.index = out.index
        for column in merged.columns:
            if column not in {"session_close", "session", "available_at", "index"}:
                out[column] = merged[column].to_numpy()
        flow_columns = [column for column in out.columns if column.startswith("flow_")]
        if flow_columns:
            same_session = merged["available_at"].eq(merged["session_close"]).to_numpy()
            out.loc[~same_session, flow_columns] = np.nan
            if {"flow_cvd_qty", "flow_total_qty", "flow_dom_imbalance_1pct"}.issubset(out.columns):
                out["flow_cvd_ratio"] = out["flow_cvd_qty"] / out["flow_total_qty"].replace(0, np.nan)
                for window in (3, 5):
                    cvd_sum = out["flow_cvd_qty"].rolling(window, min_periods=window).sum()
                    volume_sum = out["flow_total_qty"].rolling(window, min_periods=window).sum()
                    out[f"flow_cvd_ratio_{window}d"] = cvd_sum / volume_sum.replace(0, np.nan)
                    out[f"flow_dom_imbalance_1pct_{window}d"] = (
                        out["flow_dom_imbalance_1pct"].rolling(window, min_periods=window).mean()
                    )
                out["flow_dom_imbalance_1pct_change"] = out["flow_dom_imbalance_1pct"].diff()
    return out.replace([np.inf, -np.inf], np.nan)


def build_forward_targets(prices: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Create future targets; rows near the end remain NaN until outcomes mature."""
    if horizon not in (1, 2, 3, 4, 5):
        raise ValueError("horizon must be between 1 and 5")
    close = prices["close"]
    future_close = close.shift(-horizon)
    future_low = prices["low"].shift(-1).rolling(horizon, min_periods=horizon).min().shift(-(horizon - 1))
    future_high = prices["high"].shift(-1).rolling(horizon, min_periods=horizon).max().shift(-(horizon - 1))
    return pd.DataFrame({
        "forward_return": future_close / close - 1,
        "forward_low_return": future_low / close - 1,
        "forward_high_return": future_high / close - 1,
    }, index=prices.index)
