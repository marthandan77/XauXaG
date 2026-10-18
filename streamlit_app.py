from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from metals_forecast.ledger import ForecastLedger
from metals_forecast.strategy import build_trade_plan
from metals_forecast.orderflow import (
    fetch_orderflow,
    load_daily_orderflow_features,
    record_orderflow,
)
from metals_forecast.coinbase_provider import fetch_paxg_daily
from metals_forecast.rolling import generate_rolling_forecasts
from metals_forecast.scorecards import build_scorecards

st.set_page_config(page_title="PAXG Gold Forecast", layout="wide")
st.title("Gold (PAXG/USD) Rolling Forecast")
st.caption("Five-session research forecasts using Coinbase PAXG/USD prices and order flow. Research only.")
st.info(
    "Uses Coinbase's public PAXG-USD daily candles, level-2 book, and recent trades. "
    "PAXG is a gold-backed token proxy, not spot XAUUSD or your broker's feed. "
    "Coinbase is public and needs no API key. This dashboard forecasts gold only."
)
models = st.multiselect(
    "Models to run",
    ["hist_gradient_boosting", "ridge", "random_forest", "extra_trees"],
    default=["hist_gradient_boosting", "ridge"],
    help="Each model produces its own forecast and scorecard.",
)

if st.button("Calculate latest forecast", type="primary", use_container_width=True):
    try:
        current_flow = None
        with st.spinner("Refreshing current Coinbase CVD and market depth..."):
            for symbol in ("XAUUSD",):
                try:
                    current_flow = fetch_orderflow(symbol)
                    record_orderflow(current_flow)
                    st.session_state["latest_strategy_flow"] = current_flow
                except Exception as exc:
                    st.warning(f"Could not refresh {symbol} live flow; using saved data if available: {type(exc).__name__}: {exc}")
                    current_flow = st.session_state.get("coinbase_orderflow", {}).get(symbol)

        orderflow_context = load_daily_orderflow_features()
        with st.spinner("Refreshing Coinbase PAXG/USD prices, processing data, and calculating forecasts..."):
            fetched = fetch_paxg_daily()
            asset = "xauusd"
            ledger = ForecastLedger(Path("data/ledger/coinbase") / asset)
            newly_scored = ledger.score_matured(fetched.frame)
            flow = orderflow_context.get(asset, pd.DataFrame())
            rows = generate_rolling_forecasts(
                fetched.frame,
                context=flow if not flow.empty else None,
                model_names=models or ("hist_gradient_boosting",),
                ledger=ledger,
                asset=asset,
                min_rows=100,
            )
            trade_plan = build_trade_plan(fetched.frame, current_flow, rows)
            outcomes = ledger.outcomes()
            st.session_state["results"] = {
                asset: {
                    "rows": rows,
                    "latest": str(fetched.frame.index[-1].date()),
                    "provider": fetched.snapshot.provider,
                    "scorecards": build_scorecards(outcomes),
                    "recent_outcomes": outcomes[-20:],
                    "newly_scored": newly_scored,
                    "trade_plan": trade_plan,
                }
            }
        st.success("Forecasts calculated; live order flow refreshed when available.")
    except Exception as exc:
        st.error(f"Forecast blocked: {type(exc).__name__}: {exc}")

for asset, result in st.session_state.get("results", {}).items():
    st.subheader("Gold | PAXG/USD proxy")
    st.caption(f"Latest completed session: {result['latest']} | Source: {result['provider']}")
    plan = result.get("trade_plan")
    if plan:
        st.subheader("Simple trade plan")
        st.caption(
            "Daily trend-pullback setup using Coinbase PAXG/USD. Levels are proxy prices, not your broker's "
            "XAUUSD quotes; verify the equivalent level on your trading platform before acting."
        )
        if plan["setup"].startswith("Conditional"):
            st.success(plan["setup"])
        elif plan["setup"].startswith("Wait"):
            st.warning(plan["setup"])
        status = st.columns(3)
        status[0].metric("Trend bias", plan["bias"])
        status[1].metric("Live CVD + DOM", plan["flow_state"])
        status[2].metric("Short-horizon ML", plan["model_bias"])
        if plan["entry_low"] is not None:
            levels = st.columns(5)
            levels[0].metric(
                "Entry zone", f"{plan['entry_low']:,.2f} – {plan['entry_high']:,.2f}"
            )
            levels[1].metric("Protective stop", f"{plan['stop']:,.2f}")
            levels[2].metric("Target 1 | 1R", f"{plan['target_1']:,.2f}")
            levels[3].metric("Target 2 | 2R", f"{plan['target_2']:,.2f}")
            levels[4].metric("Risk from entry ref.", f"${plan['risk_per_unit']:,.2f} / PAXG")
        st.markdown("**Why:** " + " ".join(plan["reasons"]))
        st.caption(
            "Targets are 1R and 2R from the 20-session EMA entry reference. Set position size from your "
            "account risk and stop distance; wait for price to hold the zone with flow confirmation."
        )
    rows = result["rows"]
    if rows:
        columns = {
            "target_session": "Session", "model_name": "Model", "horizon": "Days ahead",
            "predicted_low": "Forecast low", "expected_close": "Forecast close",
            "predicted_high": "Forecast high", "probability_up": "Up probability",
            "orderflow_features_enabled": "Coinbase flow used",
            "orderflow_rows_used": "Flow history days",
            "calibration_status": "Calibration",
            "calibration_sample_count": "Calibration samples",
            "close_bias_correction": "Close correction ($)",
            "low_bound_expansion": "Low range expansion ($)",
            "high_bound_expansion": "High range expansion ($)",
        }
        table = pd.DataFrame(rows)[list(columns)].rename(columns=columns)
        table["Up probability"] = (table["Up probability"] * 100).round(0).astype(str) + "%"
        for column in ("Forecast low", "Forecast close", "Forecast high"):
            table[column] = table[column].round(2)
        for column in ("Close correction ($)", "Low range expansion ($)", "High range expansion ($)"):
            table[column] = table[column].round(2)
        st.dataframe(table, use_container_width=True, hide_index=True)
        st.caption(
            "Coinbase CVD and DOM features enter training after at least 100 completed sessions contain "
            "both signals; until then, forecasts use PAXG price-history features."
        )
    else:
        st.info("No forecast rows available.")
    if result["scorecards"]:
        st.subheader("Model scoreboard")
        st.caption(
            f"{result['newly_scored']} new outcomes scored this run. Models retrain on completed price history; "
            "after 30 matched outcomes, recent close bias and range misses adjust later forecasts."
        )
        cards = pd.DataFrame(result["scorecards"])
        scoreboard_columns = {
            "model_name": "Model", "horizon": "Days ahead", "data_mode": "Inputs",
            "matured_forecasts": "Scored", "mean_abs_close_error": "Close MAE ($)",
            "direction_accuracy": "Direction accuracy", "range_coverage": "Full-range coverage",
            "lower_bound_miss_rate": "Low-side misses", "upper_bound_miss_rate": "High-side misses",
            "recent_20_abs_close_error": "Recent 20 MAE ($)",
            "recent_20_range_coverage": "Recent 20 coverage",
        }
        visible = cards[list(scoreboard_columns)].rename(columns=scoreboard_columns)
        percentage_columns = (
            "Direction accuracy", "Full-range coverage", "Low-side misses", "High-side misses",
            "Recent 20 coverage",
        )
        for column in percentage_columns:
            visible[column] = visible[column].map(
                lambda value: "—" if pd.isna(value) else f"{value:.0%}"
            )
        for column in ("Close MAE ($)", "Recent 20 MAE ($)"):
            visible[column] = visible[column].round(2)
        st.dataframe(visible, use_container_width=True, hide_index=True)

        detailed = [
            row for row in result.get("recent_outcomes", [])
            if "actual_low" in row and "actual_high" in row
        ]
        if detailed:
            st.markdown("**Recent forecast range vs actual range**")
            comparison = pd.DataFrame(detailed)
            comparison["Forecast range"] = comparison.apply(
                lambda row: f"{row['predicted_low']:,.2f} - {row['predicted_high']:,.2f}", axis=1
            )
            comparison["Actual range"] = comparison.apply(
                lambda row: f"{row['actual_low']:,.2f} - {row['actual_high']:,.2f}", axis=1
            )
            comparison = comparison[[
                "target_session", "model_name", "horizon", "Forecast range", "Actual range",
                "close_error", "close_diagnosis", "range_miss_diagnosis",
            ]].rename(columns={
                "target_session": "Session", "model_name": "Model", "horizon": "Days ahead",
                "close_error": "Close error ($)", "close_diagnosis": "Close diagnosis",
                "range_miss_diagnosis": "Range diagnosis",
            })
            comparison["Close error ($)"] = comparison["Close error ($)"].round(2)
            st.dataframe(comparison, use_container_width=True, hide_index=True)
            st.caption(
                "Positive close error means the forecast was too high; negative means it was too low. "
                "Low-side and high-side miss rates show which range boundary needs adjustment."
            )
    else:
        st.info("Scorecards will populate after target sessions complete.")


st.header("Live Coinbase market depth and order flow")
st.caption(
    "Coinbase Exchange PAXG/USD proxy data, not spot XAUUSD or a consolidated gold book. CVD is calculated "
    "from the latest 1,000 public trades returned by Coinbase Exchange; it resets to that rolling sample "
    "on each refresh. Quantities are PAXG units."
)
refresh_choice = st.selectbox("Refresh interval", ["Manual", "5 seconds", "15 seconds", "30 seconds"])
refresh_interval = {"Manual": None, "5 seconds": "5s", "15 seconds": "15s", "30 seconds": "30s"}[refresh_choice]


@st.fragment(run_every=refresh_interval)
def show_coinbase_orderflow() -> None:
    st.button("Refresh Coinbase DOM and CVD", key="refresh_coinbase_orderflow")
    with st.spinner("Loading Coinbase depth and trades..."):
        try:
            current = {}
            for symbol in ("XAUUSD",):
                current[symbol] = fetch_orderflow(symbol)
                record_orderflow(current[symbol])
            st.session_state["coinbase_orderflow"] = current
        except Exception as exc:
            st.error(f"Coinbase data unavailable: {type(exc).__name__}: {exc}")
            return

    gold, = st.tabs(["Gold | PAXG-USD"])
    for tab, symbol in ((gold, "XAUUSD"),):
        with tab:
            data = st.session_state["coinbase_orderflow"][symbol]
            best_bid = data["bids"][0]["Price"] if data["bids"] else 0
            best_ask = data["asks"][0]["Price"] if data["asks"] else 0
            spread = best_ask - best_bid if best_bid and best_ask else 0
            bid_depth = sum(level["Quantity"] for level in data["bids"][:10])
            ask_depth = sum(level["Quantity"] for level in data["asks"][:10])
            depth_total = bid_depth + ask_depth
            imbalance = (bid_depth - ask_depth) / depth_total * 100 if depth_total else 0

            metrics = st.columns(6)
            metrics[0].metric("Best bid", f"{best_bid:,.3f}")
            metrics[1].metric("Best ask", f"{best_ask:,.3f}")
            metrics[2].metric("Spread", f"{spread:,.3f}")
            metrics[3].metric("Top 10 depth imbalance", f"{imbalance:+.1f}%")
            metrics[4].metric(
                "DOM imbalance | 1%", f"{data['depth_bands']['imbalance_1pct'] * 100:+.1f}%"
            )
            metrics[5].metric("CVD | sample", f"{data['cvd_qty']:+,.3f}")
            metrics = st.columns(3)
            metrics[0].metric("Aggressive buy volume", f"{data['buy_qty']:,.3f}")
            metrics[1].metric("Aggressive sell volume", f"{data['sell_qty']:,.3f}")
            metrics[2].metric("Traded notional | USD", f"{data['total_quote']:,.0f}")
            bid_col, ask_col = st.columns(2)
            with bid_col:
                st.markdown("**Bids**")
                st.dataframe(pd.DataFrame(data["bids"][:20]), use_container_width=True, hide_index=True)
            with ask_col:
                st.markdown("**Asks**")
                st.dataframe(pd.DataFrame(data["asks"][:20]), use_container_width=True, hide_index=True)
            st.caption(
                f"Read at {data['updated_at'].strftime('%Y-%m-%d %H:%M:%S UTC')} | "
                f"CVD uses {data['trade_count']} recent aggregate trades."
            )


show_coinbase_orderflow()
