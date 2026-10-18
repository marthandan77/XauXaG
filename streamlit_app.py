from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from metals_forecast.ledger import ForecastLedger
from metals_forecast.orderflow import (
    fetch_orderflow,
    load_daily_orderflow_features,
    record_orderflow,
)
from metals_forecast.metals_providers import fetch_xagusd, fetch_xauusd
from metals_forecast.rolling import generate_rolling_forecasts
from metals_forecast.scorecards import build_scorecards

st.set_page_config(page_title="Metals Forecast", page_icon="📈", layout="wide")
st.title("Gold and Silver Rolling Forecast")
st.caption("Five-session research forecasts and live Bybit order flow. Research only.")
st.info(
    "Refreshes completed daily XAUUSD/XAGUSD bars from Dukascopy's public historical feed. "
    "Live DOM and signed trade volume come from Bybit XAUUSDT/XAGUSDT linear perpetual markets. "
    "Both feeds are public and need no API key. Prices and order flow can differ from your broker's feed."
)
models = st.multiselect(
    "Models to run",
    ["hist_gradient_boosting", "ridge", "random_forest", "extra_trees"],
    default=["hist_gradient_boosting", "ridge"],
    help="Each model produces its own forecast and scorecard.",
)

if st.button("Calculate latest forecast", type="primary", use_container_width=True):
    try:
        with st.spinner("Refreshing current Bybit CVD and market depth..."):
            for symbol in ("XAUUSDT", "XAGUSDT"):
                try:
                    live_flow = fetch_orderflow(symbol)
                    record_orderflow(live_flow)
                except Exception as exc:
                    st.warning(f"Could not refresh {symbol} live flow; using saved data if available: {type(exc).__name__}: {exc}")

        orderflow_context = load_daily_orderflow_features()
        with st.spinner("Refreshing Dukascopy prices, processing data, and calculating forecasts..."):
            results = {}
            for asset, fetcher in (("xauusd", fetch_xauusd), ("xagusd", fetch_xagusd)):
                fetched = fetcher()
                ledger = ForecastLedger(Path("data/ledger/dukascopy") / asset)
                rows = generate_rolling_forecasts(
                    fetched.frame,
                    context=orderflow_context[asset] if not orderflow_context[asset].empty else None,
                    model_names=models or ("hist_gradient_boosting",),
                    ledger=ledger,
                    asset=asset,
                    min_rows=100,
                )
                ledger.score_matured(fetched.frame)
                results[asset] = {
                    "rows": rows,
                    "latest": str(fetched.frame.index[-1].date()),
                    "provider": fetched.snapshot.provider,
                    "scorecards": build_scorecards(ledger.outcomes()),
                }
            st.session_state["results"] = results
        st.success("Prices and order flow refreshed; latest forecasts calculated.")
    except Exception as exc:
        st.error(f"Forecast blocked: {type(exc).__name__}: {exc}")

for asset, result in st.session_state.get("results", {}).items():
    st.subheader(asset.upper())
    st.caption(f"Latest completed session: {result['latest']} | Source: {result['provider']}")
    rows = result["rows"]
    if rows:
        columns = {
            "target_session": "Session", "model_name": "Model", "horizon": "Days ahead",
            "predicted_low": "Forecast low", "expected_close": "Forecast close",
            "predicted_high": "Forecast high", "probability_up": "Up probability",
            "orderflow_features_enabled": "Bybit flow used",
            "orderflow_rows_used": "Flow history days",
        }
        table = pd.DataFrame(rows)[list(columns)].rename(columns=columns)
        table["Up probability"] = (table["Up probability"] * 100).round(0).astype(str) + "%"
        for column in ("Forecast low", "Forecast close", "Forecast high"):
            table[column] = table[column].round(2)
        st.dataframe(table, use_container_width=True, hide_index=True)
        st.caption(
            "Bybit flow features enter training only after at least 100 completed sessions contain both CVD and DOM; "
            "until then forecasts use price-history features."
        )
    else:
        st.info("No forecast rows available.")
    if result["scorecards"]:
        st.subheader("Model scorecards")
        st.dataframe(result["scorecards"], use_container_width=True, hide_index=True)
    else:
        st.info("Scorecards will populate after target sessions complete.")


st.header("Live Bybit market depth and order flow")
st.caption(
    "Venue-specific Bybit perpetual data, not a consolidated gold/silver book. CVD is calculated "
    "from the latest 1,000 public trades returned by Bybit; it resets to that rolling sample "
    "on each refresh. Quantities are contract units."
)
refresh_choice = st.selectbox("Refresh interval", ["Manual", "5 seconds", "15 seconds", "30 seconds"])
refresh_interval = {"Manual": None, "5 seconds": "5s", "15 seconds": "15s", "30 seconds": "30s"}[refresh_choice]


@st.fragment(run_every=refresh_interval)
def show_bybit_orderflow() -> None:
    st.button("Refresh Bybit DOM and CVD", key="refresh_bybit_orderflow")
    with st.spinner("Loading Bybit depth and trades..."):
        try:
            current = {}
            for symbol in ("XAUUSDT", "XAGUSDT"):
                current[symbol] = fetch_orderflow(symbol)
                record_orderflow(current[symbol])
            st.session_state["bybit_orderflow"] = current
        except Exception as exc:
            st.error(f"Bybit data unavailable: {type(exc).__name__}: {exc}")
            return

    gold, silver = st.tabs(["Gold · XAUUSDT", "Silver · XAGUSDT"])
    for tab, symbol in ((gold, "XAUUSDT"), (silver, "XAGUSDT")):
        with tab:
            data = st.session_state["bybit_orderflow"][symbol]
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
                "DOM imbalance · 1%", f"{data['depth_bands']['imbalance_1pct'] * 100:+.1f}%"
            )
            metrics[5].metric("CVD · sample", f"{data['cvd_qty']:+,.3f}")
            metrics = st.columns(3)
            metrics[0].metric("Aggressive buy volume", f"{data['buy_qty']:,.3f}")
            metrics[1].metric("Aggressive sell volume", f"{data['sell_qty']:,.3f}")
            metrics[2].metric("Traded notional · USDT", f"{data['total_quote']:,.0f}")
            bid_col, ask_col = st.columns(2)
            with bid_col:
                st.markdown("**Bids**")
                st.dataframe(pd.DataFrame(data["bids"][:20]), use_container_width=True, hide_index=True)
            with ask_col:
                st.markdown("**Asks**")
                st.dataframe(pd.DataFrame(data["asks"][:20]), use_container_width=True, hide_index=True)
            st.caption(
                f"Read at {data['updated_at'].strftime('%Y-%m-%d %H:%M:%S UTC')} · "
                f"CVD uses {data['trade_count']} recent aggregate trades."
            )


show_bybit_orderflow()
