"""Streamlit shell for the rolling UNG forecast system."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from ung_forecast.ledger import ForecastLedger
from ung_forecast.context_data import combine_context_sources, fetch_cftc_positioning, fetch_eia_storage
from ung_forecast.metals_providers import fetch_xagusd, fetch_xauusd
from ung_forecast.providers import compare_sources, fetch_nasdaq_ung, fetch_yahoo_ung
from ung_forecast.rolling import generate_rolling_forecasts
from ung_forecast.scorecards import build_scorecards
from ung_forecast.weights import compute_model_weights

st.set_page_config(page_title="UNG Forecast", page_icon="📈", layout="wide")
st.title("UNG Rolling Forecast")
st.caption("Daily and five-session research forecasts. Not investment advice.")

run_report_path = Path("data/ledger/latest-run.json")
if run_report_path.exists():
    try:
        persisted_report = json.loads(run_report_path.read_text(encoding="utf-8"))
        st.caption(
            f"Last scheduled run: {persisted_report.get('run_at', 'unknown')} | "
            f"Status: {persisted_report.get('status', 'unknown')}"
        )
    except (OSError, json.JSONDecodeError):
        st.warning("The latest run report exists but could not be read.")

st.info("Forecasts are provisional research outputs until enough matured scorecard history exists.")
selected_models = st.multiselect(
    "Models to run",
    ["ridge", "hist_gradient_boosting", "random_forest", "extra_trees"],
    default=["ridge"],
    help="Ridge is fastest. Tree challengers provide independent scorecards but take longer.",
)
if st.button("Calculate latest forecast", type="primary", use_container_width=True):
    with st.spinner("Fetching validated UNG data and calculating individual forecasts..."):
        try:
            fetched = fetch_nasdaq_ung()
            # Nasdaq is the primary source. Yahoo is an independent health check,
            # but rate limiting there must not prevent a valid primary forecast.
            try:
                confirmation = fetch_yahoo_ung()
                agreement = compare_sources(fetched, confirmation)
                provider_status = "CONFIRMED"
            except Exception as yahoo_exc:
                agreement = f"Nasdaq primary accepted; Yahoo confirmation unavailable: {yahoo_exc}"
                provider_status = "PRIMARY ONLY"
                st.warning(agreement)
            context_frames = []
            context_status = {}
            for name, fetcher in (("EIA storage", fetch_eia_storage), ("CFTC positioning", fetch_cftc_positioning)):
                try:
                    context_frames.append(fetcher())
                    context_status[name] = "AVAILABLE"
                except Exception as context_exc:
                    context_status[name] = f"UNAVAILABLE: {context_exc}"
            ledger = ForecastLedger()
            generated = generate_rolling_forecasts(
                fetched.frame, context=combine_context_sources(context_frames),
                model_names=selected_models or ("ridge",), ledger=ledger
            )
            ledger.score_matured(fetched.frame)
            st.session_state["latest_forecasts"] = generated
            st.session_state["latest_bars"] = fetched.frame.tail(120)
            st.session_state["run_quality"] = {
                "status": "READY" if len(context_frames) == 2 and provider_status == "CONFIRMED" else "PROVISIONAL",
                "provider_status": provider_status,
                "agreement": agreement,
                **context_status,
            }
            st.success(f"Generated {len(generated)} individual model forecasts.")
        except Exception as exc:
            st.error(f"Forecast blocked: {type(exc).__name__}: {exc}")

forecasts = st.session_state.get("latest_forecasts", [])
quality = st.session_state.get("run_quality")
if quality:
    st.subheader("Latest run quality")
    q1, q2, q3 = st.columns(3)
    q1.metric("Status", quality["status"])
    q2.metric("UNG providers", quality.get("provider_status", "UNKNOWN"))
    q3.metric("Context sources", f"{sum(value == 'AVAILABLE' for key, value in quality.items() if key not in ('status', 'agreement'))}/2")
    st.caption(quality["agreement"])
    for key, value in quality.items():
        if key not in ("status", "agreement"):
            st.write(f"**{key}:** {value}")
if forecasts:
    st.subheader("Next five trading sessions")
    st.caption("Low and high show the model's expected trading range. Close is the expected closing price.")
    forecast_columns = {
        "target_session": "Session", "model_name": "Model", "horizon": "Days ahead",
        "predicted_low": "Forecast low", "expected_close": "Forecast close",
        "predicted_high": "Forecast high", "probability_up": "Up probability",
    }
    forecast_table = pd.DataFrame(forecasts)[list(forecast_columns)].rename(columns=forecast_columns)
    forecast_table["Up probability"] = (forecast_table["Up probability"] * 100).round(0).astype(str) + "%"
    for column in ("Forecast low", "Forecast close", "Forecast high"):
        forecast_table[column] = forecast_table[column].round(2)
    st.dataframe(forecast_table, use_container_width=True, hide_index=True)
    chart_rows = {
        row["target_session"]: {
            "predicted_low": row["predicted_low"],
            "expected_close": row["expected_close"],
            "predicted_high": row["predicted_high"],
        }
        for row in forecasts if row["model_name"] == "ridge"
    }
    if chart_rows:
        st.subheader("Forecast range graphic — Ridge")
        st.line_chart(chart_rows, use_container_width=True)
        st.caption("The area between Forecast low and Forecast high is the expected price range; Forecast close is the middle estimate.")
    cards = build_scorecards(ForecastLedger().outcomes())
    st.subheader("Individual model scorecards")
    if cards:
        scorecard_columns = [
            "model_name", "horizon", "matured_forecasts", "range_coverage",
            "mean_abs_low_error", "mean_abs_high_error", "average_range_width", "status",
        ]
        st.dataframe(
            [{key: row.get(key) for key in scorecard_columns} for row in cards],
            use_container_width=True,
            hide_index=True,
        )
        st.caption("Range coverage = percentage of sessions where the actual low and high stayed inside the forecast range.")
        st.caption("Lower low/high errors are better. Narrower ranges are better only when coverage remains high.")
        scorecard_frame = pd.DataFrame(cards)
        st.subheader("Range accuracy at a glance")
        chart_frame = scorecard_frame.pivot(index="model_name", columns="horizon", values="range_coverage")
        chart_frame.columns = [f"{int(value)}d" for value in chart_frame.columns]
        st.bar_chart(chart_frame, y_label="Range coverage (1.0 = 100%)")
    else:
        st.info("No matured forecasts yet; scorecards will populate as target sessions complete.")
    weights = compute_model_weights(ForecastLedger().outcomes())
    if weights:
        st.subheader("Learned model weights")
        st.dataframe(weights, use_container_width=True, hide_index=True)
    else:
        st.info("Model weights remain neutral until sufficient matured out-of-sample evidence exists.")

st.subheader("Planned output")
st.write("Each run displays separate forecasts for each model and horizon; no model is hidden behind an ensemble.")

st.divider()
st.header("Gold and silver research forecasts")
st.caption("Free public research feed: Yahoo COMEX futures proxy (GC=F/SI=F). Validate against your broker's spot feed before trading.")
metal_models = st.multiselect(
    "Metals models",
    ["hist_gradient_boosting", "ridge", "random_forest", "extra_trees"],
    default=["hist_gradient_boosting", "ridge"],
    key="metal_models",
    help="HistGradientBoosting is the nonlinear champion; Ridge is the transparent benchmark.",
)
if st.button("Calculate XAUUSD and XAGUSD forecasts", type="secondary", use_container_width=True):
    with st.spinner("Fetching gold and silver history and calculating five-session ranges..."):
        try:
            metal_results = {}
            for asset, fetcher in (("xauusd", fetch_xauusd), ("xagusd", fetch_xagusd)):
                fetched = fetcher()
                ledger = ForecastLedger(Path("data/ledger") / asset)
                rows = generate_rolling_forecasts(
                    fetched.frame, model_names=metal_models or ("hist_gradient_boosting",),
                    ledger=ledger, asset=asset, min_rows=100,
                )
                ledger.score_matured(fetched.frame)
                metal_results[asset] = {
                    "rows": rows,
                    "latest": str(fetched.frame.index[-1].date()),
                    "provider": fetched.snapshot.provider,
                    "scorecards": build_scorecards(ledger.outcomes()),
                }
            st.session_state["metal_results"] = metal_results
            st.success("Gold and silver forecasts calculated.")
        except Exception as exc:
            st.error(f"Metals forecast blocked: {type(exc).__name__}: {exc}")

metal_results = st.session_state.get("metal_results", {})
for asset, result in metal_results.items():
    st.subheader(asset.upper())
    st.caption(f"Latest completed session: {result['latest']} | Feed: {result['provider']}")
    rows = result["rows"]
    if not rows:
        st.info("No forecast rows available.")
        continue
    metal_table = pd.DataFrame(rows)[[
        "target_session", "model_name", "horizon", "predicted_low", "expected_close",
        "predicted_high", "probability_up",
    ]].rename(columns={
        "target_session": "Session", "model_name": "Model", "horizon": "Days ahead",
        "predicted_low": "Forecast low", "expected_close": "Forecast close",
        "predicted_high": "Forecast high", "probability_up": "Up probability",
    })
    metal_table["Up probability"] = (metal_table["Up probability"] * 100).round(0).astype(str) + "%"
    for column in ("Forecast low", "Forecast close", "Forecast high"):
        metal_table[column] = metal_table[column].round(2)
    st.dataframe(metal_table, use_container_width=True, hide_index=True)
    if result["scorecards"]:
        st.caption("Range coverage is the percentage of completed sessions whose actual low and high stayed inside the forecast range.")
        st.dataframe(result["scorecards"], use_container_width=True, hide_index=True)
    else:
        st.info("Metals scorecards will populate after the forecast target sessions complete.")
