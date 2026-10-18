"""Refresh Coinbase PAXG/USD and run the daily five-session gold forecast."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from metals_forecast.orderflow import fetch_orderflow, load_daily_orderflow_features, record_orderflow
from metals_forecast.ledger import ForecastLedger
from metals_forecast.coinbase_provider import fetch_paxg_daily
from metals_forecast.rolling import generate_rolling_forecasts


def main() -> None:
    report: dict = {"run_at": datetime.now(timezone.utc).isoformat(), "assets": {}}
    try:
        record_orderflow(fetch_orderflow("XAUUSD"))
    except Exception as exc:
        report["orderflow_refresh_error"] = f"{type(exc).__name__}: {exc}"
    orderflow_context = load_daily_orderflow_features()
    asset = "xauusd"
    fetched = fetch_paxg_daily()
    ledger = ForecastLedger(Path("data/ledger/coinbase") / asset)
    flow = orderflow_context.get(asset)
    rows = generate_rolling_forecasts(
        fetched.frame,
        context=flow if flow is not None and not flow.empty else None,
        model_names=("hist_gradient_boosting", "ridge"),
        ledger=ledger, asset=asset, min_rows=100,
    )
    scored = ledger.score_matured(fetched.frame)
    report["assets"][asset] = {
        "provider": fetched.snapshot.provider,
        "latest_session": str(fetched.frame.index[-1].date()),
        "history_rows": len(fetched.frame), "forecasts": len(rows), "scored": scored,
        "ledger_forecasts": len(ledger.forecasts()), "ledger_outcomes": len(ledger.outcomes()),
    }
    Path("data/ledger").mkdir(parents=True, exist_ok=True)
    Path("data/ledger/metals-latest-run.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
