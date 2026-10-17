"""Run daily five-session XAUUSD/XAGUSD proxy research forecasts."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from ung_forecast.ledger import ForecastLedger
from ung_forecast.metals_providers import fetch_xagusd, fetch_xauusd
from ung_forecast.rolling import generate_rolling_forecasts


def main() -> None:
    report: dict = {"run_at": datetime.now(timezone.utc).isoformat(), "assets": {}}
    for asset, fetcher in (("xauusd", fetch_xauusd), ("xagusd", fetch_xagusd)):
        fetched = fetcher()
        ledger = ForecastLedger(Path("data/ledger") / asset)
        # Daily default: champion + transparent benchmark. The heavier tree
        # challengers remain available from Streamlit for periodic audits.
        rows = generate_rolling_forecasts(
            fetched.frame, model_names=("hist_gradient_boosting", "ridge"),
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
