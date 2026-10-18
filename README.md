# Metals Rolling Forecast

Research system for daily low, close, and high forecasts for gold (XAUUSD) and silver (XAGUSD) across the next five trading sessions.

## Market data

The forecast engine refreshes historical daily bid bars from Dukascopy for XAUUSD and XAGUSD. The first run downloads the available history into `data/dukascopy/`; later runs update the current year and reuse the local cache.

The dashboard reads Coinbase Exchange PAXG-USD public market data without an API key. PAXG is a gold-backed token and is a proxy for XAUUSD, not the same instrument or venue. The app reads the aggregated level-2 order book and the latest public trades. It displays bid/ask depth, top-10 imbalance, depth imbalance within 0.2% and 1% of mid-price, aggressive buy/sell volume, and CVD. Coinbase currently provides the live flow panel for gold only; silver forecasts continue using Dukascopy prices without order-flow features.

Live trades and DOM snapshots are deduplicated and stored in `data/coinbase/orderflow.sqlite3`. The Streamlit panel records while it is open; to collect independently, run `python collect_orderflow.py --interval 5` in another terminal. Coinbase's public-trades endpoint returns a rolling recent sample, so the model builds its gold flow history from saved live observations.

The daily gold model receives same-session CVD, traded volume, depth imbalance, spread, and microprice features. Order-flow features are enabled after at least 100 complete gold sessions; silver forecasts use price-history features only. Coinbase flow history starts accumulating when the app or collector runs, and the model requires at least 720 observed DOM minutes for a session before treating that day's flow as complete.

## Install and run

```powershell
python -m pip install -e ".[dev]"
streamlit run streamlit_app.py
```

Use **Calculate latest forecast** to fetch and save current Coinbase PAXG/USD DOM/CVD, refresh Dukascopy prices, process the combined data, and calculate forecasts in one run. If Coinbase is unavailable, the dashboard warns and continues with saved flow data when available. The Coinbase panel can also refresh independently. The collector command remains optional and helps gather flow between forecast runs. No API key is needed. To run forecasts from the command line:

```powershell
$env:PYTHONPATH = "src"
python run_metals_forecast.py
```

The daily runner evaluates HistGradientBoosting and Ridge. Random Forest and Extra Trees are available in the dashboard. Forecast ledgers are stored under `data/ledger/dukascopy/`.

The system is for research only and does not execute trades.
