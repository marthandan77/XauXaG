# Metals Rolling Forecast

Research system for daily low, close, and high forecasts for gold (XAUUSD) and silver (XAGUSD) across the next five trading sessions.

## Market data

The forecast engine refreshes historical daily bid bars from Dukascopy for XAUUSD and XAGUSD. The first run downloads the available history into `data/dukascopy/`; later runs update the current year and reuse the local cache.

The dashboard reads OKX perpetual XAU-USDT-SWAP and XAG-USDT-SWAP public market data without an API key: a 400-level order-book snapshot (the display shows the nearest 20 levels) and up to 500 recent public trades. It displays bid/ask depth, top-10 imbalance, depth imbalance within 0.2% and 1% of mid-price, aggressive buy/sell volume, and CVD. This venue-specific book is not the same as your broker's OTC XAUUSD/XAGUSD book or COMEX futures depth.

Live trades and DOM snapshots are deduplicated and stored in `data/okx/orderflow.sqlite3`. The Streamlit panel records while it is open; to collect independently, run `python collect_orderflow.py --interval 5` in another terminal. OKX's public-trades endpoint only returns a rolling recent sample, so the model builds its OKX flow history from saved live observations.

The daily model receives same-session CVD, traded volume, depth imbalance, spread, and microprice features. The order-flow model trains only on rows where both CVD and 1% DOM imbalance are present, and is enabled after at least 100 such sessions and when the latest price session has flow data. Until then, forecasts fall back to price-history features. The OKX flow history starts accumulating when the app or collector runs; the model requires at least 720 observed DOM minutes for a session before treating that day's flow as complete.

## Install and run

```powershell
python -m pip install -e ".[dev]"
streamlit run streamlit_app.py
```

Use **Calculate latest forecast** to fetch and save current OKX DOM/CVD, refresh Dukascopy prices, process the combined data, and calculate forecasts in one run. If OKX is unavailable, the dashboard warns and continues with saved flow data when available. The OKX panel can also refresh independently. The collector command remains optional and helps gather flow between forecast runs. No API key is needed. To run forecasts from the command line:

```powershell
$env:PYTHONPATH = "src"
python run_metals_forecast.py
```

The daily runner evaluates HistGradientBoosting and Ridge. Random Forest and Extra Trees are available in the dashboard. Forecast ledgers are stored under `data/ledger/dukascopy/`.

The system is for research only and does not execute trades.
