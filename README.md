# Metals Rolling Forecast

Research system for daily low, close, and high forecasts for gold (XAUUSD) and silver (XAGUSD) across the next five trading sessions.

## Market data

The forecast engine refreshes historical daily bid bars from Dukascopy for XAUUSD and XAGUSD. The first run downloads the available history into `data/dukascopy/`; later runs update the current year and reuse the local cache.

The dashboard also reads Binance USDⓈ-M XAUUSDT and XAGUSDT perpetual public market data without an API key: a 1,000-level snapshot (the display shows the nearest 20 levels) and the latest 1,000 aggregate trades. It displays bid/ask depth, top-10 imbalance, depth imbalance within 0.2% and 1% of mid-price, aggressive buy/sell volume, and CVD. Binance's perpetual order book is venue-specific and is not the same as your broker's OTC XAUUSD/XAGUSD book or COMEX futures depth. Public feeds may be unavailable in some regions.

Run `python backfill_binance_history.py` once to import available Binance history since the XAUUSDT/XAGUSDT contracts launched in January 2026. It combines signed-volume proxies from 1-minute futures klines with archived 30-second depth bands. Live trades and DOM snapshots are also deduplicated and stored in `data/binance/orderflow.sqlite3`; live CVD uses aggregate trades, and live DOM uses current book levels. The Streamlit panel records while it is open; to keep collecting independently, run `python collect_binance_orderflow.py --interval 5` in another terminal. Historical archive data covers more sessions than the dashboard's live polling, but DOM is an archived percentage-band summary rather than a replayable full order book.

The daily model receives same-session CVD, traded volume, depth imbalance, spread, and microprice features. The order-flow model trains only on rows where both CVD and 1% DOM imbalance are present, and is enabled after at least 100 such sessions and when the latest price session has flow data. Until then, forecasts fall back to price-history features. Binance's daily kline CVD proxy and live trade-derived CVD are closely related but come from different archive resolutions; Binance-specific flow remains a proxy for XAUUSD broker liquidity.

## Install and run

```powershell
python -m pip install -e ".[dev]"
streamlit run streamlit_app.py
```

Use **Calculate latest forecast** to sync available Binance history, fetch and save current Binance DOM/CVD, refresh Dukascopy prices, process the combined data, and calculate forecasts in one run. The first archive sync can take a few minutes; later clicks update a recent overlap window to pick up delayed archive files. If Binance is unavailable, the dashboard warns and continues with saved flow data when available. The Binance panel can also refresh independently. The collector command remains optional and helps gather live flow between forecast runs. No API key is needed. To run forecasts from the command line:

```powershell
$env:PYTHONPATH = "src"
python run_metals_forecast.py
```

The daily runner evaluates HistGradientBoosting and Ridge. Random Forest and Extra Trees are available in the dashboard. Forecast ledgers are stored under `data/ledger/dukascopy/`.

The system is for research only and does not execute trades.
