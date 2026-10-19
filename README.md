# XauXaG

Research dashboard for five-session gold range forecasts. Price and order-flow inputs use Coinbase Exchange's public PAXG-USD market, a gold-backed token proxy. PAXG is not spot XAUUSD and its price, liquidity, and trading hours can differ from a broker's gold feed. The project now forecasts gold only; it does not forecast silver.

## Data and forecast inputs

The dashboard downloads completed daily OHLCV candles for PAXG-USD directly from Coinbase when **Calculate latest forecast** is pressed. The public candles endpoint requires no API key and accepts daily granularity. On first use, the app fetches the full history available since PAXG trading began in chunks of 280 days, below Coinbase's 300-candle limit, then caches it in `data/coinbase/paxg_daily.csv`. Later calculations refresh the most recent 280 days and merge them into the cache. Weekend candles are excluded so training labels line up with XAUUSD trading sessions. The model requires at least 100 completed daily rows.

The app also loads Coinbase's public level-2 order book and recent trades for PAXG-USD. It calculates CVD, aggressive buy/sell volume, depth imbalance, spread, and microprice features. Trades and depth snapshots are archived in `data/coinbase/orderflow.sqlite3`. The Streamlit panel records while it is open; to collect between page visits, run `python collect_orderflow.py --interval 5`.

Daily order-flow features join the price history by UTC date and are passed into the same ML feature pipeline as OHLCV-derived price features. Flow features are used in model training once at least 100 completed sessions have sufficient DOM and CVD coverage. DOM completeness currently means at least 720 observed minutes in a session, so leave the collector running around the clock to build this dataset. Before enough flow history is collected, the model falls back to price-history features. Coinbase's public trades endpoint returns a rolling recent sample, so saved historical CVD builds from when the app or collector starts.

## Forecast scoreboard and feedback

The scoreboard compares each matured forecast with that session's actual high, low, and close. It reports close MAE and directional accuracy, full-range coverage, recent 20-forecast results, and separate low-side/high-side miss rates. A detail table shows the predicted range beside the actual range and diagnoses close over/under-prediction and which range edge missed.

Each time a forecast is calculated, the app first scores any forecasts whose target sessions have completed, then retrains on the updated price history. Once there are 30 matched outcomes for the same model, horizon, and input mode, it applies a rolling 60-outcome median close-bias correction and expands range edges from observed one-sided residuals. The forecast table shows the calibration sample and adjustments used. This feedback advances when **Calculate latest forecast** is pressed or the command-line runner is run; it is not a background job while the app is idle.

## Trade plan panel

The dashboard also shows a conditional trend-pullback plan. It checks the 20/50-session EMA trend, current PAXG price versus an EMA20 pullback zone, the latest trade-sample CVD ratio, 1% DOM imbalance, and the short-horizon ML direction. A setup remains **Wait** unless trend, price location, live CVD/DOM confirmation, and the ML direction do not conflict. The plan uses recent five-session swing structure for a protective stop and displays 1R and 2R targets from its EMA20 entry reference. If a stop would be wider than two ATR, the plan says to wait.

All displayed trade-plan levels are PAXG/USD proxy prices. They are not broker XAUUSD quotes; verify levels against the XAUUSD instrument you trade. The panel is a research aid and does not place orders.

## Run

Python 3.12 or later is required. Install the project and launch Streamlit:

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Press **Calculate latest forecast** to refresh live Coinbase trades and DOM, save available flow data, download PAXG-USD candles, load the archived flow features, and calculate the latest model forecasts. The next trading session's predicted low-to-high range and expected close appear near the top of the results, with one range per selected model. Live DOM/CVD can also refresh independently. No account or API key is needed.

To run the daily HistGradientBoosting and Ridge forecasts from a terminal:

```bash
python run_metals_forecast.py
```

Forecast ledgers are stored under `data/ledger/coinbase/xauusd/`. The app also offers Random Forest and Extra Trees. This is a research tool, not a trading instruction or execution system.

## Source documentation

- [Coinbase Exchange candles](https://docs.cdp.coinbase.com/api-reference/exchange-api/rest-api/products/get-product-candles)
- [Coinbase Exchange level-2 book](https://docs.cdp.coinbase.com/api-reference/exchange-api/rest-api/products/get-product-book)
- [Coinbase Exchange trades](https://docs.cdp.coinbase.com/api-reference/exchange-api/rest-api/products/get-product-trades)
