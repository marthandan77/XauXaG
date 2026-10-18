# XauXaG

Research dashboard for five-session gold range forecasts. Price and order-flow inputs use Coinbase Exchange's public PAXG-USD market, a gold-backed token proxy. PAXG is not spot XAUUSD and its price, liquidity, and trading hours can differ from a broker's gold feed. The project now forecasts gold only; it does not forecast silver.

## Data and forecast inputs

The dashboard downloads completed daily OHLCV candles for PAXG-USD directly from Coinbase when **Calculate latest forecast** is pressed. The public candles endpoint requires no API key and accepts daily granularity. On first use, the app fetches the full history available since PAXG trading began in chunks of 280 days, below Coinbase's 300-candle limit, then caches it in `data/coinbase/paxg_daily.csv`. Later calculations refresh the most recent 280 days and merge them into the cache. Weekend candles are excluded so training labels line up with XAUUSD trading sessions. The model requires at least 100 completed daily rows.

The app also loads Coinbase's public level-2 order book and recent trades for PAXG-USD. It calculates CVD, aggressive buy/sell volume, depth imbalance, spread, and microprice features. Trades and depth snapshots are archived in `data/coinbase/orderflow.sqlite3`. The Streamlit panel records while it is open; to collect between page visits, run `python collect_orderflow.py --interval 5`.

Daily order-flow features join the price history by UTC date and are passed into the same ML feature pipeline as OHLCV-derived price features. Flow features are used in model training once at least 100 completed sessions have sufficient DOM and CVD coverage. DOM completeness currently means at least 720 observed minutes in a session, so leave the collector running around the clock to build this dataset. Before enough flow history is collected, the model falls back to price-history features. Coinbase's public trades endpoint returns a rolling recent sample, so saved historical CVD builds from when the app or collector starts.

## Run

Python 3.12 or later is required. Install the project and launch Streamlit:

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Press **Calculate latest forecast** to refresh live Coinbase trades and DOM, save available flow data, download PAXG-USD candles, load the archived flow features, and calculate the latest model forecasts. Live DOM/CVD can also refresh independently. No account or API key is needed.

To run the daily HistGradientBoosting and Ridge forecasts from a terminal:

```bash
python run_metals_forecast.py
```

Forecast ledgers are stored under `data/ledger/coinbase/xauusd/`. The app also offers Random Forest and Extra Trees. This is a research tool, not a trading instruction or execution system.

## Source documentation

- [Coinbase Exchange candles](https://docs.cdp.coinbase.com/api-reference/exchange-api/rest-api/products/get-product-candles)
- [Coinbase Exchange level-2 book](https://docs.cdp.coinbase.com/api-reference/exchange-api/rest-api/products/get-product-book)
- [Coinbase Exchange trades](https://docs.cdp.coinbase.com/api-reference/exchange-api/rest-api/products/get-product-trades)
