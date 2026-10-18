"""Public Coinbase PAXG/USD depth and trade data."""
from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd


BASE_URL = "https://api.exchange.coinbase.com"
SYMBOLS = {"XAUUSD": "Gold (PAXG/USD proxy)"}
INSTRUMENTS = {"XAUUSD": "PAXG-USD"}
DEFAULT_DB = Path("data/coinbase/orderflow.sqlite3")


def _get(path: str, params: dict[str, str | int]) -> list | dict:
    query = urlencode(params)
    url = f"{BASE_URL}{path}" + (f"?{query}" if query else "")
    request = Request(url, headers={"User-Agent": "MetalsForecast/0.1", "Accept": "application/json"})
    try:
        with urlopen(request, timeout=10) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Coinbase Exchange public market data request failed: {exc}") from exc
    return payload


def fetch_orderflow(symbol: str) -> dict:
    """Fetch a public Coinbase PAXG/USD book and recent trades."""
    key = symbol.upper()
    if key not in SYMBOLS:
        raise ValueError(f"Unsupported Coinbase metals symbol: {symbol}")
    product = INSTRUMENTS[key]
    depth_payload = _get(f"/products/{product}/book", {"level": 2})
    trade_payload = _get(f"/products/{product}/trades", {})
    trades = trade_payload if isinstance(trade_payload, list) else None
    if not isinstance(depth_payload, dict) or not isinstance(trades, list):
        raise RuntimeError(f"Unexpected Coinbase response for {product}")

    bids = [{"Price": float(level[0]), "Quantity": float(level[1])} for level in depth_payload["bids"]]
    asks = [{"Price": float(level[0]), "Quantity": float(level[1])} for level in depth_payload["asks"]]
    best_bid = bids[0]["Price"] if bids else 0.0
    best_ask = asks[0]["Price"] if asks else 0.0
    mid = (best_bid + best_ask) / 2 if best_bid and best_ask else 0.0
    band_depths = {}
    for band in (0.002, 0.01):
        bid_band = sum(level["Quantity"] for level in bids if level["Price"] >= mid * (1 - band))
        ask_band = sum(level["Quantity"] for level in asks if level["Price"] <= mid * (1 + band))
        denominator = bid_band + ask_band
        label = "02pct" if band == 0.002 else "1pct"
        band_depths[f"bid_depth_{label}"] = bid_band
        band_depths[f"ask_depth_{label}"] = ask_band
        band_depths[f"imbalance_{label}"] = (bid_band - ask_band) / denominator if denominator else 0.0
    buy_qty = sell_qty = buy_quote = sell_quote = 0.0
    normalized_trades = []
    for trade in trades:
        quantity = float(trade["size"])
        price = float(trade["price"])
        quote = price * quantity
        sell_aggressor = trade["side"].lower() == "sell"
        if sell_aggressor:
            sell_qty += quantity
            sell_quote += quote
        else:
            buy_qty += quantity
            buy_quote += quote
        trade_key = str(trade.get("trade_id") or f"{trade['time']}:{trade['price']}:{trade['size']}:{trade['side']}")
        digest = hashlib.blake2b(trade_key.encode("utf-8"), digest_size=8).digest()
        normalized_trades.append({
            "id": int.from_bytes(digest, "big") & ((1 << 63) - 1),
            "timestamp": int(datetime.fromisoformat(trade["time"].replace("Z", "+00:00")).timestamp() * 1000),
            "price": price,
            "quantity": quantity,
            "buyer_is_maker": sell_aggressor,
        })

    return {
        "symbol": key,
        "name": SYMBOLS[key],
        "source": "Coinbase Exchange",
        "bids": bids,
        "asks": asks,
        "depth_bands": band_depths,
        "updated_at": datetime.now(timezone.utc),
        "trade_count": len(normalized_trades),
        "trades": normalized_trades,
        "buy_qty": buy_qty,
        "sell_qty": sell_qty,
        "buy_quote": buy_quote,
        "sell_quote": sell_quote,
        "cvd_qty": buy_qty - sell_qty,
        "cvd_quote": buy_quote - sell_quote,
        "total_quote": buy_quote + sell_quote,
    }


def record_orderflow(data: dict, db_path: str | Path = DEFAULT_DB) -> None:
    """Persist deduplicated aggregate trades and timestamped DOM snapshots."""
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    bids, asks = data["bids"], data["asks"]
    if not bids or not asks:
        return
    captured_time = data["updated_at"].astimezone(timezone.utc)
    captured_time = captured_time.replace(second=(captured_time.second // 30) * 30, microsecond=0)
    captured_at = captured_time.isoformat()
    bid_qty = sum(level["Quantity"] for level in bids[:10])
    ask_qty = sum(level["Quantity"] for level in asks[:10])
    depth_total = bid_qty + ask_qty
    imbalance = (bid_qty - ask_qty) / depth_total if depth_total else 0.0
    best_bid, best_ask = bids[0]["Price"], asks[0]["Price"]
    mid = (best_bid + best_ask) / 2
    spread_bps = (best_ask - best_bid) / mid * 10_000 if mid else 0.0
    bid_top, ask_top = bids[0]["Quantity"], asks[0]["Quantity"]
    microprice = (best_ask * bid_top + best_bid * ask_top) / (bid_top + ask_top) if bid_top + ask_top else mid
    microprice_edge_bps = (microprice - mid) / mid * 10_000 if mid else 0.0
    band_depths = {}
    for band in (0.002, 0.01):
        bid_band = sum(level["Quantity"] for level in bids if level["Price"] >= mid * (1 - band))
        ask_band = sum(level["Quantity"] for level in asks if level["Price"] <= mid * (1 + band))
        denominator = bid_band + ask_band
        label = "02pct" if band == 0.002 else "1pct"
        band_depths[f"bid_depth_{label}"] = bid_band
        band_depths[f"ask_depth_{label}"] = ask_band
        band_depths[f"imbalance_{label}"] = (bid_band - ask_band) / denominator if denominator else 0.0

    trade_ids = [int(trade["id"]) for trade in data["trades"]]
    existing_ids: set[int] = set()
    with sqlite3.connect(path) as connection:
        # This is the first database access on a fresh install. Create the
        # table before querying it; the complete schema is initialized below.
        connection.execute("""
            CREATE TABLE IF NOT EXISTS aggregate_trades (
                symbol TEXT NOT NULL,
                trade_id INTEGER NOT NULL,
                trade_time_ms INTEGER NOT NULL,
                price REAL NOT NULL,
                quantity REAL NOT NULL,
                buyer_is_maker INTEGER NOT NULL,
                PRIMARY KEY (symbol, trade_id)
            )
        """)
        for offset in range(0, len(trade_ids), 500):
            batch = trade_ids[offset:offset + 500]
            if not batch:
                continue
            placeholders = ",".join("?" for _ in batch)
            rows = connection.execute(
                f"SELECT trade_id FROM aggregate_trades WHERE symbol=? AND trade_id IN ({placeholders})",
                [data["symbol"], *batch],
            ).fetchall()
            existing_ids.update(int(row[0]) for row in rows)
    new_trades = [trade for trade in data["trades"] if int(trade["id"]) not in existing_ids]
    daily_updates: dict[str, dict[str, float | int]] = {}
    for trade in new_trades:
        session = datetime.fromtimestamp(trade["timestamp"] / 1000, tz=timezone.utc).date().isoformat()
        metrics = daily_updates.setdefault(session, {
            "flow_buy_qty": 0.0, "flow_sell_qty": 0.0, "flow_cvd_qty": 0.0,
            "flow_cvd_quote": 0.0, "flow_total_quote": 0.0, "flow_trade_count": 0,
            "flow_max_trade_qty": 0.0,
        })
        quantity = float(trade["quantity"])
        quote = float(trade["price"]) * quantity
        sign = -1.0 if trade["buyer_is_maker"] else 1.0
        metrics["flow_buy_qty"] += quantity if sign > 0 else 0.0
        metrics["flow_sell_qty"] += quantity if sign < 0 else 0.0
        metrics["flow_cvd_qty"] += quantity * sign
        metrics["flow_cvd_quote"] += quote * sign
        metrics["flow_total_quote"] += quote
        metrics["flow_trade_count"] += 1
        metrics["flow_max_trade_qty"] = max(float(metrics["flow_max_trade_qty"]), quantity)

    with sqlite3.connect(path) as connection:
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("""
            CREATE TABLE IF NOT EXISTS aggregate_trades (
                symbol TEXT NOT NULL,
                trade_id INTEGER NOT NULL,
                trade_time_ms INTEGER NOT NULL,
                price REAL NOT NULL,
                quantity REAL NOT NULL,
                buyer_is_maker INTEGER NOT NULL,
                PRIMARY KEY (symbol, trade_id)
            )
        """)
        connection.execute("""
            CREATE TABLE IF NOT EXISTS depth_snapshots (
                symbol TEXT NOT NULL,
                captured_at TEXT NOT NULL,
                bids_json TEXT NOT NULL,
                asks_json TEXT NOT NULL,
                bid_depth_10 REAL NOT NULL,
                ask_depth_10 REAL NOT NULL,
                imbalance_10 REAL NOT NULL,
                spread_bps REAL NOT NULL,
                microprice_edge_bps REAL NOT NULL,
                bid_depth_02pct REAL,
                ask_depth_02pct REAL,
                imbalance_02pct REAL,
                bid_depth_1pct REAL,
                ask_depth_1pct REAL,
                imbalance_1pct REAL,
                PRIMARY KEY (symbol, captured_at)
            )
        """)
        for column in (
            "bid_depth_02pct", "ask_depth_02pct", "imbalance_02pct",
            "bid_depth_1pct", "ask_depth_1pct", "imbalance_1pct",
        ):
            try:
                connection.execute(f"ALTER TABLE depth_snapshots ADD COLUMN {column} REAL")
            except sqlite3.OperationalError as exc:
                if "duplicate column" not in str(exc).lower():
                    raise
        connection.execute("""
            CREATE TABLE IF NOT EXISTS daily_archive_features (
                symbol TEXT NOT NULL,
                session TEXT NOT NULL,
                features_json TEXT NOT NULL,
                PRIMARY KEY (symbol, session)
            )
        """)
        connection.execute("""
            CREATE TABLE IF NOT EXISTS live_daily_features (
                symbol TEXT NOT NULL,
                session TEXT NOT NULL,
                flow_buy_qty REAL NOT NULL,
                flow_sell_qty REAL NOT NULL,
                flow_cvd_qty REAL NOT NULL,
                flow_cvd_quote REAL NOT NULL,
                flow_total_quote REAL NOT NULL,
                flow_trade_count INTEGER NOT NULL,
                flow_max_trade_qty REAL NOT NULL,
                PRIMARY KEY (symbol, session)
            )
        """)
        connection.executemany("""
            INSERT OR IGNORE INTO aggregate_trades
                (symbol, trade_id, trade_time_ms, price, quantity, buyer_is_maker)
            VALUES (?, ?, ?, ?, ?, ?)
        """, [
            (data["symbol"], trade["id"], trade["timestamp"], trade["price"],
             trade["quantity"], int(trade["buyer_is_maker"]))
            for trade in data["trades"]
        ])
        connection.execute("""
            INSERT OR IGNORE INTO depth_snapshots
                (symbol, captured_at, bids_json, asks_json, bid_depth_10, ask_depth_10,
                 imbalance_10, spread_bps, microprice_edge_bps, bid_depth_02pct,
                 ask_depth_02pct, imbalance_02pct, bid_depth_1pct, ask_depth_1pct, imbalance_1pct)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            data["symbol"], captured_at, "[]", "[]", bid_qty,
            ask_qty, imbalance, spread_bps, microprice_edge_bps,
            band_depths["bid_depth_02pct"], band_depths["ask_depth_02pct"],
            band_depths["imbalance_02pct"], band_depths["bid_depth_1pct"],
            band_depths["ask_depth_1pct"], band_depths["imbalance_1pct"],
        ))
        for session, metrics in daily_updates.items():
            total_qty = float(metrics["flow_buy_qty"]) + float(metrics["flow_sell_qty"])
            metrics["flow_delta_ratio"] = float(metrics["flow_cvd_qty"]) / total_qty if total_qty else 0.0
            connection.execute("""
                INSERT INTO live_daily_features
                    (symbol, session, flow_buy_qty, flow_sell_qty, flow_cvd_qty, flow_cvd_quote,
                     flow_total_quote, flow_trade_count, flow_max_trade_qty)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(symbol, session) DO UPDATE SET
                    flow_buy_qty=flow_buy_qty + excluded.flow_buy_qty,
                    flow_sell_qty=flow_sell_qty + excluded.flow_sell_qty,
                    flow_cvd_qty=flow_cvd_qty + excluded.flow_cvd_qty,
                    flow_cvd_quote=flow_cvd_quote + excluded.flow_cvd_quote,
                    flow_total_quote=flow_total_quote + excluded.flow_total_quote,
                    flow_trade_count=flow_trade_count + excluded.flow_trade_count,
                    flow_max_trade_qty=max(flow_max_trade_qty, excluded.flow_max_trade_qty)
            """, (
                data["symbol"], session, metrics["flow_buy_qty"], metrics["flow_sell_qty"],
                metrics["flow_cvd_qty"], metrics["flow_cvd_quote"], metrics["flow_total_quote"],
                metrics["flow_trade_count"], metrics["flow_max_trade_qty"],
            ))
        # Trade IDs are only needed as a recent cursor for gap recovery; daily
        # totals above preserve the historical model features with less storage.
        cutoff_ms = int(datetime.now(timezone.utc).timestamp() * 1000) - 48 * 60 * 60 * 1000
        connection.execute("DELETE FROM aggregate_trades WHERE trade_time_ms < ?", (cutoff_ms,))


def load_daily_orderflow_features(db_path: str | Path = DEFAULT_DB) -> dict[str, pd.DataFrame]:
    """Return UTC-day CVD/trade and DOM summary features for both metals."""
    path = Path(db_path)
    if not path.exists():
        return {asset: pd.DataFrame() for asset in ("xauusd", "xagusd")}

    def read_table(connection: sqlite3.Connection, table: str) -> pd.DataFrame:
        try:
            return pd.read_sql_query(f"SELECT * FROM {table}", connection)
        except pd.errors.DatabaseError:
            return pd.DataFrame()

    with sqlite3.connect(path) as connection:
        trades = read_table(connection, "aggregate_trades")
        depth = read_table(connection, "depth_snapshots")
        live_features = read_table(connection, "live_daily_features")
    result = {}
    for symbol, asset in (("XAUUSD", "xauusd"), ("XAGUSD", "xagusd")):
        symbol_live = live_features.loc[live_features["symbol"] == symbol].copy() if not live_features.empty else pd.DataFrame()
        symbol_trades = trades.loc[trades["symbol"] == symbol].copy() if not trades.empty else pd.DataFrame()
        if not symbol_live.empty:
            symbol_live["session"] = pd.to_datetime(symbol_live["session"])
            flow = symbol_live.set_index("session").drop(columns=["symbol"])
            total_qty = flow["flow_buy_qty"] + flow["flow_sell_qty"]
            flow["flow_total_qty"] = total_qty
            flow["flow_delta_ratio"] = flow["flow_cvd_qty"] / total_qty.replace(0, float("nan"))
        elif not symbol_trades.empty:
            ts = pd.to_datetime(symbol_trades["trade_time_ms"], unit="ms", utc=True)
            symbol_trades["session"] = ts.dt.tz_convert(None).dt.normalize()
            sign = symbol_trades["buyer_is_maker"].map({1: -1.0, 0: 1.0})
            symbol_trades["quote_qty"] = symbol_trades["price"] * symbol_trades["quantity"]
            symbol_trades["signed_qty"] = symbol_trades["quantity"] * sign
            symbol_trades["signed_quote"] = symbol_trades["quote_qty"] * sign
            symbol_trades["flow_buy_qty"] = symbol_trades["quantity"].where(
                symbol_trades["buyer_is_maker"] == 0, 0.0
            )
            symbol_trades["flow_sell_qty"] = symbol_trades["quantity"].where(
                symbol_trades["buyer_is_maker"] == 1, 0.0
            )
            flow = symbol_trades.groupby("session").agg(
                flow_buy_qty=("flow_buy_qty", "sum"),
                flow_sell_qty=("flow_sell_qty", "sum"),
                flow_cvd_qty=("signed_qty", "sum"),
                flow_cvd_quote=("signed_quote", "sum"),
                flow_total_quote=("quote_qty", "sum"),
                flow_trade_count=("trade_id", "count"),
                flow_max_trade_qty=("quantity", "max"),
                flow_total_qty=("quantity", "sum"),
            )
            flow["flow_delta_ratio"] = flow["flow_cvd_qty"] / flow["flow_total_qty"].replace(0, float("nan"))
        else:
            flow = pd.DataFrame()

        symbol_depth = depth.loc[depth["symbol"] == symbol].copy() if not depth.empty else pd.DataFrame()
        if not symbol_depth.empty:
            capture_times = pd.to_datetime(symbol_depth["captured_at"], utc=True)
            symbol_depth["session"] = capture_times.dt.tz_convert(None).dt.normalize()
            symbol_depth["minute"] = capture_times.dt.floor("min")
            dom = symbol_depth.groupby("session").agg(
                flow_dom_bid_depth_10=("bid_depth_10", "mean"),
                flow_dom_ask_depth_10=("ask_depth_10", "mean"),
                flow_dom_imbalance_10=("imbalance_10", "mean"),
                flow_dom_imbalance_10_last=("imbalance_10", "last"),
                flow_dom_bid_depth_02pct=("bid_depth_02pct", "mean"),
                flow_dom_ask_depth_02pct=("ask_depth_02pct", "mean"),
                flow_dom_imbalance_02pct=("imbalance_02pct", "mean"),
                flow_dom_bid_depth_1pct=("bid_depth_1pct", "mean"),
                flow_dom_ask_depth_1pct=("ask_depth_1pct", "mean"),
                flow_dom_imbalance_1pct=("imbalance_1pct", "mean"),
                flow_dom_imbalance_1pct_last=("imbalance_1pct", "last"),
                flow_spread_bps=("spread_bps", "mean"),
                flow_microprice_edge_bps=("microprice_edge_bps", "mean"),
                flow_dom_snapshots=("captured_at", "count"),
                flow_dom_minutes=("minute", "nunique"),
            )
        else:
            dom = pd.DataFrame()

        daily = flow.join(dom, how="outer") if not flow.empty or not dom.empty else pd.DataFrame()
        if not daily.empty:
            daily.index = pd.DatetimeIndex(daily.index, name="session")
            daily["available_at"] = daily.index + pd.Timedelta(days=1)
            if "flow_dom_minutes" not in daily.columns:
                daily["flow_dom_minutes"] = 0
            # Partial intraday capture is not a representative daily DOM/CVD
            # feature. Keep the coverage count visible and mask thin sessions.
            thin = daily["flow_dom_minutes"].fillna(0) < 720
            flow_columns = [column for column in daily.columns if column.startswith("flow_") and column != "flow_dom_minutes"]
            daily.loc[thin, flow_columns] = float("nan")
        result[asset] = daily
    return result
