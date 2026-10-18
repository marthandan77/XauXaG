"""Backfill Binance XAU/XAG 1-minute taker-flow and depth-band archives."""
from __future__ import annotations

import argparse
import io
import sqlite3
import time
import zipfile
from datetime import date, datetime, timedelta, timezone
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd

from metals_forecast.binance_orderflow import ARCHIVE_URL, DEFAULT_DB, store_daily_archive_features


LAUNCH_DATES = {"XAUUSDT": date(2026, 1, 5), "XAGUSDT": date(2026, 1, 7)}
KLINE_COLUMNS = [
    "open_time", "open", "high", "low", "close", "volume", "close_time",
    "quote_volume", "trade_count", "taker_buy_volume", "taker_buy_quote", "ignore",
]


def _download_csv(url: str, *, has_header: bool) -> pd.DataFrame | None:
    request = Request(url, headers={"User-Agent": "MetalsForecast/0.1"})
    try:
        with urlopen(request, timeout=45) as response:
            content = response.read()
    except HTTPError as exc:
        if exc.code == 404:
            return None
        raise
    except URLError:
        raise
    try:
        with zipfile.ZipFile(io.BytesIO(content)) as archive:
            names = [name for name in archive.namelist() if name.lower().endswith(".csv")]
            if not names:
                return None
            return pd.read_csv(archive.open(names[0]), header=0 if has_header else None,
                               names=None if has_header else KLINE_COLUMNS)
    except zipfile.BadZipFile:
        return None


def _kline_daily_features(frame: pd.DataFrame) -> dict[str, dict[str, float | int]]:
    if frame.empty:
        return {}
    frame = frame.copy()
    frame["session"] = pd.to_datetime(frame["open_time"], unit="ms", utc=True).dt.strftime("%Y-%m-%d")
    rows = {}
    for session, group in frame.groupby("session", sort=True):
        total_volume = float(group["volume"].sum())
        buy_volume = float(group["taker_buy_volume"].sum())
        total_quote = float(group["quote_volume"].sum())
        buy_quote = float(group["taker_buy_quote"].sum())
        delta = 2 * buy_volume - total_volume
        rows[session] = {
            "flow_buy_qty": buy_volume,
            "flow_sell_qty": max(0.0, total_volume - buy_volume),
            "flow_total_qty": total_volume,
            "flow_cvd_qty": delta,
            "flow_cvd_quote": 2 * buy_quote - total_quote,
            "flow_total_quote": total_quote,
            "flow_trade_count": int(group["trade_count"].sum()),
            "flow_delta_ratio": delta / total_volume if total_volume else 0.0,
        }
    return rows


def _bookdepth_daily_features(frame: pd.DataFrame) -> dict[str, dict[str, float | int]]:
    if frame.empty:
        return {}
    required = {"timestamp", "percentage", "depth", "notional"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Unexpected bookDepth columns: {list(frame.columns)}")
    frame = frame.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    frame["percentage"] = frame["percentage"].astype(float).round(2)
    frame["session"] = frame["timestamp"].dt.strftime("%Y-%m-%d")
    output = {}
    for session, group in frame.groupby("session", sort=True):
        depth = group.pivot_table(index="timestamp", columns="percentage", values="depth", aggfunc="last")
        features: dict[str, float | int] = {
            "flow_dom_snapshots": int(depth.index.nunique()),
            "flow_dom_minutes": int(depth.index.floor("min").nunique()),
        }
        for band, label in ((0.2, "02pct"), (1.0, "1pct")):
            bid_col, ask_col = -band, band
            if bid_col not in depth.columns or ask_col not in depth.columns:
                continue
            bid = depth[bid_col].astype(float)
            ask = depth[ask_col].astype(float)
            denom = bid + ask
            imbalance = (bid - ask) / denom.replace(0, float("nan"))
            features[f"flow_dom_bid_depth_{label}"] = float(bid.mean())
            features[f"flow_dom_ask_depth_{label}"] = float(ask.mean())
            features[f"flow_dom_imbalance_{label}"] = float(imbalance.mean())
            features[f"flow_dom_imbalance_{label}_last"] = float(imbalance.iloc[-1])
        output[session] = features
    return output


def _fetch_month_klines(symbol: str, month: date) -> pd.DataFrame | None:
    month_name = month.strftime("%Y-%m")
    filename = f"{symbol}-1m-{month_name}.zip"
    url = f"{ARCHIVE_URL}/monthly/klines/{symbol}/1m/{filename}"
    return _download_csv(url, has_header=False)


def _fetch_day_klines(symbol: str, day: date) -> pd.DataFrame | None:
    day_name = day.isoformat()
    filename = f"{symbol}-1m-{day_name}.zip"
    url = f"{ARCHIVE_URL}/daily/klines/{symbol}/1m/{filename}"
    return _download_csv(url, has_header=False)


def _fetch_day_bookdepth(symbol: str, day: date) -> pd.DataFrame | None:
    day_name = day.isoformat()
    filename = f"{symbol}-bookDepth-{day_name}.zip"
    url = f"{ARCHIVE_URL}/daily/bookDepth/{symbol}/{filename}"
    return _download_csv(url, has_header=True)


def _month_starts(start: date, end: date) -> list[date]:
    cursor = date(start.year, start.month, 1)
    result = []
    while cursor <= end:
        result.append(cursor)
        cursor = date(cursor.year + (cursor.month == 12), cursor.month % 12 + 1, 1)
    return result


def _latest_archived_session(symbol: str) -> date | None:
    if not DEFAULT_DB.exists():
        return None
    try:
        with sqlite3.connect(DEFAULT_DB) as connection:
            row = connection.execute(
                "SELECT MAX(session) FROM daily_archive_features WHERE symbol=?", (symbol,)
            ).fetchone()
    except sqlite3.OperationalError:
        return None
    return date.fromisoformat(row[0]) if row and row[0] else None


def backfill(end: date, *, incremental: bool = True) -> None:
    for symbol, launch in LAUNCH_DATES.items():
        latest = _latest_archived_session(symbol) if incremental else None
        start = max(launch, latest - timedelta(days=3)) if latest else launch
        print(f"{symbol}: loading Binance archive from {start} through {end}")
        trade_days: dict[str, dict[str, float | int]] = {}
        current_month = date(end.year, end.month, 1)
        for month in _month_starts(start, end):
            if month < current_month:
                klines = _fetch_month_klines(symbol, month)
                if klines is not None:
                    trade_days.update(_kline_daily_features(klines))
                time.sleep(0.15)
            else:
                cursor = max(start, month)
                while cursor <= end:
                    klines = _fetch_day_klines(symbol, cursor)
                    if klines is not None:
                        trade_days.update(_kline_daily_features(klines))
                    cursor += timedelta(days=1)
                    time.sleep(0.1)

        saved = 0
        cursor = start
        while cursor <= end:
            session = cursor.isoformat()
            combined = dict(trade_days.get(session, {}))
            bookdepth = _fetch_day_bookdepth(symbol, cursor)
            if bookdepth is not None:
                combined.update(_bookdepth_daily_features(bookdepth).get(session, {}))
                time.sleep(0.1)
            if combined:
                store_daily_archive_features(symbol, session, combined)
                saved += 1
            cursor += timedelta(days=1)
        print(f"{symbol}: saved {saved} daily feature rows to data/binance/orderflow.sqlite3")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--end", type=date.fromisoformat,
        default=datetime.now(timezone.utc).date() - timedelta(days=1),
        help="Last UTC date to import (default: yesterday)",
    )
    args = parser.parse_args()
    backfill(args.end)


if __name__ == "__main__":
    main()
