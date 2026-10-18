"""Daily PAXG/USD price history from Coinbase Exchange's public REST API."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd

from .contracts import DataValidationError, FetchedFrame, SourceSnapshot, validate_ohlcv


BASE_URL = "https://api.exchange.coinbase.com"
PRODUCT = "PAXG-USD"
GRANULARITY_SECONDS = 86_400
HISTORY_DAYS = 280
HISTORY_START = datetime(2019, 1, 1, tzinfo=timezone.utc)
CACHE_PATH = Path("data/coinbase/paxg_daily.csv")


def _fetch_candles(start: datetime, end: datetime) -> pd.DataFrame:
    """Fetch one Coinbase daily-candle range, which must be under 300 days."""
    params = urlencode({
        "granularity": GRANULARITY_SECONDS,
        "start": int(start.timestamp()),
        "end": int(end.timestamp()),
    })
    url = f"{BASE_URL}/products/{PRODUCT}/candles?{params}"
    request = Request(
        url,
        headers={"User-Agent": "XauXaG-Forecast/0.1", "Accept": "application/json"},
    )
    try:
        with urlopen(request, timeout=20) as response:
            raw = response.read()
        payload = json.loads(raw.decode("utf-8"))
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise DataValidationError(f"Coinbase PAXG/USD candle request failed: {exc}") from exc
    if not isinstance(payload, list):
        raise DataValidationError("Coinbase returned an unexpected PAXG/USD candle response")
    if not payload:
        return pd.DataFrame(columns=("open", "high", "low", "close", "volume"))
    try:
        frame = pd.DataFrame(
            payload,
            columns=("time", "low", "high", "open", "close", "volume"),
        )
        frame["time"] = pd.to_datetime(frame["time"], unit="s", utc=True)
        frame = frame.set_index("time")
        frame.index = frame.index.tz_convert(None).normalize()
        frame = frame[["open", "high", "low", "close", "volume"]].sort_index()
        return validate_ohlcv(frame, instrument=PRODUCT)
    except (TypeError, ValueError, KeyError, pd.errors.ParserError) as exc:
        raise DataValidationError(f"Coinbase PAXG/USD candles failed validation: {exc}") from exc


def fetch_paxg_daily() -> FetchedFrame:
    """Refresh and return all available daily PAXG-USD OHLCV history.

    Coinbase caps a candle request at 300 rows. On first use, request 280-day
    ranges back to PAXG's launch. Cache the resulting history locally; later
    calculations refresh only the most recent 280 days and merge new candles.
    """
    now = datetime.now(timezone.utc)
    today = now.date()
    end = datetime.combine(today, datetime.min.time(), tzinfo=timezone.utc)
    recent_start = end - timedelta(days=HISTORY_DAYS)
    chunks: list[pd.DataFrame] = []
    if CACHE_PATH.exists():
        cached = pd.read_csv(CACHE_PATH, index_col="time", parse_dates=True)
        cached.index = pd.DatetimeIndex(cached.index).tz_localize(None).normalize()
        chunks.append(validate_ohlcv(cached, instrument=PRODUCT))
        refreshed = _fetch_candles(recent_start, end)
        if not refreshed.empty:
            chunks.append(refreshed)
    else:
        chunk_end = end
        while chunk_end > HISTORY_START:
            chunk_start = max(HISTORY_START, chunk_end - timedelta(days=HISTORY_DAYS))
            refreshed = _fetch_candles(chunk_start, chunk_end)
            if refreshed.empty:
                break
            chunks.append(refreshed)
            chunk_end = chunk_start
    if not chunks:
        raise DataValidationError("Coinbase returned no daily PAXG/USD candles")
    frame = pd.concat(chunks).sort_index()
    frame = frame.loc[~frame.index.duplicated(keep="last")]
    frame = validate_ohlcv(frame, instrument=PRODUCT)
    frame = frame.loc[frame.index.date < today]
    # Align the daily training labels with XAUUSD trading sessions. PAXG trades
    # on weekends, but spot gold does not publish weekend trading sessions.
    frame = frame.loc[frame.index.dayofweek < 5]
    if len(frame) < 100:
        raise DataValidationError(
            f"Coinbase returned too little completed PAXG/USD history for the model: {len(frame)} rows"
        )

    normalized_raw = frame.to_csv(index_label="time").encode("utf-8")
    latest = frame.index[-1].date()
    if (today - latest).days > 5:
        raise DataValidationError(f"Coinbase PAXG/USD candles are stale: latest completed session is {latest}")
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_bytes(normalized_raw)
    snapshot = SourceSnapshot(
        provider="coinbase_exchange",
        instrument=PRODUCT,
        retrieved_at=now,
        published_through=latest,
        content_sha256=hashlib.sha256(normalized_raw).hexdigest(),
        source_url=f"{BASE_URL}/products/{PRODUCT}/candles",
    )
    return FetchedFrame(frame, snapshot, normalized_raw)
