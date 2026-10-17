"""Free public OHLC adapters for XAUUSD and XAGUSD research."""
from __future__ import annotations

import hashlib
import json
import urllib.parse
import urllib.request
from datetime import date, datetime, timezone
from typing import Any

import pandas as pd

from .contracts import DataValidationError, FetchedFrame, SourceSnapshot, validate_ohlcv


# Yahoo's public chart service does not reliably expose XAUUSD=X/XAGUSD=X.
# These continuous COMEX futures are free, liquid research proxies; the app
# must not present them as a broker's executable spot feed.
METAL_SYMBOLS = {"xauusd": "GC=F", "xagusd": "SI=F"}


def _download(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "UNG-Forecast-metals/0.1"})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            raw = response.read()
    except Exception as exc:
        raise DataValidationError(f"metals download failed: {url}: {exc}") from exc
    if not raw:
        raise DataValidationError(f"metals feed returned an empty response: {url}")
    return raw


def fetch_yahoo_metal(asset: str, *, start: date = date(2003, 1, 1)) -> FetchedFrame:
    """Fetch completed daily spot bars from Yahoo's public chart endpoint.

    This is a free research feed, not a broker execution feed. Yahoo can throttle
    requests, so callers should cache results and avoid repeated button presses.
    """
    key = asset.lower()
    if key not in METAL_SYMBOLS:
        raise ValueError(f"unsupported metal asset: {asset}")
    symbol = METAL_SYMBOLS[key]
    query = urllib.parse.urlencode({
        "period1": int(datetime(start.year, start.month, start.day, tzinfo=timezone.utc).timestamp()),
        "period2": int(datetime.now(timezone.utc).timestamp()),
        "interval": "1d", "events": "div,splits",
    })
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?{query}"
    raw = _download(url)
    try:
        payload: dict[str, Any] = json.loads(raw)
        result = payload["chart"]["result"][0]
        quote = result["indicators"]["quote"][0]
        frame = pd.DataFrame(
            {key: quote[key] for key in ("open", "high", "low", "close")},
            index=pd.to_datetime(result["timestamp"], unit="s", utc=True).tz_convert(None),
        )
        # Spot feeds do not provide exchange volume; 1 is an explicit sentinel,
        # so volume-derived features cannot be mistaken for real contract volume.
        frame["volume"] = 1.0
        frame = frame.dropna()
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise DataValidationError(f"{asset} response schema/coverage invalid") from exc
    # Public continuous-futures history occasionally contains malformed legacy
    # bars. Drop them rather than silently repairing prices.
    valid = (
        (frame["high"] >= frame[["open", "close"]].max(axis=1))
        & (frame["low"] <= frame[["open", "close"]].min(axis=1))
    )
    frame = validate_ohlcv(frame.loc[valid])
    today = datetime.now(timezone.utc).date()
    frame = frame.loc[frame.index.date < today]
    if len(frame) < 500:
        raise DataValidationError(f"{asset} returned too little history: {len(frame)} rows")
    latest = frame.index[-1].date()
    if (today - latest).days > 5:
        raise DataValidationError(f"{asset} history is stale: {latest}")
    snapshot = SourceSnapshot(
        provider="yahoo_cme_proxy", instrument=key, retrieved_at=datetime.now(timezone.utc),
        published_through=latest, content_sha256=hashlib.sha256(raw).hexdigest(), source_url=url,
    )
    return FetchedFrame(frame, snapshot, raw)


def fetch_xauusd() -> FetchedFrame:
    return fetch_yahoo_metal("xauusd")


def fetch_xagusd() -> FetchedFrame:
    return fetch_yahoo_metal("xagusd")
