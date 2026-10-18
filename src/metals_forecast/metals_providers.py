"""Automated daily XAUUSD/XAGUSD history from Dukascopy's public feed."""
from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from dukascopy_downloader import Downloader

from .contracts import DataValidationError, FetchedFrame, SourceSnapshot, validate_ohlcv


SYMBOLS = {"xauusd": "XAU-USD", "xagusd": "XAG-USD"}
DATA_DIR = Path("data/dukascopy")
HISTORY_START = "2000-01-01"


def fetch_dukascopy_metal(asset: str) -> FetchedFrame:
    """Refresh recent daily bars and return the cached Dukascopy history.

    Dukascopy's public feed needs no account or API key. The downloader keeps a
    local history cache; on later runs only the current year's daily file is
    refreshed, keeping requests to a minimum.
    """
    key = asset.lower()
    if key not in SYMBOLS:
        raise ValueError(f"unsupported metal asset: {asset}")

    data_dir = DATA_DIR
    downloader = Downloader(str(data_dir))
    symbol = SYMBOLS[key]
    try:
        try:
            cached = downloader.load(symbol, "d1")
        except Exception:
            cached = pd.DataFrame()
        has_history = len(cached) >= 500
        start = f"{datetime.now(timezone.utc).year}-01-01" if has_history else HISTORY_START
        results = downloader.download([symbol], tf="d1", start=start)
        failures = [error for result in results for _, error in result.errors]
        if failures:
            raise RuntimeError("; ".join(failures))
        frame = downloader.load(symbol, "d1")
    except Exception as exc:
        raise DataValidationError(f"Dukascopy {key} history download failed: {exc}") from exc
    finally:
        downloader.close()

    if frame.empty:
        raise DataValidationError(f"Dukascopy returned no daily history for {key}")
    frame = frame.copy()
    frame.columns = [str(column).strip().lower() for column in frame.columns]
    if "time" in frame.columns:
        frame = frame.set_index("time")
    elif "date" in frame.columns:
        frame = frame.set_index("date")
    if not isinstance(frame.index, pd.DatetimeIndex):
        frame.index = pd.to_datetime(frame.index, utc=True, errors="coerce")
    frame.index = pd.to_datetime(frame.index, utc=True).tz_convert(None).normalize()
    frame = frame[["open", "high", "low", "close", "volume"]]
    flat_zero_volume = (
        frame["open"].eq(frame["high"])
        & frame["open"].eq(frame["low"])
        & frame["open"].eq(frame["close"])
        & frame["volume"].eq(0)
    )
    frame = frame.loc[~flat_zero_volume]
    frame = validate_ohlcv(frame, instrument=key)

    today = datetime.now(timezone.utc).date()
    frame = frame.loc[frame.index.date < today]
    if len(frame) < 500:
        raise DataValidationError(f"Dukascopy returned too little history for {key}: {len(frame)} rows")
    latest = frame.index[-1].date()
    if (today - latest).days > 5:
        raise DataValidationError(f"Dukascopy {key} data is stale: latest completed bar is {latest}")

    raw = frame.to_csv(index_label="time").encode("utf-8")
    snapshot = SourceSnapshot(
        provider="dukascopy",
        instrument=key,
        retrieved_at=datetime.now(timezone.utc),
        published_through=latest,
        content_sha256=hashlib.sha256(raw).hexdigest(),
        source_url="https://www.dukascopy.com/api/data/get/historical-data-export",
    )
    return FetchedFrame(frame, snapshot, raw)


def fetch_xauusd() -> FetchedFrame:
    return fetch_dukascopy_metal("xauusd")


def fetch_xagusd() -> FetchedFrame:
    return fetch_dukascopy_metal("xagusd")
