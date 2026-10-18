"""Typed contracts and fail-closed validation for market data."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Final, Literal

import numpy as np
import pandas as pd

HORIZONS: Final[tuple[int, ...]] = (1, 2, 3, 4, 5)
OHLCV_COLUMNS: Final[tuple[str, ...]] = ("open", "high", "low", "close", "volume")
RunStatus = Literal["READY", "PROVISIONAL", "BLOCKED"]


class DataValidationError(ValueError):
    """Raised when data cannot safely support downstream calculations."""


@dataclass(frozen=True, slots=True)
class SourceSnapshot:
    provider: str
    instrument: str
    retrieved_at: datetime
    published_through: date | None
    content_sha256: str
    source_url: str

    def __post_init__(self) -> None:
        if not self.provider.strip() or not self.instrument.strip():
            raise ValueError("provider and instrument are required")
        if self.retrieved_at.tzinfo is None or self.retrieved_at.utcoffset() is None:
            raise ValueError("retrieved_at must be timezone-aware")
        if len(self.content_sha256) != 64:
            raise ValueError("content_sha256 must be SHA-256")
        if not self.source_url.startswith(("http://", "https://")):
            raise ValueError("source_url must be HTTP(S)")


@dataclass(frozen=True, slots=True)
class FetchedFrame:
    frame: pd.DataFrame
    snapshot: SourceSnapshot
    raw_bytes: bytes


def validate_ohlcv(frame: pd.DataFrame, *, instrument: str = "instrument") -> pd.DataFrame:
    """Return a normalized OHLCV frame or raise; never repair silently."""
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise DataValidationError(f"{instrument}: index must be DatetimeIndex")
    if frame.index.tz is not None:
        frame = frame.copy()
        frame.index = frame.index.tz_convert(None)
    frame = frame.copy()
    frame.index = frame.index.normalize()
    if frame.index.has_duplicates or not frame.index.is_monotonic_increasing:
        raise DataValidationError(f"{instrument}: duplicate or unsorted sessions")
    missing = [column for column in OHLCV_COLUMNS if column not in frame.columns]
    if missing:
        raise DataValidationError(f"{instrument}: missing columns {missing}")
    frame = frame.loc[:, list(OHLCV_COLUMNS)].apply(pd.to_numeric, errors="coerce")
    if frame.empty or frame.isna().any().any():
        raise DataValidationError(f"{instrument}: empty or incomplete OHLCV data")
    if (frame[["open", "high", "low", "close", "volume"]] <= 0).any().any():
        raise DataValidationError(f"{instrument}: OHLCV values must be positive")
    if (frame["high"] < frame[["open", "close"]].max(axis=1)).any():
        raise DataValidationError(f"{instrument}: high below open/close")
    if (frame["low"] > frame[["open", "close"]].min(axis=1)).any():
        raise DataValidationError(f"{instrument}: low above open/close")
    return frame.sort_index()
