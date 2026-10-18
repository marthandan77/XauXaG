import pandas as pd
import pytest

from metals_forecast.contracts import DataValidationError, validate_ohlcv


def valid_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {"open": [10, 11], "high": [12, 13], "low": [9, 10], "close": [11, 12], "volume": [100, 120]},
        index=pd.date_range("2026-01-01", periods=2, freq="B"),
    )


def test_ohlcv_is_normalized() -> None:
    result = validate_ohlcv(valid_frame())
    assert list(result.columns) == ["open", "high", "low", "close", "volume"]


def test_invalid_ohlc_is_rejected() -> None:
    frame = valid_frame()
    frame.loc[frame.index[0], "high"] = 8
    with pytest.raises(DataValidationError):
        validate_ohlcv(frame)
