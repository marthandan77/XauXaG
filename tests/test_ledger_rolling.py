import pandas as pd

from metals_forecast.ledger import ForecastLedger
from metals_forecast.rolling import generate_rolling_forecasts


def prices(n: int = 220) -> pd.DataFrame:
    index = pd.date_range("2020-01-01", periods=n, freq="B")
    close = pd.Series(range(100, 100 + n), index=index, dtype=float)
    return pd.DataFrame({"open": close - 1, "high": close + 2, "low": close - 2, "close": close, "volume": 1000}, index=index)


def test_rolling_engine_creates_five_forecasts_per_model(tmp_path) -> None:
    ledger = ForecastLedger(tmp_path)
    records = generate_rolling_forecasts(prices(), model_names=("ridge",), ledger=ledger)
    assert len(records) == 5
    assert [row["horizon"] for row in records] == [1, 2, 3, 4, 5]
    assert len(ledger.forecasts()) == 5
    assert len(ledger.forecasts()) == len({row["forecast_id"] for row in ledger.forecasts()})


def test_matured_forecasts_are_scored_once(tmp_path) -> None:
    ledger = ForecastLedger(tmp_path)
    records = generate_rolling_forecasts(prices(), model_names=("ridge",), ledger=ledger)
    bars = prices().copy()
    target = pd.Timestamp(records[0]["target_session"])
    bars.loc[target] = [320, 325, 315, 322, 1000]
    assert ledger.score_matured(bars) == 1
    assert ledger.score_matured(bars) == 0
    assert len(ledger.outcomes()) == 1
