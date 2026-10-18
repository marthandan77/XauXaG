import numpy as np
import pandas as pd

from metals_forecast.features import build_features, build_forward_targets
from metals_forecast.models import walk_forward_evaluate


def prices(n=180):
    index = pd.date_range("2020-01-01", periods=n, freq="B")
    close = pd.Series(100 + np.arange(n) * 0.3 + np.sin(np.arange(n)), index=index)
    return pd.DataFrame({"open": close - .1, "high": close + 1, "low": close - 1, "close": close, "volume": 1000 + np.arange(n)}, index=index)


def test_walk_forward_is_expanding_and_future_blind() -> None:
    frame = prices()
    rows = walk_forward_evaluate("ridge", build_features(frame), build_forward_targets(frame, 1), horizon=1, min_train_rows=100, step=10)
    assert rows
    assert all(row["model_name"] == "ridge" for row in rows)
    assert all("actual_return" in row and "absolute_error" in row for row in rows)
