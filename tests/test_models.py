import pandas as pd

from metals_forecast.features import build_features, build_forward_targets
from metals_forecast.models import fit_individual_forecast, score_forecast


def frame() -> pd.DataFrame:
    index = pd.date_range("2020-01-01", periods=180, freq="B")
    close = pd.Series(range(100, 280), index=index, dtype=float)
    return pd.DataFrame({"open": close - 1, "high": close + 2, "low": close - 2, "close": close, "volume": 1000}, index=index)


def test_models_return_individual_forecast() -> None:
    prices = frame()
    forecast = fit_individual_forecast("ridge", build_features(prices), build_forward_targets(prices, 1), reference_close=279, target_session="2026-12-31", horizon=1)
    assert forecast.model_name == "ridge"
    assert forecast.predicted_low <= forecast.predicted_high
    assert 0.0 < forecast.probability_up < 1.0
    assert forecast.uncertainty >= 0.0


def test_scorecard_is_separate_from_forecast() -> None:
    scored = score_forecast({"predicted_low": 9, "expected_close": 10, "predicted_high": 11}, 9.5, 10.2, 10.8)
    assert scored["range_contains_day"] is True
    assert scored["predicted_range_width"] == 2.0


def test_tree_models_are_individual_models() -> None:
    prices = frame()
    features = build_features(prices)
    targets = build_forward_targets(prices, 1)
    for name in ("hist_gradient_boosting", "random_forest", "extra_trees"):
        forecast = fit_individual_forecast(name, features, targets, reference_close=279, target_session="2026-12-31", horizon=1)
        assert forecast.model_name == name
