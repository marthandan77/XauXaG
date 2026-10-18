import numpy as np
import pandas as pd

from metals_forecast.features import build_features, build_forward_targets


def prices() -> pd.DataFrame:
    index = pd.date_range("2026-01-01", periods=30, freq="B")
    close = pd.Series(np.arange(10.0, 40.0), index=index)
    return pd.DataFrame({"open": close - .1, "high": close + .5, "low": close - .5, "close": close, "volume": 1000}, index=index)


def test_features_are_point_in_time() -> None:
    frame = prices()
    features = build_features(frame)
    changed = frame.copy()
    changed.iloc[-1, changed.columns.get_loc("close")] = 1000
    changed_features = build_features(changed)
    assert features.iloc[-2]["return_1"] == changed_features.iloc[-2]["return_1"]


def test_forward_targets_have_expected_horizons() -> None:
    targets = build_forward_targets(prices(), 5)
    assert targets.iloc[0]["forward_return"] > 0
    assert targets.iloc[-1].isna().all()


def test_release_aware_context_is_not_used_early() -> None:
    frame = prices()
    context = pd.DataFrame(
        {"storage_bcf": [1000], "available_at": [pd.Timestamp("2026-01-20 15:30", tz="UTC")]},
        index=pd.DatetimeIndex([pd.Timestamp("2026-01-16")], name="observation_date"),
    )
    features = build_features(frame, context=context)
    assert pd.isna(features.loc[pd.Timestamp("2026-01-19"), "storage_bcf"])
    assert features.loc[pd.Timestamp("2026-01-21"), "storage_bcf"] == 1000
