import pandas as pd

from metals_forecast.sessions import next_sessions


def test_friday_rolls_to_monday() -> None:
    result = next_sessions(pd.Timestamp("2026-10-16"), 2)
    assert result == [pd.Timestamp("2026-10-19"), pd.Timestamp("2026-10-20")]


def test_good_friday_is_skipped() -> None:
    result = next_sessions(pd.Timestamp("2026-04-02"), 1)
    assert result == [pd.Timestamp("2026-04-06")]
