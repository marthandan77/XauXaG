from metals_forecast.scorecards import build_scorecards, empirical_range_calibration


def outcomes(n=30):
    return [{
        "model_name": "ridge", "horizon": 1, "close_error": 1.0,
        "low_error": -0.5, "high_error": 0.5, "range_contains_day": True,
        "predicted_range_width": 2.0,
    } for _ in range(n)]


def test_scorecard_is_individual() -> None:
    cards = build_scorecards(outcomes())
    assert cards[0]["model_name"] == "ridge"
    assert cards[0]["status"] == "EVALUATED"
    assert cards[0]["range_coverage"] == 1.0
    assert cards[0]["average_range_width"] == 2.0


def test_calibration_requires_minimum_history() -> None:
    assert empirical_range_calibration(outcomes(2), model_name="ridge", horizon=1)["status"] == "INSUFFICIENT_DATA"
    assert empirical_range_calibration(outcomes(), model_name="ridge", horizon=1)["status"] == "CALIBRATED"


def test_scorecard_exposes_recent_windows() -> None:
    card = build_scorecards(outcomes(30))[0]
    assert card["recent_20_abs_close_error"] is not None
    assert card["recent_60_abs_close_error"] is None
