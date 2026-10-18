from metals_forecast.calibration import calibrate_range


def outcome(low=-1.0, high=2.0):
    return {"model_name": "ridge", "horizon": 1, "low_error": low, "high_error": high}


def test_calibration_waits_for_matured_history() -> None:
    forecast = {"model_name": "ridge", "horizon": 1, "predicted_low": 10.0, "predicted_high": 12.0}
    result = calibrate_range(forecast, [outcome()] * 2)
    assert result["calibration_status"] == "UNCALIBRATED"
    assert result["predicted_low"] == 10.0


def test_calibration_expands_bounds_from_matured_errors() -> None:
    forecast = {"model_name": "ridge", "horizon": 1, "predicted_low": 10.0, "predicted_high": 12.0}
    result = calibrate_range(forecast, [outcome(-1.0, 2.0)] * 30)
    assert result["calibration_status"] == "CALIBRATED"
    assert result["predicted_low"] < 10.0
    assert result["predicted_high"] > 12.0
