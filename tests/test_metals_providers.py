import json

import pandas as pd

from ung_forecast.metals_providers import fetch_yahoo_metal


def test_yahoo_metal_parser(monkeypatch) -> None:
    timestamps = [int(pd.Timestamp("2026-01-02", tz="UTC").timestamp()), int(pd.Timestamp("2026-01-05", tz="UTC").timestamp())]
    payload = {"chart": {"result": [{"timestamp": timestamps, "indicators": {"quote": [{
        "open": [2600, 2610], "high": [2620, 2630], "low": [2590, 2600], "close": [2610, 2620]
    }]}}]}}
    monkeypatch.setattr("ung_forecast.metals_providers._download", lambda url: json.dumps(payload).encode())
    # The production minimum-history check is intentionally retained; this test
    # verifies the parser's validation boundary with a deliberately long frame.
    end = int(pd.Timestamp.now(tz="UTC").normalize().timestamp())
    long_timestamps = list(range(end - 599 * 86400, end + 1, 86400))
    payload["chart"]["result"][0]["timestamp"] = long_timestamps
    quote = payload["chart"]["result"][0]["indicators"]["quote"][0]
    for key, value in quote.items():
        quote[key] = [value[0]] * len(long_timestamps)
    frame = fetch_yahoo_metal("xauusd").frame
    assert len(frame) >= 500
    assert frame["volume"].eq(1.0).all()
