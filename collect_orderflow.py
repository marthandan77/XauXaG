"""Continuously archive public Coinbase Exchange metals trades and depth snapshots."""
from __future__ import annotations

import argparse
import time

from metals_forecast.orderflow import fetch_orderflow, record_orderflow


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interval", type=float, default=5.0, help="Polling interval in seconds (minimum 5; default: 5)")
    args = parser.parse_args()
    if args.interval < 5:
        parser.error("--interval must be at least five seconds to stay within public API rate limits")

    print("Collecting public Coinbase Exchange PAXG-USD flow data. Press Ctrl+C to stop.")
    while True:
        started = time.monotonic()
        for symbol in ("XAUUSD",):
            try:
                data = fetch_orderflow(symbol)
                record_orderflow(data)
                print(
                    f"{symbol}: saved {data['trade_count']} recent trades and depth snapshot "
                    f"at {data['updated_at'].isoformat()}"
                )
            except Exception as exc:
                print(f"{symbol}: collection failed: {type(exc).__name__}: {exc}")
        time.sleep(max(0.0, args.interval - (time.monotonic() - started)))


if __name__ == "__main__":
    main()
