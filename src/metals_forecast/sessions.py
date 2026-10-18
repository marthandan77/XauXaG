"""Market-session helpers for forecast horizons."""
from __future__ import annotations

from datetime import date

import pandas as pd
from pandas.tseries.holiday import GoodFriday, USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay


def _calendar_days(start: str | date, end: str | date) -> pd.DatetimeIndex:
    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    holidays = USFederalHolidayCalendar().holidays(start=start_ts, end=end_ts)
    good_fridays = GoodFriday.dates(start_ts, end_ts)
    return pd.DatetimeIndex(sorted(set(holidays.tolist()) | set(good_fridays.tolist())))


def next_sessions(last_session: pd.Timestamp | str, count: int) -> list[pd.Timestamp]:
    """Return the next US sessions; this is a conservative daily forecast calendar."""
    if count < 1:
        raise ValueError("count must be positive")
    start = pd.Timestamp(last_session).normalize()
    end = start + pd.Timedelta(days=count * 4 + 30)
    holiday_dates = _calendar_days(start, end)
    offset = CustomBusinessDay(holidays=holiday_dates)
    return [start + offset * step for step in range(1, count + 1)]
