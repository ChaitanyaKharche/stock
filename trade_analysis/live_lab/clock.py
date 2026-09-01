"""The lab's single source of "now".

ThetaData stamps every timestamp in **Exchange Time (ET)**. The machine running the lab
need not be in ET -- the development machine is MST, two hours behind -- so calling
`datetime.now()` and treating the result as ET silently shifts every clock decision:
bar-completeness admission, the 09:30/15:55 session boundaries, and time-based exits.

At a 2-hour offset the runner would admit bars two hours late and keep trading past the
close. This module exists so that mistake cannot recur: nothing in the lab may call
`datetime.now()` directly.

All lab timestamps are **naive ET**, matching the feed.
"""
from __future__ import annotations

import datetime as dt
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")

RTH_OPEN = dt.time(9, 30)
RTH_LAST_BAR = dt.time(15, 59)
RTH_CLOSE = dt.time(16, 0)
EOD_FLAT = dt.time(15, 55)


def now_et() -> dt.datetime:
    """Current exchange time, naive, matching the feed's stamps."""
    return dt.datetime.now(ET).replace(tzinfo=None)


def today_et() -> dt.date:
    return now_et().date()


def is_rth(when: dt.datetime | None = None) -> bool:
    w = when or now_et()
    return w.weekday() < 5 and RTH_OPEN <= w.time() <= RTH_CLOSE


def offset_hours() -> float:
    """Local clock minus exchange clock, in hours. 0.0 when the machine is on ET."""
    local = dt.datetime.now()
    return round((local - now_et()).total_seconds() / 3600.0, 2)
