"""US equity trading calendar.

Every downloader iterates dates, and requesting a non-session is pure waste: at
one request per symbol per day across ~100 symbols, the ~10 market holidays a
year plus weekends are the difference between 25k and 37k requests annually.
Worse, a holiday returns NO_DATA, which is indistinguishable from a genuine
coverage gap once it is written to the manifest - so filtering them out up front
keeps "missing" meaning something.

Uses pandas_market_calendars when installed (authoritative, includes the
one-off closures). Falls back to a hardcoded holiday table otherwise, which is
maintained back to 2004 because the entitlement prober searches that far. The
fallback also handles the early closes it knows about, but early closes only
shorten a session - they never remove one - so they do not affect which dates
get requested.
"""
from __future__ import annotations

from datetime import date, timedelta
from functools import lru_cache

# Full-day closures. Weekends are handled separately. Sources: NYSE holiday
# archive; includes the irregular closures (Reagan/Ford/Bush funerals, Sandy,
# 9/11, Carter 2025) that a weekday-only rule would wrongly treat as sessions.
_HOLIDAYS = {
    # 2004
    "2004-01-01", "2004-01-19", "2004-02-16", "2004-04-09", "2004-05-31",
    "2004-06-11", "2004-07-05", "2004-09-06", "2004-11-25", "2004-12-24",
    # 2005
    "2005-01-17", "2005-02-21", "2005-03-25", "2005-05-30", "2005-07-04",
    "2005-09-05", "2005-11-24", "2005-12-26",
    # 2006
    "2006-01-02", "2006-01-16", "2006-02-20", "2006-04-14", "2006-05-29",
    "2006-07-04", "2006-09-04", "2006-11-23", "2006-12-25",
    # 2007
    "2007-01-01", "2007-01-02", "2007-01-15", "2007-02-19", "2007-04-06",
    "2007-05-28", "2007-07-04", "2007-09-03", "2007-11-22", "2007-12-25",
    # 2008
    "2008-01-01", "2008-01-21", "2008-02-18", "2008-03-21", "2008-05-26",
    "2008-07-04", "2008-09-01", "2008-11-27", "2008-12-25",
    # 2009
    "2009-01-01", "2009-01-19", "2009-02-16", "2009-04-10", "2009-05-25",
    "2009-07-03", "2009-09-07", "2009-11-26", "2009-12-25",
    # 2010
    "2010-01-01", "2010-01-18", "2010-02-15", "2010-04-02", "2010-05-31",
    "2010-07-05", "2010-09-06", "2010-11-25", "2010-12-24",
    # 2011
    "2011-01-17", "2011-02-21", "2011-04-22", "2011-05-30", "2011-07-04",
    "2011-09-05", "2011-11-24", "2011-12-26",
    # 2012 (Sandy closed 29-30 Oct)
    "2012-01-02", "2012-01-16", "2012-02-20", "2012-04-06", "2012-05-28",
    "2012-07-04", "2012-09-03", "2012-10-29", "2012-10-30", "2012-11-22",
    "2012-12-25",
    # 2013
    "2013-01-01", "2013-01-21", "2013-02-18", "2013-03-29", "2013-05-27",
    "2013-07-04", "2013-09-02", "2013-11-28", "2013-12-25",
    # 2014
    "2014-01-01", "2014-01-20", "2014-02-17", "2014-04-18", "2014-05-26",
    "2014-07-04", "2014-09-01", "2014-11-27", "2014-12-25",
    # 2015
    "2015-01-01", "2015-01-19", "2015-02-16", "2015-04-03", "2015-05-25",
    "2015-07-03", "2015-09-07", "2015-11-26", "2015-12-25",
    # 2016
    "2016-01-01", "2016-01-18", "2016-02-15", "2016-03-25", "2016-05-30",
    "2016-07-04", "2016-09-05", "2016-11-24", "2016-12-26",
    # 2017
    "2017-01-02", "2017-01-16", "2017-02-20", "2017-04-14", "2017-05-29",
    "2017-07-04", "2017-09-04", "2017-11-23", "2017-12-25",
    # 2018 (Bush funeral 5 Dec)
    "2018-01-01", "2018-01-15", "2018-02-19", "2018-03-30", "2018-05-28",
    "2018-07-04", "2018-09-03", "2018-11-22", "2018-12-05", "2018-12-25",
    # 2019
    "2019-01-01", "2019-01-21", "2019-02-18", "2019-04-19", "2019-05-27",
    "2019-07-04", "2019-09-02", "2019-11-28", "2019-12-25",
    # 2020
    "2020-01-01", "2020-01-20", "2020-02-17", "2020-04-10", "2020-05-25",
    "2020-07-03", "2020-09-07", "2020-11-26", "2020-12-25",
    # 2021
    "2021-01-01", "2021-01-18", "2021-02-15", "2021-04-02", "2021-05-31",
    "2021-07-05", "2021-09-06", "2021-11-25", "2021-12-24",
    # 2022 (Juneteenth observed from 2022)
    "2022-01-17", "2022-02-21", "2022-04-15", "2022-05-30", "2022-06-20",
    "2022-07-04", "2022-09-05", "2022-11-24", "2022-12-26",
    # 2023
    "2023-01-02", "2023-01-16", "2023-02-20", "2023-04-07", "2023-05-29",
    "2023-06-19", "2023-07-04", "2023-09-04", "2023-11-23", "2023-12-25",
    # 2024 (Carter funeral was 2025; 2024 had none extra)
    "2024-01-01", "2024-01-15", "2024-02-19", "2024-03-29", "2024-05-27",
    "2024-06-19", "2024-07-04", "2024-09-02", "2024-11-28", "2024-12-25",
    # 2025 (Carter national day of mourning 9 Jan)
    "2025-01-01", "2025-01-09", "2025-01-20", "2025-02-17", "2025-04-18",
    "2025-05-26", "2025-06-19", "2025-07-04", "2025-09-01", "2025-11-27",
    "2025-12-25",
    # 2026
    "2026-01-01", "2026-01-19", "2026-02-16", "2026-04-03", "2026-05-25",
    "2026-06-19", "2026-07-03", "2026-09-07", "2026-11-26", "2026-12-25",
}

# Sessions ending 13:00 ET. Not used to skip requests - a short session is still
# a session - but exported so a downstream bar-count check does not flag them as
# missing data. A 1s layer legitimately has 12600 rows on these days, not 23400.
_EARLY_CLOSES = {
    "2004-11-26", "2004-12-23", "2005-11-25", "2006-07-03", "2006-11-24",
    "2007-07-03", "2007-11-23", "2007-12-24", "2008-07-03", "2008-11-28",
    "2008-12-24", "2009-11-27", "2009-12-24", "2010-11-26", "2010-12-24",
    "2011-11-25", "2012-07-03", "2012-11-23", "2012-12-24", "2013-07-03",
    "2013-11-29", "2013-12-24", "2014-07-03", "2014-11-28", "2014-12-24",
    "2015-11-27", "2015-12-24", "2016-11-25", "2017-07-03", "2017-11-24",
    "2018-07-03", "2018-11-23", "2018-12-24", "2019-07-03", "2019-11-29",
    "2019-12-24", "2020-11-27", "2020-12-24", "2021-11-26", "2022-11-25",
    "2023-07-03", "2023-11-24", "2024-07-03", "2024-11-29", "2024-12-24",
    "2025-07-03", "2025-11-28", "2025-12-24", "2026-11-27", "2026-12-24",
}


def _try_market_calendar(start: date, end: date):
    """Authoritative calendar if the library is present, else None."""
    try:
        import pandas_market_calendars as mcal
    except ImportError:
        return None
    try:
        sched = mcal.get_calendar("NYSE").schedule(
            start_date=start.isoformat(), end_date=end.isoformat())
        return [d.date() for d in sched.index]
    except Exception:
        # A calendar library that errors is worse than not having one: fall back
        # rather than let an exception decide the date range.
        return None


# Unbounded on purpose. The key space is the set of distinct date RANGES the
# downloaders ask for - a few hundred at most - while each miss rebuilds a
# full exchange schedule. A 64-entry cache thrashed on a 128-month task
# build and turned a dry run into minutes of calendar construction.
@lru_cache(maxsize=None)
def trading_days(start, end) -> list[date]:
    """Every US equity session in [start, end] inclusive, ascending."""
    start = _as_date(start)
    end = _as_date(end)
    if start > end:
        return []
    via_lib = _try_market_calendar(start, end)
    if via_lib is not None:
        return via_lib
    out = []
    d = start
    while d <= end:
        if d.weekday() < 5 and d.isoformat() not in _HOLIDAYS:
            out.append(d)
        d += timedelta(days=1)
    return out


@lru_cache(maxsize=None)
def _sessions_in_year(year: int) -> frozenset[str]:
    return frozenset(x.isoformat() for x in
                     trading_days(date(year, 1, 1), date(year, 12, 31)))


def is_trading_day(d) -> bool:
    """Cached per year: this is called once per (symbol, date) pair, which is
    ~10^5 times on a full task build."""
    d = _as_date(d)
    return d.isoformat() in _sessions_in_year(d.year)


def is_early_close(d) -> bool:
    return _as_date(d).isoformat() in _EARLY_CLOSES


def nearest_trading_day(d, direction: int = -1) -> date:
    """Snap to a session. direction -1 = on/before, +1 = on/after."""
    d = _as_date(d)
    for _ in range(12):
        if is_trading_day(d):
            return d
        d += timedelta(days=direction)
    return d


def month_chunks(start, end) -> list[tuple[date, date]]:
    """Split [start, end] into calendar-month spans.

    The gateway rejects any bulk request spanning more than a month
    ("Bulk history requests are limited to no more than 1 month"), and calendar
    months are used rather than rolling 31-day windows so a re-run produces
    byte-identical chunk boundaries - otherwise resumption keys drift and
    already-downloaded spans get re-fetched under a new key.
    """
    start, end = _as_date(start), _as_date(end)
    out = []
    cur = start.replace(day=1)
    while cur <= end:
        nxt = (cur.replace(day=28) + timedelta(days=4)).replace(day=1)
        lo = max(cur, start)
        hi = min(nxt - timedelta(days=1), end)
        if lo <= hi and trading_days(lo, hi):
            out.append((lo, hi))
        cur = nxt
    return out


def _as_date(d) -> date:
    if isinstance(d, date):
        return d
    s = str(d)[:10]
    if "-" in s:
        y, m, dd = s.split("-")
    else:
        s = str(d)[:8]
        y, m, dd = s[:4], s[4:6], s[6:8]
    return date(int(y), int(m), int(dd))


if __name__ == "__main__":
    import sys
    a = sys.argv[1] if len(sys.argv) > 1 else "2024-01-01"
    b = sys.argv[2] if len(sys.argv) > 2 else "2024-12-31"
    days = trading_days(a, b)
    print(f"{a} -> {b}: {len(days)} sessions "
          f"({'pandas_market_calendars' if _try_market_calendar(_as_date(a), _as_date(a)) is not None else 'builtin table'})")
    print(f"first {days[0]}  last {days[-1]}")
    print(f"early closes in range: "
          f"{sum(1 for d in days if is_early_close(d))}")
    print(f"month chunks: {len(month_chunks(a, b))}")
