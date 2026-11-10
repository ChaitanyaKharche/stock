"""NYSE full-day closures, by rule, so a replay can tell a holiday from a dead feed.

2026-11-10: 23 regular 2022 sessions came back with zero bars during a vendor outage and
`backfill.run` logged each as "not a session, skipped" (its only test was the bar count).
A weekday with no bars that is not on this calendar is a feed problem and must raise.
"""
from __future__ import annotations

import datetime as dt


def easter(year: int) -> dt.date:
    """Gregorian Easter Sunday (anonymous algorithm)."""
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    l = (32 + 2 * e + 2 * i - h - k) % 7                      # noqa: E741
    m = (a + 11 * h + 22 * l) // 451
    month, day = divmod(h + l - 7 * m + 114, 31)
    return dt.date(year, month, day + 1)


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> dt.date:
    d = dt.date(year, month, 1)
    d += dt.timedelta(days=(weekday - d.weekday()) % 7)
    return d + dt.timedelta(weeks=n - 1)


def _last_weekday(year: int, month: int, weekday: int) -> dt.date:
    d = dt.date(year + (month == 12), (month % 12) + 1, 1) - dt.timedelta(days=1)
    return d - dt.timedelta(days=(d.weekday() - weekday) % 7)


def _observed(d: dt.date) -> dt.date | None:
    """Saturday holidays are observed Friday, Sunday ones Monday; a Saturday New Year's Day
    is not observed at all (NYSE rule)."""
    if d.weekday() == 5:
        return None if (d.month, d.day) == (1, 1) else d - dt.timedelta(days=1)
    if d.weekday() == 6:
        return d + dt.timedelta(days=1)
    return d


SPECIAL_CLOSURES = {
    dt.date(2012, 10, 29), dt.date(2012, 10, 30),             # Hurricane Sandy
    dt.date(2018, 12, 5),                                     # G. H. W. Bush funeral
    dt.date(2025, 1, 9),                                      # J. Carter funeral
}


def holidays(year: int) -> set[dt.date]:
    out = set()
    for d in (dt.date(year, 1, 1), dt.date(year, 7, 4), dt.date(year, 12, 25)):
        o = _observed(d)
        if o:
            out.add(o)
    if year >= 2022:
        o = _observed(dt.date(year, 6, 19))
        if o:
            out.add(o)
    out.add(_nth_weekday(year, 1, 0, 3))                      # MLK Day
    out.add(_nth_weekday(year, 2, 0, 3))                      # Presidents Day
    out.add(easter(year) - dt.timedelta(days=2))              # Good Friday
    out.add(_last_weekday(year, 5, 0))                        # Memorial Day
    out.add(_nth_weekday(year, 9, 0, 1))                      # Labor Day
    out.add(_nth_weekday(year, 11, 3, 4))                     # Thanksgiving
    out |= {d for d in SPECIAL_CLOSURES if d.year == year}
    return out


def is_closed(d: dt.date) -> bool:
    return d.weekday() >= 5 or d in holidays(d.year)


def early_close(d: dt.date) -> bool:
    """Scheduled 13:00 closes: the day after Thanksgiving, July 3 and December 24 when they
    fall Monday-Thursday (a Friday one is either a full day or the observed holiday)."""
    if d.month == 11 and d == _nth_weekday(d.year, 11, 3, 4) + dt.timedelta(days=1):
        return True
    return (d.month, d.day) in ((7, 3), (12, 24)) and d.weekday() <= 3
