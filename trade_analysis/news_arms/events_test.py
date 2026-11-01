"""15.4: the overnight marker marks exactly the 14 listed rows of the frozen file."""
from __future__ import annotations

from . import events as E

EXPECTED = ["2017-04-24", "2018-05-29", "2019-03-25", "2019-05-06", "2020-11-09", "2021-03-08",
            "2022-02-22", "2022-08-08", "2023-02-06", "2024-06-10", "2024-07-22", "2025-01-21",
            "2025-07-07", "2025-07-28"]


def test_exactly_the_fourteen_listed_rows_are_overnight():
    rows = E.load_geopolitical()
    marked = sorted(r["date"] for r in rows if E.in_session_break(r) and E.is_overnight(r))
    assert marked == EXPECTED


def test_the_half_day_row_is_not_marked_and_nothing_else_has_an_unmatched_bracket():
    rows = E.load_geopolitical()
    unmatched = [r for r in rows if E.in_session_break(r) and "(" in r["headline"]
                 and not E.is_overnight(r)]
    assert [r["date"] for r in unmatched] == ["2020-12-24"]
    assert "half-day" in unmatched[0]["headline"]


def test_event_minute_rules():
    assert E.event_minute({"break_time_et": "14:31", "headline": "x (Sunday)"}) is None
    assert E.event_minute({"break_time_et": "08:30", "headline": "x"}) is None
    assert E.event_minute({"break_time_et": "unknown", "headline": "x"}) is None
    assert E.event_minute({"break_time_et": "14:31", "headline": "x"}) == "14:31"
    assert E.event_minute({"break_time_et": "10:00", "headline": "x (half-day session)"}) == "10:00"
