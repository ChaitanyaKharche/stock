"""17.2 / 14.11: market-hours refusal and the no-0DTE halt rule."""
from __future__ import annotations

import datetime as dt

from . import replay_chunks as R


def test_refuses_inside_market_hours_on_a_weekday():
    assert R.in_market_hours(dt.datetime(2026, 11, 4, 10, 0))          # Wednesday 10:00
    assert not R.in_market_hours(dt.datetime(2026, 11, 4, 17, 0))
    assert not R.in_market_hours(dt.datetime(2026, 11, 7, 10, 0))      # Saturday


def test_halt_only_where_a_zero_dte_was_expected():
    has = {"2021-03-09": False,   # Tuesday 2021: no 0DTE expected -> fine
           "2021-03-10": False,   # Wednesday 2021: expected -> halt
           "2022-11-03": False,   # Thursday after 2022-11-01: expected -> halt
           "2023-06-14": True}
    assert R.halt_reasons(has) == ["2021-03-10", "2022-11-03"]
