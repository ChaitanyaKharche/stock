"""Tests for backfill.HistoryFeed -- the one component that could smuggle the future in.

A replay is only as honest as its feed. The runner code is production code; what is new is
a feed that holds the WHOLE day in memory while pretending it is 10:02. Each of these was
verified to fail with the corresponding guard removed from HistoryFeed.

  1. an unclosed bar served on the replayed day
  2. any date after the replayed day served at all
  3. a quote or option quote stamped after the simulated clock
  4. the output landing inside live_lab_data/, where the dashboard would count it
"""
from __future__ import annotations

import datetime as dt

import pytest

from trade_analysis.live_lab import backfill as B

DAY = dt.date(2026, 9, 23)


def _t(h, m, s=0):
    return dt.datetime.combine(DAY, dt.time(h, m, s))


class _Real:
    """Canned history: the full day, as the vendor serves it after the close."""

    def _fetch_minute_bars(self, sym, day):
        return [{"ts": _t(9, 30) + dt.timedelta(minutes=i), "open": 1, "high": 1, "low": 1,
                 "close": 1, "volume": 1} for i in range(390)]

    def extended_bars(self, sym, day, start, end):
        return [{"ts": _t(4, 0) + dt.timedelta(minutes=i), "open": 1, "high": 1, "low": 1,
                 "close": 1, "volume": 1} for i in range(16 * 60)]

    def _get_csv(self, path, **k):
        if path == "/stock/history/quote":
            return [{"timestamp": (_t(9, 30) + dt.timedelta(minutes=i)).isoformat(),
                     "bid": "100", "ask": "100.02", "bid_size": "1", "ask_size": "1"}
                    for i in range(391)]
        if path == "/option/history/quote":
            return [{"strike": "100.000", "right": "CALL",
                     "timestamp": (_t(9, 30) + dt.timedelta(minutes=i)).isoformat(),
                     "bid": str(1 + i / 100), "ask": str(1.02 + i / 100),
                     "bid_size": "1", "ask_size": "1"} for i in range(391)]
        if path == "/option/list/strikes":
            return [{"strike": "100.000"}]
        raise AssertionError(path)


@pytest.fixture
def feed(tmp_path, monkeypatch):
    monkeypatch.setattr(B, "CACHE", tmp_path)          # never reuse a real cache in a test
    clock = B.SimClock()
    clock.now = _t(10, 2, 2)
    return B.HistoryFeed(_Real(), clock, DAY), clock


def test_the_replayed_day_is_cut_at_the_clock_even_without_now(feed):
    f, clock = feed
    for bars in (f.minute_bars("QQQ", DAY), f.minute_bars("QQQ", DAY, now=_t(15, 59))):
        assert bars[-1]["ts"] == _t(10, 1), "a bar that had not closed was served"


def test_prior_days_are_whole_and_future_days_are_refused(feed):
    f, _ = feed
    assert len(f.minute_bars("QQQ", DAY - dt.timedelta(days=1))) == 390
    with pytest.raises(AssertionError):
        f.minute_bars("QQQ", DAY + dt.timedelta(days=1))
    with pytest.raises(AssertionError):
        f.extended_bars("QQQ", DAY + dt.timedelta(days=1))


def test_todays_premarket_is_only_what_has_closed(feed):
    f, clock = feed
    clock.now = _t(9, 31, 2)
    pre = f.extended_bars("QQQ", DAY, dt.time(4, 0), dt.time(9, 30))
    assert pre[-1]["ts"] == _t(9, 29)
    clock.now = _t(9, 0, 30)
    assert f.extended_bars("QQQ", DAY, dt.time(4, 0), dt.time(9, 30))[-1]["ts"] == _t(8, 59)


def test_quotes_never_postdate_the_clock(feed):
    f, clock = feed
    q = f.stock_quote("QQQ")
    assert q["ts"] <= clock.now and q["ts"] == _t(10, 2)
    rows = f.chain_quotes("QQQ", DAY)
    assert rows and all(r["ts"] <= clock.now for r in rows)
    assert rows[0]["bid"] == pytest.approx(1 + 32 / 100), "not the quote as of 10:02"


def test_backfill_output_is_outside_the_live_record():
    live = B.ROOT / "live_lab_data"
    assert live not in B.OUT.parents and B.OUT != live, \
        "backfill writing into live_lab_data would be archived and counted as prospective"
