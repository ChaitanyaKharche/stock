"""Tests for freeze-and-catch-up (catchup.py) and the runners' frozen-entry hooks.

What can lie here, in order of danger:

  1. A BATCH AFTER A GAP. Missed minutes must be replayed one at a time, each at its own
     instant, against history -- never admitted in one lump and priced at the reconnect.
  2. AN ENTRY WHILE AWAY. Nothing the runner could not see live may be traded.
  3. A CATCH-UP THAT BREAKS THE RUNNER. If history is unreachable the live feed and the
     store must come back exactly as they were, and the runner must stay frozen.
  4. RE-MANAGING THE PAST. Bars seen before the gap are admitted silently, never managed.
  5. A FALSE ALARM. Normal poll jitter must never look like a gap.
"""
from __future__ import annotations

import datetime as dt
import json
from types import SimpleNamespace

import pytest

from trade_analysis.live_lab import catchup as C
from trade_analysis.live_lab.feed import FeedOutage
from trade_analysis.live_lab.session import SessionState

DAY = dt.date(2026, 9, 28)


def _t(h, m, s=0):
    return dt.datetime.combine(DAY, dt.time(h, m, s))


def _bars(n=390):
    return [{"ts": _t(9, 30) + dt.timedelta(minutes=i), "open": 100, "high": 100.1,
             "low": 99.9, "close": 100, "volume": 1} for i in range(n)]


class _LiveFeed:
    """Stands in for ThetaLiveFeed's HISTORY endpoints (what HistoryFeed calls)."""

    def __init__(self, fail=False):
        self.fail = fail

    def _fetch_minute_bars(self, sym, day):
        if self.fail:
            raise FeedOutage("HTTP 503: Unable to resolve host mdds-01.thetadata.us")
        return _bars()

    def _get_csv(self, path, **k):
        if self.fail:
            raise FeedOutage("HTTP 503")
        return [{"timestamp": (_t(9, 30) + dt.timedelta(minutes=i)).isoformat(),
                 "bid": "100", "ask": "100.02"} for i in range(391)]


class _Store:
    def __init__(self):
        self.trades, self.events, self.outages = [], [], []

    def write_trade(self, t):
        self.trades.append(t)

    def event(self, kind, **f):
        self.events.append((kind, f))

    def outage(self, kind, detail, **f):
        self.outages.append(kind)


class _Runner:
    """Just enough of a runner: a store, a feed, sessions, and a _tick that records what
    it saw and 'exits' a position once."""

    def __init__(self, feed):
        self.feed, self.store = feed, _Store()
        self.sessions = {"QQQ": SessionState("QQQ", DAY, {}, None)}
        self.ticks = []
        self.catchup = C.CatchUp(self, "t")

    def _tick(self, now, day):
        sess = self.sessions["QQQ"]
        admitted = sess.accept_bars(self.feed.minute_bars("QQQ", day, now=now), now)
        q = self.feed.stock_quote("QQQ")
        self.ticks.append((now, type(self.feed).__name__, len(admitted), q and q["ts"]))
        if now == _t(10, 25, 2):
            self.store.write_trade({"setup_id": "S", "exit_ts": now.isoformat()})


def _seen_until(r, t):
    """The live runner had admitted every bar that closed by `t`."""
    r.sessions["QQQ"].accept_bars(_bars(), t)


# ------------------------------------------------------------ 5. false alarms

def test_jitter_is_not_a_gap_and_a_long_silence_is():
    c = C.CatchUp(SimpleNamespace(), "t")
    c.note_good(_t(10, 0, 0))
    assert not c.away(_t(10, 1, 15))
    assert c.away(_t(10, 1, 31))


# ------------------------------------------------------------ 1 + 4. one minute at a time

def test_missed_minutes_are_replayed_one_at_a_time_on_history():
    r = _Runner(_LiveFeed())
    _seen_until(r, _t(10, 20, 7))
    r.catchup.note_good(_t(10, 20, 7))
    assert r.catchup.run(_t(10, 47, 30), DAY)
    times = [x[0] for x in r.ticks]
    assert times[0] == _t(10, 21, 2) and times[-1] == _t(10, 47, 2)
    assert len(times) == 27, "every missed minute gets its own tick"
    assert {x[1] for x in r.ticks} == {"HistoryFeed"}, "a catch-up tick used the live feed"
    assert all(x[2] == 1 for x in r.ticks), "a tick admitted a batch instead of one bar"
    assert all(q is not None and q <= t for t, _, _, q in r.ticks), "quote after its minute"


def test_bars_seen_before_the_gap_are_not_managed_again():
    r = _Runner(_LiveFeed())          # fresh session: a RESTART that saw nothing yet
    r.catchup.note_good(_t(10, 20, 7))
    r.catchup.run(_t(10, 23, 30), DAY)
    assert r.ticks[0][2] == 1, "pre-gap bars were handed to _tick instead of pre-admitted"
    assert r.sessions["QQQ"].bars_1m[0]["ts"] == _t(9, 30)


def test_catch_up_exits_are_labelled_and_the_feed_is_restored():
    live = _LiveFeed()
    r = _Runner(live)
    _seen_until(r, _t(10, 20, 7))
    r.catchup.note_good(_t(10, 20, 7))
    r.catchup.run(_t(10, 30, 0), DAY)
    assert r.store.trades[0]["exit_mode"] == "caught_up_from_history"
    assert r.store.trades[0]["caught_up_window"][0] == "2026-09-28T10:20:07"
    assert r.feed is live and not r.catchup.frozen
    assert r.store.events[-1][0] == "caught_up"


def test_a_catch_up_after_the_close_stops_at_the_close():
    r = _Runner(_LiveFeed())
    _seen_until(r, _t(15, 39, 33))
    r.catchup.note_good(_t(15, 39, 33))
    r.catchup.run(_t(16, 19, 9), DAY)
    assert r.ticks[-1][0] < _t(16, 0), "replayed past the close"
    assert r.ticks[-1][0] == _t(15, 59, 2)


# ------------------------------------------------------------ 3. history unreachable

def test_unreachable_history_leaves_the_runner_frozen_and_intact():
    live = _LiveFeed(fail=True)
    r = _Runner(live)
    write = r.store.write_trade
    r.catchup.note_good(_t(10, 20, 7))
    assert r.catchup.run(_t(10, 47, 30), DAY) is False
    assert r.feed is live and r.store.write_trade == write
    assert r.catchup.frozen, "must stay frozen until history is back"
    assert "catch_up_blocked" in r.store.outages
    assert r.catchup.last_good == _t(10, 20, 7), "progress claimed that did not happen"


# ------------------------------------------------------------ 2. no entries while away

def test_options_runner_refuses_entries_while_frozen(tmp_path, monkeypatch):
    from trade_analysis.live_lab import runner as R
    lab = R.LiveLab(["QQQ"], lab_dir=tmp_path)
    lab.feed.close()
    opened = []
    lab._open = lambda *a, **k: opened.append(a)
    monkeypatch.setattr(R, "ALL_SETUPS", [SimpleNamespace(
        id="S", timeframe="1m", max_per_day=9, max_per_direction=None,
        evaluate=lambda ctx: SimpleNamespace(direction="long", state={}))])
    sess = SimpleNamespace(context=lambda *a: SimpleNamespace())
    lab.catchup.frozen = True
    lab._evaluate("QQQ", sess, _t(10, 0), "1m", {"mid": 1}, _t(10, 1, 2), DAY)
    assert opened == [] and lab.catchup.suppressed == 1
    lab.catchup.frozen = False
    lab._evaluate("QQQ", sess, _t(10, 0), "1m", {"mid": 1}, _t(10, 1, 2), DAY)
    assert len(opened) == 1


def test_shares_runner_refuses_entries_while_frozen(tmp_path):
    from trade_analysis.live_lab import shares_runner as SR
    lab = SR.SharesLab(["QQQ"], lab_dir=tmp_path)
    lab.feed.close()
    lab.catchup.frozen = True
    bar = {"ts": _t(10, 0), "open": 1, "high": 1, "low": 1, "close": 1, "volume": 1}
    lab._enter("QQQ", None, bar, SimpleNamespace(), None,
               {"ts": _t(10, 1), "bid": 1, "ask": 1.01, "mid": 1.005}, _t(10, 1, 2))
    assert lab.open_pos == [] and lab.catchup.suppressed == 1


# ------------------------------------------------------------ restarts

def test_a_restart_catches_up_from_the_last_checkpoint(tmp_path):
    (tmp_path / "positions_open.json").write_text(json.dumps(
        {"saved_at": "2026-09-28T10:31:05", "session_date": "2026-09-28", "positions": []}))
    assert C.checkpoint_time(tmp_path, DAY) == _t(10, 31, 5)
    assert C.checkpoint_time(tmp_path, dt.date(2026, 9, 29)) is None, "yesterday's book"
    (tmp_path / "positions_open.json").write_text(json.dumps(
        {"saved_at": "2026-09-28T09:10:00", "session_date": "2026-09-28", "positions": []}))
    assert C.checkpoint_time(tmp_path, DAY) is None, "a pre-open save is not a gap"


# ------------------------------------------------------------ decided once

def _decisions(tmp_path, rows):
    (tmp_path / "signals.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_decision_keys_are_todays_decisions_only(tmp_path):
    from trade_analysis.live_lab.store import LabStore
    _decisions(tmp_path, [
        {"phase": "DECISION", "setup_id": "Six_Lines", "symbol": "SPY", "bar_ts": "2026-09-28T09:39:00"},
        {"phase": "FILL", "setup_id": "Six_Lines", "symbol": "SPY", "bar_ts": "2026-09-28T09:40:00"},
        {"phase": "DECISION", "setup_id": "ORB_5min", "symbol": "QQQ", "bar_ts": "2026-09-25T09:41:00"},
    ])
    assert LabStore(tmp_path).decision_keys(DAY) == {("Six_Lines", "SPY", "2026-09-28T09:39:00")}


def test_the_options_runner_never_decides_the_same_signal_twice(tmp_path):
    from trade_analysis.live_lab import runner as R
    from trade_analysis.live_lab.setups import by_id
    lab = R.LiveLab(["SPY"], lab_dir=tmp_path)
    lab.feed.close()
    lab._decided = {("Six_Lines", "SPY", "2026-09-28T09:39:00")}
    sig = SimpleNamespace(direction="long", state={})
    ctx = SimpleNamespace(price=1, vwap=1, bars_1m=(), bars_5m=(), tf="1m")
    lab._open(by_id["Six_Lines"], sig, "SPY", ctx, {"mid": 1}, _t(9, 40, 2), _t(9, 39), DAY)
    rows = [json.loads(l) for l in (tmp_path / "signals.jsonl").read_text().splitlines()]
    assert [r["phase"] for r in rows] == ["SKIP"] and rows[0]["skip_reason"] == "already_decided"


def test_the_shares_runner_never_decides_the_same_signal_twice(tmp_path, monkeypatch):
    from trade_analysis.live_lab import shares_runner as SR
    from trade_analysis.live_lab.session import Signal
    lab = SR.SharesLab(["SPY"], lab_dir=tmp_path)
    lab.feed.close()
    monkeypatch.setattr(SR, "ALL_SETUPS", [SimpleNamespace(
        id="S", timeframe="1m", max_per_day=9, max_per_direction=None,
        evaluate=lambda ctx: Signal("S", "long", stop=None, target=None))])
    lab._decided = {("S", "SPY", "2026-09-28T10:00:00")}
    bar = {"ts": _t(10, 0), "open": 1, "high": 1, "low": 1, "close": 1, "volume": 1}
    lab._enter("SPY", None, bar, SimpleNamespace(), None,
               {"ts": _t(10, 1), "bid": 1, "ask": 1.01, "mid": 1.005}, _t(10, 1, 2))
    assert lab.open_pos == [], "a signal already decided was taken again"
