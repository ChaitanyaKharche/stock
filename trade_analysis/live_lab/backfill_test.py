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
import json

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


def test_a_whole_shares_session_replays_and_leaves_a_log(tmp_path, monkeypatch):
    """The full-day harness, end to end, on canned history: warmup, 390 ticks, flatten,
    daily summary, and the runner's own console output captured as that day's log."""
    from trade_analysis.live_lab import shares_runner as SR
    monkeypatch.setattr(B, "CACHE", tmp_path / "cache")
    monkeypatch.setattr(SR, "ThetaLiveFeed", lambda *a, **k: type(
        "F", (_Real,), {"close": lambda self: None, "calls": 0})())
    out = tmp_path / "shares"
    summary = B.run(DAY, DAY, symbols=("QQQ",), out=out, arm="shares")
    assert len(summary) == 1 and summary[0]["date"] == DAY.isoformat()
    log = (out / "logs" / f"{DAY}.log").read_text(encoding="utf-8")
    assert "NOT the prospective record" in log and "warmup QQQ" in log
    assert (out / "daily" / f"{DAY}.json").exists()
    assert json.loads((out / "BACKFILL.json").read_text())["arm"] == "shares"


def test_reconcile_bins_every_trade_once_and_the_identity_holds():
    """live - replay splits exactly into matched differences + live_only - replay_only.
    A trade counted twice, or dropped, breaks the identity."""
    def t(setup, bar, exit_ts, reason, pnl):
        return {"setup_id": setup, "symbol": "QQQ", "direction": "long",
                "entry_bar_ts": f"2026-09-23T{bar}:00", "exit_ts": f"2026-09-23T{exit_ts}",
                "exit_reason": reason, "pnl_net": pnl}
    live = [t("A", "10:00", "10:30:05", "stop", -5),       # same
            t("B", "10:05", "15:55:02", "eod", 3),         # moved (live exited late)
            t("C", "09:31", "09:40:00", "target", 7),      # live only (stale entry)
            t("A", "10:00", "11:00:00", "stop", -1),       # a duplicate decision live
            t("E", "11:01", "11:30:00", "stop", 2)]        # shifted: revised bar in history
    replay = [t("A", "10:00", "10:30:02", "stop", -4),
              t("B", "10:05", "14:10:02", "stop", -2),
              t("E", "11:02", "11:31:02", "stop", 4),
              t("D", "09:35", "09:50:02", "bars", 11)]     # replay only (blind open live)
    r = B.reconcile(live, replay, detail=True)
    assert [r[k]["n"] for k in ("same", "moved", "shifted", "live_only", "replay_only")]         == [1, 1, 1, 2, 1]
    assert [x["setup_id"] for x in r["_live_only"]] == ["C", "A"]
    assert [x["setup_id"] for x in r["_replay_only"]] == ["D"]
    lhs = sum(x["pnl_net"] for x in live) - sum(x["pnl_net"] for x in replay)
    rhs = sum(r[k]["live"] - r[k]["replay"] for k in ("same", "moved", "shifted"))         + r["live_only"]["live"] - r["replay_only"]["replay"]
    assert lhs == pytest.approx(rhs)
