"""The replay's information barrier, tested mechanism by mechanism.

    pytest trade_analysis/live_lab/backfill_barrier_test.py

research/news_arms/preregistration.md section 1.2 tests (a)-(g), 14.1 (e-extended, i),
specs/003 (cache) and specs/004 (proxy). Constructed rows: these prove the MECHANICS keep
future data out. What a vendor 1-minute row stamped T contains is probe (h), run on the
laptop by backfill_probe.py, not here.
"""
from __future__ import annotations

import datetime as dt

import pytest

from . import backfill as B
from . import store as S

D = dt.date(2023, 6, 14)
T = lambda h, m, s=0: dt.datetime.combine(D, dt.time(h, m, s))     # noqa: E731


class FakeReal:
    """Stands in for ThetaLiveFeed. Serves whatever rows the test put in."""

    def __init__(self, bars=None, quotes=None, chain=None, strikes=3):
        self.bars = bars or []
        self.quotes = quotes or []          # rows for /stock/history/quote
        self.chain = chain or []            # rows for /option/history/quote
        self.strikes = strikes
        self.calls = 0

    def _fetch_minute_bars(self, symbol, day):
        return [b for b in self.bars if b["ts"].date() == day]

    def extended_bars(self, symbol, day, start, end):
        return [b for b in self.bars if b["ts"].date() == day]

    def _get_csv(self, path, **params):
        self.calls += 1
        if path == "/stock/history/quote":
            return self.quotes
        if path == "/option/history/quote":
            return self.chain
        if path == "/option/list/strikes":
            return [{"strike": 100 + i} for i in range(self.strikes)]
        return []

    def stats(self):
        return {}

    def wait_for_upstream(self, *a, **k):
        return True

    def close(self):
        pass


def _bars(day, n=390):
    out, t = [], dt.datetime.combine(day, dt.time(9, 30))
    for _ in range(n):
        out.append({"ts": t, "open": 100.0, "high": 100.5, "low": 99.5, "close": 100.0,
                    "volume": 1000.0})
        t += dt.timedelta(minutes=1)
    return out


def _q(ts, bid, ask):
    return {"timestamp": ts.isoformat(), "bid": str(bid), "ask": str(ask),
            "bid_size": "10", "ask_size": "10"}


def _c(ts, strike, right, bid, ask):
    return {**_q(ts, bid, ask), "strike": str(strike), "right": right}


def _feed(real, now):
    clock = B.SimClock()
    clock.now = now
    return B.HistoryFeed(real, clock, D, disk_cache=False), clock


# ------------------------------------------------------------------ (a) bars close first

def test_a_bar_is_served_only_after_it_has_closed():
    feed, clock = _feed(FakeReal(bars=_bars(D)), T(10, 0, 30))
    served = feed.minute_bars("QQQ", D)
    assert served[-1]["ts"] == T(9, 59)            # the 10:00 bar closes at 10:01; not yet
    clock.now = T(10, 1, 0)
    assert feed.minute_bars("QQQ", D)[-1]["ts"] == T(10, 0)


def test_the_same_cut_applies_to_extended_and_batch_bars():
    feed, _ = _feed(FakeReal(bars=_bars(D)), T(10, 0, 30))
    assert feed.extended_bars("QQQ", D, dt.time(4, 0), dt.time(16, 0))[-1]["ts"] == T(9, 59)
    many, _ = feed.minute_bars_many(["QQQ"], D)
    assert many["QQQ"][-1]["ts"] == T(9, 59)


# ------------------------------------------------------------------ (b) no later day

def test_a_request_for_a_later_day_raises():
    feed, _ = _feed(FakeReal(bars=_bars(D)), T(10, 0))
    with pytest.raises(AssertionError):
        feed.minute_bars("QQQ", D + dt.timedelta(days=1))
    with pytest.raises(AssertionError):
        feed.extended_bars("QQQ", D + dt.timedelta(days=1), dt.time(4, 0), dt.time(16, 0))
    with pytest.raises(AssertionError):
        feed.zero_dte("SPY", D + dt.timedelta(days=1))


def test_an_earlier_day_is_served_whole():
    prev = D - dt.timedelta(days=1)
    feed, _ = _feed(FakeReal(bars=_bars(prev)), T(9, 31))
    assert len(feed.minute_bars("QQQ", prev)) == 390       # yesterday is complete history


# ------------------------------------------------------------------ (c) underlying NBBO

def test_the_quote_served_is_the_last_one_at_or_before_the_clock():
    real = FakeReal(quotes=[_q(T(10, 0), 100.0, 100.02), _q(T(10, 1), 101.0, 101.02)])
    feed, clock = _feed(real, T(10, 0, 30))
    assert feed.stock_quote("SPY")["bid"] == 100.0
    clock.now = T(10, 1, 0)
    assert feed.stock_quote("SPY")["bid"] == 101.0
    clock.now = T(9, 59, 59)
    assert feed.stock_quote("SPY") is None               # nothing yet


# ------------------------------------------------------------------ (d) chain at the clock

def test_a_chain_row_stamped_after_the_clock_is_not_served():
    real = FakeReal(chain=[_c(T(13, 59), 440, "C", 1.0, 1.1), _c(T(14, 1), 440, "C", 2.0, 2.1)])
    feed, clock = _feed(real, T(14, 0))
    assert [r["bid"] for r in feed.chain_quotes("SPY", D)] == [1.0]
    clock.now = T(14, 1)
    assert [r["bid"] for r in feed.chain_quotes("SPY", D)] == [2.0]


# ------------------------------------------------------------------ (e) store stamps

def test_every_store_record_carries_the_simulated_clock(tmp_path, monkeypatch):
    clock = B.SimClock()
    clock.now = T(10, 17)
    monkeypatch.setattr(S, "_now", clock)
    store = S.LabStore(tmp_path)
    store.outage("tick", "test")
    row = (tmp_path / "outages.jsonl").read_text(encoding="utf-8").strip().splitlines()[-1]
    assert '"ts":"2023-06-14T10:17:00"' in row.replace(" ", "")
    clock.now = T(10, 18)
    store.outage("tick", "test2")
    rows = (tmp_path / "outages.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert '"ts":"2023-06-14T10:18:00"' in rows[-1].replace(" ", "") and len(rows) == 2


# ------------------------------------------------------------------ (f) contract universe

def test_a_contract_first_quoted_at_11_is_absent_before_11():
    real = FakeReal(chain=[_c(T(9, 31), 440, "C", 1.0, 1.1), _c(T(11, 0), 445, "C", 0.5, 0.6)])
    feed, clock = _feed(real, T(10, 59))
    assert {r["strike"] for r in feed.chain_quotes("SPY", D)} == {440.0}
    clock.now = T(11, 0)
    assert {r["strike"] for r in feed.chain_quotes("SPY", D)} == {440.0, 445.0}


# ------------------------------------------------------------------ (g) strike choice

def test_zero_dte_answers_existence_only_and_never_a_strike_list():
    feed, _ = _feed(FakeReal(strikes=5), T(10, 0))
    assert feed.zero_dte("SPY", D) == D
    feed2, _ = _feed(FakeReal(strikes=0), T(10, 0))
    assert feed2.zero_dte("SPY", D) is None


def test_atm_choice_at_the_decision_minute_cannot_see_a_later_listed_strike():
    # spot 444: the 445 strike would be ATM, but it is first quoted at 11:00
    real = FakeReal(chain=[_c(T(9, 31), 440, "C", 1.0, 1.1), _c(T(9, 31), 450, "C", 0.2, 0.3),
                           _c(T(11, 0), 445, "C", 0.5, 0.6)])
    feed, _ = _feed(real, T(10, 0))
    chain = feed.chain_quotes("SPY", D)
    atm = min(chain, key=lambda r: abs(r["strike"] - 444.0))
    assert atm["strike"] == 440.0                      # 445 does not exist yet at 10:00


# ------------------------------------------------------------------ (i) the proxy, specs/004

def test_proxy_allows_only_the_guarded_surface():
    feed, _ = _feed(FakeReal(bars=_bars(D)), T(10, 1))
    proxy = B.GuardedFeed(feed)
    assert proxy.minute_bars("QQQ", D)[-1]["ts"] == T(10, 0)          # guarded path works
    assert proxy.wait_for_upstream() is True and isinstance(proxy.stats(), dict)
    for forbidden in ("_get_csv", "_fetch_minute_bars", "real", "_rth", "_c", "_closed"):
        with pytest.raises(AttributeError):
            getattr(proxy, forbidden)
    with pytest.raises(AttributeError):
        proxy.calls = 5                                                  # read-only


def test_run_attaches_the_proxy_not_the_history_feed():
    class Lab:
        feed = None
    feed, _ = _feed(FakeReal(), T(10, 0))
    lab = Lab()
    B._attach_feed(lab, feed)
    assert isinstance(lab.feed, B.GuardedFeed)
    with pytest.raises(AttributeError):
        lab.feed._get_csv("/x")


def test_the_surface_is_exactly_what_the_runners_use():
    assert B.GUARDED_SURFACE == {"minute_bars", "extended_bars", "stock_quote", "chain_quotes",
                                 "zero_dte", "open_interest", "minute_bars_many",
                                 "stock_quote_many", "calls", "stats", "wait_for_upstream",
                                 "close"}


# ------------------------------------------------------------------ specs/003 cache

def test_cache_never_stores_an_empty_response(tmp_path, monkeypatch):
    monkeypatch.setattr(B, "CACHE", tmp_path)
    assert B._cached("rth_X_2023-06-14", lambda: []) == []
    assert B._cached("zero_Y", lambda: 0) == 0
    assert list(tmp_path.glob("*.pkl")) == []


def test_cache_propagates_errors_and_stores_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(B, "CACHE", tmp_path)

    def boom():
        raise RuntimeError("HTTP 503")
    with pytest.raises(RuntimeError):
        B._cached("rth_X", boom)
    assert list(tmp_path.glob("*.pkl")) == []


def test_cache_stores_real_data(tmp_path, monkeypatch):
    monkeypatch.setattr(B, "CACHE", tmp_path)
    assert B._cached("rth_X", lambda: [1, 2]) == [1, 2]
    assert len(list(tmp_path.glob("rth_X.pkl"))) == 1
    assert B._cached("rth_X", lambda: [9]) == [1, 2]        # served from cache
