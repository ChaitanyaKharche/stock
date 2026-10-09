"""Tests for the day charts (day_charts.py). Offline; the drawing test skips without matplotlib.

What can lie here, in order of danger:

  1. A LINE IN THE WRONG PLACE. The six lines must be the runner's (first logged event),
     or rebuilt from the same bars by the same rule, and a 09:30 bar is never premarket.
  2. A STALE OR WRONG "YESTERDAY". Today's bars are never cached mid-session, an empty
     answer is never cached as a holiday, and "yesterday" is chosen exactly as the runner
     chooses it.
  3. AN INDICATOR THAT CHANGES WITH THE TIMEFRAME. VWAP is a property of the session, not
     of the bar size.
  4. A RUN THAT SAYS IT WORKED WHEN IT DID NOT, or that competes with the live runners
     for the data terminal.
"""
from __future__ import annotations

import datetime as dt
import json

import pytest

from trade_analysis.live_lab import day_charts as D

DAY = dt.date(2026, 10, 8)          # a Thursday


def _bar(t, o, h, l, c, v=100.0, day=DAY):
    return {"ts": dt.datetime.combine(day, t), "open": o, "high": h, "low": l, "close": c,
            "volume": v}


def _session(day=DAY, start=dt.time(4, 0), end=dt.time(16, 0), px=100.0, vol=100.0):
    out, t = [], dt.datetime.combine(day, start)
    while t.time() < end:
        out.append({"ts": t, "open": px, "high": px + 0.1, "low": px - 0.1, "close": px,
                    "volume": vol})
        t += dt.timedelta(minutes=1)
    return out


class FakeFeed:
    def __init__(self, by_day):
        self.by_day, self.calls = by_day, []

    def extended_bars(self, sym, day, start, end):
        self.calls.append(day)
        return list(self.by_day.get(day, []))


# ------------------------------------------------------------ bars and indicators

def test_resample_aligns_to_the_clock_and_survives_a_missing_minute():
    bars = [_bar(dt.time(9, 29), 10, 11, 9, 10.5), _bar(dt.time(9, 30), 20, 21, 19, 20.5, 5),
            _bar(dt.time(9, 31), 20.5, 23, 20, 22, 5), _bar(dt.time(9, 33), 22, 22, 18, 19, 5),
            _bar(dt.time(9, 34), 19, 19.5, 18.5, 19.2, 5), _bar(dt.time(9, 35), 19.2, 20, 19, 19.9)]
    five = D.resample(bars, 5)
    assert [b["ts"].time() for b in five] == [dt.time(9, 25), dt.time(9, 30), dt.time(9, 35)]
    b = five[1]                          # 09:30-09:34, with 09:32 missing
    assert (b["open"], b["high"], b["low"], b["close"], b["volume"]) == (20, 23, 18, 19.2, 20)
    assert b["last_ts"].time() == dt.time(9, 34)


def test_timeframes_that_do_not_divide_30_are_refused():
    # A 4-minute grid has no 09:30 boundary: the 09:28 bucket swallowed the open.
    with pytest.raises(ValueError):
        D.resample(_session(), 4)
    with pytest.raises(SystemExit) as e:
        D.main(["--offline", "--tf", "4"])
    assert e.value.code == 2


def test_rsi_is_wilders_and_a_flat_series_reads_50():
    # The StockCharts worked example (their averages are rounded, hence the tolerance).
    closes = [44.34, 44.09, 44.15, 43.61, 44.33, 44.83, 45.10, 45.42, 45.84, 46.08, 45.89,
              46.03, 45.61, 46.28, 46.28, 46.00, 46.03, 46.41, 46.22, 45.64]
    r = D.rsi(closes, 14)
    assert r[13] is None and abs(r[14] - 70.53) < 0.1 and abs(r[15] - 66.32) < 0.15
    assert D.rsi([5.0] * 20, 14)[-1] == 50.0


def test_ema_is_seeded_with_the_first_value():
    assert D.ema([1.0, 2.0, 3.0], 2) == pytest.approx([1.0, 5 / 3, 23 / 9])


def test_vwap_ignores_premarket_and_is_the_same_on_every_timeframe():
    bars = [_bar(dt.time(9, 25), 50, 51, 49, 50, 10_000)]       # premarket: must not count
    t = dt.datetime.combine(DAY, dt.time(9, 30))
    for i in range(30):
        px = 100 + (i % 7) - 3
        bars.append({"ts": t + dt.timedelta(minutes=i), "open": px, "high": px + 1,
                     "low": px - 1, "close": px, "volume": 100 + 37 * i})
    vw = D.vwap_1m(bars)
    assert min(vw) == t, "VWAP started before 09:30"
    first = bars[1]
    assert vw[t] == pytest.approx((first["high"] + first["low"] + first["close"]) / 3)
    at_0939 = t + dt.timedelta(minutes=9)
    two, ten = D.resample(bars, 2), D.resample(bars, 10)
    v2 = dict(zip((b["last_ts"] for b in two), D.vwap_series(bars, two)))
    v10 = dict(zip((b["last_ts"] for b in ten), D.vwap_series(bars, ten)))
    assert v2[at_0939] == v10[at_0939] == vw[at_0939]
    assert v10[dt.datetime.combine(DAY, dt.time(9, 25))] is None


# ------------------------------------------------------------ cache and "yesterday"

def test_todays_bars_are_not_cached_before_the_session_is_over(tmp_path):
    feed = FakeFeed({DAY: _session()})
    during = dt.datetime.combine(DAY, dt.time(15, 0))
    assert D.load_bars("QQQ", DAY, tmp_path, feed, now=during)
    assert not list(tmp_path.glob("*.pkl")), "a mid-session snapshot was cached"
    after = dt.datetime.combine(DAY, dt.time(16, 10))
    D.load_bars("QQQ", DAY, tmp_path, feed, now=after)
    assert (tmp_path / f"QQQ_{DAY}.pkl").exists()


def test_an_empty_answer_is_never_cached_as_a_holiday(tmp_path):
    feed = FakeFeed({})
    after = dt.datetime.combine(DAY, dt.time(20, 0))
    assert D.load_bars("QQQ", DAY, tmp_path, feed, now=after) == []
    assert not list(tmp_path.glob("*.pkl"))


def test_prior_session_is_chosen_exactly_as_the_runner_chooses_it(tmp_path):
    from trade_analysis.live_lab.levels_live import LevelsLoader
    mon = dt.date(2026, 10, 5)
    feed = FakeFeed({dt.date(2026, 10, 2): [],                                  # holiday
                     dt.date(2026, 10, 1): _session(dt.date(2026, 10, 1), end=dt.time(13, 0)),
                     dt.date(2026, 9, 30): _session(dt.date(2026, 9, 30))})
    later = dt.datetime(2026, 10, 9, 20, 0)
    got = D.prior_session("QQQ", mon, tmp_path, feed, now=later)
    assert got is not None and got[0] == dt.date(2026, 9, 30)

    class Store:
        def outage(self, *a, **k):
            pass
    loader = LevelsLoader(Store())
    weekdays = [mon - dt.timedelta(days=i) for i in range(1, 8)
                if (mon - dt.timedelta(days=i)).weekday() < 5]
    assert loader.at_warmup(feed, "QQQ", weekdays) and loader.yday_date["QQQ"] == got[0]


# ------------------------------------------------------------ the six lines

def _levels_event(ts, price, sym="QQQ"):
    return {"kind": "levels", "ts": ts, "symbol": sym,
            "lines": [{"name": "R1", "price": price, "source": "today premarket high", "side": "R"}]}


def test_the_first_logged_lines_win(tmp_path):
    # A restart logs the day again; the first event is what the setups traded on, and the
    # options arm's copy is read before the shares arm's.
    (tmp_path / "shares").mkdir()
    (tmp_path / "events.jsonl").write_text(
        json.dumps(_levels_event("2026-10-08T09:31:08", 758.20)) + "\n"
        + json.dumps(_levels_event("2026-10-08T15:07:30", 760.00)) + "\n", encoding="utf-8")
    (tmp_path / "shares" / "events.jsonl").write_text(
        json.dumps(_levels_event("2026-10-08T09:31:02", 759.00)) + "\n", encoding="utf-8")
    D._logged_levels.cache_clear()
    try:
        assert D.recorded_lines(tmp_path, "QQQ", DAY)[0]["price"] == 758.20
        assert D.recorded_lines(tmp_path, "SPY", DAY) is None
    finally:
        D._logged_levels.cache_clear()


def test_compare_lines_flags_a_revision_and_ignores_rounding():
    def L(p, src="today premarket high"):
        return [{"name": "R3", "price": p, "source": src, "side": "R"}]
    assert D.compare_lines(L(747.06), L(747.10)) and D.compare_lines(L(747.06), L(747.06, "yday market high"))
    assert D.compare_lines(L(747.06), L(747.064)) == []


def test_a_0930_bar_never_becomes_todays_premarket_high():
    yday = _session(dt.date(2026, 10, 7))
    today = _session(end=dt.time(9, 30)) + [_bar(dt.time(9, 30), 100, 105, 99, 104)]
    lines = D.rebuilt_lines(yday, today)
    assert lines and not any(l["price"] == 105 for l in lines)


def test_set_at_names_the_minute_that_made_a_level_and_nothing_for_a_revised_one():
    bars = _session(end=dt.time(9, 30))
    spike = next(b for b in bars if b["ts"].time() == dt.time(8, 5))
    spike["high"] = 103.0
    line = {"name": "R1", "price": 103.0, "source": "today premarket high", "side": "R"}
    assert D.set_at(line, [], bars) == "08:05"
    assert D.set_at({**line, "price": 103.04}, [], bars) == ""


def test_mst_labels_follow_daylight_time():
    assert D._mst(dt.datetime(2026, 10, 8, 7, 0)).time() == dt.time(4, 0)
    assert D._mst(dt.datetime(2026, 12, 15, 7, 0)).time() == dt.time(5, 0)


# ------------------------------------------------------------ the command line

def test_fetch_only_reports_missing_bars_as_a_failure(tmp_path):
    assert D.main(["--offline", "--fetch-only", "--day", "2026-10-08", "--out", str(tmp_path)]) == 1


def test_a_network_run_is_refused_during_the_session(tmp_path):
    during = dt.datetime.combine(DAY, dt.time(10, 0))
    assert D.main(["--day", "2026-10-08", "--out", str(tmp_path)], now=during) == 2


# ------------------------------------------------------------ the drawing

def test_premarket_volume_is_visible():
    # The opaque premarket shade used to sit on the price axes, above the volume axes, and
    # hid every premarket volume bar and the start of its average.
    pytest.importorskip("matplotlib")
    from matplotlib.colors import to_rgb
    yday = _session(dt.date(2026, 10, 7))
    bars = _session(vol=5000.0)
    for b in bars:
        b["close"] = b["open"] + 0.05            # up bars: volume drawn in the up colour
    fig, ax = D.build_figure("QQQ", DAY, 15, bars, None, "", 100.0, warm=yday)
    fig.canvas.draw()
    vol = ax["volume"]
    x, y = vol.transData.transform((0, 2000.0))   # first visible (07:00) bar, mid-height
    import numpy as np
    img = np.asarray(fig.canvas.buffer_rgba())
    px = [int(v) for v in img[img.shape[0] - int(y), int(x), :3]]
    shade = tuple(round(c * 255) for c in to_rgb(D.STYLE["pre"]))
    assert tuple(px) != shade and px[1] > px[0], f"premarket volume bar not visible: {tuple(px)}"


def test_set_at_looks_only_in_the_window_the_level_came_from():
    # Yesterday's MARKET high of 101 was made at 10:15; an 08:00 premarket print hit 101 too.
    yday = _session(dt.date(2026, 10, 7))
    for b in yday:
        if b["ts"].time() in (dt.time(8, 0), dt.time(10, 15)):
            b["high"] = 101.0
    line = {"name": "R2", "price": 101.0, "source": "yday market high", "side": "R"}
    assert D.set_at(line, yday, []) == "10:15"
