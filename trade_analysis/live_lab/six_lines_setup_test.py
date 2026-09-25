"""Tests for the LIVE Six_Lines setups (setups.py, levels_live.py). Amendment 2026-09-25.

What can lie here, in order of danger:

  1. THE LIVE RULE DRIFTING FROM THE BACKTEST. The forward record only extends the
     2,438-trade six-line result if the live setup picks the same bar, the same line and
     the same direction as six_lines.first_break_trade. Checked on 400 synthetic days
     against that function itself, not against a restatement of it.
  2. A SECOND CHANCE. "The first break" must not become "a break": if the first break is
     missed, nothing later may fire. That is the exact distortion that let Crabel_Stretch
     trade the first bar a blind runner could see.
  3. SEEING THE FUTURE. Both runners hand a setup the whole admitted buffer; on a batch
     admission that includes bars after the one being judged.
  4. A PARTIAL LEVEL SET. Five lines are not his six. No premarket -> no lines -> no trade,
     recorded, never silently degraded.
  5. THE FREEZE. The 13 original definitions must be byte-identical to what was frozen, or
     the amendment's central claim is false.
"""
from __future__ import annotations

import datetime as dt
import json
import random
from pathlib import Path
from types import SimpleNamespace

import pytest

from trade_analysis.live_lab import six_lines as research
from trade_analysis.live_lab.levels_live import LevelsLoader, lines_as_dicts
from trade_analysis.live_lab.setups import ALL_SETUPS, SixLines, SixLinesNoCap

LAB = Path(__file__).resolve().parents[2] / "live_lab_data"
YDAY, DAY = dt.date(2026, 9, 21), dt.date(2026, 9, 22)


def _walk(day, t0, t1, px, rng, sd=0.0006):
    out, t = [], dt.datetime.combine(day, t0)
    end = dt.datetime.combine(day, t1)
    while t < end:
        o = px
        c = max(1.0, o * (1 + rng.gauss(0, sd)))
        h = max(o, c) * (1 + abs(rng.gauss(0, sd / 2)))
        l = min(o, c) * (1 - abs(rng.gauss(0, sd / 2)))
        out.append({"ts": t, "open": o, "high": h, "low": l, "close": c, "volume": 100.0})
        px, t = c, t + dt.timedelta(minutes=1)
    return out, px


def _day(seed):
    rng = random.Random(seed)
    yday, px = _walk(YDAY, dt.time(4, 0), dt.time(16, 0), 100.0, rng)
    pre, px = _walk(DAY, dt.time(4, 0), dt.time(9, 30), px * (1 + rng.gauss(0, 0.004)), rng)
    rth, _ = _walk(DAY, dt.time(9, 30), dt.time(16, 0), px, rng, sd=0.0009)
    return yday, pre, rth


def _ctx(bars_1m, bar_ts, levels):
    return SimpleNamespace(tf="1m", bars_1m=tuple(bars_1m), bar_ts=bar_ts, levels=levels)


def _live_signals(setup, rth, levels):
    """Evaluate bar by bar, as a runner does, collecting (bar_ts, signal)."""
    out = []
    for i, b in enumerate(rth):
        sig = setup.evaluate(_ctx(rth[: i + 1], b["ts"], levels))
        if sig is not None:
            out.append((b["ts"], sig))
    return out


# ------------------------------------------------------------ 1. parity with research

@pytest.mark.parametrize("seed", range(400))
def test_live_entry_matches_six_lines_first_break_trade(seed):
    yday, pre, rth = _day(seed)
    lines = research.build_six(yday, pre)
    assert lines, "synthetic day should always yield six lines"
    rec = research.breaks_today(rth, lines)
    want = research.first_break_trade(rth, lines, rec)
    got = _live_signals(SixLines(), rth, lines_as_dicts(lines))

    if want is None:
        # Research also returns None when the break is on the LAST 10m bar (no next bar to
        # fill on). Live would decide at 16:00, which no runner evaluates. Either way: no trade
        # inside the session window.
        assert all(ts.time() >= dt.time(15, 59) for ts, _ in got), (seed, got)
        return
    assert len(got) == 1, f"seed {seed}: live fired {len(got)} times, research once"
    ts, sig = got[0]
    bucket = dt.datetime.fromisoformat(sig.state["signal_bucket"])
    assert bucket.strftime("%H:%M") == want["signal_at"], (seed, bucket, want)
    assert ts == bucket + dt.timedelta(minutes=9), "decided before the 10m bar closed"
    assert sig.state["line"] == want["line"]
    assert sig.direction == want["direction"]


@pytest.mark.parametrize("seed", range(400))
def test_live_failed_exit_matches_research(seed):
    """NoCap exits only on 'failed' or 15:55 -- compare the failed bar with research's."""
    yday, pre, rth = _day(seed)
    lines = research.build_six(yday, pre)
    want = research.first_break_trade(rth, lines, research.breaks_today(rth, lines))
    got = _live_signals(SixLinesNoCap(), rth, lines_as_dicts(lines))
    if want is None or not got:
        return
    entry_ts, sig = got[0]
    setup = SixLinesNoCap()
    pos = {"direction": sig.direction, "state": sig.state}
    failed_at = None
    for i, b in enumerate(rth):
        if b["ts"] <= entry_ts:
            continue
        if setup.manage(pos, _ctx(rth[: i + 1], b["ts"], lines_as_dicts(lines))):
            failed_at = b
            break
    if want["exit_reason"] == "failed":
        assert failed_at is not None, f"seed {seed}: research failed, live never did"
        # research exits at that 10m bar's close, which is this 1m bar's close
        entry = want["entry"]
        live_bp = (failed_at["close"] / entry - 1) * 10_000 * (1 if sig.direction == "long" else -1)
        assert live_bp == pytest.approx(want["move_bp"], abs=0.011), (seed, live_bp, want)
    else:
        assert failed_at is None or failed_at["ts"].time() >= dt.time(15, 50), (seed, failed_at)


# ------------------------------------------------------------ 2. no second chance

def _pick_breaking_day():
    for seed in range(2000):
        yday, pre, rth = _day(seed)
        lines = research.build_six(yday, pre)
        want = research.first_break_trade(rth, lines, research.breaks_today(rth, lines))
        if want and want["signal_at"] < "12:00":
            return rth, lines, want
    raise AssertionError("no breaking synthetic day found")


def test_a_missed_first_break_is_not_replaced_by_a_later_one():
    rth, lines, want = _pick_breaking_day()
    signal_close = dt.datetime.combine(DAY, dt.time.fromisoformat(want["signal_at"])) \
        + dt.timedelta(minutes=9)
    later = [s for s in _live_signals(SixLines(), rth, lines_as_dicts(lines))
             if s[0] > signal_close]
    assert later == [], "a later break fired after the first -- that is 'a break', not his rule"


# ------------------------------------------------------------ 3. no future

def test_a_batch_buffer_cannot_leak_future_bars():
    """Same bar judged with the whole day in the buffer must give the same answer."""
    rth, lines, want = _pick_breaking_day()
    lv = lines_as_dicts(lines)
    for i, b in enumerate(rth):
        alone = SixLines().evaluate(_ctx(rth[: i + 1], b["ts"], lv))
        batch = SixLines().evaluate(_ctx(rth, b["ts"], lv))
        assert (alone is None) == (batch is None), f"future bars changed the {b['ts']:%H:%M} verdict"


# ------------------------------------------------------------ targets

def test_cap_is_20bp_from_the_signal_close_and_nocap_has_none():
    rth, lines, _ = _pick_breaking_day()
    lv = lines_as_dicts(lines)
    (ts, capped), = _live_signals(SixLines(), rth, lv)
    (_, uncapped), = _live_signals(SixLinesNoCap(), rth, lv)
    close = next(b for b in rth if b["ts"] == ts)["close"]
    sign = 1 if capped.direction == "long" else -1
    assert capped.target == pytest.approx(close * (1 + sign * 0.0020))
    assert uncapped.target is None and uncapped.stop is None


def test_pre_broken_lines_take_no_part():
    """A line the open is already past can never be 'the break'."""
    open_ = dt.datetime.combine(DAY, dt.time(9, 30))
    rth = [{"ts": open_ + dt.timedelta(minutes=m), "open": 105.0, "high": 105.2,
            "low": 104.8, "close": 105.0, "volume": 1.0} for m in range(20)]
    levels = ({"name": "R1", "price": 101.0, "source": "yday premarket high", "side": "R"},
              {"name": "R2", "price": 110.0, "source": "yday market high", "side": "R"},
              {"name": "R3", "price": 111.0, "source": "today premarket high", "side": "R"},
              {"name": "S1", "price": 99.0, "source": "yday premarket low", "side": "S"},
              {"name": "S2", "price": 98.0, "source": "yday market low", "side": "S"},
              {"name": "S3", "price": 97.0, "source": "today premarket low", "side": "S"})
    assert _live_signals(SixLines(), rth, levels) == [], "fired on a line passed before the open"


def test_no_levels_means_no_trade():
    rth, _, _ = _pick_breaking_day()
    assert _live_signals(SixLines(), rth, None) == []
    assert _live_signals(SixLines(), rth, ()) == []


# ------------------------------------------------------------ 4. the loader

class _Feed:
    def __init__(self, yday, pre, fail_pre=0):
        self.yday, self.pre, self.fail_pre, self.calls = yday, pre, fail_pre, []

    def extended_bars(self, sym, day, start, end):
        from trade_analysis.live_lab.feed import FeedOutage
        self.calls.append((day, start, end))
        if day == DAY:
            if self.fail_pre:
                self.fail_pre -= 1
                raise FeedOutage("HTTP 503")
            return [b for b in self.pre if start <= b["ts"].time() < end]
        if day == YDAY:
            return [b for b in self.yday if start <= b["ts"].time() < end]
        return []


class _Store:
    def __init__(self):
        self.outages, self.events = [], []

    def outage(self, kind, detail, **f):
        self.outages.append(kind)

    def event(self, kind, **f):
        self.events.append((kind, f))


def _sess():
    return SimpleNamespace(symbol="QQQ", day=DAY, levels=None)


def test_loader_builds_exactly_the_research_lines_and_only_after_0931():
    yday, pre, _ = _day(7)
    feed, store, sess = _Feed(yday, pre), _Store(), _sess()
    ld = LevelsLoader(store)
    assert ld.at_warmup(feed, "QQQ", [dt.date(2026, 9, 18), YDAY])
    ld.at_tick(feed, sess, dt.datetime.combine(DAY, dt.time(9, 30, 30)))
    assert sess.levels is None, "premarket read before its last bar could be published"
    ld.at_tick(feed, sess, dt.datetime.combine(DAY, dt.time(9, 31, 2)))
    assert sess.levels == lines_as_dicts(research.build_six(yday, pre))
    assert len(sess.levels) == 6 and store.events[0][0] == "levels"


def test_loader_retries_a_failed_fetch_and_reports_it_once():
    yday, pre, _ = _day(7)
    feed, store, sess = _Feed(yday, pre, fail_pre=2), _Store(), _sess()
    ld = LevelsLoader(store)
    ld.at_warmup(feed, "QQQ", [YDAY])
    for s in (2, 7, 12):
        ld.at_tick(feed, sess, dt.datetime.combine(DAY, dt.time(9, 31, s)))
    assert sess.levels is not None and len(sess.levels) == 6
    assert store.outages.count("levels_fetch") == 1


def test_loader_with_no_premarket_trades_nothing_and_says_so():
    yday, _, _ = _day(7)
    feed, store, sess = _Feed(yday, []), _Store(), _sess()
    ld = LevelsLoader(store)
    ld.at_warmup(feed, "QQQ", [YDAY])
    ld.at_tick(feed, sess, dt.datetime.combine(DAY, dt.time(9, 32)))
    assert sess.levels == () and "levels_unavailable" in store.outages


# ------------------------------------------------------------ 5. the freeze

@pytest.mark.parametrize("frozen, freeze, new_hash", [
    ("config/1f7247d7839d9950.json", "FREEZE.json", "ccd00ac71f8243e9"),
    ("shares/config/b53ca8a58aa11718.json", "shares/FREEZE.json", "7c6414fc549315f3"),
])
def test_the_13_originals_are_byte_identical_and_the_new_hash_is_accepted(frozen, freeze, new_hash):
    old = json.loads((LAB / frozen).read_text(encoding="utf-8"))["setups"]
    cur = [s.describe() for s in ALL_SETUPS]
    assert len(old) == 13 and len(cur) == 15
    for a, b in zip(old, cur[:13]):
        assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True), a["id"]
    assert [s.id for s in ALL_SETUPS[13:]] == ["Six_Lines", "Six_Lines_NoCap"]
    fz = json.loads((LAB / freeze).read_text(encoding="utf-8"))
    assert new_hash in fz["accepted_config_hashes"]
    assert fz["family_size"] == 15


def test_the_new_hashes_are_what_the_runners_compute():
    from trade_analysis.live_lab import runner, shares_runner
    assert runner.build_config(["QQQ", "SPY"], 1500)["config_hash"] == "ccd00ac71f8243e9"
    assert shares_runner.build_config(list(shares_runner.BROAD_UNIVERSE))["config_hash"] \
        == "7c6414fc549315f3"
