"""Does the veto classify the day the trader was looking at, and refuse the future?

    python -m trade_analysis.live_lab.orb_veto_test
    pytest trade_analysis/live_lab/orb_veto_test.py

Frozen by `research/orb_veto_preregistration.md`. Bars are hand-written here, so every
check states the shape it is testing rather than depending on an archive.

The reference case is **2026-09-17 QQQ**, the chart that produced the hypothesis:
premarket ran ~710 -> ~716.3, RTH chopped inside ~715.0-716.7, and one bar spiked to ~718
on the session's largest volume before closing back inside. That day must come out VETOED
on a close basis and NOT vetoed on a touch basis -- it is the whole reason §2 of the
pre-registration picks closes, and if the two rules agreed here the distinction would be
cosmetic.

The other half of the file is about the lookahead. `decide` may only read bars whose close
is known by T; `post` may read the session. A test that cannot tell them apart would let
the defect that cost this project 88-95.7% of a measured edge back in through a filter.
"""
from __future__ import annotations

import datetime as dt
import sys

from .orb_veto import (BUFFER, DECIDE_AT, classify_day, resample_10m,
                       touch_based_for_comparison)

DAY = dt.date(2026, 9, 17)


def _bar(hhmm, o, h, l, c, v=1000.0):
    h_, m_ = hhmm
    return {"ts": dt.datetime.combine(DAY, dt.time(h_, m_)),
            "open": o, "high": h, "low": l, "close": c, "volume": v}


def _minutes(start, end, o, h, l, c, v=1000.0):
    """1-minute bars from `start` to `end` exclusive, all the same shape."""
    out = []
    t = dt.datetime.combine(DAY, dt.time(*start))
    stop = dt.datetime.combine(DAY, dt.time(*end))
    while t < stop:
        out.append({"ts": t, "open": o, "high": h, "low": l, "close": c, "volume": v})
        t += dt.timedelta(minutes=1)
    return out


# --------------------------------------------------------------- the reference day

def _premarket_20260917():
    """~710 -> ~716.3, the run that spent the day's momentum before the bell."""
    out, px = [], 710.0
    t = dt.datetime.combine(DAY, dt.time(4, 0))
    stop = dt.datetime.combine(DAY, dt.time(9, 30))
    n = int((stop - t).total_seconds() // 60)
    for i in range(n):
        nxt = 710.0 + (716.30 - 710.0) * (i + 1) / n
        out.append({"ts": t, "open": px, "high": max(px, nxt), "low": min(px, nxt),
                    "close": nxt, "volume": 5000.0})
        px = nxt
        t += dt.timedelta(minutes=1)
    return out


def _rth_20260917(spike_close_inside=True):
    """Contained 715.0-716.7, with one 718 spike near 14:30.

    `spike_close_inside=False` gives the counterfactual: the same spike CLOSING above,
    which must flip the verdict. Without that companion case, a passing close-based test
    proves only that the band was never exceeded, not that closes are what is read.
    """
    bars = []
    # Opening range 09:30-09:39: 715.90-716.70, the values off the chart's OHLC readout.
    bars += _minutes((9, 30), (9, 40), 715.98, 716.70, 715.90, 716.27)
    # Chop until the spike. Stays inside the band on both high and close.
    bars += _minutes((9, 40), (14, 30), 715.80, 716.60, 715.05, 716.10)
    # The spike: high 718.00 on huge volume. Close is the variable under test.
    spike_close = 716.30 if spike_close_inside else 717.90
    bars += _minutes((14, 30), (14, 40), 716.20, 718.00, 716.10, spike_close, 90000.0)
    # Fade to 715.3.
    bars += _minutes((14, 40), (16, 0), 716.00, 716.20, 715.20, 715.30)
    return bars


def test_the_reference_day_is_vetoed_on_closes():
    """2026-09-17 QQQ. The day the trader called untradeable."""
    v = classify_day(_premarket_20260917(), _rth_20260917())
    assert v.usable, v.reason
    assert v.decide is True, "the 11:00 gate did not veto a contained morning"
    assert v.post is True, f"a close broke the band at {v.first_break_ts}"
    assert v.gate() is True
    # The band is the union: premarket high 716.30 vs opening-range high 716.70.
    assert abs(v.band_high - 716.70) < 1e-9, v.band_high
    assert abs(v.band_low - 710.0) < 1e-9, v.band_low


def test_the_same_day_is_NOT_vetoed_on_touches():
    """The reason §2 picks closes. If these agreed, the choice would be cosmetic.

    Note this is the POST comparison: the 718 spike is at 14:30, after T, so it cannot
    affect the 11:00 gate either way. What it changes is whether the SESSION counts as
    having expanded -- which is exactly the judgement the trader was making off a
    finished chart.
    """
    pm, rth = _premarket_20260917(), _rth_20260917()
    close_based = classify_day(pm, rth)
    assert close_based.post is True, "close rule saw expansion"

    # Touch, evaluated over the session rather than to T, to isolate the spike.
    ten = [b for b in resample_10m(rth) if b["ts"].time() >= dt.time(9, 30)]
    band_high = max(716.30, ten[0]["high"])
    touched = any(b["high"] > band_high * (1 + BUFFER) for b in ten[1:])
    assert touched is True, "the 718 spike did not exceed the band; fixture is wrong"


def test_a_close_above_the_band_is_expansion():
    """The counterfactual. Same bars, spike CLOSES above -> no longer a contained day."""
    v = classify_day(_premarket_20260917(), _rth_20260917(spike_close_inside=False))
    assert v.post is False, "a close above the band was not read as expansion"
    assert v.first_break_dir == "up", v.first_break_dir
    assert v.first_break_ts.time() == dt.time(14, 30), v.first_break_ts


# ------------------------------------------------------------------- the lookahead

def test_decide_ignores_everything_after_T():
    """The core separation. An afternoon breakout must NOT un-veto the 11:00 gate.

    This is the check that keeps the filter tradeable. If it fails, the veto is reading
    the future and every P&L number downstream is void -- the same defect class that
    supplied 88-95.7% of a measured edge here once already.
    """
    pm = _premarket_20260917()
    rth = _minutes((9, 30), (9, 40), 715.98, 716.70, 715.90, 716.27)
    rth += _minutes((9, 40), (13, 0), 715.80, 716.60, 715.05, 716.10)
    rth += _minutes((13, 0), (16, 0), 717.00, 725.00, 716.90, 724.00)   # huge afternoon
    v = classify_day(pm, rth)
    assert v.decide is True, "an afternoon breakout leaked into the 11:00 decision"
    assert v.post is False, "the session plainly expanded; post should be False"
    assert v.first_break_ts.time() >= dt.time(13, 0)


def test_a_break_before_T_lifts_the_veto():
    """The other direction: a genuine morning expansion must be tradeable."""
    pm = _premarket_20260917()
    rth = _minutes((9, 30), (9, 40), 715.98, 716.70, 715.90, 716.27)
    rth += _minutes((9, 40), (10, 30), 717.00, 719.00, 716.90, 718.50)
    rth += _minutes((10, 30), (16, 0), 718.00, 718.50, 717.50, 718.00)
    v = classify_day(pm, rth)
    assert v.decide is False, "a 09:40 breakout did not lift the veto"
    assert v.post is False


def test_the_bar_boundary_at_T_is_inclusive_of_its_close():
    """The 10:50 bar closes at 11:00, so it counts. The 11:00 bar does not.

    Off-by-one here is worth a test on its own: getting it wrong in one direction reads
    ten minutes of the future, and in the other throws away the last usable bar.
    """
    pm = _premarket_20260917()
    base = _minutes((9, 30), (9, 40), 715.98, 716.70, 715.90, 716.27)
    chop = _minutes((9, 40), (10, 50), 715.80, 716.60, 715.05, 716.10)

    # Break in the 10:50-10:59 bar -> close known at 11:00 -> must lift the veto.
    at_t = base + chop + _minutes((10, 50), (11, 0), 717.0, 719.0, 716.9, 718.5) \
        + _minutes((11, 0), (16, 0), 718.0, 718.2, 717.8, 718.0)
    assert classify_day(pm, at_t).decide is False, "the 10:50 bar's close was ignored"

    # Break in the 11:00-11:09 bar -> not complete at 11:00 -> veto stands.
    after_t = base + chop + _minutes((10, 50), (11, 0), 715.8, 716.6, 715.05, 716.1) \
        + _minutes((11, 0), (11, 10), 717.0, 719.0, 716.9, 718.5) \
        + _minutes((11, 10), (16, 0), 718.0, 718.2, 717.8, 718.0)
    assert classify_day(pm, after_t).decide is True, "the 11:00 bar leaked into the gate"


def test_the_opening_range_bar_cannot_break_its_own_band():
    """It helps define the band. Counting it would make every wide open read as expansion."""
    pm = [{"ts": dt.datetime.combine(DAY, dt.time(9, 0)), "open": 715.0, "high": 715.5,
           "low": 714.5, "close": 715.0, "volume": 100.0}]
    rth = _minutes((9, 30), (9, 40), 715.0, 720.0, 710.0, 719.0)      # enormous OR
    rth += _minutes((9, 40), (16, 0), 715.0, 716.0, 714.0, 715.0)     # then quiet
    v = classify_day(pm, rth)
    assert v.decide is True, "the opening-range bar broke the band it defined"
    assert v.post is True


# ---------------------------------------------------------------------- degenerate

def test_a_day_with_no_premarket_still_classifies_on_the_opening_range():
    """Premarket is often thin or absent. The band degrades to the OR and says so."""
    rth = _minutes((9, 30), (9, 40), 715.0, 716.0, 714.0, 715.5)
    rth += _minutes((9, 40), (16, 0), 715.0, 715.9, 714.1, 715.4)
    v = classify_day([], rth)
    assert v.usable is True
    assert v.premarket_high is None
    assert abs(v.band_high - 716.0) < 1e-9
    assert "no premarket" in v.reason
    assert v.decide is True


def test_an_unusable_day_never_vetoes():
    """A data gap must not become a strategy decision.

    If `gate()` returned True on an unclassifiable day, the filter would suppress trades
    because the feed was thin and the resulting P&L would be credited to the veto.
    """
    v = classify_day([], [])
    assert v.usable is False
    assert v.gate() is False, "an unclassifiable day vetoed"


def test_resample_aligns_the_opening_range_to_the_open():
    """09:30-09:39 must be ONE bucket.

    Bucketing by `minute // 10` would split the session open and make the opening range a
    9-minute bar on some days -- a silently different reference band per day.
    """
    rth = _minutes((9, 30), (10, 0), 715.0, 716.0, 714.0, 715.5)
    ten = resample_10m(rth)
    assert [b["ts"].time() for b in ten] == [dt.time(9, 30), dt.time(9, 40),
                                             dt.time(9, 50)], [b["ts"] for b in ten]
    assert all(b["volume"] == 10000.0 for b in ten), "volume did not aggregate"


def test_the_buffer_is_required_to_be_exceeded():
    """A close exactly on the band is not expansion. Equality must not trigger."""
    pm = [{"ts": dt.datetime.combine(DAY, dt.time(9, 0)), "open": 700.0, "high": 700.0,
           "low": 700.0, "close": 700.0, "volume": 1.0}]
    rth = _minutes((9, 30), (9, 40), 700.0, 700.0, 700.0, 700.0)
    exact = 700.0 * (1 + BUFFER)
    rth += _minutes((9, 40), (10, 0), 700.0, exact, 700.0, exact)
    rth += _minutes((10, 0), (16, 0), 700.0, 700.0, 700.0, 700.0)
    assert classify_day(pm, rth).post is True, "a close exactly at the buffer triggered"

    rth2 = _minutes((9, 30), (9, 40), 700.0, 700.0, 700.0, 700.0)
    rth2 += _minutes((9, 40), (10, 0), 700.0, exact * 1.001, 700.0, exact * 1.001)
    rth2 += _minutes((10, 0), (16, 0), 700.0, 700.0, 700.0, 700.0)
    assert classify_day(pm, rth2).post is False, "a close past the buffer did not trigger"


def test_frozen_parameters_match_the_preregistration():
    """The pre-registration is the contract. Drift here silently changes the experiment."""
    assert DECIDE_AT == dt.time(11, 0)
    assert BUFFER == 0.0005
    from . import orb_veto as M
    assert M.INTERVAL_MIN == 10
    assert M.OR_END == dt.time(9, 40)
    assert M.PREMARKET_FROM == dt.time(4, 0)


CHECKS = [(n, f) for n, f in sorted(globals().items())
          if n.startswith("test_") and callable(f)]


def main() -> int:
    ok = True
    print("=" * 78)
    print("CLOSE-BASED RANGE-EXPANSION VETO")
    print("=" * 78)
    for name, fn in CHECKS:
        try:
            fn()
            print(f"  [PASS] {name[5:].replace('_', ' ')}")
        except AssertionError as exc:
            ok = False
            print(f"  [FAIL] {name[5:].replace('_', ' ')}\n         {exc}")
        except Exception as exc:                              # noqa: BLE001
            ok = False
            print(f"  [FAIL] {name[5:].replace('_', ' ')}\n         raised {exc!r}")
    print(f"\n  {len(CHECKS)} checks on hand-written bars")
    print(f"  RESULT: {'PASS' if ok else 'FAIL'}")
    print("\n  The rule may be late. It may not be early.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
