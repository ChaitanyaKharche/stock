"""Does the 8-line breakout fire where the trader's chart says it should, and nowhere else?

    python -m trade_analysis.live_lab.breakout_levels_test
    pytest trade_analysis/live_lab/breakout_levels_test.py

Frozen by `research/breakout_options_preregistration.md`. Hand-written bars throughout,
so each check states the shape it tests rather than depending on an archive.

The reference case is **2026-09-18 QQQ**, the day he said the move was there: price broke
its premarket high late in the session and ran to ~722.5. The veto's reference case,
2026-09-17, is here too as the negative -- a day that must produce NO signal, because a
rule that fires on both is not reading anything.

Most of the file is lookahead. This signal has three separate ways to read the future --
a "prior" session that is today, today's premarket used before it exists, and a level
re-firing all the way up a trend -- and the third is the sneaky one: it does not look
like lookahead, it looks like a high win rate.
"""
from __future__ import annotations

import datetime as dt
import sys

from .breakout_levels import (COST_FLOOR_BP, Level, build_levels, find_breakouts,
                              move_bp, nearest_unbroken)

DAY = dt.date(2026, 9, 18)
PREV = dt.date(2026, 9, 17)


def _mins(day, start, end, o, h, l, c, v=1000.0):
    out, t = [], dt.datetime.combine(day, dt.time(*start))
    stop = dt.datetime.combine(day, dt.time(*end))
    while t < stop:
        out.append({"ts": t, "open": o, "high": h, "low": l, "close": c, "volume": v})
        t += dt.timedelta(minutes=1)
    return out


def _session(day, hi, lo, close=None):
    """A full extended session 04:00-20:00 whose extremes are `hi`/`lo`."""
    mid = (hi + lo) / 2
    bars = _mins(day, (4, 0), (9, 30), mid, mid, mid, mid)
    bars += _mins(day, (9, 30), (10, 0), mid, hi, lo, close or mid)
    bars += _mins(day, (10, 0), (20, 0), mid, mid, mid, mid)
    return bars


# --------------------------------------------------------------------- the level set

def test_eight_lines_three_prior_days_plus_todays_premarket():
    priors = [_session(dt.date(2026, 9, 15), 720.0, 710.0),
              _session(dt.date(2026, 9, 16), 719.0, 712.0),
              _session(PREV, 718.0, 714.0)]
    pm = _mins(DAY, (4, 0), (9, 30), 716.0, 717.5, 715.5, 717.0)
    lv = build_levels(priors, pm)
    assert len(lv) == 8, [x.name for x in lv]
    names = {x.name for x in lv}
    assert names == {"D-1 high", "D-1 low", "D-2 high", "D-2 low",
                     "D-3 high", "D-3 low", "premarket high", "premarket low"}
    by = {x.name: x.price for x in lv}
    assert by["D-1 high"] == 718.0 and by["D-1 low"] == 714.0, by   # yesterday
    assert by["D-3 high"] == 720.0, by                              # oldest of the three
    assert by["premarket high"] == 717.5 and by["premarket low"] == 715.5


def test_the_six_line_variant_is_the_prior_days_only():
    """Secondary S1 -- what he literally said. One flag, no other behaviour change."""
    priors = [_session(dt.date(2026, 9, 15), 720.0, 710.0),
              _session(dt.date(2026, 9, 16), 719.0, 712.0),
              _session(PREV, 718.0, 714.0)]
    pm = _mins(DAY, (4, 0), (9, 30), 716.0, 717.5, 715.5, 717.0)
    lv = build_levels(priors, pm, include_premarket=False)
    assert len(lv) == 6
    assert not any("premarket" in x.name for x in lv)


def test_only_the_most_recent_three_sessions_are_used():
    """A fourth session must not add lines, or the lookback is not the lookback."""
    priors = [_session(dt.date(2026, 9, 11), 999.0, 1.0)] + [
        _session(d, 718.0, 714.0) for d in
        (dt.date(2026, 9, 15), dt.date(2026, 9, 16), PREV)]
    lv = build_levels(priors, [], include_premarket=False)
    assert len(lv) == 6
    assert all(x.price in (718.0, 714.0) for x in lv), \
        "a session older than the lookback contributed a level"


def test_levels_span_the_extended_session_not_just_rth():
    """He marks premarket extremes too. An RTH-only high is not the line he drew."""
    day = dt.date(2026, 9, 16)
    bars = _mins(day, (4, 0), (9, 30), 700.0, 730.0, 690.0, 700.0)   # extreme premarket
    bars += _mins(day, (9, 30), (16, 0), 705.0, 706.0, 704.0, 705.0)  # quiet RTH
    lv = build_levels([bars], [], include_premarket=False)
    by = {x.name: x.price for x in lv}
    assert by["D-1 high"] == 730.0, "the premarket high was ignored"
    assert by["D-1 low"] == 690.0


# ------------------------------------------------------------------- the two reference days

def test_20260918_breaks_the_premarket_high_and_signals_long():
    """The day he said the move was there."""
    priors = [_session(d, 719.0, 714.0) for d in
              (dt.date(2026, 9, 15), dt.date(2026, 9, 16), PREV)]
    pm = _mins(DAY, (4, 0), (9, 30), 717.0, 718.1, 716.5, 717.8)
    rth = _mins(DAY, (9, 30), (9, 40), 717.8, 718.1, 717.5, 718.0)
    rth += _mins(DAY, (9, 40), (15, 0), 717.0, 718.0, 716.4, 717.5)   # inside all day
    rth += _mins(DAY, (15, 0), (15, 30), 718.0, 720.0, 717.9, 719.8)  # the break
    rth += _mins(DAY, (15, 30), (16, 0), 720.0, 722.6, 719.9, 722.5)  # the run
    sigs = find_breakouts(rth, build_levels(priors, pm))
    assert sigs, "no signal on the day the move happened"
    first = sigs[0]
    assert first.direction == "long"
    assert first.level.name in ("premarket high", "D-1 high"), first.level.name
    assert first.fill_ts > first.ts, "fill is not on a later bar than the signal"


def test_20260917_produces_no_signal_at_all():
    """The contained day. A rule that fires on both days is reading nothing."""
    priors = [_session(d, 719.0, 710.0) for d in
              (dt.date(2026, 9, 14), dt.date(2026, 9, 15), dt.date(2026, 9, 16))]
    pm = _mins(PREV, (4, 0), (9, 30), 712.0, 716.3, 710.0, 716.2)
    rth = _mins(PREV, (9, 30), (9, 40), 715.98, 716.70, 715.90, 716.27)
    rth += _mins(PREV, (9, 40), (16, 0), 715.8, 716.6, 715.05, 716.1)
    assert find_breakouts(rth, build_levels(priors, pm)) == []


# -------------------------------------------------------------------------- lookahead

def test_a_level_fires_at_most_once_per_day():
    """The sneaky one. It does not look like lookahead, it looks like a high win rate.

    Without this, a trending day re-signals on every bar above the level and the "win
    rate" becomes a function of how long the trend lasted.
    """
    priors = [_session(d, 718.0, 714.0) for d in
              (dt.date(2026, 9, 15), dt.date(2026, 9, 16), PREV)]
    rth = _mins(DAY, (9, 30), (9, 40), 717.0, 717.5, 716.5, 717.0)
    rth += _mins(DAY, (9, 40), (15, 30), 719.0, 725.0, 718.9, 724.0)   # far above, hours
    sigs = find_breakouts(rth, build_levels(priors, [], include_premarket=False))
    names = [s.level.name for s in sigs]
    assert len(names) == len(set(names)), f"a level fired twice: {names}"
    assert names.count("D-1 high") == 1


def test_the_fill_is_the_next_bar_never_the_signal_bar():
    """A close is only knowable once the bar ends. This is the one-bar lookahead that
    supplied 88-95.7% of a previously measured edge in this repo."""
    priors = [_session(d, 718.0, 714.0) for d in
              (dt.date(2026, 9, 15), dt.date(2026, 9, 16), PREV)]
    rth = _mins(DAY, (9, 30), (9, 40), 717.0, 717.5, 716.5, 717.0)
    rth += _mins(DAY, (9, 40), (9, 50), 717.0, 719.5, 716.9, 719.4)
    rth += _mins(DAY, (9, 50), (16, 0), 719.4, 719.6, 719.2, 719.4)
    s = find_breakouts(rth, build_levels(priors, [], include_premarket=False))[0]
    assert s.ts.time() == dt.time(9, 40)
    assert s.fill_ts.time() == dt.time(9, 50)
    assert (s.fill_ts - s.ts).total_seconds() == 600


def test_a_signal_with_no_next_bar_is_not_emitted():
    """The last bar of the window has nothing to fill on, so it is not a trade."""
    priors = [_session(d, 718.0, 714.0) for d in
              (dt.date(2026, 9, 15), dt.date(2026, 9, 16), PREV)]
    rth = _mins(DAY, (9, 30), (9, 40), 717.0, 717.5, 716.5, 717.0)
    rth += _mins(DAY, (9, 40), (9, 50), 719.0, 719.5, 718.9, 719.4)   # the LAST bar
    assert find_breakouts(rth, build_levels(priors, [], include_premarket=False)) == []


def test_the_signal_window_excludes_the_opening_range_and_the_close():
    """09:40-15:30. Earlier is the opening range; later has no room before the flatten."""
    priors = [_session(d, 718.0, 714.0) for d in
              (dt.date(2026, 9, 15), dt.date(2026, 9, 16), PREV)]
    lv = build_levels(priors, [], include_premarket=False)

    early = _mins(DAY, (9, 30), (9, 40), 719.0, 720.0, 718.9, 719.5)   # OR bar breaks
    early += _mins(DAY, (9, 40), (16, 0), 717.0, 717.5, 716.5, 717.0)  # then back inside
    assert find_breakouts(early, lv) == [], "the opening-range bar produced a signal"

    late = _mins(DAY, (9, 30), (15, 30), 717.0, 717.5, 716.5, 717.0)
    late += _mins(DAY, (15, 30), (16, 0), 719.0, 720.0, 718.9, 719.5)
    got = [s for s in find_breakouts(late, lv) if s.ts.time() > dt.time(15, 30)]
    assert got == [], f"signalled after 15:30: {[s.ts for s in got]}"


def test_the_buffer_must_be_exceeded_not_merely_touched():
    lv = [Level("D-1 high", 700.0, "up")]
    exact = 700.0 * 1.0005
    rth = _mins(DAY, (9, 30), (9, 40), 699.0, 699.0, 699.0, 699.0)
    rth += _mins(DAY, (9, 40), (9, 50), exact, exact, exact, exact)
    rth += _mins(DAY, (9, 50), (16, 0), 699.0, 699.0, 699.0, 699.0)
    assert find_breakouts(rth, lv) == [], "a close exactly at the buffer fired"

    rth2 = _mins(DAY, (9, 30), (9, 40), 699.0, 699.0, 699.0, 699.0)
    rth2 += _mins(DAY, (9, 40), (9, 50), 701.0, 701.0, 701.0, 701.0)
    rth2 += _mins(DAY, (9, 50), (16, 0), 701.0, 701.0, 701.0, 701.0)
    assert len(find_breakouts(rth2, lv)) == 1


def test_a_wick_through_a_level_is_not_a_breakout():
    """2026-09-17's 718 print, in miniature. Close-based means close-based."""
    lv = [Level("D-1 high", 700.0, "up")]
    rth = _mins(DAY, (9, 30), (9, 40), 699.0, 699.5, 698.5, 699.0)
    rth += _mins(DAY, (9, 40), (9, 50), 699.0, 710.0, 698.0, 699.2)   # huge wick, closes in
    rth += _mins(DAY, (9, 50), (16, 0), 699.0, 699.5, 698.5, 699.0)
    assert find_breakouts(rth, lv) == []


# ------------------------------------------------------------------------- reporting

def test_move_bp_is_signed_by_trade_direction():
    """A short that profits must report a POSITIVE move. Getting this backwards would
    invert the sign of the whole claim-A result."""
    assert abs(move_bp(100.0, 101.0, "long") - 100.0) < 1e-9
    assert abs(move_bp(100.0, 99.0, "short") - 100.0) < 1e-9
    assert move_bp(100.0, 99.0, "long") < 0
    assert move_bp(100.0, 101.0, "short") < 0


def test_the_cost_floor_is_the_one_from_the_repo():
    """5 bp, from shares_runner.py. Claim B dies on the underlying if most signals miss it."""
    assert COST_FLOOR_BP == 5.0
    assert move_bp(700.0, 700.0 * 1.0004, "long") < COST_FLOOR_BP    # 4bp: does not clear
    assert move_bp(700.0, 700.0 * 1.0006, "long") > COST_FLOOR_BP    # 6bp: clears


def test_nearest_unbroken_picks_the_closest_level_in_the_direction():
    lv = [Level("a", 720.0, "up"), Level("b", 725.0, "up"), Level("c", 710.0, "down")]
    assert nearest_unbroken(lv, 718.0, "long").name == "a"
    assert nearest_unbroken(lv, 718.0, "short").name == "c"
    assert nearest_unbroken(lv, 730.0, "long") is None


# ------------------------------------------------- the sweep's outcome measurement

def _ten(bars):
    from .orb_veto import resample_10m
    return [b for b in resample_10m(bars) if b["ts"].time() >= dt.time(9, 30)]


def _one_signal(rth, levels):
    sigs = find_breakouts(rth, levels)
    assert sigs, "fixture produced no signal"
    return sigs[0]


def test_a_failed_breakout_exits_when_price_closes_back_inside():
    """Exit 2 in the pre-registration. The thesis was the break, so the thesis dying
    is the exit -- not an invented stop distance."""
    from .breakout_sweep import evaluate_signal
    lv = [Level("D-1 high", 700.0, "up")]
    rth = _mins(DAY, (9, 30), (9, 40), 699.0, 699.5, 698.5, 699.0)
    rth += _mins(DAY, (9, 40), (9, 50), 700.0, 702.0, 699.9, 701.5)   # break
    rth += _mins(DAY, (9, 50), (10, 0), 701.5, 702.0, 701.0, 701.8)   # fill bar
    rth += _mins(DAY, (10, 0), (16, 0), 699.0, 699.5, 698.0, 698.5)   # back inside
    r = evaluate_signal(_one_signal(rth, lv), _ten(rth))
    assert r["exit_reason"] == "failed", r
    assert r["exit_ts"][11:16] == "10:00", r["exit_ts"]
    assert r["move_bp"] < 0, r["move_bp"]


def test_a_trade_that_never_fails_runs_to_the_eod_flatten():
    from .breakout_sweep import evaluate_signal
    lv = [Level("D-1 high", 700.0, "up")]
    rth = _mins(DAY, (9, 30), (9, 40), 699.0, 699.5, 698.5, 699.0)
    rth += _mins(DAY, (9, 40), (9, 50), 700.0, 702.0, 699.9, 701.5)
    rth += _mins(DAY, (9, 50), (16, 0), 702.0, 710.0, 701.5, 709.0)
    r = evaluate_signal(_one_signal(rth, lv), _ten(rth))
    assert r["exit_reason"] == "eod", r
    assert r["move_bp"] > 0 and r["mfe_bp"] >= r["move_bp"], r


def test_mfe_is_measured_on_the_bar_extreme_not_the_close():
    """A profit cap is a limit order -- it fills on the extreme, not the close.

    Measuring MFE on closes would understate how often the +50% cap was reachable and
    would bias claim C toward 'the cap never triggers'.
    """
    from .breakout_sweep import evaluate_signal
    lv = [Level("D-1 high", 700.0, "up")]
    rth = _mins(DAY, (9, 30), (9, 40), 699.0, 699.5, 698.5, 699.0)
    rth += _mins(DAY, (9, 40), (9, 50), 700.0, 701.0, 699.9, 700.9)
    # Spikes to 715 intrabar, closes back at 701. MFE must see the 715.
    rth += _mins(DAY, (9, 50), (10, 0), 701.0, 715.0, 700.9, 701.0)
    rth += _mins(DAY, (10, 0), (16, 0), 701.0, 701.5, 700.8, 701.0)
    r = evaluate_signal(_one_signal(rth, lv), _ten(rth))
    assert r["mfe_bp"] > 150.0, f"MFE missed the intrabar extreme: {r['mfe_bp']}"
    assert r["move_bp"] < 50.0, r["move_bp"]


def test_a_short_signal_reports_a_favourable_move_as_positive():
    """Sign errors here would invert claim A for every down-break."""
    from .breakout_sweep import evaluate_signal
    lv = [Level("D-1 low", 700.0, "down")]
    rth = _mins(DAY, (9, 30), (9, 40), 701.0, 701.5, 700.5, 701.0)
    rth += _mins(DAY, (9, 40), (9, 50), 700.0, 700.1, 698.0, 698.5)   # break down
    rth += _mins(DAY, (9, 50), (16, 0), 698.0, 698.2, 690.0, 690.5)   # keeps falling
    r = evaluate_signal(_one_signal(rth, lv), _ten(rth))
    assert r["direction"] == "short"
    assert r["move_bp"] > 0, f"a profitable short reported a negative move: {r}"
    assert r["mfe_bp"] > 0


def test_post_switch_signals_are_flagged_not_dropped():
    """Prereg §4: after 13:00 the strategy wants 1DTE, which cannot be priced. Those
    signals are EXCLUDED from P&L and REPORTED, never silently discarded."""
    from .breakout_sweep import evaluate_signal
    lv = [Level("D-1 high", 700.0, "up")]
    rth = _mins(DAY, (9, 30), (13, 20), 699.0, 699.5, 698.5, 699.0)
    rth += _mins(DAY, (13, 20), (13, 30), 700.0, 702.0, 699.9, 701.5)
    rth += _mins(DAY, (13, 30), (16, 0), 702.0, 703.0, 701.5, 702.5)
    r = evaluate_signal(_one_signal(rth, lv), _ten(rth))
    assert r["post_switch"] is True, r["fill_ts"]

    lv2 = [Level("D-1 high", 700.0, "up")]
    early = _mins(DAY, (9, 30), (9, 40), 699.0, 699.5, 698.5, 699.0)
    early += _mins(DAY, (9, 40), (9, 50), 700.0, 702.0, 699.9, 701.5)
    early += _mins(DAY, (9, 50), (16, 0), 702.0, 703.0, 701.5, 702.5)
    assert evaluate_signal(_one_signal(early, lv2), _ten(early))["post_switch"] is False


def test_the_bootstrap_clusters_by_day_not_by_signal():
    """A trending day throws several signals and they are not independent.

    Eight days, each with three identical signals. Resampling DAYS must leave the CI
    wide; resampling signals would narrow it by sqrt(3) and manufacture significance --
    the same ~300x error already corrected once in the VRP arm.
    """
    from .breakout_sweep import boot_mean
    by_day = {f"d{i}": [10.0] * 3 if i % 2 else [-10.0] * 3 for i in range(12)}
    obs, lo, hi, p = boot_mean(by_day, reps=2000)
    assert abs(obs) < 1e-9, obs
    assert hi - lo > 6.0, f"CI width {hi - lo:.2f} too tight for 12 clusters"
    assert p > 0.05


def test_the_bootstrap_refuses_too_few_days():
    from .breakout_sweep import boot_mean
    assert boot_mean({f"d{i}": [1.0] for i in range(5)}) == (None, None, None, None)


def test_the_profit_cap_exits_on_an_intrabar_touch():
    """The exit the whole strategy is built around, and the one the first version of
    `evaluate_signal` OMITTED -- it measured 'hold until the breakout fails', which is
    close to the opposite, and reported a -13.79bp median on QQQ as if that were his
    strategy.

    A bar that spikes to +40bp and closes back at breakeven must still fill a +20bp
    limit. Uncapped sees ~0; capped sees +20.
    """
    from .breakout_sweep import CAP50_BP, evaluate_signal
    lv = [Level("D-1 high", 700.0, "up")]
    rth = _mins(DAY, (9, 30), (9, 40), 699.0, 699.5, 698.5, 699.0)
    rth += _mins(DAY, (9, 40), (9, 50), 700.0, 701.0, 699.9, 700.9)      # break
    entry = 700.9
    tp = entry * (1 + CAP50_BP / 10_000.0)
    rth += _mins(DAY, (9, 50), (10, 0), entry, tp * 1.002, entry, entry)  # spike, close flat
    rth += _mins(DAY, (10, 0), (16, 0), entry, entry, entry, entry)
    r = evaluate_signal(_one_signal(rth, lv), _ten(rth))
    assert r["cap_hit"] is True, r
    assert abs(r["move_capped_bp"] - CAP50_BP) < 0.5, r["move_capped_bp"]
    assert abs(r["move_bp"]) < 1.0, f"uncapped should be ~flat, got {r['move_bp']}"


def test_the_cap_beats_the_uncapped_exit_when_price_reverses():
    """The QQQ shape: goes +34bp in your favour, then gives it all back and more.

    This is the case that makes claim A and claim C compatible -- no terminal drift, and
    a reachable favourable excursion. If this test ever fails, the two series have been
    collapsed into one and the contrast is gone.
    """
    from .breakout_sweep import evaluate_signal
    lv = [Level("D-1 high", 700.0, "up")]
    rth = _mins(DAY, (9, 30), (9, 40), 699.0, 699.5, 698.5, 699.0)
    rth += _mins(DAY, (9, 40), (9, 50), 700.0, 701.0, 699.9, 700.9)
    rth += _mins(DAY, (9, 50), (10, 0), 700.9, 703.5, 700.8, 703.0)   # runs +30bp
    rth += _mins(DAY, (10, 0), (16, 0), 703.0, 703.0, 695.0, 696.0)   # gives it all back
    r = evaluate_signal(_one_signal(rth, lv), _ten(rth))
    assert r["cap_hit"] is True
    assert r["move_capped_bp"] > 0, r["move_capped_bp"]
    assert r["move_bp"] < 0, r["move_bp"]
    assert r["move_capped_bp"] > r["move_bp"], r


def test_a_trade_that_never_reaches_the_cap_reports_the_uncapped_exit():
    """No cap fill means the capped series must equal the control, not a better number."""
    from .breakout_sweep import evaluate_signal
    lv = [Level("D-1 high", 700.0, "up")]
    rth = _mins(DAY, (9, 30), (9, 40), 699.0, 699.5, 698.5, 699.0)
    rth += _mins(DAY, (9, 40), (9, 50), 700.0, 700.6, 699.9, 700.5)
    rth += _mins(DAY, (9, 50), (16, 0), 700.5, 700.6, 695.0, 696.0)   # straight down
    r = evaluate_signal(_one_signal(rth, lv), _ten(rth))
    assert r["cap_hit"] is False, r
    assert r["move_capped_bp"] == r["move_bp"], r
    assert r["move_bp"] < 0


def test_a_short_signals_cap_is_below_the_entry():
    """A sign error here would make every down-break's cap unreachable and silently
    halve the sample the cap applies to."""
    from .breakout_sweep import CAP50_BP, evaluate_signal
    lv = [Level("D-1 low", 700.0, "down")]
    rth = _mins(DAY, (9, 30), (9, 40), 701.0, 701.5, 700.5, 701.0)
    rth += _mins(DAY, (9, 40), (9, 50), 700.0, 700.1, 699.0, 699.3)   # break down
    entry = 699.3
    tp = entry * (1 - CAP50_BP / 10_000.0)
    rth += _mins(DAY, (9, 50), (10, 0), entry, entry, tp * 0.998, entry)
    rth += _mins(DAY, (10, 0), (16, 0), entry, entry, entry, entry)
    r = evaluate_signal(_one_signal(rth, lv), _ten(rth))
    assert r["direction"] == "short"
    assert r["cap_hit"] is True, r
    assert abs(r["move_capped_bp"] - CAP50_BP) < 0.5, r["move_capped_bp"]


def test_the_cap50_threshold_is_stated_not_folklore():
    """+50% on an ATM option needs ~20bp of underlying at delta 0.5. The constant is
    auditable so the reported fraction can be checked rather than believed."""
    from .breakout_sweep import CAP50_BP, SWITCH_AT
    assert CAP50_BP == 20.0
    assert SWITCH_AT == dt.time(13, 0)   # primary; 14:00 is secondary S2


CHECKS = [(n, f) for n, f in sorted(globals().items())
          if n.startswith("test_") and callable(f)]


def main() -> int:
    ok = True
    print("=" * 78)
    print("8-LINE BREAKOUT SIGNAL")
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
    print("\n  A level that fires twice is a trend detector wearing a win rate.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
