"""Tests for the early-R1 hold-to-close experiment.

What can lie here, in order of danger:

  1. LOOKAHEAD ON ENTRY. Entering at the break bar's own open instead of the next bar's
     open buys before the signal existed. A 1-minute lookahead once supplied 88-95.7% of
     a measured edge in this project, so this gets the sharpest test.
  2. ARM B QUIETLY KEEPING THE STOP. If B ever exits early the whole experiment measures
     nothing -- B would just be A with extra steps and (B - A) would read zero.
  3. A SESSION LANDING IN BOTH THE TREATMENT AND THE BENCHMARK. That would put the same
     day on both sides of the comparison the verdict rests on.

Each of those has a test below that was verified to fail against the corresponding bug.
"""
from __future__ import annotations

import datetime as dt

import pytest

from trade_analysis.live_lab.early_r1_hold import (
    CONTROL_BAR,
    _bp,
    _boot_diff,
    _boot_mean,
    evaluate_session,
)

DAY = dt.date(2024, 3, 14)


def _bars(prices, start="09:30"):
    """1-minute bars from a list of prices, one per minute, o=h=l=c."""
    h, m = (int(x) for x in start.split(":"))
    t0 = dt.datetime.combine(DAY, dt.time(h, m))
    return [{"ts": t0 + dt.timedelta(minutes=i), "open": p, "high": p,
             "low": p, "close": p, "volume": 1.0} for i, p in enumerate(prices)]


def _flat(n, p):
    return [p] * n


def _rec(price=100.0, close_break="09:30", pre_broken=False):
    return {"R1": {"price": price, "source": "x", "side": "R",
                   "pre_broken": pre_broken, "close_break": close_break,
                   "touch_break": close_break}}


# --------------------------------------------------------------------- basic maths

def test_bp_is_a_long_move_in_basis_points():
    assert _bp(100.0, 101.0) == pytest.approx(100.0)
    assert _bp(100.0, 99.0) == pytest.approx(-100.0)
    assert _bp(740.0, 740.74) == pytest.approx(10.0, abs=0.01)


# ------------------------------------------------------ 1. LOOKAHEAD ON ENTRY

def test_entry_is_the_next_bars_open_never_the_break_bars_own():
    """Break bar 09:30 closes at 101. The 09:40 bar OPENS at 150.

    Entry must be 150 -- the price you could actually get after seeing the close.
    Entering at the 09:30 bar's open (100) would be buying before the signal existed and
    would hand the strategy a free 50-point head start.
    """
    prices = _flat(10, 101.0) + _flat(10, 150.0) + _flat(370, 150.0)
    got = evaluate_session(_bars(prices), _rec(price=100.0, close_break="09:30"))
    assert got["entry"] == 150.0, "entry must come from the bar AFTER the break"
    assert got["entry_bar"] == "09:40"


def test_a_0940_break_enters_at_0950():
    prices = _flat(10, 99.0) + _flat(10, 101.0) + _flat(10, 120.0) + _flat(360, 120.0)
    got = evaluate_session(_bars(prices), _rec(price=100.0, close_break="09:40"))
    assert got["entry_bar"] == CONTROL_BAR
    assert got["entry"] == 120.0


# ------------------------------------------------ 2. ARM B MUST IGNORE THE STOP

def test_arm_b_holds_through_a_dip_that_stops_arm_a_out():
    """THE CASE THE WHOLE EXPERIMENT IS ABOUT.

    R1 = 100. Break at 09:30, enter 101 at 09:40. Price dips to 98 (closing back inside,
    so A is stopped out at -297 bp) and then runs to 110 by the close.

    A must book the dip. B must book the close. If B ever reads A's number the
    experiment is measuring nothing.
    """
    prices = (_flat(10, 101.0)          # 09:30 break bar
              + _flat(10, 101.0)        # 09:40 entry bar, opens 101
              + _flat(10, 98.0)         # 09:50 closes back inside R1 -> A exits
              + _flat(360, 110.0))      # recovers and closes at 110
    got = evaluate_session(_bars(prices), _rec(price=100.0, close_break="09:30"))
    assert got["a_exit"] == "failed"
    assert got["a_bp"] == pytest.approx(_bp(101.0, 98.0), abs=0.01)
    assert got["b_bp"] == pytest.approx(_bp(101.0, 110.0), abs=0.01)
    assert got["b_bp"] > got["a_bp"] + 1000, "B must not inherit A's stop"


def test_arm_b_also_takes_the_loss_when_price_never_recovers():
    """B is not a free option. When the dip keeps going, B is WORSE than A."""
    prices = _flat(10, 101.0) + _flat(10, 101.0) + _flat(10, 98.0) + _flat(360, 80.0)
    got = evaluate_session(_bars(prices), _rec(price=100.0, close_break="09:30"))
    assert got["a_exit"] == "failed"
    assert got["b_bp"] < got["a_bp"], "holding through a real breakdown must hurt"
    assert got["b_bp"] == pytest.approx(_bp(101.0, 80.0), abs=0.01)


def test_a_and_b_agree_exactly_when_the_stop_never_fires():
    prices = _flat(10, 101.0) + _flat(380, 105.0)
    got = evaluate_session(_bars(prices), _rec(price=100.0, close_break="09:30"))
    assert got["a_exit"] == "eod"
    assert got["a_bp"] == got["b_bp"]


def test_the_entry_bar_itself_cannot_trigger_the_stop():
    """Matches six_lines.first_break_trade: you cannot be stopped out on the bar you
    entered on. Without this the rule stops out on its own entry tick."""
    prices = _flat(10, 101.0) + _flat(10, 99.0) + _flat(370, 105.0)
    got = evaluate_session(_bars(prices), _rec(price=100.0, close_break="09:30"))
    assert got["a_exit"] == "eod", "the 09:40 entry bar closing at 99 must not stop A"


# ------------------------------------------- 3. TREATMENT AND BENCHMARK DISJOINT

def test_an_early_break_never_produces_a_benchmark_row():
    for t in ("09:30", "09:40"):
        got = evaluate_session(_bars(_flat(390, 105.0)), _rec(close_break=t))
        assert got["arm"] == "AB", t
        assert "c_bp" not in got


def test_a_late_break_is_a_benchmark_row_not_a_treatment_row():
    for t in ("09:50", "11:39", "15:30", None):
        got = evaluate_session(_bars(_flat(390, 105.0)), _rec(close_break=t))
        assert got["arm"] == "C", t
        assert "a_bp" not in got and "b_bp" not in got


def test_benchmark_enters_at_0950_open_and_exits_at_the_close():
    prices = _flat(20, 100.0) + _flat(10, 103.0) + _flat(360, 108.0)
    got = evaluate_session(_bars(prices), _rec(close_break=None))
    assert got["entry_bar"] == CONTROL_BAR
    assert got["entry"] == 103.0
    assert got["c_bp"] == pytest.approx(_bp(103.0, 108.0), abs=0.01)


def test_a_pre_broken_r1_is_dropped_from_every_arm():
    for t in ("09:30", "10:30", None):
        assert evaluate_session(_bars(_flat(390, 105.0)),
                                _rec(close_break=t, pre_broken=True)) is None


def test_a_session_too_short_to_hold_is_dropped():
    assert evaluate_session(_bars(_flat(15, 100.0)), _rec()) is None


def test_three_ten_minute_bars_is_the_floor_for_every_arm():
    """Three bars is the minimum that lets all arms fill, and it is why the
    'no next bar to enter on' branch is unreachable for an early break.

    Arm C enters at 09:50, index 2. A 09:40 break also enters at 09:50, index 2. So any
    session that clears `len(ten) >= 3` always has a bar to fill on, and any session
    that does not is dropped before the arms are reached.

    An earlier version of this test asserted that 20 one-minute bars produced a trade.
    That is 2 ten-minute bars, below the floor, so the test failed for a reason that had
    nothing to do with the code.
    """
    rec = _rec(close_break="09:30")
    assert evaluate_session(_bars(_flat(20, 100.0)), rec) is None, "2 bars is below floor"
    got = evaluate_session(_bars(_flat(30, 100.0)), rec)
    assert got is not None and got["entry_bar"] == "09:40"
    # arm C needs the 09:50 bar specifically
    assert evaluate_session(_bars(_flat(20, 100.0)), _rec(close_break=None)) is None
    assert evaluate_session(_bars(_flat(30, 100.0)), _rec(close_break=None)) is not None


# ------------------------------------------------------------------- inference

def test_boot_mean_recovers_a_known_mean_and_brackets_it():
    xs = [10.0] * 50 + [0.0] * 50
    o, lo, hi, p = _boot_mean(xs)
    assert o == pytest.approx(5.0)
    assert lo < 5.0 < hi


def test_boot_mean_on_pure_noise_does_not_claim_significance():
    xs = [(-1.0) ** i * 10.0 for i in range(200)]
    _o, lo, hi, p = _boot_mean(xs)
    assert lo < 0 < hi
    assert p > 0.05


def test_boot_diff_signs_follow_the_groups():
    hi_grp = [10.0] * 60
    lo_grp = [2.0] * 60
    o, lo, hi, p = _boot_diff(hi_grp, lo_grp)
    assert o == pytest.approx(8.0)
    assert lo > 0 and p < 0.05
    o2, _, _, _ = _boot_diff(lo_grp, hi_grp)
    assert o2 == pytest.approx(-8.0)


def test_boot_returns_nan_rather_than_a_number_on_a_tiny_sample():
    o, _lo, _hi, _p = _boot_mean([1.0, 2.0, 3.0])
    assert o != o, "too few points must return nan, not a confident mean"
