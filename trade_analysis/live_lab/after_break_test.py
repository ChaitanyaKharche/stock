"""Tests for the after-the-break sweep.

The thing most likely to lie here is direction. Every R-side comparison flips on the S
side -- reaching R2 means `rth_high >= price`, reaching S2 means `rth_low <= price` -- and
a copy-paste that leaves one of them pointing up would produce a clean-looking table of
completely wrong numbers with no crash. That is this project's standard failure mode, so
the S-side tests below are built to fail if any comparison is flipped.
"""
from __future__ import annotations

import math

import pytest

from trade_analysis.live_lab.after_break import (
    mins,
    round_dist,
    side_rows,
    two_prop_p,
    wilson,
)


def _session(day="2020-01-02", *, lines, rth_high, rth_low, trade=None):
    return {"day": day, "rth_high": rth_high, "rth_low": rth_low,
            "lines": lines, "trade": trade}


def _L(price, *, side, pre_broken=False, close_break=None):
    return {"price": price, "source": "x", "side": side,
            "pre_broken": pre_broken, "close_break": close_break,
            "touch_break": close_break}


def _six(r=(100.0, 105.0, 110.0), s=(95.0, 90.0, 85.0), *,
         r_break=(None, None, None), s_break=(None, None, None),
         r_pre=(False,) * 3, s_pre=(False,) * 3):
    out = {}
    for i in range(3):
        out[f"R{i + 1}"] = _L(r[i], side="R", pre_broken=r_pre[i], close_break=r_break[i])
        out[f"S{i + 1}"] = _L(s[i], side="S", pre_broken=s_pre[i], close_break=s_break[i])
    return out


# ------------------------------------------------------------------- round numbers

def test_round_dist_is_distance_to_the_nearest_multiple_either_way():
    assert round_dist(100.0, 25) == 0.0
    assert round_dist(101.0, 25) == 1.0
    assert round_dist(124.0, 25) == pytest.approx(1.0)   # nearest is 125, above
    assert round_dist(112.5, 25) == pytest.approx(12.5)  # exactly halfway
    assert round_dist(737.0, 25) == pytest.approx(12.0)  # 725 is 12 below, 750 is 13 above


def test_round_dist_never_exceeds_half_the_multiple():
    for p in (0.1, 7.3, 99.9, 512.4, 739.99):
        for m in (1, 5, 10, 25):
            assert 0.0 <= round_dist(p, m) <= m / 2 + 1e-9


def test_round_dist_of_a_uniform_grid_averages_to_m_over_4():
    """The null the report leans on: no clustering => mean distance is M/4."""
    m = 25
    pts = [400 + i * 0.01 for i in range(2500)]     # one full period, evenly spaced
    avg = sum(round_dist(p, m) for p in pts) / len(pts)
    assert avg == pytest.approx(m / 4, abs=0.02)


# ------------------------------------------------------------------------- stats

def test_wilson_handles_the_zero_and_all_cases():
    p, lo, hi = wilson(0, 50)
    assert p == 0.0 and lo == 0.0 and hi > 0.0
    p, lo, hi = wilson(50, 50)
    assert p == 1.0 and hi == 1.0 and lo < 1.0


def test_wilson_interval_narrows_as_n_grows():
    _, lo_a, hi_a = wilson(30, 60)
    _, lo_b, hi_b = wilson(300, 600)
    assert (hi_b - lo_b) < (hi_a - lo_a)


def test_two_prop_p_is_large_when_rates_match_and_small_when_they_differ():
    assert two_prop_p(50, 100, 100, 200) > 0.9
    assert two_prop_p(90, 100, 50, 200) < 0.001


def test_mins_parses_the_bar_clock():
    assert mins("09:30") == 570
    assert mins("16:00") == 960
    assert mins("13:20") - mins("09:40") == 220


# ------------------------------------------- R side: reaching a level means going UP

def test_r_side_reach2_needs_the_high_to_get_there():
    hit = _session(lines=_six(r_break=("09:30", None, None)),
                   rth_high=105.0, rth_low=99.0)
    miss = _session(lines=_six(r_break=("09:30", None, None)),
                    rth_high=104.99, rth_low=99.0)
    assert side_rows([hit], "R")[0]["reach2"] is True
    assert side_rows([miss], "R")[0]["reach2"] is False


def test_r_side_full_reversal_means_price_fell_through_s1():
    rev = _session(lines=_six(r_break=("09:30", None, None)),
                   rth_high=106.0, rth_low=94.99)
    no = _session(lines=_six(r_break=("09:30", None, None)),
                  rth_high=106.0, rth_low=95.01)
    assert side_rows([rev], "R")[0]["full_reversal"] is True
    assert side_rows([no], "R")[0]["full_reversal"] is False


# ------------------- S side: EVERY comparison flips. These fail if one points up.

def test_s_side_reach2_needs_the_low_to_get_there_not_the_high():
    """S2 sits BELOW S1. Reaching it means rth_low <= S2, never rth_high >= S2."""
    hit = _session(lines=_six(s_break=("09:30", None, None)),
                   rth_high=101.0, rth_low=90.0)
    miss = _session(lines=_six(s_break=("09:30", None, None)),
                    rth_high=101.0, rth_low=90.01)
    assert side_rows([hit], "S")[0]["reach2"] is True
    assert side_rows([miss], "S")[0]["reach2"] is False


def test_s_side_reach2_is_not_satisfied_by_a_big_rally():
    """A session that rockets up has a huge rth_high. If the comparison were pointing
    the wrong way that alone would mark S2 'reached'."""
    rally = _session(lines=_six(s_break=("09:30", None, None)),
                     rth_high=500.0, rth_low=94.0)
    assert side_rows([rally], "S")[0]["reach2"] is False


def test_s_side_full_reversal_means_price_rose_through_r1():
    rev = _session(lines=_six(s_break=("09:30", None, None)),
                   rth_high=100.01, rth_low=94.0)
    no = _session(lines=_six(s_break=("09:30", None, None)),
                  rth_high=99.99, rth_low=94.0)
    assert side_rows([rev], "S")[0]["full_reversal"] is True
    assert side_rows([no], "S")[0]["full_reversal"] is False


def test_s_side_distance_is_positive_even_though_s2_is_below_s1():
    row = side_rows([_session(lines=_six(s_break=("09:30", None, None)),
                              rth_high=101.0, rth_low=88.0)], "S")[0]
    assert row["dist_12_bp"] > 0


# --------------------------------------------------------------- population rules

def test_a_pre_broken_r1_is_excluded_entirely():
    s = _session(lines=_six(r_break=("09:30", None, None), r_pre=(True, False, False)),
                 rth_high=106.0, rth_low=99.0)
    assert side_rows([s], "R") == []


def test_a_session_where_r1_never_broke_is_excluded():
    s = _session(lines=_six(), rth_high=106.0, rth_low=99.0)
    assert side_rows([s], "R") == []


def test_early_is_the_first_two_bars_only():
    for t, want in (("09:30", True), ("09:40", True), ("09:50", False),
                    ("11:39", False), ("15:50", False)):
        s = _session(lines=_six(r_break=(t, None, None)), rth_high=106.0, rth_low=99.0)
        assert side_rows([s], "R")[0]["early"] is want, t


def test_gap_min_is_the_clock_distance_between_the_two_breaks():
    s = _session(lines=_six(r_break=("09:40", "10:20", None)),
                 rth_high=106.0, rth_low=99.0)
    assert side_rows([s], "R")[0]["gap_min"] == 40


def test_gap_min_is_none_when_r2_never_broke():
    s = _session(lines=_six(r_break=("09:40", None, None)),
                 rth_high=106.0, rth_low=99.0)
    assert side_rows([s], "R")[0]["gap_min"] is None


# ----------------------------------------------------- trade record is side-matched

def test_trade_fields_attach_only_when_the_trade_was_on_this_very_line():
    """The session's trade is its FIRST break, which may be the other side's line.
    Attaching an S1 trade's MFE to an R1 row would mix a short's numbers into a long."""
    s = _session(
        lines=_six(r_break=("09:40", None, None), s_break=("09:30", None, None)),
        rth_high=106.0, rth_low=94.0,
        trade={"line": "S1", "exit_reason": "failed", "mfe_bp": 9.0, "move_bp": -3.0})
    r_row = side_rows([s], "R")[0]
    s_row = side_rows([s], "S")[0]
    assert "mfe_bp" not in r_row, "an S1 trade must not decorate the R1 row"
    assert s_row["mfe_bp"] == 9.0
    assert s_row["giveback_bp"] == 12.0


def test_mfe_reached_2_compares_against_the_line_gap():
    far = _session(lines=_six(r=(100.0, 105.0, 110.0), r_break=("09:30", None, None)),
                   rth_high=106.0, rth_low=99.0,
                   trade={"line": "R1", "exit_reason": "eod",
                          "mfe_bp": 600.0, "move_bp": 500.0})
    short = _session(lines=_six(r=(100.0, 105.0, 110.0), r_break=("09:30", None, None)),
                     rth_high=106.0, rth_low=99.0,
                     trade={"line": "R1", "exit_reason": "eod",
                            "mfe_bp": 100.0, "move_bp": 50.0})
    # R1 100 -> R2 105 is ~488 bp
    assert side_rows([far], "R")[0]["dist_12_bp"] == pytest.approx(500.0, abs=15)
    assert side_rows([far], "R")[0]["mfe_reached_2"] is True
    assert side_rows([short], "R")[0]["mfe_reached_2"] is False
