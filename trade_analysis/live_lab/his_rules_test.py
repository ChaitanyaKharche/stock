"""Hand-written bars against the rules in his_rules.py.

    pytest trade_analysis/live_lab/his_rules_test.py

Each test encodes one sentence of research/his_rules_preregistration.md section 3 and was
run against a deliberately wrong rule first (filter inverted, window shifted, ladder
order swapped) to confirm it fails there.
"""
from __future__ import annotations

import datetime as dt

from . import his_rules as H
from .six_lines import Line

D = dt.date(2023, 6, 14)


def _bar(hm, o, h, l, c, v=1000.0):
    return {"ts": dt.datetime.combine(D, dt.time(*hm)), "open": o, "high": h, "low": l,
            "close": c, "volume": v}


def _ten(spec):
    """spec: list of (hm, o, h, l, c[, v]). Fills the rest of the day flat at the last close."""
    out = [_bar(*s) for s in spec]
    last = out[-1]
    t = last["ts"] + dt.timedelta(minutes=10)
    while t.time() < dt.time(16, 0):
        out.append({"ts": t, "open": last["close"], "high": last["close"], "low": last["close"],
                    "close": last["close"], "volume": 1000.0})
        t += dt.timedelta(minutes=10)
    return out


def _quiet_day(px=100.0):
    """A whole session of 1-point, 1000-volume bars from 09:30."""
    out, t = [], dt.datetime.combine(D, dt.time(9, 30))
    while t.time() < dt.time(16, 0):
        out.append({"ts": t, "open": px, "high": px + 0.5, "low": px - 0.5, "close": px,
                    "volume": 1000.0})
        t += dt.timedelta(minutes=10)
    return out


def _base():
    """Baselines full: median range 1.0, median volume 1000 at every clock time."""
    b = H.Baselines()
    for _ in range(H.LOOKBACK_SESSIONS):
        day = _quiet_day()
        for bar in day:
            b.rate(bar)
        b.end_session(day)
    return b


SIX = [Line("R1", 101.0, "", "R"), Line("R2", 103.0, "", "R"), Line("R3", 105.0, "", "R"),
       Line("S1", 99.0, "", "S"), Line("S2", 97.0, "", "S"), Line("S3", 95.0, "", "S")]


# ------------------------------------------------------------ the filters

def test_a_big_high_volume_close_above_R1_qualifies_and_enters_next_bar_open():
    ten = _ten([((9, 30), 100, 100.5, 99.5, 100), ((9, 40), 100, 102, 99.9, 101.8, 2000),
                ((9, 50), 101.9, 102.5, 101.5, 102.2)])
    sig = H.morning_signal(ten, H.rate_bars(ten, _base()), SIX)
    assert sig and sig["line"] == "R1" and sig["direction"] == "long"
    assert sig["signal_i"] == 1 and sig["entry_i"] == 2
    assert sig["range_ratio"] >= H.RANGE_MULT and sig["vol_ratio"] >= H.VOL_MULT


def test_a_small_candle_through_R1_is_not_a_signal():
    # range 1.2 points = 1.2x median, under 1.5x; volume fine
    ten = _ten([((9, 30), 100, 100.5, 99.5, 100), ((9, 40), 100.6, 101.7, 100.5, 101.6, 2000)])
    assert H.morning_signal(ten, H.rate_bars(ten, _base()), SIX) is None


def test_a_big_candle_on_normal_volume_is_not_a_signal():
    ten = _ten([((9, 30), 100, 100.5, 99.5, 100), ((9, 40), 100, 102, 99.9, 101.8, 1200)])
    assert H.morning_signal(ten, H.rate_bars(ten, _base()), SIX) is None


def test_no_baseline_means_no_signal():
    ten = _ten([((9, 30), 100, 100.5, 99.5, 100), ((9, 40), 100, 102, 99.9, 101.8, 5000)])
    assert H.morning_signal(ten, H.rate_bars(ten, H.Baselines()), SIX) is None


def test_a_bar_does_not_rate_against_its_own_range():
    b = H.Baselines()
    for bar in _quiet_day():
        b.rate(bar)
    rr, _ = b.rate(_bar((9, 30), 100, 110, 90, 100))
    assert rr == 20.0           # 20 / median 1.0, the wide bar itself not yet in the deque


def test_S1_break_is_a_short():
    ten = _ten([((9, 30), 100, 100.5, 99.5, 100), ((9, 40), 100, 100.1, 98, 98.2, 2000)])
    sig = H.morning_signal(ten, H.rate_bars(ten, _base()), SIX)
    assert sig and sig["line"] == "S1" and sig["direction"] == "short"


def test_a_session_that_opens_above_R1_has_not_broken_R1():
    # opens at 102 (R1=101 is pre-broken); a big bar through R2 must not count as an R1 break
    ten = _ten([((9, 30), 102, 102.5, 101.5, 102), ((9, 40), 102, 104, 101.9, 103.8, 2000)])
    assert H.morning_signal(ten, H.rate_bars(ten, _base()), SIX) is None


# ------------------------------------------------------------ the windows

def test_a_signal_whose_entry_would_open_after_10_45_is_skipped():
    ten = _ten([((9, 30), 100, 100.5, 99.5, 100)] +
               [((10, 40), 100, 102, 99.9, 101.8, 2000)])
    # fill the gap 09:40-10:30 flat
    ten = sorted(ten + [_bar((h, m), 100, 100.5, 99.5, 100) for h, m in
                        [(9, 40), (9, 50), (10, 0), (10, 10), (10, 20), (10, 30)]],
                 key=lambda b: b["ts"])
    ten = [b for i, b in enumerate(ten) if i == 0 or b["ts"] != ten[i - 1]["ts"]]
    assert H.morning_signal(ten, H.rate_bars(ten, _base()), SIX) is None


def test_the_last_morning_entry_is_the_10_40_bar():
    bars = [((9, 30), 100, 100.5, 99.5, 100)]
    bars += [((h, m), 100, 100.5, 99.5, 100) for h, m in
             [(9, 40), (9, 50), (10, 0), (10, 10), (10, 20)]]
    bars += [((10, 30), 100, 102, 99.9, 101.8, 2000)]
    ten = _ten(bars)
    sig = H.morning_signal(ten, H.rate_bars(ten, _base()), SIX)
    assert sig and ten[sig["entry_i"]]["ts"].time() == dt.time(10, 40)


def test_afternoon_takes_a_line_not_broken_in_the_morning_and_skips_one_that_was():
    bars = [((9, 30), 100, 100.5, 99.5, 100)]
    t = dt.time(9, 40)
    while t < dt.time(12, 30):
        bars.append(((t.hour, t.minute), 100, 100.5, 99.5, 100))
        t = (dt.datetime.combine(D, t) + dt.timedelta(minutes=10)).time()
    # 12:30 bar breaks R1 big and loud; R1 is NOT yet broken, so it is live in the afternoon
    bars.append(((12, 30), 100, 102, 99.9, 101.8, 2000))
    ten = _ten(bars)
    rated = H.rate_bars(ten, _base())
    sig = H.afternoon_signal(ten, rated, SIX, [])
    assert sig and sig["line"] == "R1" and sig["window"] == "afternoon"
    assert ten[sig["entry_i"]]["ts"].time() == dt.time(12, 40)
    # same day but R1 already close-broke at 09:40 (quietly): the afternoon must skip it
    bars[1] = ((9, 40), 100, 101.3, 99.9, 101.2)
    ten2 = _ten(bars)
    assert H.afternoon_signal(ten2, H.rate_bars(ten2, _base()), SIX, []) is None


def test_temporary_lines_are_the_morning_range_only_when_inside_the_six():
    rth = [{"ts": dt.datetime.combine(D, dt.time(9, 30)), "high": 102.0, "low": 99.5},
           {"ts": dt.datetime.combine(D, dt.time(11, 0)), "high": 101.0, "low": 98.5},
           {"ts": dt.datetime.combine(D, dt.time(13, 0)), "high": 110.0, "low": 90.0}]
    temps = H.temp_lines(rth, SIX)
    names = {t.name: t.price for t in temps}
    assert names == {"TR": 102.0, "TS": 98.5}        # 13:00 bar excluded; both inside R3/S3
    wide = [{"ts": dt.datetime.combine(D, dt.time(10, 0)), "high": 106.0, "low": 94.0}]
    assert H.temp_lines(wide, SIX) == []             # outside R3 and S3: no temp lines


def test_afternoon_can_break_a_temporary_line():
    bars = [((9, 30), 100, 100.5, 99.5, 100)]
    t = dt.time(9, 40)
    while t < dt.time(12, 30):
        bars.append(((t.hour, t.minute), 100, 100.5, 99.5, 100))
        t = (dt.datetime.combine(D, t) + dt.timedelta(minutes=10)).time()
    bars.append(((12, 30), 100, 101.6, 99.9, 100.9, 2000))   # closes above TR=100.5, below R1
    ten = _ten(bars)
    rth = [{"ts": b["ts"], "high": b["high"], "low": b["low"]} for b in ten]
    temps = H.temp_lines(rth[:-1], SIX)
    sig = H.afternoon_signal(ten, H.rate_bars(ten, _base()), SIX, temps)
    assert sig and sig["line"] == "TR"


# ------------------------------------------------------------ the pullback arm

def test_pullback_arm_enters_on_the_return_to_R1_after_R2_was_reached():
    ten = _ten([((9, 30), 100, 100.5, 99.5, 100),
                ((9, 40), 100.5, 101.4, 100.4, 101.3),      # quiet close-break of R1
                ((9, 50), 101.3, 103.2, 101.2, 102.9),      # reaches R2 within 30 min
                ((10, 0), 102.9, 103.0, 101.02, 101.5),     # dips to within 5 bp of R1, closes above
                ((10, 10), 101.5, 102, 101.4, 101.9)])
    sig = H.pullback_signal(ten, SIX)
    assert sig and sig["arm"] == "B" and sig["signal_i"] == 3 and sig["entry_i"] == 4


def test_pullback_arm_needs_R2_to_have_been_reached():
    ten = _ten([((9, 30), 100, 100.5, 99.5, 100),
                ((9, 40), 100.5, 101.4, 100.4, 101.3),
                ((9, 50), 101.3, 102.0, 101.2, 101.9),      # never reaches R2 = 103
                ((10, 0), 101.9, 102.0, 101.02, 101.5)])
    assert H.pullback_signal(ten, SIX) is None


# ------------------------------------------------------------ exits, underlying

def test_ladder_sells_2_2_1_at_20_30_40_bp_in_order():
    # entry 100 at 09:50; one bar sweeps to +45 bp
    ten = _ten([((9, 30), 100, 100.5, 99.5, 100), ((9, 40), 100, 102, 99.9, 101.8, 2000),
                ((9, 50), 100.0, 100.46, 100.0, 100.4)])
    r = H.simulate_underlying(ten, 2, "long", 101.0, "R")
    assert [(l["n"], l["why"]) for l in r["legs"]] == [(2, "target1"), (2, "target2"), (1, "target3")]
    assert [round(l["bp"]) for l in r["legs"]] == [20, 30, 40]
    assert round(r["weighted_bp"], 1) == round((2 * 20 + 2 * 30 + 40) / 5, 1)


def test_first_close_back_inside_sells_three_and_a_second_sells_the_rest():
    ten = _ten([((9, 30), 100, 100.5, 99.5, 100), ((9, 40), 100, 102, 99.9, 101.8, 2000),
                ((9, 50), 101.8, 101.9, 100.6, 100.7),      # closes back under R1=101 -> 3 out
                ((10, 0), 100.7, 101.5, 100.6, 101.2),      # back above: hold the 2
                ((10, 10), 101.2, 101.3, 100.2, 100.3)])    # second close inside -> rest out
    r = H.simulate_underlying(ten, 2, "long", 101.0, "R")
    assert [(l["n"], l["why"]) for l in r["legs"]] == [(3, "stop"), (2, "stop_rest")]
    assert r["legs"][0]["ts"].endswith("09:50:00") and r["legs"][1]["ts"].endswith("10:10:00")


def test_what_is_left_at_the_close_exits_at_the_last_bar():
    ten = _ten([((9, 30), 100, 100.5, 99.5, 100), ((9, 40), 100, 102, 99.9, 101.8, 2000),
                ((9, 50), 101.8, 101.9, 101.5, 101.6)])
    r = H.simulate_underlying(ten, 2, "long", 101.0, "R")
    assert r["legs"][-1]["why"] == "eod" and r["legs"][-1]["n"] == 5


def test_ladder_fills_before_a_close_stop_inside_the_same_bar():
    # the bar reaches +20 bp intrabar then closes back inside the line
    ten = _ten([((9, 30), 100, 100.5, 99.5, 100), ((9, 40), 100, 102, 99.9, 101.8, 2000),
                ((9, 50), 100.0, 100.25, 99.0, 99.5)])
    r = H.simulate_underlying(ten, 2, "long", 100.0, "R")
    assert [(l["n"], l["why"]) for l in r["legs"]][:2] == [(2, "target1"), (3, "stop")]


def test_short_side_mirrors():
    ten = _ten([((9, 30), 100, 100.5, 99.5, 100), ((9, 40), 100, 100.1, 98, 98.2, 2000),
                ((9, 50), 100.0, 100.0, 99.55, 99.6)])
    r = H.simulate_underlying(ten, 2, "short", 99.0, "S")
    assert r["legs"][0] == {**r["legs"][0], "n": 2, "why": "target1"}
    assert round(r["legs"][0]["bp"]) == 20


# ------------------------------------------------------------ exits, options

def _book(path, strike=100.0, right="C", bid_ask=(0.05,)):
    rows = []
    for mins, bid in path:
        ts = dt.datetime.combine(D, dt.time(9, 30)) + dt.timedelta(minutes=mins)
        rows.append({"ts": ts, "strike": strike, "right": right, "bid": bid, "ask": bid + bid_ask[0]})
    for k in (99.0, 101.0):
        rows.append({"ts": dt.datetime.combine(D, dt.time(9, 30)), "strike": k, "right": right,
                     "bid": 0.5, "ask": 0.55})
    return H.OptionBook(rows)


def test_option_entry_is_the_ask_and_ladder_legs_fill_at_the_limit():
    # entry minute 20 (09:50 open). bid path: 1.00 at entry -> 1.6 -> 1.8 -> 2.1
    book = _book([(0, 1.0), (20, 1.0), (25, 1.6), (30, 1.8), (35, 2.1)])
    ten = _ten([((9, 30), 100, 100.5, 99.5, 100), ((9, 40), 100, 102, 99.9, 101.8, 2000),
                ((9, 50), 100.0, 100.46, 100.0, 100.4)])
    r = H.simulate_option(book, ten, 2, "long", 100.0, "R")
    assert r["entry_px"] == 1.05 and r["strike"] == 100.0
    assert [(l["n"], l["px"]) for l in r["legs"]] == [(2, 1.575), (2, 1.8375), (1, 2.1)]
    gross = (2 * (1.575 - 1.05) + 2 * (1.8375 - 1.05) + 1 * (2.1 - 1.05)) * 100
    assert r["pnl_gross"] == round(gross, 2)
    assert r["pnl_net"] == round(gross - 2 * H.FEE_PER_CONTRACT_SIDE * 5, 2)


def test_option_stop_sells_at_the_bid_when_the_bar_closes_back_inside():
    # entry bar 09:50 closes back under the line: verdict known at 10:00, 3 sold at that bid.
    # 10:00 bar closes back above: the 2 are held. 10:10 bar closes inside again: rest at 10:20.
    book = _book([(0, 1.0), (20, 1.0), (25, 0.9), (30, 0.7), (40, 0.65), (50, 0.6), (385, 0.3)])
    ten = _ten([((9, 30), 100, 100.5, 99.5, 100), ((9, 40), 99.5, 101.2, 99.4, 100.9, 2000),
                ((9, 50), 100.0, 100.3, 99.6, 99.7), ((10, 0), 99.7, 100.4, 99.6, 100.2),
                ((10, 10), 100.2, 100.3, 99.4, 99.5)])
    r = H.simulate_option(book, ten, 2, "long", 100.0, "R")
    assert [(l["n"], l["px"], l["why"]) for l in r["legs"]] == [(3, 0.7, "stop"), (2, 0.6, "stop_rest")]
    assert r["legs"][0]["ts"].endswith("10:00:00") and r["legs"][1]["ts"].endswith("10:20:00")


def test_option_leftovers_exit_at_the_last_bid_before_15_55():
    book = _book([(0, 1.0), (20, 1.0), (380, 1.2), (390, 9.9)])       # 390 min = 16:00, ignored
    ten = _ten([((9, 30), 100, 100.5, 99.5, 100), ((9, 40), 99.5, 101.2, 99.4, 100.9, 2000),
                ((9, 50), 100.0, 100.3, 99.9, 100.2)])
    r = H.simulate_option(book, ten, 2, "long", 100.0, "R")
    assert r["legs"] == [{"n": 5, "px": 1.2, "why": "eod", "ts": r["legs"][0]["ts"]}]


def test_no_quote_within_two_minutes_of_entry_means_unpriced():
    book = _book([(0, 1.0), (30, 1.0)])
    ten = _ten([((9, 30), 100, 100.5, 99.5, 100), ((9, 40), 100, 102, 99.9, 101.8, 2000),
                ((9, 50), 100.0, 100.46, 100.0, 100.4)])
    assert H.simulate_option(book, ten, 2, "long", 101.0, "R") is None


def test_strike_offset_moves_out_of_the_money_on_both_sides():
    book = _book([(0, 1.0)])
    assert H.pick_strike(book, "C", 100.2) == 100.0
    assert H.pick_strike(book, "C", 100.2, +1) == 101.0
    rows = [{"ts": dt.datetime.combine(D, dt.time(9, 30)), "strike": k, "right": "P", "bid": 1, "ask": 1.1}
            for k in (99.0, 100.0, 101.0)]
    assert H.pick_strike(H.OptionBook(rows), "P", 100.2, +1) == 99.0


# ------------------------------------------------------------ one session, two trades max

def test_a_session_gives_at_most_one_morning_and_one_afternoon_trade():
    bars = [((9, 30), 100, 100.5, 99.5, 100), ((9, 40), 100, 102, 99.9, 101.8, 2000),
            ((9, 50), 101.8, 101.9, 100.6, 100.7), ((10, 0), 100.7, 100.8, 100.5, 100.6),
            ((10, 10), 100.6, 102.5, 100.5, 102.3, 2000)]            # a second big R1 break: ignored
    t = dt.time(10, 20)
    while t < dt.time(12, 30):
        bars.append(((t.hour, t.minute), 100.6, 100.9, 100.3, 100.6))
        t = (dt.datetime.combine(D, t) + dt.timedelta(minutes=10)).time()
    bars.append(((12, 30), 100.6, 100.7, 98.0, 98.3, 2000))          # S1 break in the afternoon
    ten = _ten(bars)
    rth = [{"ts": b["ts"], "high": b["high"], "low": b["low"]} for b in ten]
    trades = H.session_trades(ten, rth, SIX, _base())
    assert [(t_["window"], t_["line"]) for t_ in trades] == [("morning", "R1"), ("afternoon", "S1")]


def test_the_afternoon_waits_for_the_morning_position_to_close():
    # morning long from 101.8 drifts sideways: never +20 bp (102.0036), never back under R1.
    # At 12:30 a wide, loud bar closes above the temporary line TR=101.85 without touching
    # the ladder or the stop, so the morning position is still open at 12:40: no second trade.
    bars = [((9, 30), 100, 100.5, 99.5, 100), ((9, 40), 100, 101.85, 99.9, 101.8, 2000)]
    t = dt.time(9, 50)
    while t < dt.time(12, 30):
        bars.append(((t.hour, t.minute), 101.8, 101.85, 101.5, 101.8))
        t = (dt.datetime.combine(D, t) + dt.timedelta(minutes=10)).time()
    bars.append(((12, 30), 101.8, 101.95, 100.4, 101.9, 2000))
    ten = _ten(bars)
    rth = [{"ts": b["ts"], "high": b["high"], "low": b["low"]} for b in ten]
    temps = H.temp_lines(rth, SIX)
    assert {t_.name: t_.price for t_ in temps} == {"TR": 101.85, "TS": 99.5}
    assert H.afternoon_signal(ten, H.rate_bars(ten, _base()), SIX, temps)["line"] == "TR"
    trades = H.session_trades(ten, rth, SIX, _base())
    assert [t_["window"] for t_ in trades] == ["morning"]


def test_the_report_refuses_a_verdict_under_150_trades():
    rows = [{"day": "2023-01-0%d" % (i % 9 + 1), "arm": "A", "window": "morning",
             "underlying": {"weighted_bp": 1.0}, "underlying_flat": {"weighted_bp": 1.0},
             "opt_atm": {"pnl_net": 5.0, "entry_spread_pct": 3.0}, "opt_atm_flat": None,
             "opt_otm1": None, "opt_itm1": None} for i in range(30)]
    assert H.report(rows, options=True)["verdict"].startswith("UNDERPOWERED")


def test_the_run_refuses_to_price_options_into_the_held_out_2024_block():
    import pytest
    with pytest.raises(SystemExit):
        H.run("SPY", dt.date(2023, 12, 1), dt.date(2024, 1, 5), options=True)
