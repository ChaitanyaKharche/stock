"""Are there exactly six lines, from the right places, rolling the right way?

    python -m trade_analysis.live_lab.six_lines_test
    pytest trade_analysis/live_lab/six_lines_test.py

Hand-written bars. The spec, from 2026-09-20:

    yesterday's premarket high/low  +  yesterday's market high/low
                                    +  today's premarket high/low   =  6

and rolling into tomorrow, yesterday's four drop off. The check that matters most is
`test_yesterdays_premarket_survives_inside_its_rth_range`: the previous implementation
took yesterday's extremes over the whole 04:00-20:00 session, which silently DELETES the
premarket line whenever it sits inside the RTH range -- which is most days. Six lines
became four without anything failing.
"""
from __future__ import annotations

import datetime as dt
import sys

from .six_lines import breaks_today, build_six

D0 = dt.date(2026, 9, 17)      # "yesterday"
D1 = dt.date(2026, 9, 18)      # "today"


def _mins(day, start, end, o, h, l, c, v=1000.0):
    out, t = [], dt.datetime.combine(day, dt.time(*start))
    stop = dt.datetime.combine(day, dt.time(*end))
    while t < stop:
        out.append({"ts": t, "open": o, "high": h, "low": l, "close": c, "volume": v})
        t += dt.timedelta(minutes=1)
    return out


def _yday(pre_h, pre_l, rth_h, rth_l):
    """A full extended session with separately controlled premarket and RTH extremes."""
    b = _mins(D0, (4, 0), (9, 30), (pre_h + pre_l) / 2, pre_h, pre_l,
              (pre_h + pre_l) / 2)
    b += _mins(D0, (9, 30), (16, 0), (rth_h + rth_l) / 2, rth_h, rth_l,
               (rth_h + rth_l) / 2)
    b += _mins(D0, (16, 0), (20, 0), rth_h, rth_h, rth_l, rth_h)   # after hours
    return b


def _today_pre(h, l):
    return _mins(D1, (4, 0), (9, 30), (h + l) / 2, h, l, (h + l) / 2)


# ------------------------------------------------------------------- the six lines

def test_exactly_six_lines_from_the_three_stated_places():
    lines = build_six(_yday(712.0, 708.0, 718.0, 710.0), _today_pre(716.0, 714.0))
    assert len(lines) == 6, [l.name for l in lines]
    assert {l.name for l in lines} == {"R1", "R2", "R3", "S1", "S2", "S3"}
    assert {l.source for l in lines} == {
        "yday premarket high", "yday market high", "today premarket high",
        "yday premarket low", "yday market low", "today premarket low"}


def test_yesterdays_premarket_survives_inside_its_rth_range():
    """The bug that turned six lines into four.

    Yesterday's premarket ran 712/708 entirely INSIDE its RTH range of 718/710. Taking
    one extreme over 04:00-20:00 returns 718/708 and both premarket lines vanish. They
    must both still be here.
    """
    lines = build_six(_yday(712.0, 708.0, 718.0, 710.0), _today_pre(716.0, 714.0))
    prices = {l.source: l.price for l in lines}
    assert prices["yday premarket high"] == 712.0, prices
    assert prices["yday market high"] == 718.0, prices
    assert prices["yday market low"] == 710.0, prices
    assert prices["yday premarket low"] == 708.0, prices
    assert len({l.price for l in lines}) == 6, "two lines collapsed onto one price"


def test_R1_is_the_nearest_resistance_and_S1_the_nearest_support():
    """R1 is the lowest of the three highs -- the first one price meets going up.
    S1 is the highest of the three lows. Getting this inverted would make R1 the level
    that almost never breaks and R3 the one that almost always does."""
    lines = build_six(_yday(712.0, 708.0, 718.0, 710.0), _today_pre(716.0, 714.0))
    by = {l.name: l.price for l in lines}
    assert by["R1"] == 712.0 and by["R2"] == 716.0 and by["R3"] == 718.0, by
    assert by["S1"] == 714.0 and by["S2"] == 710.0 and by["S3"] == 708.0, by
    assert by["R1"] < by["R2"] < by["R3"]
    assert by["S1"] > by["S2"] > by["S3"]


def test_R_lines_break_upward_and_S_lines_downward():
    lines = build_six(_yday(712.0, 708.0, 718.0, 710.0), _today_pre(716.0, 714.0))
    assert all(l.side == "R" for l in lines if l.name.startswith("R"))
    assert all(l.side == "S" for l in lines if l.name.startswith("S"))


def test_a_day_missing_premarket_yields_no_lines_rather_than_four():
    """Four lines is not this strategy. Better to record nothing than a different setup."""
    assert build_six(_yday(712.0, 708.0, 718.0, 710.0), []) == []
    b = _mins(D0, (9, 30), (16, 0), 715.0, 718.0, 710.0, 716.0)   # no yday premarket
    assert build_six(b, _today_pre(716.0, 714.0)) == []


def test_the_roll_drops_yesterday_and_keeps_today():
    """D+1's six must come from D's four plus D+1's premarket two, with D-1 gone."""
    yday = _yday(712.0, 708.0, 718.0, 710.0)
    today_pre = _today_pre(716.0, 714.0)
    today_rth = _mins(D1, (9, 30), (16, 0), 716.0, 722.0, 713.0, 720.0)

    tomorrow_pre = _mins(dt.date(2026, 9, 21), (4, 0), (9, 30),
                         721.0, 723.0, 719.0, 721.0)
    rolled = build_six(today_pre + today_rth, tomorrow_pre)
    prices = {l.price for l in rolled}
    assert prices == {716.0, 714.0, 722.0, 713.0, 723.0, 719.0}, prices
    # every level unique to D-1 is gone
    assert not ({712.0, 708.0, 718.0, 710.0} & prices), prices


# ------------------------------------------------------------------------- breaking

def test_a_close_beyond_a_line_is_a_close_break_and_a_touch_break():
    lines = build_six(_yday(712.0, 708.0, 718.0, 710.0), _today_pre(716.0, 714.0))
    rth = _mins(D1, (9, 30), (9, 40), 715.0, 715.5, 714.5, 715.0)
    rth += _mins(D1, (9, 40), (9, 50), 716.5, 717.0, 716.2, 716.9)   # above R2=716
    rth += _mins(D1, (9, 50), (16, 0), 716.9, 717.0, 716.5, 716.9)
    rec = breaks_today(rth, lines)
    assert rec["R1"]["pre_broken"] is True, rec["R1"]       # 712 is below the 715 open
    assert rec["R2"]["close_break"] == "09:40", rec["R2"]
    assert rec["R3"]["close_break"] is None, rec["R3"]      # 718 never reached


def test_a_wick_through_a_line_is_a_touch_break_only():
    """2026-09-17's 718 print. The whole reason both are recorded."""
    lines = build_six(_yday(712.0, 708.0, 718.0, 710.0), _today_pre(716.0, 714.0))
    rth = _mins(D1, (9, 30), (9, 40), 715.0, 715.5, 714.5, 715.0)
    rth += _mins(D1, (9, 40), (9, 50), 715.0, 719.0, 714.9, 715.2)   # wick over 718
    rth += _mins(D1, (9, 50), (16, 0), 715.2, 715.5, 714.8, 715.0)
    rec = breaks_today(rth, lines)
    assert rec["R3"]["touch_break"] == "09:40", rec["R3"]
    assert rec["R3"]["close_break"] is None, rec["R3"]


def test_each_line_records_only_its_FIRST_break():
    """Otherwise 'how many lines broke' becomes 'how long the trend lasted'."""
    lines = build_six(_yday(712.0, 708.0, 718.0, 710.0), _today_pre(716.0, 714.0))
    rth = _mins(D1, (9, 30), (9, 40), 715.0, 715.5, 714.5, 715.0)
    rth += _mins(D1, (9, 40), (16, 0), 719.0, 725.0, 718.9, 724.0)   # far above, all day
    rec = breaks_today(rth, lines)
    assert rec["R3"]["close_break"] == "09:40", rec["R3"]
    # R1 is pre-broken at the 715 open, so two of the three R lines are in play.
    assert sum(1 for v in rec.values() if v["close_break"]) == 2, rec


def test_a_contained_session_breaks_nothing():
    """The trader's no-trade day, in this vocabulary: zero of six.

    R1=712 is below the 715 open, so it is PRE-BROKEN and takes no part. Without that
    rule this session would read "1 of 6 broke" on a day where price never left a
    1.8-point band -- the tally would be measuring the overnight gap, not the session.

    S1=714 is genuinely below the open and stays in play; price simply never reaches it.
    That asymmetry is the point: pre-broken is about which SIDE of the open a level is
    on, not about how close it is.
    """
    lines = build_six(_yday(712.0, 708.0, 718.0, 710.0), _today_pre(716.0, 714.0))
    rth = _mins(D1, (9, 30), (16, 0), 715.0, 715.9, 714.1, 715.0)
    rec = breaks_today(rth, lines)
    assert rec["R1"]["pre_broken"] is True, rec["R1"]     # 712 is below the 715 open
    assert rec["S1"]["pre_broken"] is False, rec["S1"]    # 714 is below it: still live
    assert sum(1 for v in rec.values() if v["pre_broken"]) == 1, rec
    assert all(v["close_break"] is None for v in rec.values()), rec
    assert all(v["touch_break"] is None for v in rec.values()), rec


def test_a_pre_broken_line_can_never_register_a_break():
    """Even if price later crosses back and forth over it. It was never in play."""
    lines = build_six(_yday(712.0, 708.0, 718.0, 710.0), _today_pre(716.0, 714.0))
    rth = _mins(D1, (9, 30), (9, 40), 715.0, 715.5, 714.5, 715.0)
    rth += _mins(D1, (9, 40), (16, 0), 711.0, 711.5, 710.5, 711.0)   # back under R1=712
    rec = breaks_today(rth, lines)
    assert rec["R1"]["pre_broken"] is True
    assert rec["R1"]["close_break"] is None and rec["R1"]["touch_break"] is None, rec["R1"]


def test_downside_breaks_are_recorded_on_the_S_lines():
    lines = build_six(_yday(712.0, 708.0, 718.0, 710.0), _today_pre(716.0, 714.0))
    rth = _mins(D1, (9, 30), (9, 40), 715.0, 715.5, 714.5, 715.0)
    rth += _mins(D1, (9, 40), (16, 0), 713.0, 713.5, 707.0, 707.5)   # through all lows
    rec = breaks_today(rth, lines)
    assert rec["S1"]["close_break"] == "09:40", rec["S1"]
    assert rec["S2"]["close_break"] == "09:40", rec["S2"]
    assert rec["S3"]["close_break"] == "09:40", rec["S3"]
    assert all(v["close_break"] is None for k, v in rec.items()
               if k.startswith("R")), {k: v for k, v in rec.items() if k[0] == "R"}


# ------------------------------------------------- one trade per session, first break

def test_the_first_break_is_the_one_taken_and_only_one_per_session():
    """92% of sessions break something, so taking every break is the weather, not a
    strategy. One per day matches the journal's 1-2."""
    from .six_lines import first_break_trade
    lines = build_six(_yday(712.0, 708.0, 718.0, 710.0), _today_pre(716.0, 714.0))
    rth = _mins(D1, (9, 30), (9, 40), 715.0, 715.5, 714.5, 715.0)
    rth += _mins(D1, (9, 40), (9, 50), 716.5, 717.0, 716.2, 716.9)    # R2=716 first
    rth += _mins(D1, (9, 50), (16, 0), 719.0, 725.0, 718.9, 724.0)    # R3 later
    rec = breaks_today(rth, lines)
    tr = first_break_trade(rth, lines, rec)
    assert tr is not None
    assert tr["line"] == "R2", tr            # the FIRST, not the biggest
    assert tr["signal_at"] == "09:40", tr
    assert tr["direction"] == "long"


def test_the_entry_is_the_bar_after_the_break():
    from .six_lines import first_break_trade
    lines = build_six(_yday(712.0, 708.0, 718.0, 710.0), _today_pre(716.0, 714.0))
    rth = _mins(D1, (9, 30), (9, 40), 715.0, 715.5, 714.5, 715.0)
    rth += _mins(D1, (9, 40), (9, 50), 716.5, 717.0, 716.2, 716.9)    # break bar
    rth += _mins(D1, (9, 50), (16, 0), 718.0, 718.5, 717.5, 718.0)    # fill bar opens 718
    tr = first_break_trade(rth, lines, breaks_today(rth, lines))
    assert tr["entry"] == 718.0, tr          # next bar's OPEN, not the break close


def test_a_session_with_no_break_produces_no_trade():
    """Not a zero. A no-trade day is absent from the P&L series, counted separately."""
    from .six_lines import first_break_trade
    lines = build_six(_yday(712.0, 708.0, 718.0, 710.0), _today_pre(716.0, 714.0))
    rth = _mins(D1, (9, 30), (16, 0), 715.0, 715.9, 714.1, 715.0)
    assert first_break_trade(rth, lines, breaks_today(rth, lines)) is None


def test_a_short_trade_reports_a_favourable_move_as_positive():
    from .six_lines import first_break_trade
    lines = build_six(_yday(712.0, 708.0, 718.0, 710.0), _today_pre(716.0, 714.0))
    rth = _mins(D1, (9, 30), (9, 40), 715.0, 715.5, 714.5, 715.0)
    rth += _mins(D1, (9, 40), (9, 50), 713.5, 713.8, 713.0, 713.2)    # under S1=714
    rth += _mins(D1, (9, 50), (16, 0), 713.0, 713.1, 705.0, 705.5)    # keeps falling
    tr = first_break_trade(rth, lines, breaks_today(rth, lines))
    assert tr["direction"] == "short", tr
    assert tr["move_bp"] > 0, tr
    assert tr["cap_hit"] is True, tr


def test_the_cap_cannot_fill_after_the_trade_exited():
    """Same lookahead that voided a published number on 2026-09-19. Pinned here too."""
    from .six_lines import first_break_trade
    lines = build_six(_yday(712.0, 708.0, 718.0, 710.0), _today_pre(716.0, 714.0))
    rth = _mins(D1, (9, 30), (9, 40), 715.0, 715.5, 714.5, 715.0)
    rth += _mins(D1, (9, 40), (9, 50), 716.2, 716.4, 716.1, 716.3)    # breaks R2=716
    rth += _mins(D1, (9, 50), (10, 0), 716.3, 716.4, 716.2, 716.3)    # fill bar
    rth += _mins(D1, (10, 0), (10, 10), 715.0, 715.2, 714.0, 714.5)   # back under: exit
    rth += _mins(D1, (10, 10), (16, 0), 715.0, 780.0, 714.0, 778.0)   # huge run AFTER
    tr = first_break_trade(rth, lines, breaks_today(rth, lines))
    assert tr["exit_reason"] == "failed", tr
    assert tr["cap_hit"] is False, "the cap filled after the trade was closed"
    assert tr["move_capped_bp"] == tr["move_bp"], tr


CHECKS = [(n, f) for n, f in sorted(globals().items())
          if n.startswith("test_") and callable(f)]


def main() -> int:
    ok = True
    print("=" * 78)
    print("SIX LINES -- YDAY PREMARKET + YDAY MARKET + TODAY PREMARKET")
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
    print("\n  Six at a time. Never four, never eight.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
