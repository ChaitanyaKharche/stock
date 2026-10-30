"""task_id: R-2026-10-30-04 | parent: R-2026-10-30-01 | role: feature-script-agent | repo_sha: a367245 | status: done | verified_by: null

Tests for day_features, written against research/news_impact/preregistration.md §4, with
hand-built 1-minute bars. No feed, no network, no repo files (lab_days runs on tmp_path).
"""
from __future__ import annotations

import csv
import datetime as dt
import json

import pytest

from trade_analysis.news_impact import day_features as df
from trade_analysis.news_impact.day_features import (
    COLUMNS, LAB_COLUMNS, LOOKBACK, build_rows, features_for_session, lab_days, push_history,
)

YDAY = dt.date(2026, 3, 2)      # Monday
TODAY = dt.date(2026, 3, 3)


# ------------------------------------------------------------------ bar builders

def bars(day, t0, t1, px, marks=None):
    """One flat 1-minute bar per minute from t0 to t1 inclusive at `px`. `marks` is
    {time: (open, high, low, close)} for bars that should differ."""
    marks = marks or {}
    out, cur = [], dt.datetime.combine(day, t0)
    end = dt.datetime.combine(day, t1)
    while cur <= end:
        o, h, l, c = marks.get(cur.time(), (px, px, px, px))
        out.append({"ts": cur, "open": o, "high": h, "low": l, "close": c, "volume": 1.0})
        cur += dt.timedelta(minutes=1)
    return out


def pre(day, px=100.0, hi=100.5, lo=99.5):
    return bars(day, dt.time(4, 0), dt.time(9, 29), px,
                {dt.time(6, 0): (px, hi, lo, px)})


def rth(day, px=100.0, marks=None, until=dt.time(15, 59)):
    return bars(day, dt.time(9, 30), until, px, marks)


def yday_ext(close=100.0, pre_hi=100.5, pre_lo=99.5, rth_hi=101.0, rth_lo=99.0):
    """Yesterday's full extended session: premarket extremes, RTH extremes, RTH close."""
    marks = {dt.time(12, 0): (close, rth_hi, rth_lo, close)}
    return pre(YDAY, close, pre_hi, pre_lo) + rth(YDAY, close, marks)


def hist(n=LOOKBACK, rng=100.0, gap=10.0):
    return [(rng, gap)] * n


# The default lines from yday_ext() + pre(TODAY, hi=100.3, lo=99.7):
#   R1 100.3  R2 100.5  R3 101.0  |  S1 99.7  S2 99.5  S3 99.0
TODAY_PRE = pre(TODAY, 100.0, 100.3, 99.7)


# ------------------------------------------------------------------ 1. arithmetic

def test_gap_range_co_arithmetic():
    y = rth(YDAY, 400.0)                                   # prev close 400, no premarket
    t = rth(TODAY, 404.0, {dt.time(9, 30): (404.0, 404.0, 404.0, 404.0),
                           dt.time(11, 0): (404.0, 408.0, 400.0, 404.0),
                           dt.time(15, 59): (404.0, 404.0, 402.0, 402.0)})
    r = features_for_session(y, [], t, [])
    assert r["prev_close"] == 400.0 and r["open"] == 404.0 and r["close"] == 402.0
    assert r["high"] == 408.0 and r["low"] == 400.0
    assert r["gap_bp"] == pytest.approx(100.0)             # 404/400 - 1
    assert r["range_bp"] == pytest.approx(200.0)           # 408/400 - 1
    assert r["co_bp"] == pytest.approx((402 / 404 - 1) * 1e4)
    assert r["date"] == "2026-03-03" and r["thin"] is False
    assert list(r) == COLUMNS


def test_prev_close_is_yesterdays_rth_close_not_a_premarket_print():
    y_rth = rth(YDAY, 100.0, {dt.time(15, 59): (100, 100, 100, 100.7)})
    y_pre = pre(YDAY, 99.0, 99.0, 99.0)
    for y in (y_pre + y_rth, y_rth + y_pre):                   # list order must not matter
        r = features_for_session(y, [], rth(TODAY, 101.0), [])
        assert r["prev_close"] == 100.7 and r["prev_session_date"] == "2026-03-02"


def test_first60_window_is_0930_to_1029_and_excludes_1030():
    t = rth(TODAY, 100.0, {dt.time(10, 0): (100, 100.5, 99.8, 100),
                           dt.time(10, 29): (100, 100.6, 100, 100),
                           dt.time(10, 30): (100, 102.0, 98.0, 100)})
    r = features_for_session(None, [], t, [])
    assert r["first60_hi_bp"] == pytest.approx(60.0)       # 10:29 counts
    assert r["first60_lo_bp"] == pytest.approx(-20.0)
    assert r["high"] == 102.0 and r["low"] == 98.0         # 10:30 is in the day, not in first60
    assert r["pre_high"] is None and r["pre_low"] is None  # no premarket passed


def test_premarket_extremes():
    r = features_for_session(None, pre(TODAY, 100, 100.9, 99.1), rth(TODAY), [])
    assert (r["pre_high"], r["pre_low"]) == (100.9, 99.1)


# ------------------------------------------------------------------ 2. med20 excludes today

def test_med20_excludes_today_and_uses_only_the_last_20():
    # 20 varied ranges whose median is exactly 100; a 21st value of 300 would move it to 101
    ranges = [float(x) for x in list(range(90, 100)) + list(range(101, 111))]
    history = [(1000.0, 500.0)] * 5 + [(x, 10.0) for x in ranges]      # 5 stale, 20 live
    y = rth(YDAY, 100.0)
    t = rth(TODAY, 100.0, {dt.time(11, 0): (100.0, 103.0, 100.0, 100.0)})   # 300 bp
    before = list(history)
    r = features_for_session(y, [], t, history)
    assert r["med20_range_bp"] == pytest.approx(100.0)
    assert r["rel_range"] == pytest.approx(3.0)
    assert r["med20_absgap_bp"] == pytest.approx(10.0)
    assert history == before                               # pure: caller owns the history


def test_med20_absgap_uses_absolute_gaps():
    history = [(100.0, -10.0)] * 10 + [(100.0, 10.0)] * 10
    r = features_for_session(rth(YDAY, 100.0), [], rth(TODAY, 100.2), history)
    assert r["med20_absgap_bp"] == pytest.approx(10.0)     # not 0
    assert r["rel_gap"] == pytest.approx(2.0)


# ------------------------------------------------------------------ 3. short history

def test_fewer_than_20_prior_sessions_gives_none_but_a_row():
    r = features_for_session(yday_ext(), TODAY_PRE, rth(TODAY, 100.0,
                             {dt.time(11, 0): (100, 100.2, 99.8, 100)}), hist(LOOKBACK - 1))
    for k in ("med20_range_bp", "med20_absgap_bp", "rel_range", "rel_gap", "ranged_b",
              "day_class", "gap_reversal", "rel_range_band", "rel_gap_band", "impact"):
        assert r[k] is None, k
    assert r["range_bp"] == pytest.approx((100.2 / 99.8 - 1) * 1e4)   # what can be computed, is
    assert r["n_line_breaks"] == 0 and r["ranged_a"] is True
    assert r["date"] == "2026-03-03"
    assert list(r) == COLUMNS


# ------------------------------------------------------------------ 4. ranged_a from the six lines

def test_ranged_a_true_when_no_line_close_breaks():
    t = rth(TODAY, 100.0, {dt.time(11, 0): (100, 100.29, 99.71, 100)})   # inside R1/S1
    r = features_for_session(yday_ext(), TODAY_PRE, t, hist())
    assert r["n_line_breaks"] == 0 and r["ranged_a"] is True


def test_ranged_a_false_when_one_line_close_breaks():
    # a whole 10-minute bucket closes above R1 = 100.3 (today's premarket high)
    marks = {dt.time(11, m): (100.4, 100.4, 100.4, 100.4) for m in range(0, 10)}
    r = features_for_session(yday_ext(), TODAY_PRE, rth(TODAY, 100.0, marks), hist())
    assert r["n_line_breaks"] == 1 and r["ranged_a"] is False


def test_a_touch_without_a_close_beyond_is_not_a_break():
    marks = {dt.time(11, 0): (100.0, 100.45, 100.0, 100.0)}           # wick over R1, closes inside
    r = features_for_session(yday_ext(), TODAY_PRE, rth(TODAY, 100.0, marks), hist())
    assert r["n_line_breaks"] == 0 and r["ranged_a"] is True


def test_no_premarket_means_no_lines_and_none_not_zero():
    r = features_for_session(yday_ext(), [], rth(TODAY, 100.0), hist())
    assert r["n_line_breaks"] is None and r["ranged_a"] is None
    r = features_for_session(None, TODAY_PRE, rth(TODAY, 100.0), hist())
    assert r["n_line_breaks"] is None and r["ranged_a"] is None and r["prev_close"] is None


# ------------------------------------------------------------------ 5. ranged_b strict

def test_ranged_b_strict_at_0_6():
    on = rth(TODAY, 100.0, {dt.time(11, 0): (100, 100.6, 100, 100)})      # 60 bp / 100 = 0.6
    under = rth(TODAY, 100.0, {dt.time(11, 0): (100, 100.59, 100, 100)})  # 0.59
    assert features_for_session(yday_ext(), TODAY_PRE, on, hist())["ranged_b"] is False
    assert features_for_session(yday_ext(), TODAY_PRE, under, hist())["ranged_b"] is True


# ------------------------------------------------------------------ 6. day_class

def _day(open_, close, hi, lo, history, break_r1=False):
    marks = {dt.time(9, 30): (open_, max(open_, hi), min(open_, lo), open_),
             dt.time(11, 0): (open_, hi, lo, open_),
             dt.time(15, 59): (close, max(close, hi), min(close, lo), close)}
    if break_r1:
        marks.update({dt.time(13, m): (100.4, 100.4, 100.4, 100.4) for m in range(0, 10)})
    return features_for_session(yday_ext(), TODAY_PRE, rth(TODAY, open_, marks), history)


def test_day_class_trend_at_exactly_0_6_and_wins_over_ranged():
    # co 18 bp, range 30 bp -> 0.6, no line broken (ranged_a True), rel_range 0.3 (ranged_b True)
    r = _day(100.0, 100.18, 100.3, 100.0, hist())
    assert r["ranged_a"] is True and r["ranged_b"] is True
    assert r["day_class"] == "trend"
    r = _day(100.0, 100.17, 100.3, 100.0, hist())         # 0.567 -> not trend -> ranged
    assert r["day_class"] == "ranged"


def test_day_class_ranged_by_a_or_by_b_else_chop():
    # ranged by A only: no break, rel_range 1.0 (>= 0.6), flat close
    r = _day(100.0, 100.0, 100.5, 99.5, hist(rng=100.0))
    assert r["ranged_a"] is True and r["ranged_b"] is False and r["day_class"] == "ranged"
    # ranged by B only: R1 broken, rel_range 0.4
    r = _day(100.0, 100.0, 100.4, 100.0, hist(rng=100.0), break_r1=True)
    assert r["ranged_a"] is False and r["ranged_b"] is True and r["day_class"] == "ranged"
    # neither: R1 broken, rel_range 1.0, flat close
    r = _day(100.0, 100.0, 100.5, 99.5, hist(rng=100.0), break_r1=True)
    assert r["ranged_a"] is False and r["ranged_b"] is False and r["day_class"] == "chop"


def test_day_class_none_when_lines_missing():
    r = features_for_session(yday_ext(), [], rth(TODAY, 100.0,
                             {dt.time(11, 0): (100, 100.5, 99.5, 100)}), hist())
    assert r["ranged_a"] is None and r["day_class"] is None


# ------------------------------------------------------------------ 7. gap_reversal

def test_gap_reversal():
    h = hist(gap=10.0)
    y = rth(YDAY, 100.0)
    up_then_down = rth(TODAY, 100.2, {dt.time(15, 59): (100.2, 100.2, 100.1, 100.1)})
    assert features_for_session(y, [], up_then_down, h)["rel_gap"] == pytest.approx(2.0)
    assert features_for_session(y, [], up_then_down, h)["gap_reversal"] is True
    up_then_up = rth(TODAY, 100.2, {dt.time(15, 59): (100.2, 100.3, 100.2, 100.3)})
    assert features_for_session(y, [], up_then_up, h)["gap_reversal"] is False
    small_gap_flip = rth(TODAY, 100.09, {dt.time(15, 59): (100.09, 100.09, 100.0, 100.0)})
    r = features_for_session(y, [], small_gap_flip, h)
    assert r["rel_gap"] == pytest.approx(0.9) and r["gap_reversal"] is False
    down_then_up = rth(TODAY, 99.8, {dt.time(15, 59): (99.8, 99.9, 99.8, 99.9)})
    assert features_for_session(y, [], down_then_up, h)["gap_reversal"] is True


# ------------------------------------------------------------------ 8. thin days

def test_thin_day_row_written_not_pushed_to_history():
    h = hist(rng=100.0)
    half = rth(TODAY, 100.0, {dt.time(11, 0): (100, 105.0, 100.0, 100)}, until=dt.time(12, 59))
    assert len(half) == 210 < df.MIN_RTH_BARS
    r = features_for_session(yday_ext(), TODAY_PRE, half, h)
    assert r["thin"] is True
    assert r["open"] == 100.0 and r["range_bp"] == pytest.approx(500.0)   # computed, flagged
    assert r["date"] == "2026-03-03"
    assert push_history(h, r) is False and len(h) == LOOKBACK
    nxt = features_for_session(yday_ext(), TODAY_PRE, rth(TODAY, 100.0), h)
    assert nxt["med20_range_bp"] == pytest.approx(100.0)                   # unchanged


def test_full_day_is_pushed_and_moves_the_median():
    h = [(100.0, 10.0)] * LOOKBACK
    r = features_for_session(rth(YDAY, 100.0), [], rth(TODAY, 100.0,
                             {dt.time(11, 0): (100, 103.0, 100, 100)}), h)
    assert push_history(h, r) is True and h[-1] == (pytest.approx(300.0), pytest.approx(0.0))
    assert len(h) == LOOKBACK + 1


def test_at_299_bars_thin_at_300_not():
    r299 = features_for_session(None, [], rth(TODAY, until=dt.time(14, 28)), [])
    r300 = features_for_session(None, [], rth(TODAY, until=dt.time(14, 29)), [])
    assert len(rth(TODAY, until=dt.time(14, 29))) == 300
    assert r299["thin"] is True and r300["thin"] is False


def test_zero_rth_bars_gives_a_thin_row_with_no_break_tally():
    r = features_for_session(yday_ext(), TODAY_PRE, [], hist(), date=TODAY)
    assert r["thin"] is True and r["date"] == "2026-03-03"
    assert r["open"] is None and r["n_line_breaks"] is None and r["ranged_a"] is None


# ------------------------------------------------------------------ 8b. amendment A6: by-year, skip-over

def test_thin_day_counted_by_year_and_skipped_as_previous_session():
    d1, d2, d3, d4 = (dt.date(2025, 12, 30), dt.date(2025, 12, 31),
                      dt.date(2026, 1, 2), dt.date(2026, 1, 5))
    full = lambda d, px: (d, pre(d, px, px + 0.3, px - 0.3),              # noqa: E731
                          rth(d, px, {dt.time(11, 0): (px, px + 0.2, px - 0.2, px)}))
    half = (d3, pre(d3, 100.0, 100.3, 99.7),
            rth(d3, 105.0, until=dt.time(12, 59)))                   # 210 bars, thin
    rows, diag = build_rows([full(d1, 100.0), full(d2, 100.0), half, full(d4, 100.0)],
                            quiet=True)
    assert [r["date"] for r in rows] == ["2025-12-30", "2025-12-31", "2026-01-02", "2026-01-05"]
    assert diag["thin"] == 1 and diag["thin_by_year"] == {"2026": 1}
    assert rows[0]["prev_session_date"] is None
    assert rows[1]["prev_session_date"] == "2025-12-30"
    assert rows[2]["thin"] is True and rows[2]["prev_session_date"] == "2025-12-31"
    assert rows[3]["prev_session_date"] == "2025-12-31"          # skips the half day
    assert rows[3]["prev_close"] == 100.0                        # not the half day's 105
    assert rows[3]["gap_bp"] == pytest.approx(0.0)
    assert diag["rows_written"] == 4 and diag["days_tried"] == 4


def test_thin_by_year_counts_every_year_separately():
    days = [(dt.date(2016, 11, 25), [], rth(dt.date(2016, 11, 25), until=dt.time(12, 59))),
            (dt.date(2016, 12, 23), [], rth(dt.date(2016, 12, 23))),
            (dt.date(2017, 7, 3), [], rth(dt.date(2017, 7, 3), until=dt.time(12, 59))),
            (dt.date(2017, 11, 24), [], rth(dt.date(2017, 11, 24), until=dt.time(12, 59))),
            (dt.date(2018, 1, 2), [], [])]                          # no bars: no row, no thin
    rows, diag = build_rows(days, quiet=True)
    assert diag["thin_by_year"] == {"2016": 1, "2017": 2} and diag["thin"] == 3
    assert diag["no_bars"] == 1 and len(rows) == 4


def test_prev_session_date_in_the_row_matches_the_bars_that_built_the_lines():
    r = features_for_session(yday_ext(), TODAY_PRE, rth(TODAY, 100.0), hist())
    assert r["prev_session_date"] == "2026-03-02"
    assert features_for_session(None, TODAY_PRE, rth(TODAY), hist())["prev_session_date"] is None
    assert "prev_session_date" in COLUMNS


# ------------------------------------------------------------------ 9. bands and impact

def test_bands_edges_per_preregistration_table():
    assert [df.band(x, df.BANDS["rel_range"]) for x in (0.79, 0.8, 1.5, 1.51)] == \
        ["low", "medium", "medium", "high"]
    assert [df.band(x, df.BANDS["rel_gap"]) for x in (0.99, 1.0, 2.5, 2.51)] == \
        ["low", "medium", "medium", "high"]
    assert df.band(None, df.BANDS["rel_gap"]) is None


def test_impact_is_the_higher_band():
    h = hist(rng=100.0, gap=10.0)
    y = rth(YDAY, 100.0)
    # rel_range 1.0 (medium) + rel_gap 3.0 (high) -> high
    t = rth(TODAY, 100.3, {dt.time(11, 0): (100.3, 100.3, 99.3, 100.3)})    # range ~100.7 bp
    r = features_for_session(y, [], t, h)
    assert r["rel_range_band"] == "medium" and r["rel_gap_band"] == "high"
    assert r["impact"] == "high"
    # rel_range 0.5 (low) + rel_gap 1.5 (medium) -> medium
    t = rth(TODAY, 100.15, {dt.time(11, 0): (100.15, 100.65, 100.15, 100.15)})
    r = features_for_session(y, [], t, h)
    assert r["rel_range_band"] == "low" and r["rel_gap_band"] == "medium"
    assert r["impact"] == "medium"
    # high on rel_range alone, low gap -> high
    t = rth(TODAY, 100.0, {dt.time(11, 0): (100.0, 102.0, 100.0, 100.0)})
    r = features_for_session(y, [], t, h)
    assert r["rel_gap_band"] == "low" and r["impact"] == "high"
    assert df.impact_of("medium", None) is None


# ------------------------------------------------------------------ 10. lab join

def _jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def test_lab_days_sums_flags_and_last_non_opened_outcome(tmp_path):
    lab = tmp_path / "live_lab_data"
    _jsonl(lab / "trades.jsonl", [
        {"arm": "ATM", "entry_ts": "2026-09-01T09:45:04", "pnl_net": 5.0},
        {"arm": "ATM", "entry_ts": "2026-09-01T10:45:04", "pnl_net": -2.0},
        {"arm": "ATM-1", "entry_ts": "2026-09-01T09:45:04", "pnl_net": 100.0},   # other arm
        {"arm": "ATM", "entry_ts": "2026-09-01T11:45:04", "pnl_net": None},      # still open
        {"arm": "ATM", "entry_ts": "2026-09-03T09:45:04", "pnl_net": -1.0},
        {"arm": "ATM", "entry_ts": "2026-09-08T09:45:04", "pnl_net": 0.0},       # flat is not green
    ])
    _jsonl(lab / "shares" / "trades.jsonl", [
        {"arm": "SHARES", "entry_ts": "2026-09-01T09:39:07", "pnl_net": -4.0},
        {"arm": "SHARES", "entry_ts": "2026-09-02T09:39:07", "pnl_net": 1.5},
        {"arm": "SHARES", "entry_ts": "2026-09-03T09:39:07", "pnl_net": 2.0},
    ])
    _jsonl(lab / "session_ledger.jsonl", [
        {"date": "2026-09-01", "outcome": "OPENED"},
        {"date": "2026-09-01", "outcome": "PARTIAL"},
        {"date": "2026-09-02", "outcome": "OPENED"},
        {"date": "2026-09-02", "outcome": "INTERRUPTED"},        # superseded by the retry below
        {"date": "2026-09-02", "outcome": "OPENED"},
        {"date": "2026-09-02", "outcome": "COLLECTED"},
        {"date": "2026-09-02", "outcome": "OPENED"},             # a later restart, not an outcome
        {"date": "2026-09-05", "outcome": "NOT_A_SESSION"},
        {"date": "2026-09-08", "outcome": "ABORTED"},            # session with no trades
    ])
    out = tmp_path / "lab_days.csv"
    rows = lab_days(tmp_path, out=out)
    by = {r["date"]: r for r in rows}
    assert sorted(by) == ["2026-09-01", "2026-09-02", "2026-09-03", "2026-09-08"]
    d1 = by["2026-09-01"]
    assert d1["options_atm_net"] == pytest.approx(3.0) and d1["shares_net"] == pytest.approx(-4.0)
    assert (d1["options_green"], d1["shares_green"], d1["both_green"]) == (True, False, False)
    assert d1["ledger_outcome"] == "PARTIAL"
    d2 = by["2026-09-02"]
    assert d2["options_atm_net"] is None and d2["options_green"] is None
    assert d2["shares_green"] is True and d2["both_green"] is None
    assert d2["ledger_outcome"] == "COLLECTED"
    d3 = by["2026-09-03"]
    assert (d3["options_green"], d3["shares_green"], d3["both_green"]) == (False, True, False)
    assert d3["ledger_outcome"] is None
    d8 = by["2026-09-08"]
    assert d8["ledger_outcome"] == "ABORTED"
    assert d8["options_atm_net"] == 0.0 and d8["options_green"] is False
    assert d8["shares_net"] is None and d8["both_green"] is None
    with out.open(encoding="utf-8") as fh:
        rd = csv.DictReader(fh)
        assert rd.fieldnames == LAB_COLUMNS
        got = list(rd)
    assert got[1]["options_atm_net"] == "" and got[0]["both_green"] == "False"


# ------------------------------------------------------------------ output shape

def test_write_outputs_csv_order_and_json_sidecar(tmp_path):
    r = features_for_session(rth(YDAY, 100.0), [], rth(TODAY, 100.0), [])
    out = tmp_path / "day_features_QQQ.csv"
    side = df.write_outputs([r], {"days_tried": 1}, out, {"symbol": "QQQ"})
    with out.open(encoding="utf-8") as fh:
        rd = csv.DictReader(fh)
        assert rd.fieldnames == COLUMNS
        row = next(rd)
    assert row["med20_range_bp"] == "" and row["thin"] == "False" and row["date"] == "2026-03-03"
    j = json.loads(side.read_text(encoding="utf-8"))
    assert j["rows"][0]["date"] == "2026-03-03" and j["diagnostics"] == {"days_tried": 1}
    assert j["constants"]["RANGED_B_REL"] == 0.6 and j["constants"]["LOOKBACK"] == 20
    assert "git_sha" in j and j["columns"] == COLUMNS


def test_frozen_constants_match_preregistration():
    assert df.RANGED_B_REL == 0.6 and df.TREND_MIN == 0.6
    assert df.BANDS == {"rel_range": (0.8, 1.5), "rel_gap": (1.0, 2.5)}
    assert df.MIN_RTH_BARS == 300 and df.LOOKBACK == 20
