"""setup_baseline: the per-setup table, the pick-then-check split, the tail view of a live arm
and the off-ladder detector, on hand-built rows."""
from __future__ import annotations

from . import setup_baseline as S


def _t(setup, day, pnl, **kw):
    return {"setup_id": setup, "entry_ts": f"{day}T10:00:00", "pnl_net": pnl, **kw}


def test_per_setup_totals_rates_and_years():
    trades = [_t("A", "2019-01-02", 10.0), _t("A", "2020-01-02", -4.0), _t("B", "2019-01-03", -1.0)]
    p = S.per_setup(trades, n_days=3)
    assert list(p) == ["A", "B"]                                   # sorted by total, best first
    assert p["A"] == {"trades": 2, "total": 6.0, "per_trade": 3.0, "per_session": 2.0, "win_rate": 0.5,
                      "by_year": {"2019": 10.0, "2020": -4.0}}


def test_split_chooses_on_the_first_part_and_scores_only_the_second():
    days = ["2019-01-02", "2019-01-03", "2020-01-02", "2020-01-03"]
    trades = [_t("A", "2019-01-02", 10.0), _t("B", "2019-01-03", -5.0),      # first part: A ahead, B behind
              _t("A", "2020-01-02", -8.0), _t("B", "2020-01-03", 20.0)]     # second part: the reverse
    s = S.split_test(trades, days, "2019-12-31")
    assert s["chosen"] == ["A"] and s["first_sessions"] == 2 and s["second_sessions"] == 2
    assert s["chosen_first_per_session"] == 5.0                     # hindsight view
    assert s["chosen_second_per_session"] == -4.0                   # what choosing A actually earned
    assert s["book_second_per_session"] == 6.0 and s["chosen_second_each"] == {"A": -8.0}
    none = S.split_test([_t("A", "2019-01-02", -1.0), _t("A", "2020-01-02", 9.0)], days, "2019-12-31")
    assert none["chosen"] == [] and none["chosen_second_per_session"] == 0.0


def test_tail_summary_removes_the_largest_days_in_order():
    trades = [_t("A", "2026-09-01", 1000.0), _t("A", "2026-09-02", -100.0), _t("A", "2026-09-03", -100.0),
              _t("A", "2026-09-04", 600.0), _t("B", "2026-09-04", -50.0)]
    t = S.tail_summary(trades, target=500.0)
    assert t["sessions"] == 4 and t["total"] == 1350.0 and t["median_session"] == 225.0
    assert t["days_at_or_above_target"] == 2 and t["days_at_or_below_minus_target"] == 0
    assert [d["date"] for d in t["top_days"]] == ["2026-09-01", "2026-09-04", "2026-09-02"]
    assert t["without_top"]["1"] == {"total": 350.0, "per_session": 116.67}
    assert t["without_top"]["2"] == {"total": -200.0, "per_session": -100.0}


def test_live_per_setup_reports_the_total_without_the_excluded_days():
    trades = [_t("A", "2026-09-01", 1000.0), _t("A", "2026-09-02", -100.0), _t("B", "2026-09-01", 5.0)]
    p = S.live_per_setup(trades, {"2026-09-01"})
    assert p["A"]["total"] == 900.0 and p["A"]["total_without_top_days"] == -100.0 and p["A"]["win_rate"] == 0.5
    assert p["B"]["total_without_top_days"] == 0.0


def test_off_ladder_flags_a_far_atm_or_a_far_wing_and_nothing_else():
    def sig(day, sid, und, atm, lo, hi):
        return [{"entry_ts": f"{day}T10:00:00", "signal_id": sid, "arm": a, "strike": k, "entry_underlying": und}
                for a, k in (("ATM", atm), ("ATM-1", lo), ("ATM+1", hi))]
    rows = (sig("2026-11-20", "s1", 642.49, 635.0, 631.0, 654.0)        # ATM 7.5 away
            + sig("2026-11-20", "s2", 635.54, 636.0, 635.0, 646.0)      # wing 10 away
            + sig("2026-11-18", "s3", 700.20, 700.0, 699.0, 701.0))     # clean
    o = S.off_ladder(rows)
    assert o == {"signals": 3, "off_ladder": 2, "by_day": {"2026-11-20": 2}}
