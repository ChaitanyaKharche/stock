"""analyze.py against the pre-registration: conditioning columns (15.4), BH/Holm, the
day-clustered bootstrap, the matched placebo (13.3, 14.Q6), zero-fill for Q7 (15.7),
pre/post (14.4, 15.4), Q6 divergence, the holdout refusal, the metric block."""
from __future__ import annotations

import datetime as dt
import json

import numpy as np
import pytest

from . import analyze as A
from . import join as J


def _rows(n=120, seed=1, cats=None):
    rng = np.random.default_rng(seed)
    rows = []
    d = dt.date(2019, 1, 2)
    i = 0
    while len(rows) < n:
        if d.weekday() < 5:
            c = cats(i) if cats else []
            rows.append({"date": d.isoformat(), "arm": "shares", "year": str(d.year), "weekday": d.weekday(),
                         "regime": ["low", "mid", "high"][i % 3], "categories": c, "n_categories": len(c),
                         "scheduled_today": bool(set(c) & set(J.SCHEDULED)), "any_event": bool(c),
                         "n_trades": 2, "net": float(rng.normal(0, 100)),
                         "setup_net": {"ORB_5min": 1.0} if i % 2 == 0 else {},
                         "setup_n": {"ORB_5min": 2} if i % 2 == 0 else {}, "config_hash": "h"})
            i += 1
        d += dt.timedelta(days=1)
    return rows


# ------------------------------------------------------------ 15.4 conditioning columns

def test_no_conditioning_column_is_built_from_the_sessions_own_prices():
    for col in A.CONDITIONING:
        for bad in A.FORBIDDEN:
            assert bad not in col, (col, bad)


# ------------------------------------------------------------ multiple testing

def test_bh_and_holm_on_known_pvalues():
    p = [0.001, 0.02, 0.03, 0.5, 0.9]
    assert A.bh(p, q=0.10) == [True, True, True, False, False]      # 0.03 <= 0.10*3/5
    assert A.bh([0.5, 0.9]) == [False, False] and A.bh([]) == []
    assert A.holm([0.01, 0.03], alpha=0.05) == [True, True]          # 0.01<=0.025, 0.03<=0.05
    assert A.holm([0.03, 0.04], alpha=0.05) == [False, False]        # 0.03 > 0.025 stops


# ------------------------------------------------------------ bootstrap

def test_bootstrap_effect_is_condition_minus_b0_with_a_ci_that_covers_it_and_is_seeded():
    z = np.array([1.0] * 40 + [0.0] * 60)
    mask = np.array([True] * 40 + [False] * 60)
    s = A.bootstrap_effect(z, mask, n_boot=500)
    assert s["effect"] == pytest.approx(0.6)                           # mean(cond)=1, mean(all)=0.4 and s["undiluted"] == pytest.approx(1.0)
    assert s["ci"][0] <= 0.6 <= s["ci"][1] and s["p"] == pytest.approx(1 / 500)
    assert A.bootstrap_effect(z, mask, n_boot=500) == s                # same seed, same numbers
    assert A.bootstrap_effect(z, np.zeros(100, bool))["effect"] is None
    few = np.zeros(100, bool); few[:3] = True
    s3 = A.bootstrap_effect(z, few)                                   # 3 sessions: effect, no CI
    assert s3["effect"] == pytest.approx(0.6) and s3["ci"] == [None, None] and s3["p"] is None


# ------------------------------------------------------------ placebo

def test_placebo_matches_year_and_weekday_and_never_draws_the_condition_itself():
    rows = _rows(200)
    net, year, weekday = A._arrays(rows)
    z = np.zeros(200)
    mask = np.zeros(200, bool)
    mask[[3, 17, 40, 88, 120, 150]] = True
    z[mask] = 50.0                                                   # only condition days are nonzero
    pl = A.placebo(z, mask, year, weekday, effect=float(z[mask].mean() - z.mean()), n_draw=300)
    assert pl["p95_same_sign"] == pytest.approx(-z.mean())            # every draw: 0 - B0; cond excluded
    assert pl["rank"] == 1.0 and pl["verdict"] == "effect"
    # weekday matching: a condition on Mondays only; a placebo on the same strata gives the
    # same statistic as a hand-made Monday draw would
    mon = np.array([r["weekday"] == 0 for r in rows])
    z2 = np.where(mon, 1.0, 0.0)
    m2 = mon.copy()
    m2[np.where(mon)[0][5:]] = False                                  # condition = first 5 Mondays
    pl2 = A.placebo(z2, m2, year, weekday, effect=float(z2[m2].mean() - z2.mean()), n_draw=200)
    assert pl2["p95_same_sign"] == pytest.approx(1.0 - z2.mean())     # every matched draw lands on Mondays


def test_placebo_calls_a_zero_effect_selection_or_chance():
    rows = _rows(200, seed=3)
    net, year, weekday = A._arrays(rows)
    z = net / net.std()
    mask = np.zeros(200, bool)
    mask[::7] = True
    eff = float(z[mask].mean() - z.mean())
    pl = A.placebo(z, mask, year, weekday, eff, n_draw=400)
    assert pl["rank"] < 0.95 and pl["verdict"] == "selection / chance"


# ------------------------------------------------------------ Q7 zero fill and minimums

def test_q7_counts_every_session_for_a_setup_and_requires_both_minimums():
    rows = _rows(120, cats=lambda i: ["CPI"] if i < 40 else [])
    tb = {}
    out = A.q7("shares", rows, tb, sigma=100.0)
    c = out["cells"]["ORB_5min|CPI"]
    assert c["n_sessions"] == 40                                       # full category count (15.7)
    assert c["n_trades"] == 40 and c["tested"] is False               # 20 trading sessions x 2 < 50
    assert c["one_sided_rate"] == 0.72 and c["low_fidelity"] is True
    rows2 = _rows(120, cats=lambda i: ["CPI"] if i < 60 else [])
    out2 = A.q7("shares", rows2, tb, sigma=100.0)
    assert out2["cells"]["ORB_5min|CPI"]["tested"] is True            # 60 sessions, 60 trades
    assert all("|financial_stress" not in k for k in out2["cells"])
    assert "ORB_5min|war_military" in out2["cells"] and out2["cells"]["ORB_5min|war_military"]["geopolitical"]


# ------------------------------------------------------------ Q5 pre/post

def test_pre_post_follows_14_4_and_15_4():
    rows = [{"date": "2021-03-17", "categories": ["FOMC"], "n_categories": 1, "scheduled_today": True,
             "year": "2021", "weekday": 2, "net": 0.0, "any_event": True},
            {"date": "2021-03-10", "categories": ["CPI"], "n_categories": 1, "scheduled_today": True,
             "year": "2021", "weekday": 2, "net": 0.0, "any_event": True},
            {"date": "2021-03-08", "categories": ["war_military"], "n_categories": 1, "scheduled_today": False,
             "year": "2021", "weekday": 0, "net": 0.0, "any_event": True},
            {"date": "2021-03-09", "categories": ["energy_shock"], "n_categories": 1, "scheduled_today": False,
             "year": "2021", "weekday": 1, "net": 0.0, "any_event": True}]
    detail = {"2021-03-17": [{"category": "FOMC", "minute": "14:00", "kind": "in_session", "surprise": False}],
              "2021-03-10": [{"category": "CPI", "minute": None, "kind": "all_post", "surprise": False}],
              "2021-03-08": [{"category": "war_military", "minute": None, "kind": "all_post", "surprise": True}],
              "2021-03-09": [{"category": "energy_shock", "minute": None, "kind": "unknown", "surprise": True}]}
    tb = {"2021-03-17": [{"entry_ts": "2021-03-17T13:58:00", "pnl_net": 1.0},
                         {"entry_ts": "2021-03-17T14:00:00", "pnl_net": 2.0}],
          "2021-03-10": [{"entry_ts": "2021-03-10T09:35:00", "pnl_net": 3.0}],
          "2021-03-08": [{"entry_ts": "2021-03-08T09:35:00", "pnl_net": 4.0}],
          "2021-03-09": [{"entry_ts": "2021-03-09T09:35:00", "pnl_net": 5.0}]}
    q = A.q5(rows, tb, detail)["pre_post"]
    assert q["FOMC"]["pre_n"] == 1 and q["FOMC"]["post_n"] == 1 and q["FOMC"]["pre_total_usd"] == 1.0
    assert q["CPI"]["pre_n"] == 0 and q["CPI"]["post_n"] == 1                    # pre-open: all post
    assert q["war_military"]["pre_n"] == 0 and q["war_military"]["post_n"] == 1   # overnight: all post
    assert q["energy_shock"]["post_n"] == 0 and q["energy_shock"]["sessions_excluded_unknown_time"] == 1


def test_first60_uses_entries_between_0930_and_1030_only():
    rows = [{"date": "2021-03-17", "scheduled_today": True, "categories": ["FOMC"], "n_categories": 1,
             "year": "2021", "weekday": 2, "net": 0.0, "any_event": True},
            {"date": "2021-03-18", "scheduled_today": False, "categories": [], "n_categories": 0,
             "year": "2021", "weekday": 3, "net": 0.0, "any_event": False}]
    tb = {"2021-03-17": [{"entry_ts": "2021-03-17T09:31:00", "pnl_net": 10.0},
                         {"entry_ts": "2021-03-17T10:30:00", "pnl_net": 99.0}],
          "2021-03-18": [{"entry_ts": "2021-03-18T10:29:00", "pnl_net": 4.0}]}
    f = A.q5(rows, tb, {"2021-03-17": [{"category": "FOMC", "minute": "14:00", "kind": "in_session",
                                        "surprise": False}]})["first60"]
    assert f["scheduled_today"]["first60_net_mean_usd"] == 10.0 and f["non_release"]["first60_net_mean_usd"] == 4.0
    assert f["difference_usd"] == 6.0


# ------------------------------------------------------------ Q6

def test_q6_divergence_needs_both_cis_to_exclude_zero_with_opposite_signs():
    def one(eff, lo, hi):
        return {"q1": {"CPI": {"effect_sigma": eff, "ci95": [lo, hi]}}}
    assert A.q6(one(0.5, 0.1, 0.9), one(-0.5, -0.9, -0.1))["CPI"]["divergence"] is True
    assert A.q6(one(0.5, -0.1, 0.9), one(-0.5, -0.9, -0.1))["CPI"]["divergence"] is False
    assert A.q6(one(0.5, 0.1, 0.9), one(0.3, 0.1, 0.5))["CPI"]["divergence"] is False


# ------------------------------------------------------------ holdout guard

def test_holdout_lock_is_shut_on_the_committed_files_and_opens_only_on_the_auditors_whole_line(tmp_path, monkeypatch):
    """Gate 2 (2026-11-10): the first lock matched a substring of results.md and was already
    open because the report described the lock. The lock reads audit_results.md, whole line."""
    audit = (A.OUT / "audit_results.md").read_text(encoding="utf-8").strip().splitlines()[-1].strip()
    assert audit in ("GATE 2 VERDICT: PASS", "GATE 2 VERDICT: FAIL")
    assert A.holdout_allowed() is (audit == "GATE 2 VERDICT: PASS")   # lock state == the auditor's last line
    monkeypatch.setattr(A, "OUT", tmp_path)
    assert A.main(["--arm", "shares", "--set", "holdout"]) == 2
    (tmp_path / "results.md").write_text("GATE 2 VERDICT: PASS\n", encoding="utf-8")
    assert A.holdout_allowed() is False                     # wrong file
    (tmp_path / "audit_results.md").write_text("the line 'GATE 2 VERDICT: PASS' would open it\n", encoding="utf-8")
    assert A.holdout_allowed() is False                     # token inside prose
    (tmp_path / "audit_results.md").write_text("...\nGATE 2 VERDICT: FAIL\n", encoding="utf-8")
    assert A.holdout_allowed() is False
    (tmp_path / "audit_results.md").write_text("...\n  GATE 2 VERDICT: PASS \n", encoding="utf-8")
    assert A.holdout_allowed() is True
    (tmp_path / "audit_results.md").write_text("GATE 2 VERDICT: PASS\nlater: GATE 2 VERDICT: FAIL\n", encoding="utf-8")
    assert A.holdout_allowed() is False                     # only the last line counts


def test_unstable_flags_sign_flips_across_regimes_or_three_years():
    base = {"effect_sigma": -0.3, "n_sessions": 40}
    reg = lambda *e: {k: {"n": 10, "effect_vs_stratum_b0": x} for k, x in zip("lmh", e)}
    yrs = lambda *e: {str(2018 + i): {"n": 8, "effect_vs_stratum_b0": x} for i, x in enumerate(e)}
    assert A._unstable({**base, "by_regime": reg(-0.1, -0.2, -0.5), "by_year": yrs(-0.1, -0.2, 0.1, -0.3, 0.2)}) is False
    assert A._unstable({**base, "by_regime": reg(-0.1, 0.2, -0.5), "by_year": yrs(-0.1, -0.2, -0.3, -0.3, -0.2)}) is True
    assert A._unstable({**base, "by_regime": reg(-0.1, -0.2, -0.5), "by_year": yrs(0.1, 0.2, 0.1, -0.3, -0.2)}) is True
    thin = {"n": 3, "effect_vs_stratum_b0": 0.9}
    assert A._unstable({**base, "by_regime": {"l": thin}, "by_year": yrs(-0.1, -0.2, -0.3, -0.3, -0.2)}) is False


def test_exclusions_list_half_days_recorded_skips_and_lost_sessions():
    rows = [{"date": d} for d in ("2022-11-21", "2022-11-22", "2022-11-23", "2022-11-29", "2022-11-30")]
    chunks = [{"skipped": [{"date": "2022-11-28", "reason": "half_day"}]}]   # Nov 28 was a full day
    ex = A.exclusions(rows, (dt.date(2022, 11, 21), dt.date(2022, 11, 30)), chunks)
    assert ex["half_days"] == ["2022-11-25"]                               # day after Thanksgiving
    assert ex["lost_regular_sessions"] == ["2022-11-28"] and ex["n_lost"] == 1   # a short full day is lost
    chunks = [{"skipped": [{"date": "2022-11-28", "reason": "holiday"}]}]   # "holiday" on an open day: lost too
    ex = A.exclusions(rows, (dt.date(2022, 11, 21), dt.date(2022, 11, 30)), chunks)
    assert ex["lost_regular_sessions"] == ["2022-11-28"] and ex["open_days_in_window"] == 7
    ex = A.exclusions(rows, (dt.date(2022, 11, 21), dt.date(2022, 11, 30)), [], no_0dte=["2022-11-28"])
    assert ex["recorded_skips"] == [{"date": "2022-11-28", "reason": "no_0dte"}] and ex["n_lost"] == 0
    ex2 = A.exclusions(rows[:-1], (dt.date(2022, 11, 21), dt.date(2022, 11, 30)), chunks)
    assert ex2["lost_regular_sessions"] == ["2022-11-30"] and ex2["open_days_in_window"] == 7
    ex3 = A.exclusions(rows[:-1], (dt.date(2022, 11, 21), dt.date(2022, 11, 30)), chunks, no_0dte=["2022-11-30"])
    assert ex3["n_lost"] == 1 and ex3["recorded_skips_by_reason"] == {"no_0dte": 1}


def test_q3_unscheduled_is_every_surprise_session_with_the_exclusive_variant_beside_it():
    rows = _rows(60, cats=lambda i: ["CPI", "war_military"] if i == 0 else (["war_military"] if i < 10 else []))
    detail = {r["date"]: [{"category": "war_military", "minute": None, "kind": "all_post", "surprise": True}]
              for r in rows if "war_military" in r["categories"]}
    tb = {}
    out = A.q3("shares", rows, tb, 100.0, detail)
    assert out["unscheduled_surprise"]["n_sessions"] == 10
    assert out["unscheduled_surprise_no_release_sensitivity"]["n_sessions"] == 9


def test_block_bootstrap_is_seeded_and_covers_the_effect():
    rng = np.random.default_rng(0)
    z = rng.normal(size=300)
    mask = np.zeros(300, bool); mask[::6] = True
    z[mask] += 1.0
    eff = z[mask].mean() - z.mean()
    lo, hi = A.block_bootstrap_ci(z, mask, block=10, n_boot=400)
    assert lo < eff < hi and A.block_bootstrap_ci(z, mask, block=10, n_boot=400) == [lo, hi]
    assert A.block_bootstrap_ci(z, np.zeros(300, bool)) == [None, None]


# ------------------------------------------------------------ metric block and overlap flag

def test_metric_block_units_per_arm_and_shutdown_exits_excluded():
    rows = [{"net": 10.0}, {"net": -4.0}]
    sh = A.metric_block("shares", [{"pnl_net": 10.0, "notional": 10000.0}, {"pnl_net": -4.0, "notional": 10000.0}],
                        rows, sigma=100.0)
    assert sh["expectancy_bp"] == pytest.approx(3.0) and sh["win_rate_trade"] == 0.5
    assert sh["max_drawdown_usd"] == -4.0 and sh["z_mean"] == pytest.approx(0.03)
    op = A.metric_block("options", [
        {"pnl_net": 20.0, "entry_ask": 2.0, "contracts": 1, "exit_reason": "failed", "entry_spread_pct": 0.01,
         "underlying_return": 0.001},
        {"pnl_net": -10.0, "entry_ask": 1.0, "contracts": 1, "exit_reason": "shutdown", "entry_spread_pct": 0.02,
         "underlying_return": -0.001}], rows, sigma=100.0)
    assert op["exit_mix"] == {"stop": 1.0} and op["exits_shutdown_excluded"] == 1
    assert op["return_on_premium_mean"] == pytest.approx((0.1 - 0.1) / 2)
    assert op["median_premium_usd"] == 150.0


def test_confounded_by_overlap_flags_sign_flip_or_half_loss():
    assert A._confounded({"effect_sigma": 0.4}, {"effect_sigma": 0.1, "n_sessions": 30}) is True
    assert A._confounded({"effect_sigma": 0.4}, {"effect_sigma": -0.3, "n_sessions": 30}) is True
    assert A._confounded({"effect_sigma": 0.4}, {"effect_sigma": 0.3, "n_sessions": 30}) is False
    assert A._confounded({"effect_sigma": 0.4}, {"effect_sigma": 0.3, "n_sessions": 2}) is None


# ------------------------------------------------------------ event detail from the files

def test_event_detail_kinds(tmp_path, monkeypatch):
    (tmp_path / "events_scheduled.csv").write_text(
        "date,session,type,time_et,source_url,note\n"
        "2021-03-17,2021-03-17,FOMC,14:00,u,\n2021-03-10,2021-03-10,CPI,08:30,u,\n"
        "2021-03-12,2021-03-15,PCE,17:00,u,after close\n", encoding="utf-8")
    monkeypatch.setattr(J, "NI", tmp_path)
    monkeypatch.setattr(A.E, "load_geopolitical", lambda: [
        {"date": "2021-03-08", "break_time_et": "unknown", "category": "war_military", "headline": "x", "surprise": "yes"},
        {"date": "2021-03-09", "break_time_et": "11:00", "category": "energy_shock", "headline": "y", "surprise": "yes"},
        {"date": "2021-03-11", "break_time_et": "11:00", "category": "us_fiscal", "headline": "z (Saturday)", "surprise": "no"}])
    d = A.event_detail()
    assert d["2021-03-17"][0]["kind"] == "in_session" and d["2021-03-17"][0]["minute"] == "14:00"
    assert d["2021-03-10"][0]["kind"] == "all_post"
    assert d["2021-03-15"][0]["kind"] == "all_post" and d["2021-03-15"][0]["category"] == "PCE"
    assert d["2021-03-08"][0]["kind"] == "unknown"
    assert d["2021-03-09"][0] == {"category": "energy_shock", "minute": "11:00", "kind": "in_session", "surprise": True}
    assert d["2021-03-11"][0]["kind"] == "all_post"                     # overnight marker beats the time


# ------------------------------------------------------------ end to end on synthetic rows

def test_q1_q3_run_end_to_end_and_mark_tested_cells(monkeypatch):
    rows = _rows(300, seed=5, cats=lambda i: ["CPI"] if i % 5 == 0 else (["war_military"] if i % 11 == 0 else []))
    tb = {r["date"]: [{"entry_ts": r["date"] + "T09:40:00", "pnl_net": r["net"] / 2, "notional": 1e4},
                      {"entry_ts": r["date"] + "T11:40:00", "pnl_net": r["net"] / 2, "notional": 1e4}] for r in rows}
    monkeypatch.setattr(A, "N_BOOT", 200)
    monkeypatch.setattr(A, "N_PLACEBO", 200)
    out = A.q1("shares", rows, tb, sigma=100.0, direction={})
    assert out["CPI"]["tested"] is True and out["CPI"]["n_sessions"] == 60
    assert out["war_military"]["n_sessions"] == 22                    # i%11==0 and not i%5==0
    assert out["war_military"]["tested"] is False and out["war_military"]["descriptive_placebo"] is False
    assert out["financial_stress"]["never_tested"] is True and out["financial_stress"]["effect_sigma"] is None
    assert out["CPI"]["bh_q10"] in (True, False) and out["war_military"]["bh_q10"] is None
    q3 = A.q3("shares", rows, tb, 100.0, {r["date"]: [{"category": "war_military", "minute": None,
                                                      "kind": "all_post", "surprise": True}]
                                          for r in rows if "war_military" in r["categories"]})
    assert q3["scheduled"]["n_sessions"] == 60 and q3["unscheduled_surprise"]["n_sessions"] == 22
    assert q3["no_event"]["n_sessions"] == 218
