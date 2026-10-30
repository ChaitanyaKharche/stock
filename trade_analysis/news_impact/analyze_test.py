"""task_id: R-2026-10-30-05 | parent: R-2026-10-30-01 | role: analysis-script-agent | repo_sha: a367245 | status: done | verified_by: null

Synthetic-data tests for analyze.py. No file under research/ is read; every CSV is built
in tmp_path. Each test encodes one rule of research/news_impact/preregistration.md §6 or
§9 and was run against a deliberately wrong implementation first (floor removed, Holm as
plain Bonferroni, geopolitical days removed from the control, join on `date`) to confirm
it fails there.

    pytest trade_analysis/news_impact/analyze_test.py
"""
from __future__ import annotations

import csv
import datetime as dt
import json

import numpy as np
import pytest

from . import analyze as A

REPS = 600            # enough for a stable CI on these samples; the determinism test uses it too
SEED = A.SEED


def _dates(n, start=dt.date(2020, 1, 6)):
    out, d = [], start
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d.isoformat())
        d += dt.timedelta(days=1)
    return out


def _row(date, rel_range=1.0, rel_gap=0.5, thin=False, ranged_a=False, ranged_b=False,
         day_class="chop", co_bp=10.0, f60_hi=20.0, f60_lo=-20.0, med20=100.0):
    return {"date": date, "rel_range": rel_range, "rel_gap": rel_gap, "thin": thin,
            "ranged_a": ranged_a, "ranged_b": ranged_b, "day_class": day_class, "co_bp": co_bp,
            "first60_hi_bp": f60_hi, "first60_lo_bp": f60_lo, "med20_range_bp": med20}


def _sample(n_event, n_non, shift, spread=0.1, type_="FOMC"):
    """Non-event rel_range evenly spread on [1-spread, 1+spread]; event days the same plus
    `shift`. Returns (features, events)."""
    dates = _dates(n_event + n_non)
    non_vals = np.linspace(1 - spread, 1 + spread, n_non)
    ev_vals = np.linspace(1 - spread, 1 + spread, n_event) + shift
    feats = [_row(d, float(v)) for d, v in zip(dates[:n_non], non_vals)]
    feats += [_row(d, float(v)) for d, v in zip(dates[n_non:], ev_vals)]
    events = [(d, type_) for d in dates[n_non:]]
    return feats, events


# 1. H1 pass / fail / floor -------------------------------------------------------------
def test_h1_pass_when_event_days_one_full_unit_wider():
    feats, events = _sample(40, 80, shift=1.0)
    r = A.analyze_symbol(feats, events, reps=REPS, seed=SEED)
    h1 = r["H1"]["rel_range"]
    assert h1["diff"] == pytest.approx(1.0, abs=1e-9)
    assert h1["ci95"][0] > 0
    assert h1["pass"] is True
    assert r["H1"]["status"] == "tested"


def test_h1_fail_on_identical_distributions():
    feats, events = _sample(40, 80, shift=0.0)
    r = A.analyze_symbol(feats, events, reps=REPS, seed=SEED)
    h1 = r["H1"]["rel_range"]
    assert abs(h1["diff"]) < 0.05
    assert h1["pass"] is False


def test_h1_fail_on_small_but_significant_difference():
    """A tight CI that excludes 0 is not enough: the pre-registration demands >= 0.2."""
    feats, events = _sample(200, 200, shift=0.1, spread=0.1)
    r = A.analyze_symbol(feats, events, reps=REPS, seed=SEED)
    h1 = r["H1"]["rel_range"]
    assert h1["diff"] == pytest.approx(0.1, abs=1e-9)
    assert h1["ci95"][0] > 0, "the CI must exclude 0 for this test to probe the floor"
    assert h1["pass"] is False


def test_h1_pass_rule_is_direct_on_the_stat():
    assert A.h1_pass({"diff": 0.2, "ci95": [0.05, 0.4]}) is True
    assert A.h1_pass({"diff": 0.19, "ci95": [0.05, 0.4]}) is False
    assert A.h1_pass({"diff": 0.5, "ci95": [-0.1, 0.9]}) is False
    assert A.h1_pass({"diff": -0.5, "ci95": [-0.9, -0.1]}) is False     # wrong direction
    assert A.h1_pass({"diff": None, "ci95": None}) is False


def test_h1_with_fewer_than_30_fomc_days_is_not_tested():
    feats, events = _sample(29, 80, shift=1.0)
    r = A.analyze_symbol(feats, events, reps=REPS, seed=SEED)
    assert r["H1"]["status"].startswith("reported, not tested")
    assert r["H1"]["rel_range"]["pass"] is False
    assert r["H1"]["rel_range"]["diff"] == pytest.approx(1.0, abs=1e-9)   # still reported


# A1: geopolitical days stay in the control group -------------------------------------
def test_geopolitical_days_stay_in_the_scheduled_control():
    dates = _dates(120)
    feats = [_row(d, 1.0) for d in dates]
    for d in dates[:30]:                      # 30 very wide geopolitical days
        feats[dates.index(d)]["rel_range"] = 5.0
    events = [(d, "war_military") for d in dates[:30]] + [(d, "FOMC") for d in dates[80:]]
    r = A.analyze_symbol(feats, events, reps=REPS, seed=SEED)
    # control = 120 - 40 FOMC = 80 sessions, including the 30 geopolitical ones
    assert r["H1"]["n_control_days"] == 80 and r["n_scheduled_control_days"] == 80
    assert r["n_nonevent_days"] == 50
    assert r["H1"]["rel_range"]["n_b"] == 80
    # the wide geo days are in the control: control median is still 1.0 but its mean is not;
    # check they are present by their count in the control values
    ctrl_vals = sorted([5.0] * 30 + [1.0] * 50)
    assert r["H1"]["rel_range"]["median_b"] == pytest.approx(float(np.median(ctrl_vals)))
    # the geopolitical line itself is compared with the 50 non-event days
    g = r["geopolitical_descriptive"]["categories"]["war_military"]
    assert g["n_control_days"] == 50 and g["rel_range"]["diff"] == pytest.approx(4.0)
    # and a FOMC day is NOT in the control of the geopolitical line nor in its own control
    assert r["H2"]["types"] == {} or all(e["n_control_days"] == 80 for e in r["H2"]["types"].values())


def test_fomc_day_is_removed_from_the_control_of_cpi():
    dates = _dates(140)
    feats = [_row(d, 1.0) for d in dates]
    events = [(d, "FOMC") for d in dates[:35]] + [(d, "CPI") for d in dates[35:70]]
    r = A.analyze_symbol(feats, events, reps=50, seed=SEED)
    assert r["H2"]["types"]["CPI"]["n_control_days"] == 70
    assert r["H3"]["types"]["CPI"]["n_control_days"] == 70
    assert r["H3"]["types"]["FOMC"]["n_control_days"] == 70


# 2. Holm ----------------------------------------------------------------------------
def test_holm_known_vector():
    # m=4. sorted: .01,.03,.04,.20 -> 4*.01=.04, 3*.03=.09, max(.09, 2*.04=.08)=.09, 1*.20=.20
    p = [0.01, 0.04, 0.03, 0.20]
    assert A.holm(p) == pytest.approx([0.04, 0.09, 0.09, 0.20])


def test_holm_caps_at_one_and_skips_none():
    assert A.holm([0.5, 0.6]) == pytest.approx([1.0, 1.0])
    out = A.holm([0.01, None, 0.02])
    assert out[1] is None
    assert out[0] == pytest.approx(0.02) and out[2] == pytest.approx(0.02)
    assert A.holm([]) == []


# 3. 29 vs 30 event days, and the families (A3) ---------------------------------------
def _with_type(n_event, type_):
    dates = _dates(n_event + 60)
    feats = [_row(d, 1.0 + 0.001 * i) for i, d in enumerate(dates)]
    events = [(d, type_) for d in dates[60:]]
    return feats, events


def test_type_with_29_days_is_reported_not_tested():
    feats, events = _with_type(29, "CPI")
    r = A.analyze_symbol(feats, events, reps=REPS, seed=SEED)
    e = r["H2"]["types"]["CPI"]
    assert e["n_event_days"] == 29 and e["tested"] is False
    assert e["status"].startswith("reported, not tested")
    assert e["rel_range"]["pass"] is None and e["rel_range"]["p_holm"] is None
    assert e["rel_range"]["ci95"] is not None            # still reported
    assert r["H2"]["family"] == [] and r["H3"]["family"] == []
    assert r["H3"]["types"]["CPI"]["tested"] is False


def test_type_with_30_days_is_tested():
    feats, events = _with_type(30, "CPI")
    r = A.analyze_symbol(feats, events, reps=REPS, seed=SEED)
    e = r["H2"]["types"]["CPI"]
    assert e["tested"] is True and e["status"] == "tested"
    assert e["rel_range"]["p_holm"] is not None and e["rel_range"]["pass"] in (True, False)
    assert r["H2"]["family"] == ["CPI"] and r["H3"]["family"] == ["CPI"]


def test_a_thin_event_session_does_not_count_toward_the_30():
    feats, events = _with_type(30, "CPI")
    feats[-1]["thin"] = True
    r = A.analyze_symbol(feats, events, reps=50, seed=SEED)
    assert r["H2"]["types"]["CPI"]["n_event_days"] == 29
    assert r["H2"]["types"]["CPI"]["tested"] is False


def test_families_fomc_only_in_h3_and_geopolitical_in_neither():
    dates = _dates(200)
    feats = [_row(d) for d in dates]
    events = ([(d, "FOMC") for d in dates[:35]] + [(d, "war_military") for d in dates[35:70]]
              + [(d, "financial_stress") for d in dates[70:105]] + [(d, "NFP") for d in dates[105:140]])
    r = A.analyze_symbol(feats, events, reps=REPS, seed=SEED)
    assert r["H2"]["family"] == ["NFP"]
    assert r["H3"]["family"] == ["FOMC", "NFP"]
    assert "war_military" not in r["H2"]["types"] and "war_military" not in r["H3"]["types"]
    g = r["geopolitical_descriptive"]["categories"]
    assert set(g) == {"war_military", "financial_stress"}
    for c in g.values():
        assert c["tested"] is False and "pass" not in c["rel_range"] and "p_holm" not in c["rel_range"]
    assert "market stress" in g["financial_stress"]["never_tested_reason"]
    assert g["war_military"]["n_event_days"] == 35          # >= 30 and still not tested


def test_h2_pass_is_holm_then_floor():
    # three CPI/NFP/PCE types, all with 35 days; one is +1.0, one +0.1 (significant but
    # below the floor), one identical
    dates = _dates(300)
    feats = [_row(d, 1.0 + 0.001 * (i % 50)) for i, d in enumerate(dates)]
    for d in dates[:35]:
        feats[dates.index(d)]["rel_range"] += 1.0
    for d in dates[35:70]:
        feats[dates.index(d)]["rel_range"] += 0.1
    events = ([(d, "CPI") for d in dates[:35]] + [(d, "NFP") for d in dates[35:70]]
              + [(d, "PCE") for d in dates[70:105]])
    r = A.analyze_symbol(feats, events, reps=REPS, seed=SEED)
    t = r["H2"]["types"]
    assert r["H2"]["family"] == ["CPI", "NFP", "PCE"]
    assert t["CPI"]["rel_range"]["pass"] is True
    assert t["NFP"]["rel_range"]["p_holm"] < 0.05 and t["NFP"]["rel_range"]["pass"] is False
    assert t["PCE"]["rel_range"]["pass"] is False
    # Holm: the adjusted p of the smallest raw p is 3x the raw p (capped)
    raw = sorted(t[k]["rel_range"]["p_raw"] for k in t)
    assert t["CPI"]["rel_range"]["p_holm"] == pytest.approx(min(1.0, 3 * raw[0]))


# 4. a date in both files -------------------------------------------------------------
def test_date_in_both_files_counts_for_both_and_is_not_nonevent():
    feats = [_row(d) for d in _dates(4)]
    d0, d1 = feats[0]["date"], feats[1]["date"]
    events = [(d0, "CPI"), (d0, "war_military"), (d1, "NFP")]
    types = A.classify_days(feats, events)
    assert types[d0] == {"CPI", "war_military"}
    assert types[d1] == {"NFP"}
    assert types[feats[2]["date"]] == set() and types[feats[3]["date"]] == set()
    nonevent = [d for d, s in types.items() if not s]
    assert d0 not in nonevent and len(nonevent) == 2
    r = A.analyze_symbol(feats, events, reps=50, seed=SEED, placebo_draws=5)
    assert r["H2"]["types"]["CPI"]["n_event_days"] == 1
    assert r["geopolitical_descriptive"]["categories"]["war_military"]["n_event_days"] == 1
    assert r["n_nonevent_days"] == 2 and r["n_event_days_any_type"] == 2
    assert r["n_scheduled_control_days"] == 2


# A2: the scheduled join is on `session`, not `date` ------------------------------------
def _write(path, header, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def test_scheduled_events_join_on_session_column(tmp_path):
    sched = tmp_path / "events_scheduled.csv"
    geo = tmp_path / "events_geopolitical.csv"
    # NFP released Friday 2020-04-10 (Good Friday, market closed) -> session Monday 2020-04-13
    _write(sched, ["date", "type", "session", "time_et", "source_url", "note"],
           [["2020-04-10", "NFP", "2020-04-13", "08:30", "https://x", "Good Friday"],
            ["2020-05-08", "NFP", "2020-05-08", "08:30", "https://x", ""]])
    _write(geo, ["date", "break_time_et", "category", "headline", "source_url", "surprise"],
           [["2020-03-09", "unknown", "financial_stress", "h", "https://x", "yes"]])
    pairs, meta = A.load_events(sched, geo)
    assert ("2020-04-13", "NFP") in pairs and ("2020-04-10", "NFP") not in pairs
    assert ("2020-05-08", "NFP") in pairs and ("2020-03-09", "financial_stress") in pairs
    assert meta["scheduled_rows_shifted_to_next_session"] == 1
    assert meta["scheduled_rows"] == 2 and meta["geopolitical_rows"] == 1 and meta["surprise_rows"] == 1
    assert meta["unknown_types"] == []


def test_scheduled_file_without_session_column_is_refused(tmp_path):
    sched = tmp_path / "events_scheduled.csv"
    geo = tmp_path / "events_geopolitical.csv"
    _write(sched, ["date", "type", "time_et", "source_url", "note"], [["2020-04-10", "NFP", "08:30", "u", ""]])
    _write(geo, ["date", "break_time_et", "category", "headline", "source_url", "surprise"], [])
    with pytest.raises(ValueError, match="session"):
        A.load_events(sched, geo)


# 5. thin and None exclusions ---------------------------------------------------------
def test_thin_and_none_rows_excluded_and_counted():
    dates = _dates(6)
    feats = [_row(dates[0], 1.0), _row(dates[1], 2.0, thin=True), _row(dates[2], None),
             _row(dates[3], 1.2), _row(dates[4], None, thin=True), _row(dates[5], 0.9, rel_gap=None)]
    usable, excl = A.usable_rows(feats)
    assert [r["date"] for r in usable] == [dates[0], dates[3], dates[5]]
    assert excl == {"n_rows": 6, "n_thin_excluded": 2, "n_rel_range_none_excluded": 1,
                    "n_usable": 3, "n_rel_gap_none_among_usable": 1, "thin_by_year": {"2020": 2}}
    r = A.analyze_symbol(feats, [], reps=50, seed=SEED)
    assert r["exclusions"]["n_usable"] == 3 and r["n_nonevent_days"] == 3
    assert r["H1"]["rel_range"]["median_b"] == pytest.approx(1.0)   # the thin 2.0 never reaches the median
    assert r["H1"]["rel_gap"]["n_b"] == 2                             # rel_gap None excluded there only


def test_event_on_a_thin_day_is_not_an_event_day_and_is_reported():
    dates = _dates(3)
    feats = [_row(dates[0], 1.0, thin=True), _row(dates[1], 1.0), _row(dates[2], 1.0)]
    r = A.analyze_symbol(feats, [(dates[0], "FOMC"), ("2019-07-04", "FOMC")], reps=50, seed=SEED)
    assert r["H1"]["n_event_days"] == 0
    assert r["event_coverage"]["FOMC"] == {"sessions_in_file": 2, "usable_event_sessions": 0,
                                           "not_usable_sessions": 2}


# impact bands (the §4 thresholds, boundaries included) ------------------------------
def test_impact_band_thresholds():
    assert A.impact_band(0.79, 0.99) == "low"
    assert A.impact_band(0.8, 0.5) == "medium"
    assert A.impact_band(1.5, 0.5) == "medium"
    assert A.impact_band(1.51, 0.5) == "high"
    assert A.impact_band(0.5, 1.0) == "medium"
    assert A.impact_band(0.5, 2.5) == "medium"
    assert A.impact_band(0.5, 2.51) == "high"
    assert A.impact_band(0.5, None) == "low"


def test_impact_column_disagreement_is_counted():
    feats = [dict(_row("2020-01-06", 2.0), impact="low"), dict(_row("2020-01-07", 1.0), impact="medium")]
    r = A.analyze_symbol(feats, [], reps=50, seed=SEED)
    assert r["exclusions"]["impact_column_disagreements"] == 1


def test_parse_bool_rejects_garbage_and_reads_strings():
    assert A.parse_bool("True") is True and A.parse_bool("false") is False
    assert A.parse_bool("") is None and A.parse_bool(None) is None
    with pytest.raises(ValueError):
        A.parse_bool("maybe")


# 6. H3 share difference --------------------------------------------------------------
def test_share_diff_arithmetic():
    s = A.share_diff_boot([True, True, False, True], [False, False, True, False, False],
                          reps=REPS, seed=SEED)
    assert s["share_a"] == pytest.approx(0.75) and s["share_b"] == pytest.approx(0.2)
    assert s["diff"] == pytest.approx(0.55)
    assert s["k_a"] == 3 and s["k_b"] == 1 and s["n_a"] == 4 and s["n_b"] == 5
    assert s["ci95"][0] <= 0.55 <= s["ci95"][1]


def test_h3_uses_affected_flag_from_impact():
    dates = _dates(8)
    # CPI days: two of three affected (rel_range > 1.5); control: one of five
    feats = [_row(dates[0], 2.0), _row(dates[1], 1.7), _row(dates[2], 1.0),
             _row(dates[3], 1.0), _row(dates[4], 1.6), _row(dates[5], 1.0),
             _row(dates[6], 1.0), _row(dates[7], 1.0)]
    events = [(dates[0], "CPI"), (dates[1], "CPI"), (dates[2], "CPI")]
    r = A.analyze_symbol(feats, events, reps=50, seed=SEED)
    e = r["H3"]["types"]["CPI"]
    assert e["share_event"] == pytest.approx(2 / 3) and e["share_control"] == pytest.approx(0.2)
    assert e["diff"] == pytest.approx(2 / 3 - 0.2)
    assert e["k_event"] == 2 and e["k_control"] == 1
    assert r["H3"]["control_affected_share"]["share"] == pytest.approx(0.2)


# 7. H5 contingency table (A5) --------------------------------------------------------
def test_contingency_counts_per_class():
    ev = [_row("2020-01-06", day_class="trend"), _row("2020-01-07", day_class="chop"),
          _row("2020-01-08", day_class="chop")]
    non = [_row("2020-01-09", day_class="ranged"), _row("2020-01-10", day_class="ranged"),
           _row("2020-01-13", day_class="trend"), _row("2020-01-14", day_class="chop"),
           _row("2020-01-15", day_class="weird")]
    t = A.contingency(ev, non)
    assert t["classes"] == ["trend", "ranged", "chop"]
    assert t["event_unaffected"] == [1, 0, 2]
    assert t["nonevent_unaffected"] == [1, 2, 1]
    assert t["unclassified_rows"] == 1
    assert t["dof"] == 2 and 0 <= t["p"] <= 1


def test_contingency_degenerate_gives_none_p():
    t = A.contingency([], [_row("2020-01-06", day_class="trend")])
    assert t["p"] is None and t["event_unaffected"] == [0, 0, 0]
    t2 = A.contingency([_row("2020-01-06", day_class="trend")], [_row("2020-01-07", day_class="trend")])
    assert t2["p"] is None          # a single non-empty column


def test_h5_compares_not_affected_event_with_not_affected_nonevent():
    dates = _dates(6)
    feats = [_row(dates[0], 2.0, day_class="trend"),        # affected event day: dropped
             _row(dates[1], 1.0, day_class="ranged"),       # event, not affected
             _row(dates[2], 1.0, day_class="chop"),         # event, not affected
             _row(dates[3], 1.0, day_class="trend"),
             _row(dates[4], 1.9, day_class="trend"),        # affected non-event day: dropped (A5)
             _row(dates[5], 1.0, day_class="chop")]
    events = [(d, "NFP") for d in dates[:3]]
    r = A.analyze_symbol(feats, events, reps=50, seed=SEED)
    assert r["H5"]["event_unaffected"] == [0, 1, 1]
    assert r["H5"]["nonevent_unaffected"] == [1, 0, 1]
    assert r["H5"]["n_event_days_affected"] == 1 and r["H5"]["n_nonevent_days_affected"] == 1


# 8. H6 lab rates --------------------------------------------------------------------
def test_lab_rates_from_tiny_lab_days():
    lab = [{"date": "2026-09-01", "options_green": True, "shares_green": False, "both_green": False,
            "options_atm_net": 10.0, "shares_net": -5.0, "ledger_outcome": "ok"},
           {"date": "2026-09-02", "options_green": True, "shares_green": True, "both_green": True,
            "options_atm_net": 20.0, "shares_net": 5.0, "ledger_outcome": "ok"},
           {"date": "2026-09-03", "options_green": False, "shares_green": None, "both_green": None,
            "options_atm_net": -3.0, "shares_net": None, "ledger_outcome": "no_shares"}]
    types = {"2026-09-02": {"CPI", "war_military"}, "2026-09-03": set()}
    h6 = A.lab_rates(lab, types)
    assert h6["n_lab_sessions"] == 3
    assert h6["event"]["n"] == 1 and h6["event"]["dates"] == ["2026-09-02"]
    assert h6["event"]["options_green"] == {"n": 1, "green": 1, "rate": 1.0}
    assert h6["event"]["both_green"]["rate"] == 1.0
    assert h6["nonevent"]["n"] == 2
    assert h6["nonevent"]["options_green"] == {"n": 2, "green": 1, "rate": 0.5}
    assert h6["nonevent"]["shares_green"] == {"n": 1, "green": 0, "rate": 0.0}   # None dropped
    assert h6["nonevent"]["both_green"]["n"] == 1
    assert h6["nonevent"]["shares_net_sum"] == pytest.approx(-5.0)
    assert h6["event_type_counts"] == {"CPI": 1, "war_military": 1}
    assert h6["nonevent"]["ledger_outcome_counts"] == {"no_shares": 1, "ok": 1}


# 9. replication flag (A4) ------------------------------------------------------------
def test_replication_flag_logic():
    prim = {"diff": 0.5, "ci95": [0.2, 0.8]}
    assert A.replication_flag(prim, {"diff": 0.3, "ci95": [0.1, 0.5]}) is True
    assert A.replication_flag(prim, {"diff": -0.3, "ci95": [-0.5, -0.1]}) is False   # opposite sign
    assert A.replication_flag(prim, {"diff": 0.3, "ci95": [-0.1, 0.5]}) is False    # CI spans 0
    assert A.replication_flag(prim, {"diff": 0.15, "ci95": [0.1, 0.2]}) is False    # below the 0.2 floor
    assert A.replication_flag(prim, {"diff": 0.15, "ci95": [0.1, 0.2]}, floor=None) is True
    assert A.replication_flag(prim, None) == "n/a"
    assert A.replication_flag(prim, {"diff": None, "ci95": None}) == "n/a"
    assert A.replication_flag({"diff": None, "ci95": None}, {"diff": 0.3, "ci95": [0.1, 0.5]}) == "n/a"


def test_build_report_replication_na_without_spy_and_true_with_spy():
    feats, events = _sample(40, 80, shift=1.0)
    rep = A.build_report({"QQQ": feats, "SPY": None}, events, reps=REPS, seed=SEED)
    assert rep["symbols"]["SPY"] is None
    assert rep["symbols"]["QQQ"]["H1"]["rel_range"]["replicated_on_SPY"] == "n/a"
    assert rep["symbols"]["QQQ"]["H3"]["types"]["FOMC"]["replicated_on_SPY"] == "n/a"
    assert rep["H6"]["status"] == "lab_days.csv not found"
    rep2 = A.build_report({"QQQ": feats, "SPY": feats}, events, reps=REPS, seed=SEED)
    assert rep2["symbols"]["QQQ"]["H1"]["rel_range"]["replicated_on_SPY"] is True
    # SPY with the opposite sign -> False
    spy = [dict(r, rel_range=(r["rel_range"] - 1.0 if r["rel_range"] > 1.5 else r["rel_range"] + 1.0))
           for r in feats]
    rep3 = A.build_report({"QQQ": feats, "SPY": spy}, events, reps=REPS, seed=SEED)
    assert rep3["symbols"]["SPY"]["H1"]["rel_range"]["diff"] < 0
    assert rep3["symbols"]["QQQ"]["H1"]["rel_range"]["replicated_on_SPY"] is False
    # SPY same sign and significant but under the floor -> False (A4)
    spy2 = [dict(r, rel_range=(r["rel_range"] - 0.9 if r["rel_range"] > 1.5 else r["rel_range"])) for r in feats]
    rep4 = A.build_report({"QQQ": feats, "SPY": spy2}, events, reps=REPS, seed=SEED)
    s = rep4["symbols"]["SPY"]["H1"]["rel_range"]
    assert s["diff"] == pytest.approx(0.1, abs=1e-9) and s["ci95"][0] > 0
    assert rep4["symbols"]["QQQ"]["H1"]["rel_range"]["replicated_on_SPY"] is False


# A1: placebo ------------------------------------------------------------------------
def test_placebo_flag_logic():
    assert A.placebo_flag(1.0, 0.5) == "selection"            # p95 == half the observed
    assert A.placebo_flag(1.0, 0.6) == "selection"
    assert A.placebo_flag(1.0, 0.49) == "placebo not reached (still descriptive)"
    assert A.placebo_flag(0.0, 0.1) == "no positive difference"
    assert A.placebo_flag(-0.3, -0.5) == "no positive difference"
    assert A.placebo_flag(None, 0.1) == "n/a" and A.placebo_flag(0.5, None) == "n/a"


def test_placebo_draws_are_matched_by_year_and_weekday():
    # pool: 2020 Mondays have rel_range 3.0, everything else 1.0; the two event days are
    # 2020 Mondays, so every placebo draw must come out at 3.0 - control median
    dates = _dates(60)
    pool = [_row(d, 3.0 if dt.date.fromisoformat(d).weekday() == 0 else 1.0) for d in dates]
    mondays = [r for r in pool if dt.date.fromisoformat(r["date"]).weekday() == 0]
    ev = mondays[:2]
    control_vals = [1.0] * 10
    out = A.placebo_diffs(ev, pool, control_vals, "rel_range", np.median, draws=50, seed=SEED)
    assert out["draws"] == 50 and out["unmatched_event_sessions"] == 0
    assert out["values"] == [pytest.approx(2.0)] * 50
    assert out["p95"] == pytest.approx(2.0) and out["median"] == pytest.approx(2.0)
    # an event whose (year, weekday) stratum has no other session is counted as unmatched
    lone = [_row("2021-01-04", 1.0)]
    out2 = A.placebo_diffs(lone, pool + lone, control_vals, "rel_range", np.median, draws=10, seed=SEED)
    assert out2["unmatched_event_sessions"] == 1 and out2["p95"] is None


def test_placebo_excludes_the_category_own_sessions_and_is_deterministic():
    dates = _dates(100)
    pool = [_row(d, 1.0) for d in dates]
    ev = pool[:10]
    for r in ev:
        r["rel_range"] = 9.0                  # only the event days are wide
    out = A.placebo_diffs(ev, pool, [1.0] * 20, "rel_range", np.median, draws=30, seed=SEED)
    assert out["p95"] == pytest.approx(0.0)   # never draws itself
    out_b = A.placebo_diffs(ev, pool, [1.0] * 20, "rel_range", np.median, draws=30, seed=SEED)
    assert out == out_b


def test_geopolitical_block_carries_placebo_and_flag():
    dates = _dates(120)
    feats = [_row(d, 1.0 + 0.001 * i) for i, d in enumerate(dates)]
    for d in dates[:20]:
        feats[dates.index(d)]["rel_range"] += 2.0
    events = [(d, "trade_tariff") for d in dates[:20]]
    r = A.analyze_symbol(feats, events, reps=50, seed=SEED, placebo_draws=40)
    g = r["geopolitical_descriptive"]["categories"]["trade_tariff"]
    assert g["rel_range"]["diff"] == pytest.approx(2.0, abs=0.1)
    assert g["rel_range"]["placebo_p95"] is not None
    assert g["rel_range"]["placebo_flag"] == "placebo not reached (still descriptive)"
    assert g["affected_share"]["placebo_flag"] in ("selection", "placebo not reached (still descriptive)",
                                                   "no positive difference")
    assert r["geopolitical_descriptive"]["placebo"]["draws"] == 40


# 10. determinism --------------------------------------------------------------------
def test_bootstrap_is_deterministic_under_the_seed():
    rng = np.random.default_rng(1)
    a, b = list(rng.normal(1.3, 0.4, 45)), list(rng.normal(1.0, 0.4, 300))
    r1 = A.median_diff_boot(a, b, reps=REPS, seed=SEED)
    r2 = A.median_diff_boot(a, b, reps=REPS, seed=SEED)
    assert r1 == r2
    s1 = A.share_diff_boot([v > 1 for v in a], [v > 1 for v in b], reps=REPS, seed=SEED)
    s2 = A.share_diff_boot([v > 1 for v in a], [v > 1 for v in b], reps=REPS, seed=SEED)
    assert s1 == s2
    assert A.share_boot([v > 1 for v in a], reps=REPS, seed=SEED) == \
        A.share_boot([v > 1 for v in a], reps=REPS, seed=SEED)
    # and a different seed is allowed to differ (so the equality above is not trivial)
    r3 = A.median_diff_boot(a, b, reps=REPS, seed=SEED + 1)
    assert r3["ci95"] != r1["ci95"]


def test_whole_report_is_identical_on_two_runs():
    feats, events = _sample(40, 80, shift=0.3)
    events += [(feats[i]["date"], "war_military") for i in range(0, 20)]
    r1 = A.analyze_symbol(feats, events, reps=REPS, seed=SEED, placebo_draws=50)
    r2 = A.analyze_symbol(feats, events, reps=REPS, seed=SEED, placebo_draws=50)
    assert json.dumps(A._round(r1), sort_keys=True) == json.dumps(A._round(r2), sort_keys=True)


def test_p_value_floor_and_two_sidedness():
    a = [10.0 + i * 0.01 for i in range(40)]
    b = [1.0 + i * 0.01 for i in range(80)]
    r = A.median_diff_boot(a, b, reps=REPS, seed=SEED)
    assert r["p"] == pytest.approx(1 / REPS)          # floored, never 0
    r0 = A.median_diff_boot(b, b, reps=REPS, seed=SEED)
    assert r0["p"] == 1.0                              # identical groups: capped at 1
    r_small = A.median_diff_boot([1.0], [1.0, 2.0], reps=REPS, seed=SEED)
    assert r_small["ci95"] is None and r_small["p"] is None and r_small["diff"] == pytest.approx(-0.5)


# H4 --------------------------------------------------------------------------------
def test_h4_shares_overall_by_year_event_vs_nonevent():
    feats = [_row("2020-01-06", ranged_a=True, ranged_b=False), _row("2020-01-07", ranged_a=False, ranged_b=True),
             _row("2021-01-05", ranged_a=True, ranged_b=True), _row("2021-01-06", ranged_a=False, ranged_b=False)]
    events = [("2021-01-05", "CPI")]
    r = A.analyze_symbol(feats, events, reps=50, seed=SEED)
    h4 = r["H4"]
    assert h4["overall"]["ranged_a"]["share"] == pytest.approx(0.5)
    assert h4["overall"]["ranged_b"]["share"] == pytest.approx(0.5)
    assert sorted(h4["by_year"]) == ["2020", "2021"]
    assert h4["by_year"]["2020"]["ranged_a"]["n"] == 2 and h4["by_year"]["2020"]["ranged_a"]["k"] == 1
    assert h4["by_year"]["2020"]["ranged_a"]["ci95"] is not None
    assert h4["event_any"]["n"] == 1 and h4["event_any"]["ranged_a"]["share"] == 1.0
    assert h4["nonevent"]["ranged_b"]["share"] == pytest.approx(1 / 3)
    assert h4["event_minus_nonevent"]["ranged_a"]["diff"] == pytest.approx(1.0 - 1 / 3)


# A7 --------------------------------------------------------------------------------
def test_early_knowable_shares_per_class():
    feats = [_row("2020-01-06", day_class="trend", f60_hi=70.0, f60_lo=-10.0, med20=100.0),   # 70 > 60: yes
             _row("2020-01-07", day_class="trend", f60_hi=10.0, f60_lo=-61.0, med20=100.0),   # 61 > 60: yes
             _row("2020-01-08", day_class="trend", f60_hi=60.0, f60_lo=-60.0, med20=100.0),   # 60 > 60: no (strict)
             _row("2020-01-09", day_class="chop", f60_hi=90.0, f60_lo=-5.0, med20=100.0),     # yes
             _row("2020-01-10", day_class="chop", f60_hi=10.0, f60_lo=-5.0, med20=100.0),     # no
             _row("2020-01-13", day_class="ranged", f60_hi=10.0, f60_lo=-5.0, med20=100.0),   # no
             _row("2020-01-14", day_class="ranged", f60_hi=None, f60_lo=-5.0, med20=100.0)]   # missing input
    out = A.early_knowable(feats, reps=50, seed=SEED)
    assert out["rows_missing_inputs"] == 1
    assert out["by_class"]["trend"]["k"] == 2 and out["by_class"]["trend"]["n"] == 3
    assert out["by_class"]["chop"]["share"] == pytest.approx(0.5)
    assert out["by_class"]["ranged"] == {"n": 1, "k": 0, "share": 0.0, "ci95": None}
    assert A.early_move_bp({"first60_hi_bp": 10.0, "first60_lo_bp": -30.0}) == 30.0


# end to end on CSV files ---------------------------------------------------------------
def test_cli_end_to_end(tmp_path, capsys):
    dates = _dates(150)
    hdr = ["date", "pre_high", "pre_low", "open", "high", "low", "close", "prev_close", "prev_session_date",
           "gap_bp", "range_bp", "co_bp", "first60_hi_bp", "first60_lo_bp", "med20_range_bp",
           "med20_absgap_bp", "rel_range", "rel_gap", "n_line_breaks", "ranged_a", "ranged_b",
           "day_class", "gap_reversal", "thin", "rel_range_band", "rel_gap_band", "impact"]

    def frow(i, d, rr, thin=False):
        rr_s = "" if rr is None else f"{rr:.4f}"
        band = "" if rr is None else ("high" if rr > 1.5 else ("medium" if rr >= 0.8 else "low"))
        return [d, 100, 99, 100, 101, 99, 100.5, 100, "2020-01-03", 5.0, 150.0, 50.0, 30.0, -20.0, 140.0, 6.0,
                rr_s, "0.8", 2, "False", "True" if (rr or 1) < 0.6 else "False",
                "trend" if i % 3 == 0 else "chop", "False", "True" if thin else "False", band, "low", band]

    for sym, bump in (("QQQ", 1.0), ("SPY", 0.9)):
        rows = []
        for i, d in enumerate(dates):
            if i < 20:
                rows.append(frow(i, d, None))                   # med20 warm-up: rel_range None
            elif i == 25:
                rows.append(frow(i, d, 3.0, thin=True))         # a half day
            elif i >= 110:
                rows.append(frow(i, d, 1.0 + bump + (i % 7) * 0.01))
            else:
                rows.append(frow(i, d, 1.0 + (i % 7) * 0.01))
        _write(tmp_path / f"day_features_{sym}.csv", hdr, rows)
    sched = [[d, "FOMC", d, "14:00", "https://x", ""] for d in dates[110:]]
    sched += [[dates[30], "CPI", dates[30], "08:30", "https://x", ""],
              ["2019-12-25", "CPI", "2019-12-26", "08:30", "https://x", "holiday, session shifted"]]
    _write(tmp_path / "events_scheduled.csv", ["date", "type", "session", "time_et", "source_url", "note"], sched)
    _write(tmp_path / "events_geopolitical.csv",
           ["date", "break_time_et", "category", "headline", "source_url", "surprise"],
           [[dates[30], "09:00", "war_military", "h", "https://x", "yes"],
            [dates[31], "unknown", "trade_tariff", "h", "https://x", "no"]])
    _write(tmp_path / "lab_days.csv", ["date", "options_atm_net", "shares_net", "options_green",
                                       "shares_green", "both_green", "ledger_outcome"],
           [[dates[30], "12.5", "-3", "True", "False", "False", "ok"],
            [dates[40], "-2", "", "False", "", "", "no_shares"]])
    out = tmp_path / "analysis.json"
    rc = A.main(["--features-dir", str(tmp_path), "--symbols", "QQQ", "SPY", "--out", str(out),
                 "--reps", "300", "--placebo-draws", "20"])
    assert rc == 0
    rep = json.loads(out.read_text())
    q = rep["symbols"]["QQQ"]
    assert q["exclusions"] == {"n_rows": 150, "n_thin_excluded": 1, "n_rel_range_none_excluded": 20,
                               "n_usable": 129, "n_rel_gap_none_among_usable": 0, "thin_by_year": {"2020": 1},
                               "impact_column_disagreements": 0}
    assert q["H1"]["n_event_days"] == 40 and q["H1"]["rel_range"]["pass"] is True
    assert q["H1"]["rel_range"]["replicated_on_SPY"] is True
    assert q["H1"]["n_control_days"] == 129 - 40 - 1               # CPI day out, geo days in
    assert q["n_nonevent_days"] == 129 - 40 - 2
    assert q["H2"]["types"]["CPI"]["status"].startswith("reported, not tested")
    assert q["event_coverage"]["CPI"] == {"sessions_in_file": 2, "usable_event_sessions": 1, "not_usable_sessions": 1}
    assert set(q["geopolitical_descriptive"]["categories"]) == {"war_military", "trade_tariff"}
    assert "trend" in q["A7_early_knowable"]["by_class"]
    assert rep["H6"]["event"]["n"] == 1 and rep["H6"]["nonevent"]["n"] == 1
    assert rep["H6"]["nonevent"]["shares_green"]["n"] == 0
    assert rep["events"]["scheduled_rows"] == 42 and rep["events"]["geopolitical_rows"] == 2
    assert rep["events"]["scheduled_rows_shifted_to_next_session"] == 1
    assert rep["bootstrap"]["reps"] == 300 and rep["bootstrap"]["seed"] == SEED
    assert rep["bootstrap"]["placebo_draws"] == 20
    text = capsys.readouterr().out
    assert text.startswith("H1 (QQQ, FOMC vs no-scheduled-event sessions, median rel_range): PASS.")
    assert "written:" in text


def test_cli_missing_files_exit_2(tmp_path, capsys):
    assert A.main(["--features-dir", str(tmp_path)]) == 2
    _write(tmp_path / "events_scheduled.csv", ["date", "type", "session", "time_et", "source_url", "note"], [])
    _write(tmp_path / "events_geopolitical.csv", ["date", "break_time_et", "category", "headline", "source_url", "surprise"], [])
    assert A.main(["--features-dir", str(tmp_path)]) == 2
    assert "missing" in capsys.readouterr().err
