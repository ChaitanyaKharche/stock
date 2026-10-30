"""task_id: R-2026-10-30-05 | parent: R-2026-10-30-01 | role: analysis-script-agent | repo_sha: a367245 | status: done | verified_by: null

Analysis script for research/news_impact/preregistration.md, sections 4, 6, 7 and the
amendment in section 9 (A1-A5, A7).

    python -m trade_analysis.news_impact.analyze --features-dir research/news_impact --symbols QQQ SPY

Reads, from --features-dir:
    day_features_{SYMBOL}.csv   one row per session (schema: preregistration §4 + A6)
    events_scheduled.csv        date,type,session,time_et,source_url,note   (joined on `session`, A2)
    events_geopolitical.csv     date,break_time_et,category,headline,source_url,surprise
                                (`date` is already the assigned session, A2)
    lab_days.csv                date,options_atm_net,shares_net,options_green,shares_green,
                                both_green,ledger_outcome   (optional; H6 only)

Writes analysis.json (every number) and prints a plain-text summary.

Every statistic is a pure function so it can be tested without files:
    classify_days, usable_rows, impact_band, median_diff_boot, share_boot, share_diff_boot,
    holm, placebo_diffs, placebo_flag, contingency, early_knowable, lab_rates,
    replication_flag, analyze_symbol, build_report.

Groups (A1):
    scheduled control  = usable sessions with no SCHEDULED event (geopolitical days stay in)
                         -> control for H1 and for CPI/NFP/GDP_ADV/PCE in H2/H3
    non-event          = usable sessions with no event of any type
                         -> control for the descriptive geopolitical lines, H4, H5 (A5), H6

Readings of the pre-registration where it leaves room (each is the conservative one):
  * Bootstrap p = 2 * min(frac <= 0, frac >= 0), floored at 1/reps and capped at 1 (A3;
    same as boot_mean in live_lab/breakout_sweep.py).
  * H2 pass = Holm-adjusted p < 0.05, then difference >= 0.2 (A3, literal). H3 pass =
    Holm-adjusted p < 0.05 AND the 95% CI excludes 0 (§6 names the CI, A3 names Holm; both
    are required).
  * The ">= 30 distinct non-thin event sessions" family rule is counted on USABLE sessions
    (non-thin AND rel_range present), which is the smaller count.
  * Replication (A4): same sign, SPY CI excludes 0, and the 0.2 floor where a floor is
    defined (median rel_range differences: H1, H2). H3 share differences have no floor in
    the pre-registration, so their flag is sign + CI only; the rule used is written into
    each entry.
  * Placebo (A1): 200 draws; for each event session of the category one random usable
    session with the same calendar year and weekday is drawn (with replacement, the
    category's own sessions excluded); statistic = median(draw) - median(non-event), same as
    the observed line. Flag "selection" when the placebo 95th percentile >= half the observed
    difference and the observed difference is > 0; an observed difference <= 0 is labelled
    "no positive difference". No geopolitical line is ever labelled "effect".
  * An event session counts only if it is a usable session of that symbol. Event rows whose
    session is not usable are counted and reported per type, never silently dropped.
  * Impact is derived from the §4 thresholds (1.5 and 2.5 exactly are "medium"); a missing
    rel_gap contributes nothing to the band. Disagreements with the file's own `impact`
    column are counted and reported.
  * H5 (A5): pooled not-affected event days (any type) vs not-affected non-event days.
  * A7 "exceeded" is strict: max(first60_hi_bp, -first60_lo_bp) > 0.6 * med20_range_bp.
  * H6 classifies lab sessions by the event files alone (any type = event).
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPS = 3000
SEED = 20260828
PLACEBO_DRAWS = 200
MIN_EVENT_DAYS = 30
MIN_DIFF = 0.2
ALPHA = 0.05
EARLY_FRACTION = 0.6
PRIMARY_TYPE = "FOMC"
SCHEDULED_TYPES = ["FOMC", "CPI", "NFP", "GDP_ADV", "PCE"]
H2_FAMILY_TYPES = ["CPI", "NFP", "GDP_ADV", "PCE"]
H3_FAMILY_TYPES = ["FOMC", "CPI", "NFP", "GDP_ADV", "PCE"]
GEO_CATEGORIES = ["war_military", "trade_tariff", "financial_stress", "us_fiscal",
                  "pandemic_health", "political_shock", "energy_shock", "cyber_infrastructure"]
NEVER_TESTED = {"financial_stress"}
GEO_LABEL = "descriptive: selected by salience; may be circular (A1); never tested"
REPLICATION_LABEL = ("robustness check on a second symbol over the same days, "
                     "not an independent sample (A4)")
DAY_CLASSES = ["trend", "ranged", "chop"]
TRUE_STRINGS = {"true", "1", "yes", "t", "y"}
FALSE_STRINGS = {"false", "0", "no", "f", "n"}


# ----------------------------------------------------------------------------- parsing
def parse_bool(s):
    """'True'/'False' strings from the feature CSV. Empty -> None. Anything else raises:
    a boolean column that holds something unexpected is a bug, not a False."""
    if s is None:
        return None
    if isinstance(s, bool):
        return s
    v = str(s).strip().lower()
    if v in ("", "none", "nan"):
        return None
    if v in TRUE_STRINGS:
        return True
    if v in FALSE_STRINGS:
        return False
    raise ValueError(f"not a boolean: {s!r}")


def parse_float(s):
    if s is None:
        return None
    if isinstance(s, (int, float)):
        return None if (isinstance(s, float) and np.isnan(s)) else float(s)
    v = str(s).strip()
    if v == "" or v.lower() in ("none", "nan"):
        return None
    return float(v)


def norm_date(s) -> str:
    """ISO date string, validated."""
    if isinstance(s, dt.date):
        return s.isoformat()
    return dt.date.fromisoformat(str(s).strip()).isoformat()


def _read_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


FLOAT_COLS = ("rel_range", "rel_gap", "range_bp", "gap_bp", "co_bp",
              "first60_hi_bp", "first60_lo_bp", "med20_range_bp", "med20_absgap_bp")
BOOL_COLS = ("thin", "ranged_a", "ranged_b", "gap_reversal")


def load_features(path: Path) -> list[dict]:
    """Rows as dicts with typed fields. Only the columns the analysis needs are typed;
    the rest (prev_session_date, bands, ...) ride along as strings."""
    out = []
    for r in _read_csv(path):
        row = dict(r)
        row["date"] = norm_date(r["date"])
        for k in FLOAT_COLS:
            if k in r:
                row[k] = parse_float(r[k])
        for k in BOOL_COLS:
            if k in r:
                row[k] = parse_bool(r[k])
        for k in ("day_class", "impact"):
            if k in r:
                row[k] = (r[k] or "").strip() or None
        out.append(row)
    return out


def load_events(scheduled_path: Path, geo_path: Path) -> tuple[list[tuple[str, str]], dict]:
    """-> ([(session, type), ...], meta). Scheduled rows join on `session` (A2), never on
    `date`; a scheduled file without a `session` column is an error. Geopolitical rows join
    on `date`, which A2 defines as the assigned session."""
    pairs, meta = [], {"scheduled_rows": 0, "geopolitical_rows": 0, "unknown_types": [],
                       "surprise_rows": 0, "scheduled_rows_shifted_to_next_session": 0}
    sched = _read_csv(scheduled_path)
    if sched and "session" not in sched[0]:
        raise ValueError(f"{scheduled_path} has no `session` column (A2); refusing to join on `date`")
    for r in sched:
        s = norm_date(r["session"])
        pairs.append((s, r["type"].strip()))
        meta["scheduled_rows"] += 1
        if norm_date(r["date"]) != s:
            meta["scheduled_rows_shifted_to_next_session"] += 1
    for r in _read_csv(geo_path):
        pairs.append((norm_date(r["date"]), r["category"].strip()))
        meta["geopolitical_rows"] += 1
        if (r.get("surprise") or "").strip().lower() == "yes":
            meta["surprise_rows"] += 1
    known = set(SCHEDULED_TYPES) | set(GEO_CATEGORIES)
    meta["unknown_types"] = sorted({t for _, t in pairs if t not in known})
    return pairs, meta


def load_lab_days(path: Path) -> list[dict]:
    out = []
    for r in _read_csv(path):
        row = dict(r)
        row["date"] = norm_date(r["date"])
        for k in ("options_green", "shares_green", "both_green"):
            row[k] = parse_bool(r.get(k))
        for k in ("options_atm_net", "shares_net"):
            row[k] = parse_float(r.get(k))
        out.append(row)
    return out


# ------------------------------------------------------------------- day classification
def classify_days(features: list[dict], events) -> dict[str, set[str]]:
    """date -> set of event types for every feature row's date. Empty set = non-event day
    (no row in EITHER events file). A date in both files is in both types' sets.
    `events` is an iterable of (session_date, type)."""
    by_date = defaultdict(set)
    for d, t in events:
        by_date[norm_date(d)].add(t)
    return {r["date"]: set(by_date.get(r["date"], ())) for r in features}


def is_scheduled(types: set) -> bool:
    return any(t in SCHEDULED_TYPES for t in types)


def usable_rows(features: list[dict]) -> tuple[list[dict], dict]:
    """Drop thin rows, then rows with rel_range None. Both counts are reported."""
    n = len(features)
    not_thin = [r for r in features if not r.get("thin")]
    n_thin = n - len(not_thin)
    usable = [r for r in not_thin if r.get("rel_range") is not None]
    n_none = len(not_thin) - len(usable)
    thin_by_year = Counter(r["date"][:4] for r in features if r.get("thin"))
    excl = {"n_rows": n, "n_thin_excluded": n_thin, "n_rel_range_none_excluded": n_none,
            "n_usable": len(usable),
            "n_rel_gap_none_among_usable": sum(1 for r in usable if r.get("rel_gap") is None),
            "thin_by_year": dict(sorted(thin_by_year.items()))}
    return usable, excl


def impact_band(rel_range, rel_gap) -> str:
    """§4 bands. rel_range: <0.8 low, 0.8-1.5 medium, >1.5 high.
    rel_gap: <1.0 low, 1.0-2.5 medium, >2.5 high. impact = the higher band."""
    order = {"low": 0, "medium": 1, "high": 2}
    br = "low" if rel_range < 0.8 else ("medium" if rel_range <= 1.5 else "high")
    if rel_gap is None:
        bg = "low"
    else:
        bg = "low" if rel_gap < 1.0 else ("medium" if rel_gap <= 2.5 else "high")
    return br if order[br] >= order[bg] else bg


def attach_impact(usable: list[dict]) -> dict:
    """Sets r['impact_derived'] and r['affected'] on every usable row. Returns counts of
    rows where a file-supplied `impact` column disagrees with the §4 derivation."""
    disagree = 0
    for r in usable:
        r["impact_derived"] = impact_band(r["rel_range"], r.get("rel_gap"))
        r["affected"] = r["impact_derived"] == "high"
        if r.get("impact") and r["impact"] != r["impact_derived"]:
            disagree += 1
    return disagree


# ---------------------------------------------------------------------------- bootstrap
def _two_sided_p(diffs: np.ndarray, reps: int) -> float:
    neg = float(np.mean(diffs <= 0))
    pos = float(np.mean(diffs >= 0))
    return float(min(1.0, max(2 * min(neg, pos), 1 / reps)))


def _percentile_ci(sorted_vals: np.ndarray) -> list[float]:
    n = len(sorted_vals)
    return [float(sorted_vals[int(.025 * n)]), float(sorted_vals[int(.975 * n)])]


def _boot_two_group(a, b, stat, reps, seed, chunk=500):
    """Resample each group independently (sessions are the unit), compute stat(a*) -
    stat(b*) per rep. Fresh generator per call, so results never depend on call order."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    rng = np.random.default_rng(seed)
    diffs = np.empty(reps)
    done = 0
    while done < reps:
        k = min(chunk, reps - done)
        ia = rng.integers(0, len(a), size=(k, len(a)))
        ib = rng.integers(0, len(b), size=(k, len(b)))
        diffs[done:done + k] = stat(a[ia], axis=1) - stat(b[ib], axis=1)
        done += k
    diffs.sort()
    return diffs


def median_diff_boot(a, b, reps: int = REPS, seed: int = SEED) -> dict:
    """Difference of medians, a minus b, 95% percentile bootstrap CI, two-sided bootstrap p.
    CI/p are None when either group has fewer than 2 rows."""
    a = [float(x) for x in a]
    b = [float(x) for x in b]
    out = {"n_a": len(a), "n_b": len(b),
           "median_a": float(np.median(a)) if a else None,
           "median_b": float(np.median(b)) if b else None,
           "diff": None, "ci95": None, "p": None}
    if a and b:
        out["diff"] = out["median_a"] - out["median_b"]
    if len(a) >= 2 and len(b) >= 2:
        diffs = _boot_two_group(a, b, np.median, reps, seed)
        out["ci95"] = _percentile_ci(diffs)
        out["p"] = _two_sided_p(diffs, reps)
    return out


def share_diff_boot(a, b, reps: int = REPS, seed: int = SEED) -> dict:
    """Difference of shares (fraction True), a minus b, same bootstrap as median_diff_boot."""
    a = [1.0 if x else 0.0 for x in a]
    b = [1.0 if x else 0.0 for x in b]
    out = {"n_a": len(a), "n_b": len(b), "k_a": int(sum(a)), "k_b": int(sum(b)),
           "share_a": float(np.mean(a)) if a else None,
           "share_b": float(np.mean(b)) if b else None,
           "diff": None, "ci95": None, "p": None}
    if a and b:
        out["diff"] = out["share_a"] - out["share_b"]
    if len(a) >= 2 and len(b) >= 2:
        diffs = _boot_two_group(a, b, np.mean, reps, seed)
        out["ci95"] = _percentile_ci(diffs)
        out["p"] = _two_sided_p(diffs, reps)
    return out


def share_boot(x, reps: int = REPS, seed: int = SEED) -> dict:
    """One-group share with a bootstrap CI (resample sessions)."""
    x = np.asarray([1.0 if v else 0.0 for v in x])
    out = {"n": int(len(x)), "k": int(x.sum()), "share": float(x.mean()) if len(x) else None,
           "ci95": None}
    if len(x) >= 2:
        rng = np.random.default_rng(seed)
        ms = np.empty(reps)
        done, chunk = 0, 500
        while done < reps:
            k = min(chunk, reps - done)
            idx = rng.integers(0, len(x), size=(k, len(x)))
            ms[done:done + k] = x[idx].mean(axis=1)
            done += k
        ms.sort()
        out["ci95"] = _percentile_ci(ms)
    return out


def holm(pvalues) -> list:
    """Holm step-down adjusted p-values, returned in the input order. None stays None and
    is not counted in the family."""
    idx = [i for i, p in enumerate(pvalues) if p is not None]
    m = len(idx)
    adj = [None] * len(pvalues)
    if m == 0:
        return adj
    order = sorted(idx, key=lambda i: pvalues[i])
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * pvalues[i])
        adj[i] = float(min(1.0, running))
    return adj


def ci_excludes_zero(ci) -> bool:
    return ci is not None and (ci[0] > 0 or ci[1] < 0)


def h1_pass(stat: dict) -> bool:
    """§6 H1: CI of the median difference excludes 0 and the difference is >= 0.2."""
    return bool(stat.get("diff") is not None and ci_excludes_zero(stat.get("ci95"))
                and stat["diff"] >= MIN_DIFF)


def replication_flag(primary: dict, repl, floor=MIN_DIFF):
    """A4. True when the replication symbol shows the same sign of difference, its CI
    excludes 0 and (when a floor is defined) |diff| >= floor. False otherwise. 'n/a' when
    there is no replication result."""
    if repl is None or repl.get("diff") is None or repl.get("ci95") is None:
        return "n/a"
    if primary.get("diff") is None:
        return "n/a"
    same_sign = primary["diff"] != 0 and np.sign(primary["diff"]) == np.sign(repl["diff"])
    meets_floor = floor is None or abs(repl["diff"]) >= floor
    return bool(same_sign and ci_excludes_zero(repl["ci95"]) and meets_floor)


# ------------------------------------------------------------------------------ placebo
def _stratum(date: str) -> tuple[str, int]:
    return date[:4], dt.date.fromisoformat(date).weekday()


def placebo_diffs(event_rows: list[dict], pool: list[dict], control_vals, key: str = "rel_range",
                  stat=np.median, draws: int = PLACEBO_DRAWS, seed: int = SEED) -> dict:
    """A1 placebo. For each of `draws`, replace every event session by a random session of
    `pool` with the same calendar year and weekday (with replacement; event sessions are
    removed from the pool first), and compute stat(drawn[key]) - stat(control_vals).
    Returns the 95th percentile and the full sorted list. Event sessions whose stratum has
    no candidate are dropped from the draw and counted."""
    ev_dates = {r["date"] for r in event_rows}
    strata = defaultdict(list)
    for r in pool:
        if r["date"] not in ev_dates and r.get(key) is not None:
            strata[_stratum(r["date"])].append(float(r[key]))
    control = float(stat(np.asarray(control_vals, dtype=float))) if len(control_vals) else None
    rng = np.random.default_rng(seed)
    out = {"draws": draws, "n_event": len(event_rows), "unmatched_event_sessions": 0,
           "p95": None, "median": None, "values": []}
    cands = []
    for r in event_rows:
        c = strata.get(_stratum(r["date"]))
        if c:
            cands.append(c)
        else:
            out["unmatched_event_sessions"] += 1
    if not cands or control is None:
        return out
    vals = []
    for _ in range(draws):
        drawn = [c[rng.integers(0, len(c))] for c in cands]
        vals.append(float(stat(np.asarray(drawn))) - control)
    vals.sort()
    out["values"] = vals
    out["p95"] = float(np.percentile(vals, 95))
    out["median"] = float(np.median(vals))
    return out


def placebo_flag(observed, p95) -> str:
    """'selection' when the placebo 95th percentile is at least half the observed positive
    difference; otherwise 'placebo not reached'. Never 'effect'."""
    if observed is None or p95 is None:
        return "n/a"
    if observed <= 0:
        return "no positive difference"
    return "selection" if p95 >= 0.5 * observed else "placebo not reached (still descriptive)"


# ------------------------------------------------------------------- per-hypothesis parts
def _vals(rows, key):
    return [r[key] for r in rows if r.get(key) is not None]


def contingency(event_unaffected: list[dict], nonevent_unaffected: list[dict]) -> dict:
    """H5 table (A5): counts of day_class per group and chi-square p (scipy). Columns whose
    total is 0 are dropped before the test; p is None if the table is degenerate."""
    ce = Counter(r.get("day_class") for r in event_unaffected)
    cn = Counter(r.get("day_class") for r in nonevent_unaffected)
    table = {"classes": list(DAY_CLASSES),
             "event_unaffected": [int(ce.get(c, 0)) for c in DAY_CLASSES],
             "nonevent_unaffected": [int(cn.get(c, 0)) for c in DAY_CLASSES],
             "unclassified_rows": int(sum(v for k, v in (ce + cn).items() if k not in DAY_CLASSES)),
             "chi2": None, "p": None, "dof": None}
    ne, nn = sum(table["event_unaffected"]), sum(table["nonevent_unaffected"])
    table["share_event_unaffected"] = [v / ne for v in table["event_unaffected"]] if ne else None
    table["share_nonevent_unaffected"] = [v / nn for v in table["nonevent_unaffected"]] if nn else None
    cols = [j for j in range(len(DAY_CLASSES))
            if table["event_unaffected"][j] + table["nonevent_unaffected"][j] > 0]
    if ne > 0 and nn > 0 and len(cols) >= 2:
        from scipy.stats import chi2_contingency
        obs = np.array([[table["event_unaffected"][j] for j in cols],
                        [table["nonevent_unaffected"][j] for j in cols]])
        chi2, p, dof, _ = chi2_contingency(obs)
        table.update(chi2=float(chi2), p=float(p), dof=int(dof))
    return table


def early_move_bp(r: dict):
    """max(first60_hi_bp, -first60_lo_bp): how far price got from the open by 10:30."""
    hi, lo = r.get("first60_hi_bp"), r.get("first60_lo_bp")
    if hi is None or lo is None:
        return None
    return max(float(hi), -float(lo))


def early_knowable(usable: list[dict], reps: int = REPS, seed: int = SEED) -> dict:
    """A7: per day_class, share of days whose first-hour move already exceeded
    0.6 * med20_range_bp. Descriptive, no test."""
    out = {"rule": f"max(first60_hi_bp, -first60_lo_bp) > {EARLY_FRACTION} * med20_range_bp by 10:30 ET",
           "rows_missing_inputs": 0, "by_class": {}}
    flags = defaultdict(list)
    for r in usable:
        mv, m20 = early_move_bp(r), r.get("med20_range_bp")
        if mv is None or m20 is None or r.get("day_class") not in DAY_CLASSES:
            out["rows_missing_inputs"] += 1
            continue
        flags[r["day_class"]].append(mv > EARLY_FRACTION * m20)
    for c in DAY_CLASSES:
        out["by_class"][c] = share_boot(flags.get(c, []), reps, seed)
    return out


def lab_rates(lab_rows: list[dict], event_types_by_date: dict) -> dict:
    """H6: green-day counts and rates per arm on event vs non-event lab sessions (any event
    type = event). Counts only; no test (A8)."""
    groups = {"event": [], "nonevent": []}
    type_counts = Counter()
    for r in lab_rows:
        types = event_types_by_date.get(r["date"], set())
        groups["event" if types else "nonevent"].append(r)
        for t in types:
            type_counts[t] += 1
    out = {"n_lab_sessions": len(lab_rows), "event_type_counts": dict(sorted(type_counts.items())),
           "note": "counts only; no day-type claim about the lab is made from this study (A8)"}
    for g, rows in groups.items():
        rep = {"n": len(rows), "dates": sorted(r["date"] for r in rows)}
        for k in ("options_green", "shares_green", "both_green"):
            known = [r[k] for r in rows if r.get(k) is not None]
            rep[k] = {"n": len(known), "green": int(sum(1 for v in known if v)),
                      "rate": (sum(1 for v in known if v) / len(known)) if known else None}
        for k in ("options_atm_net", "shares_net"):
            vals = [r[k] for r in rows if r.get(k) is not None]
            rep[k + "_sum"] = float(sum(vals)) if vals else None
        rep["ledger_outcome_counts"] = dict(sorted(Counter(
            (r.get("ledger_outcome") or "").strip() or "none" for r in rows).items()))
        out[g] = rep
    return out


def _type_block(event_rows, control_rows, reps, seed) -> dict:
    """Everything H1/H2/H3 need for one event type against its control."""
    rr = median_diff_boot(_vals(event_rows, "rel_range"), _vals(control_rows, "rel_range"), reps, seed)
    rg = median_diff_boot(_vals(event_rows, "rel_gap"), _vals(control_rows, "rel_gap"), reps, seed)
    sh = share_diff_boot([r["affected"] for r in event_rows], [r["affected"] for r in control_rows],
                         reps, seed)
    return {"n_event_days": len(event_rows), "n_control_days": len(control_rows),
            "rel_range": rr, "rel_gap": rg, "affected_share": sh}


def _not_tested(n: int) -> str:
    return f"reported, not tested: {n} usable event sessions < {MIN_EVENT_DAYS}"


def analyze_symbol(features: list[dict], events, reps: int = REPS, seed: int = SEED,
                   placebo_draws: int = PLACEBO_DRAWS) -> dict:
    """All of §6 (as amended by §9) for one symbol."""
    usable, excl = usable_rows(features)
    excl["impact_column_disagreements"] = attach_impact(usable)
    day_types = classify_days(usable, events)
    nonevent = [r for r in usable if not day_types[r["date"]]]
    sched_control = [r for r in usable if not is_scheduled(day_types[r["date"]])]
    all_event = [r for r in usable if day_types[r["date"]]]
    by_type = defaultdict(list)
    for r in usable:
        for t in day_types[r["date"]]:
            by_type[t].append(r)

    file_dates = defaultdict(set)
    for d, t in events:
        file_dates[t].add(norm_date(d))
    usable_dates = {r["date"] for r in usable}
    coverage = {t: {"sessions_in_file": len(ds), "usable_event_sessions": len(ds & usable_dates),
                    "not_usable_sessions": len(ds - usable_dates)}
                for t, ds in sorted(file_dates.items())}
    types_present = sorted(file_dates)
    sched_present = [t for t in SCHEDULED_TYPES if t in file_dates or t == PRIMARY_TYPE]
    geo_present = [t for t in types_present if t not in SCHEDULED_TYPES]

    sblocks = {t: _type_block(by_type.get(t, []), sched_control, reps, seed) for t in sched_present}
    gblocks = {t: _type_block(by_type.get(t, []), nonevent, reps, seed) for t in geo_present}

    # H1 -------------------------------------------------------------------------------
    b = sblocks[PRIMARY_TYPE]
    h1 = {"type": PRIMARY_TYPE, "control": "sessions with no scheduled event (A1)",
          "n_event_days": b["n_event_days"], "n_control_days": b["n_control_days"],
          "rel_range": dict(b["rel_range"], pass_condition="ci95 excludes 0 and diff >= 0.2; no family (A3)",
                            **{"pass": h1_pass(b["rel_range"])}),
          "rel_gap": dict(b["rel_gap"], descriptive=True),
          "status": "tested" if b["n_event_days"] >= MIN_EVENT_DAYS else _not_tested(b["n_event_days"])}
    if b["n_event_days"] < MIN_EVENT_DAYS:
        h1["rel_range"]["pass"] = False

    # H2 -------------------------------------------------------------------------------
    h2_family = [t for t in H2_FAMILY_TYPES if t in sblocks and sblocks[t]["n_event_days"] >= MIN_EVENT_DAYS]
    adj = holm([sblocks[t]["rel_range"]["p"] for t in h2_family])
    h2 = {"control": "sessions with no scheduled event (A1)", "family": list(h2_family),
          "family_size": len(h2_family),
          "pass_condition": "Holm-adjusted p < 0.05, then diff >= 0.2 (A3)", "types": {}}
    for t in [t for t in H2_FAMILY_TYPES if t in sblocks]:
        blk = sblocks[t]
        e = {"n_event_days": blk["n_event_days"], "n_control_days": blk["n_control_days"],
             "tested": t in h2_family, "rel_range": dict(blk["rel_range"]), "rel_gap": dict(blk["rel_gap"])}
        e["rel_range"]["p_raw"] = blk["rel_range"]["p"]
        if t in h2_family:
            ph = adj[h2_family.index(t)]
            e["rel_range"]["p_holm"] = ph
            e["rel_range"]["pass"] = bool(ph is not None and ph < ALPHA
                                          and blk["rel_range"]["diff"] is not None
                                          and blk["rel_range"]["diff"] >= MIN_DIFF)
            e["status"] = "tested"
        else:
            e["rel_range"]["p_holm"] = None
            e["rel_range"]["pass"] = None
            e["status"] = _not_tested(blk["n_event_days"])
        h2["types"][t] = e

    # H3 -------------------------------------------------------------------------------
    h3_family = [t for t in H3_FAMILY_TYPES if t in sblocks and sblocks[t]["n_event_days"] >= MIN_EVENT_DAYS]
    adj3 = holm([sblocks[t]["affected_share"]["p"] for t in h3_family])
    h3 = {"control": "sessions with no scheduled event (A1)", "family": list(h3_family),
          "family_size": len(h3_family),
          "pass_condition": "Holm-adjusted p < 0.05 and ci95 excludes 0 (§6 + A3)",
          "control_affected_share": share_boot([r["affected"] for r in sched_control], reps, seed),
          "types": {}}
    for t in [t for t in H3_FAMILY_TYPES if t in sblocks]:
        sh = sblocks[t]["affected_share"]
        e = {"n_event_days": sblocks[t]["n_event_days"], "n_control_days": sblocks[t]["n_control_days"],
             "tested": t in h3_family, "share_event": sh["share_a"], "share_control": sh["share_b"],
             "k_event": sh["k_a"], "k_control": sh["k_b"], "diff": sh["diff"], "ci95": sh["ci95"],
             "p_raw": sh["p"]}
        if t in h3_family:
            ph = adj3[h3_family.index(t)]
            e["p_holm"] = ph
            e["pass"] = bool(ph is not None and ph < ALPHA and ci_excludes_zero(sh["ci95"]))
            e["status"] = "tested"
        else:
            e["p_holm"] = None
            e["pass"] = None
            e["status"] = _not_tested(sblocks[t]["n_event_days"])
        h3["types"][t] = e

    # geopolitical: descriptive + placebo (A1) ----------------------------------------
    geo = {"control": "sessions with no event of any type", "label": GEO_LABEL,
           "placebo": {"draws": placebo_draws, "matching": "calendar year and weekday, with replacement",
                       "flag_rule": "selection when placebo p95 >= 0.5 * observed diff (observed > 0)"},
           "categories": {}}
    for t in geo_present:
        blk = gblocks[t]
        ev = by_type.get(t, [])
        pr = placebo_diffs(ev, usable, _vals(nonevent, "rel_range"), "rel_range", np.median,
                           placebo_draws, seed)
        ps = placebo_diffs(ev, usable, [1.0 if r["affected"] else 0.0 for r in nonevent],
                           "affected", np.mean, placebo_draws, seed)
        geo["categories"][t] = {
            "n_event_days": blk["n_event_days"], "n_control_days": blk["n_control_days"],
            "tested": False, "never_tested_reason": ("category defined by market stress (A1)"
                                                     if t in NEVER_TESTED else GEO_LABEL),
            "rel_range": dict(blk["rel_range"], placebo_p95=pr["p95"], placebo_median=pr["median"],
                              placebo_unmatched=pr["unmatched_event_sessions"],
                              placebo_flag=placebo_flag(blk["rel_range"]["diff"], pr["p95"])),
            "rel_gap": dict(blk["rel_gap"]),
            "affected_share": dict(blk["affected_share"], placebo_p95=ps["p95"], placebo_median=ps["median"],
                                   placebo_unmatched=ps["unmatched_event_sessions"],
                                   placebo_flag=placebo_flag(blk["affected_share"]["diff"], ps["p95"]))}

    # H4 -------------------------------------------------------------------------------
    def shares(rows):
        return {"n": len(rows),
                "ranged_a": share_boot([bool(r.get("ranged_a")) for r in rows], reps, seed),
                "ranged_b": share_boot([bool(r.get("ranged_b")) for r in rows], reps, seed)}
    by_year = defaultdict(list)
    for r in usable:
        by_year[r["date"][:4]].append(r)
    h4 = {"overall": shares(usable),
          "by_year": {y: shares(rows) for y, rows in sorted(by_year.items())},
          "event_any": shares(all_event), "nonevent": shares(nonevent),
          "event_minus_nonevent": {
              k: share_diff_boot([bool(r.get(k)) for r in all_event],
                                 [bool(r.get(k)) for r in nonevent], reps, seed)
              for k in ("ranged_a", "ranged_b")},
          "descriptive": True}

    # H5 (A5) --------------------------------------------------------------------------
    ev_unaff = [r for r in all_event if not r["affected"]]
    non_unaff = [r for r in nonevent if not r["affected"]]
    h5 = dict(contingency(ev_unaff, non_unaff),
              comparison="not-affected event days (any type) vs not-affected non-event days (A5)",
              n_event_days_affected=len(all_event) - len(ev_unaff),
              n_nonevent_days_affected=len(nonevent) - len(non_unaff), descriptive=True)

    return {"exclusions": excl,
            "n_event_days_any_type": len(all_event), "n_nonevent_days": len(nonevent),
            "n_scheduled_control_days": len(sched_control),
            "date_range": [usable[0]["date"], usable[-1]["date"]] if usable else None,
            "event_coverage": coverage,
            "H1": h1, "H2": h2, "H3": h3, "geopolitical_descriptive": geo, "H4": h4, "H5": h5,
            "A7_early_knowable": early_knowable(usable, reps, seed)}


def add_replication(primary: dict, repl: dict | None, sym: str) -> None:
    """A4: writes replicated_on_{sym} into the primary symbol's H1, H2 and H3 entries."""
    key = f"replicated_on_{sym}"
    primary["H1"]["rel_range"][key] = replication_flag(
        primary["H1"]["rel_range"], repl["H1"]["rel_range"] if repl else None, MIN_DIFF)
    primary["H1"]["rel_range"][key + "_rule"] = "same sign, CI excludes 0, |diff| >= 0.2; " + REPLICATION_LABEL
    for t, e in primary["H2"]["types"].items():
        r = repl["H2"]["types"].get(t) if repl else None
        e["rel_range"][key] = replication_flag(e["rel_range"], r["rel_range"] if r else None, MIN_DIFF)
        e["rel_range"][key + "_rule"] = "same sign, CI excludes 0, |diff| >= 0.2"
    for t, e in primary["H3"]["types"].items():
        r = repl["H3"]["types"].get(t) if repl else None
        e[key] = replication_flag(e, r, None)
        e[key + "_rule"] = "same sign, CI excludes 0 (no floor defined for a share difference)"


def build_report(features_by_symbol: dict, events, events_meta: dict | None = None,
                 lab_rows: list[dict] | None = None, primary_symbol: str = "QQQ",
                 replication_symbols=("SPY",), reps: int = REPS, seed: int = SEED,
                 placebo_draws: int = PLACEBO_DRAWS) -> dict:
    """features_by_symbol: {symbol: rows or None}. Missing replication symbols give 'n/a'."""
    rep = {"pre_registration": "research/news_impact/preregistration.md (incl. §9 amendment)",
           "task_id": "R-2026-10-30-05", "parent": "R-2026-10-30-01",
           "bootstrap": {"reps": reps, "seed": seed, "unit": "session", "ci": "95% percentile",
                         "p": "two-sided, 2*min(frac<=0, frac>=0), floored 1/reps",
                         "placebo_draws": placebo_draws},
           "thresholds": {"min_event_days": MIN_EVENT_DAYS, "min_median_diff": MIN_DIFF,
                          "alpha": ALPHA, "impact_rel_range": [0.8, 1.5], "impact_rel_gap": [1.0, 2.5],
                          "early_fraction": EARLY_FRACTION},
           "families": {"H1": "none (confirmatory)", "H2": H2_FAMILY_TYPES, "H3": H3_FAMILY_TYPES,
                        "geopolitical": "none; descriptive with placebo", "SPY": "none"},
           "events": events_meta or {}, "primary_symbol": primary_symbol,
           "replication_symbols": list(replication_symbols), "replication_label": REPLICATION_LABEL,
           "symbols": {}}
    if features_by_symbol.get(primary_symbol) is None:
        raise FileNotFoundError(f"no features for primary symbol {primary_symbol}")
    for sym, feats in features_by_symbol.items():
        rep["symbols"][sym] = (analyze_symbol(feats, events, reps, seed, placebo_draws)
                               if feats is not None else None)
    prim = rep["symbols"][primary_symbol]
    for sym in replication_symbols:
        add_replication(prim, rep["symbols"].get(sym), sym)
    event_types_by_date = defaultdict(set)
    for d, t in events:
        event_types_by_date[norm_date(d)].add(t)
    rep["H6"] = (lab_rates(lab_rows, event_types_by_date) if lab_rows is not None
                 else {"status": "lab_days.csv not found"})
    rep["H6"]["descriptive"] = True
    return rep


# ------------------------------------------------------------------------------ output
def _fmt(x, nd=3):
    if x is None:
        return "n/a"
    if isinstance(x, (list, tuple)):
        return "[" + ", ".join(_fmt(v, nd) for v in x) + "]"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def _verdict(p):
    return "PASS" if p is True else ("FAIL" if p is False else "not tested")


def summary_text(rep: dict) -> str:
    L = []
    prim = rep["primary_symbol"]
    s = rep["symbols"][prim]
    h1 = s["H1"]["rel_range"]
    L.append(f"H1 ({prim}, FOMC vs no-scheduled-event sessions, median rel_range): {_verdict(h1['pass'])}. "
             f"FOMC days' range was {_fmt(h1['diff'], 2)} of a normal day wider "
             f"(CI {_fmt(h1['ci95'], 2)}, {s['H1']['n_event_days']} FOMC days vs "
             f"{s['H1']['n_control_days']} control days). {s['H1']['status']}.")
    for sym in rep["replication_symbols"]:
        k = f"replicated_on_{sym}"
        if k in h1:
            L.append(f"  replicated on {sym}: {h1[k]} ({rep['replication_label']})")
    for sym, sv in rep["symbols"].items():
        if sv is None:
            L.append(f"{sym}: no features file, skipped")
            continue
        e = sv["exclusions"]
        L.append(f"{sym}: {e['n_rows']} rows, {e['n_thin_excluded']} thin excluded, "
                 f"{e['n_rel_range_none_excluded']} rel_range-None excluded, {e['n_usable']} usable; "
                 f"{sv['n_event_days_any_type']} event days, {sv['n_nonevent_days']} non-event days, "
                 f"{sv['n_scheduled_control_days']} no-scheduled-event control days"
                 + (f"; impact column disagrees on {e['impact_column_disagreements']} rows"
                    if e.get("impact_column_disagreements") else ""))
        L.append(f"  H2 (median rel_range, event minus control; Holm family of {sv['H2']['family_size']})")
        for t, en in sv["H2"]["types"].items():
            r = en["rel_range"]
            L.append(f"    {t:18s} n={en['n_event_days']:4d} diff={_fmt(r['diff'])} CI={_fmt(r['ci95'])} "
                     f"p={_fmt(r['p_raw'], 4)} holm={_fmt(r['p_holm'], 4)} "
                     f"{_verdict(r['pass']) if en['tested'] else en['status']}")
        L.append(f"  H3 (share of affected days, event minus control; Holm family of {sv['H3']['family_size']})")
        for t, en in sv["H3"]["types"].items():
            L.append(f"    {t:18s} n={en['n_event_days']:4d} {_fmt(en['share_event'])} vs "
                     f"{_fmt(en['share_control'])} diff={_fmt(en['diff'])} CI={_fmt(en['ci95'])} "
                     f"holm={_fmt(en['p_holm'], 4)} {_verdict(en['pass']) if en['tested'] else en['status']}")
        L.append(f"  geopolitical ({sv['geopolitical_descriptive']['label']})")
        for t, en in sv["geopolitical_descriptive"]["categories"].items():
            r = en["rel_range"]
            L.append(f"    {t:18s} n={en['n_event_days']:4d} diff={_fmt(r['diff'])} CI={_fmt(r['ci95'])} "
                     f"placebo p95={_fmt(r['placebo_p95'])} -> {r['placebo_flag']}")
        h4 = sv["H4"]
        L.append(f"  H4 ranged share overall: a={_fmt(h4['overall']['ranged_a']['share'])} "
                 f"CI={_fmt(h4['overall']['ranged_a']['ci95'])}, "
                 f"b={_fmt(h4['overall']['ranged_b']['share'])} CI={_fmt(h4['overall']['ranged_b']['ci95'])}; "
                 f"event a={_fmt(h4['event_any']['ranged_a']['share'])} b={_fmt(h4['event_any']['ranged_b']['share'])}, "
                 f"non-event a={_fmt(h4['nonevent']['ranged_a']['share'])} b={_fmt(h4['nonevent']['ranged_b']['share'])}")
        for y, ys in h4["by_year"].items():
            L.append(f"    {y}: n={ys['n']:3d} a={_fmt(ys['ranged_a']['share'])} b={_fmt(ys['ranged_b']['share'])}"
                     f" thin={e['thin_by_year'].get(y, 0)}")
        h5 = sv["H5"]
        L.append(f"  H5 day_class {h5['classes']}: not-affected event {h5['event_unaffected']} vs "
                 f"not-affected non-event {h5['nonevent_unaffected']}, chi2 p={_fmt(h5['p'], 4)} (descriptive)")
        a7 = sv["A7_early_knowable"]["by_class"]
        L.append("  A7 first-hour move already > 0.6 x normal range: "
                 + ", ".join(f"{c} {_fmt(a7[c]['share'])} ({a7[c]['k']}/{a7[c]['n']})" for c in DAY_CLASSES)
                 + " (descriptive)")
    h6 = rep["H6"]
    if "status" in h6:
        L.append(f"H6: {h6['status']}")
    else:
        for g in ("event", "nonevent"):
            r = h6[g]
            L.append(f"H6 lab {g:8s} n={r['n']:3d} options green {r['options_green']['green']}/{r['options_green']['n']} "
                     f"shares green {r['shares_green']['green']}/{r['shares_green']['n']} "
                     f"both green {r['both_green']['green']}/{r['both_green']['n']} (counts only, no test)")
    return "\n".join(L)


def _repo_sha(root: Path):
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=root, capture_output=True,
                              text=True, check=True, timeout=10).stdout.strip()
    except Exception:            # not a git checkout, or git missing: still write the report
        return None


def _round(obj, nd=6):
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, (float, np.floating)):
        return round(float(obj), nd)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, dict):
        return {k: _round(v, nd) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_round(v, nd) for v in obj]
    return obj


def main(argv=None) -> int:
    root = Path(__file__).resolve().parents[2]
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[2])
    ap.add_argument("--features-dir", default=str(root / "research" / "news_impact"))
    ap.add_argument("--symbols", nargs="+", default=["QQQ", "SPY"],
                    help="first is primary, the rest are replications")
    ap.add_argument("--out", default=None, help="JSON path; default <features-dir>/analysis.json")
    ap.add_argument("--reps", type=int, default=REPS)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--placebo-draws", type=int, default=PLACEBO_DRAWS)
    a = ap.parse_args(argv)
    fdir = Path(a.features_dir)
    sched, geo = fdir / "events_scheduled.csv", fdir / "events_geopolitical.csv"
    for p in (sched, geo):
        if not p.exists():
            print(f"missing: {p}", file=sys.stderr)
            return 2
    events, meta = load_events(sched, geo)
    feats = {}
    for sym in a.symbols:
        p = fdir / f"day_features_{sym}.csv"
        feats[sym] = load_features(p) if p.exists() else None
        meta.setdefault("features_files", {})[sym] = str(p) if p.exists() else None
    if feats[a.symbols[0]] is None:
        print(f"missing primary features file: {fdir / f'day_features_{a.symbols[0]}.csv'}",
              file=sys.stderr)
        return 2
    lab_p = fdir / "lab_days.csv"
    lab = load_lab_days(lab_p) if lab_p.exists() else None
    meta["lab_days_file"] = str(lab_p) if lab_p.exists() else None
    meta["unknown_types_note"] = ("types outside §3 are reported as geopolitical-style descriptive lines, never tested"
                                  if meta["unknown_types"] else "")
    rep = build_report(feats, events, meta, lab, primary_symbol=a.symbols[0],
                       replication_symbols=a.symbols[1:], reps=a.reps, seed=a.seed,
                       placebo_draws=a.placebo_draws)
    rep["run"] = {"repo_sha_at_run": _repo_sha(root), "features_dir": str(fdir),
                  "run_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")}
    out = Path(a.out) if a.out else fdir / "analysis.json"
    out.write_text(json.dumps(_round(rep), indent=1), encoding="utf-8")
    print(summary_text(rep))
    print(f"\nwritten: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
