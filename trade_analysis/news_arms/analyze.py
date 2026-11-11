"""Q1-Q7 of research/news_arms/preregistration.md, on one arm and one set, every number into
research/news_arms/analysis_<arm>_<set>.json.

    python -m trade_analysis.news_arms.analyze --arm shares --set discovery
    python -m trade_analysis.news_arms.analyze --arm options --set discovery
    python -m trade_analysis.news_arms.analyze --q6                     # both arms' discovery JSONs
    python -m trade_analysis.news_arms.analyze --arm shares --set holdout   # refuses until the
                                                                            # discovery report is committed

What is fixed here, by section of the pre-registration:
- sigma is the DISCOVERY daily-net standard deviation of the arm (15.5); the set's own sigma
  is descriptive only. Standardised return z = net / sigma.
- a condition's effect is mean z(condition) - mean z(all sessions) = condition - B0 (14.12),
  diluted by (1 - p); the undiluted condition - complement is beside it (14.Q6).
- CI: day-clustered bootstrap, sessions resampled with replacement, 3,000 reps, seed 20260828.
- placebo B1: 5,000 draws of the same number of sessions matched by year and weekday, the
  condition's own sessions excluded from the pool (13.3); "selection / chance" when the real
  effect ranks below the placebo 95th percentile of the same sign (14.Q6).
- tested only with >= 30 sessions and >= 50 trades (section 7); financial_stress never;
  geopolitical categories descriptive + placebo, never in a tested family (14.7).
- BH at q = 0.10 within each table (section 7); Q7 survivors with |effect| >= 0.25 sigma
  become the shortlist, top five by |effect| (14.7).
- a session on which a setup did not trade counts as 0 for that setup (15.7).
- pre/post (14.4, 15.4): post only when the event minute is inside 09:30-16:00 on that
  session and entry_ts >= it; pre-open releases and overnight shocks make every trade post;
  unknown break times are out of Q5.
- no conditioning column is built from the session's own prices (15.4): CONDITIONING lists
  them and a test checks the list.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from . import events as E
from . import join as J

OUT = J.OUT
SEED_BOOT, SEED_PLACEBO = 20260828, 20260828
N_BOOT, N_PLACEBO = 3000, 5000
MIN_EFFECT, BH_Q = 0.25, 0.10
MIN_CI_SESSIONS = 5                 # below this: effect reported, no CI, no p, no placebo
FIRST60 = ("09:30", "10:30")
ONE_SIDED = {"Crabel_Stretch": 0.00, "Gap_Fade": 0.08, "IntradayMomentumBoundary": 0.19,
             "Six_Lines": 0.25, "MOMO_CHASE": 0.54, "ORB_5min": 0.72, "VWAP_Reclaim": 0.79}
LOW_FIDELITY = {"VWAP_Reclaim", "ORB_5min", "MOMO_CHASE"}
EXIT_MAP = {"failed": "stop", "bars": "time", "trail_imb": "trail"}
# 15.4: the columns any condition may be built from. A test asserts none is a price field.
CONDITIONING = ("scheduled_today", "any_event", "categories", "single_category", "surprise_geo",
                "regime", "year", "weekday", "entry_before_event_minute", "entry_in_first_60")
FORBIDDEN = ("open", "high", "low", "close", "gap_bp", "range_bp", "co_bp", "first60_", "rel_",
             "impact", "day_class", "pnl", "net", "return")


# ------------------------------------------------------------------ loading

def event_detail() -> dict:
    """date -> list of {category, minute, kind, surprise}. kind: in_session (minute is the
    post boundary), all_post (pre-open release, overnight/weekend shock), unknown (out of Q5)."""
    out = defaultdict(list)
    with open(J.NI / "events_scheduled.csv", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            in_sess = r["session"] == r["date"] and "09:30" <= r["time_et"] < "16:00"
            out[r["session"]].append({"category": r["type"], "minute": r["time_et"] if in_sess else None,
                                      "kind": "in_session" if in_sess else "all_post", "surprise": False})
    for r in E.load_geopolitical():
        if r.get("break_time_et", "unknown") == "unknown":
            kind, minute = "unknown", None
        elif E.event_minute(r) is None:
            kind, minute = "all_post", None
        else:
            kind, minute = "in_session", r["break_time_et"]
        out[r["date"]].append({"category": r["category"], "minute": minute, "kind": kind,
                               "surprise": r.get("surprise") == "yes"})
    return out


def day_direction(csv_path: Path) -> dict:
    """date -> close-to-open bp of the index, descriptive twin only (Q4). Never a condition."""
    out = {}
    with open(csv_path, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            try:
                out[r["date"]] = float(r["co_bp"])
            except (ValueError, KeyError):
                pass
    return out


def load(arm: str, set_name: str):
    window = (J.DISCOVERY if set_name == "discovery" else J.HOLDOUT)[arm]
    years = range(window[0].year, window[1].year + 1)
    sessions, trades, meta = J.load_replay(arm, years)
    events = J.load_events()
    cuts = J.regime_cuts(J.NI / "day_features_QQQ.csv", *J.DISCOVERY["shares"])
    reg = J.regimes(J.NI / "day_features_QQQ.csv", cuts)
    rows = J.build_join(arm, sessions, trades, events, reg, window)
    days = {r["date"] for r in rows}
    trades = [t for t in trades if t["entry_ts"][:10] in days and t.get("pnl_net") is not None
              and (arm != "options" or t.get("arm") == "ATM")]
    return rows, trades, meta, cuts


def discovery_sigma(arm: str, rows_if_discovery=None) -> float:
    """15.5: sigma from the discovery rows; for a holdout run they are loaded for sigma only."""
    rows = rows_if_discovery if rows_if_discovery is not None else load(arm, "discovery")[0]
    return float(np.std([r["net"] for r in rows], ddof=1))


# ------------------------------------------------------------------ statistics

def bh(pvals: list[float], q: float = BH_Q) -> list[bool]:
    """Benjamini-Hochberg: True where rejected."""
    m = len(pvals)
    if m == 0:
        return []
    order = sorted(range(m), key=lambda i: pvals[i])
    out = [False] * m
    k_max = 0
    for rank, i in enumerate(order, start=1):
        if pvals[i] <= q * rank / m:
            k_max = rank
    for rank, i in enumerate(order, start=1):
        if rank <= k_max:
            out[i] = True
    return out


def holm(pvals: list[float], alpha: float = 0.05) -> list[bool]:
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    out = [False] * m
    for rank, i in enumerate(order):
        if pvals[i] <= alpha / (m - rank):
            out[i] = True
        else:
            break
    return out


def bootstrap_effect(z: np.ndarray, mask: np.ndarray, n_boot: int = N_BOOT, seed: int = SEED_BOOT):
    """condition - B0 with a day-clustered (session-resampled) bootstrap CI and two-sided p."""
    rng = np.random.default_rng(seed)
    n = len(z)
    k = int(mask.sum())
    if k == 0 or k == n:
        return {"effect": None, "ci": [None, None], "p": None, "undiluted": None}
    eff = float(z[mask].mean() - z.mean())
    und = float(z[mask].mean() - z[~mask].mean())
    if k < MIN_CI_SESSIONS:                                 # a CI on a handful of days is decoration
        return {"effect": eff, "ci": [None, None], "p": None, "undiluted": und}
    idx = rng.integers(0, n, size=(n_boot, n))
    zb = z[idx]
    mb = mask[idx]
    kb = mb.sum(axis=1)
    ok = kb > 0
    cond_mean = np.where(ok, (zb * mb).sum(axis=1) / np.maximum(kb, 1), np.nan)
    effs = cond_mean - zb.mean(axis=1)
    effs = effs[~np.isnan(effs)]
    lo, hi = (float(x) for x in np.percentile(effs, [2.5, 97.5]))
    p = 2 * min(float((effs <= 0).mean()), float((effs >= 0).mean()))
    p = max(min(p, 1.0), 1.0 / len(effs))
    return {"effect": eff, "ci": [lo, hi], "p": p, "undiluted": und}


def block_bootstrap_ci(z: np.ndarray, mask: np.ndarray, block: int = 10, n_boot: int = N_BOOT,
                       seed: int = SEED_BOOT):
    """NOT pre-registered: a moving-block bootstrap (blocks of `block` consecutive sessions)
    for the same condition - B0 statistic, added at gate 2 (2026-11-10) because |z| shows
    lag-1 autocorrelation of about +0.3 (volatility clustering). Sensitivity only."""
    n = len(z)
    k = int(mask.sum())
    if k < MIN_CI_SESSIONS or k == n or n < 2 * block:
        return [None, None]
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    starts = rng.integers(0, n - block + 1, size=(n_boot, n_blocks))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(n_boot, -1)[:, :n]
    zb, mb = z[idx], mask[idx]
    kb = mb.sum(axis=1)
    effs = np.where(kb > 0, (zb * mb).sum(axis=1) / np.maximum(kb, 1), np.nan) - zb.mean(axis=1)
    effs = effs[~np.isnan(effs)]
    return [float(x) for x in np.percentile(effs, [2.5, 97.5])]


def placebo(z: np.ndarray, mask: np.ndarray, year: np.ndarray, weekday: np.ndarray, effect: float,
            n_draw: int = N_PLACEBO, seed: int = SEED_PLACEBO, exclude_condition: bool = True) -> dict:
    """13.3 / 14.Q6: n_draw placebo conditions of the same size, matched by year and weekday,
    the real condition's sessions excluded from the pool. Each draw computes the identical
    condition - B0 statistic. Rank is the share of same-sign placebo effects below the real
    one; 'selection / chance' when the real effect is below the placebo 95th percentile."""
    rng = np.random.default_rng(seed)
    b0 = z.mean()
    strata = defaultdict(list)
    for i in range(len(z)):
        strata[(int(year[i]), int(weekday[i]))].append(i)
    totals = np.zeros(n_draw)
    k_total = 0
    fallback = 0
    for key, idxs in strata.items():
        idxs = np.array(idxs)
        k = int(mask[idxs].sum())
        if k == 0:
            continue
        pool = idxs[~mask[idxs]] if exclude_condition else idxs
        if len(pool) < k:                                   # thin stratum: same year, any weekday
            same_year = np.array([i for (y, _), ii in strata.items() if y == key[0] for i in ii])
            pool = same_year[~mask[same_year]] if exclude_condition else same_year
            fallback += 1
        if len(pool) < k:                                   # still thin: with replacement
            pick = rng.integers(0, len(pool), size=(n_draw, k))
        else:
            pick = np.argsort(rng.random((n_draw, len(pool))), axis=1)[:, :k]
        totals += z[pool[pick]].sum(axis=1)
        k_total += k
    effs = totals / k_total - b0
    sign = 1.0 if effect >= 0 else -1.0
    same = effs * sign
    real = effect * sign
    p95, p99 = (float(x) * sign for x in np.percentile(same, [95, 99]))
    rank = float((same < real).mean())
    return {"n_draws": n_draw, "mean": float(effs.mean()), "sd": float(effs.std()),
            "p95_same_sign": p95, "p99_same_sign": p99, "rank": rank,
            "strata": len(strata), "strata_fallback": fallback, "condition_excluded_from_pool": exclude_condition,
            "verdict": "effect" if rank >= 0.95 else "selection / chance"}


# ------------------------------------------------------------------ cells

def _arrays(rows):
    net = np.array([r["net"] for r in rows], dtype=float)
    year = np.array([int(r["year"]) for r in rows])
    weekday = np.array([r["weekday"] for r in rows])
    return net, year, weekday


def metric_block(arm: str, trades: list[dict], rows: list[dict], sigma: float) -> dict:
    """Section 4, per cell: trade-level and day-level metrics in the arm's own units."""
    pnl = np.array([t["pnl_net"] for t in trades], dtype=float)
    nets = np.array([r["net"] for r in rows], dtype=float)
    n = len(pnl)
    out = {"n_sessions": len(rows), "n_trades": n,
           "daily_net_mean": float(nets.mean()) if len(nets) else None,
           "daily_net_median": float(np.median(nets)) if len(nets) else None,
           "z_mean": float(nets.mean() / sigma) if len(nets) else None,
           "win_rate_trade": float((pnl > 0).mean()) if n else None,
           "win_rate_day": float((nets > 0).mean()) if len(nets) else None,
           "expectancy_usd": float(pnl.mean()) if n else None,
           "trades_per_session": n / len(rows) if rows else None,
           "avg_winner_usd": float(pnl[pnl > 0].mean()) if (pnl > 0).any() else None,
           "avg_loser_usd": float(pnl[pnl < 0].mean()) if (pnl < 0).any() else None,
           "max_drawdown_usd": _mdd(nets)}
    if arm == "shares":
        bp = np.array([t["pnl_net"] / t["notional"] * 1e4 for t in trades if t.get("notional")])
        out.update({"expectancy_bp": float(bp.mean()) if len(bp) else None,
                    "avg_winner_bp": float(bp[bp > 0].mean()) if (bp > 0).any() else None,
                    "avg_loser_bp": float(bp[bp < 0].mean()) if (bp < 0).any() else None})
    else:
        prem = np.array([t["entry_ask"] * 100 * t.get("contracts", 1) for t in trades if t.get("entry_ask")])
        rop = np.array([t["pnl_net"] / (t["entry_ask"] * 100 * t.get("contracts", 1))
                        for t in trades if t.get("entry_ask")])
        reasons = defaultdict(int)
        shutdown = 0
        for t in trades:
            r = EXIT_MAP.get(t.get("exit_reason"), t.get("exit_reason"))
            if r == "shutdown":
                shutdown += 1
                continue
            reasons[r] += 1
        tot = sum(reasons.values()) or 1
        out.update({"return_on_premium_mean": float(rop.mean()) if len(rop) else None,
                    "avg_winner_pct_premium": float(rop[rop > 0].mean()) if (rop > 0).any() else None,
                    "avg_loser_pct_premium": float(rop[rop < 0].mean()) if (rop < 0).any() else None,
                    "median_premium_usd": float(np.median(prem)) if len(prem) else None,
                    "median_entry_spread_pct": float(np.median([t["entry_spread_pct"] for t in trades
                                                                if t.get("entry_spread_pct") is not None]))
                    if trades else None,
                    "exit_mix": {k: round(v / tot, 4) for k, v in sorted(reasons.items())},
                    "exits_shutdown_excluded": shutdown,
                    "underlying_twin_bp": float(np.mean([t["underlying_return"] * 1e4 for t in trades
                                                         if t.get("underlying_return") is not None]))
                    if trades else None})
    return out


def _mdd(nets: np.ndarray):
    if len(nets) == 0:
        return None
    c = np.cumsum(nets)
    peak = np.maximum.accumulate(np.concatenate([[0.0], c]))[1:]
    return float((c - peak).min())


def cell(arm: str, name: str, rows_all: list[dict], mask: np.ndarray, trades_by_day: dict,
         sigma: float, tested_ok: bool, never: bool, geo: bool, direction: dict | None = None,
         with_placebo: bool = True) -> dict:
    net, year, weekday = _arrays(rows_all)
    z = net / sigma
    cond_rows = [r for r, m in zip(rows_all, mask) if m]
    cond_trades = [t for r in cond_rows for t in trades_by_day.get(r["date"], [])]
    n_s, n_t = len(cond_rows), len(cond_trades)
    meets = n_s >= J.MIN_SESSIONS and n_t >= J.MIN_TRADES
    stat = bootstrap_effect(z, mask)
    out = {"cell": name, "n_sessions": n_s, "n_trades": n_t, "meets_minimums": meets,
           "tested": bool(meets and tested_ok and not never and not geo),
           "descriptive_placebo": bool(meets and geo and not never), "never_tested": never,
           "effect_sigma": stat["effect"], "ci95": stat["ci"], "p_boot": stat["p"],
           "ci95_block10_post_hoc": block_bootstrap_ci(z, mask),
           "undiluted_sigma": stat["undiluted"],
           "effect_native": (stat["effect"] * sigma) if stat["effect"] is not None else None,
           "metrics": metric_block(arm, cond_trades, cond_rows, sigma),
           "by_year": _by(rows_all, mask, "year", sigma), "by_regime": _by(rows_all, mask, "regime", sigma)}
    out["unstable"] = _unstable(out)
    if with_placebo and stat["effect"] is not None and n_s >= MIN_CI_SESSIONS:
        out["placebo"] = placebo(z, mask, year, weekday, stat["effect"])
        out["placebo_pool_includes_condition_sensitivity"] = placebo(
            z, mask, year, weekday, stat["effect"], exclude_condition=False)
    if direction is not None:
        d = [direction[r["date"]] for r in cond_rows if r["date"] in direction]
        out["day_direction_co_bp_mean"] = float(np.mean(d)) if d else None
    out["post_hoc_within_regime"] = within_regime(rows_all, mask, z)
    return out


def within_regime(rows_all, mask, z) -> dict:
    """NOT pre-registered; added after the discovery tables were first read (2026-11-10)
    because shock days cluster in the high trailing-range regime, which is itself a pre-open
    condition. Per regime: condition minus that regime's own B0, same bootstrap. Descriptive."""
    out = {}
    regs = np.array([str(r.get("regime")) for r in rows_all])
    for reg in ("low", "mid", "high"):
        sel = regs == reg
        if sel.sum() < MIN_CI_SESSIONS or mask[sel].sum() == 0:
            continue
        st = bootstrap_effect(z[sel], mask[sel])
        out[reg] = {"n_regime": int(sel.sum()), "n_condition": int(mask[sel].sum()),
                    "effect_sigma": st["effect"], "ci95": st["ci"]}
    return out


def _by(rows_all, mask, key, sigma):
    """Per stratum (year or regime): the condition's n, raw mean z, and its effect against
    that stratum's OWN B0 (gate 2: a raw by-year mean read as an effect called 2020 negative
    for Fed days when it was +0.04 sigma against that year's baseline)."""
    strat_all, strat_cond = defaultdict(list), defaultdict(list)
    for r, m in zip(rows_all, mask):
        strat_all[str(r.get(key))].append(r["net"])
        if m:
            strat_cond[str(r.get(key))].append(r["net"])
    out = {}
    for k in sorted(strat_cond):
        c, a = np.array(strat_cond[k]), np.array(strat_all[k])
        out[k] = {"n": len(c), "net_mean": float(c.mean()), "z_mean": float(c.mean() / sigma),
                  "effect_vs_stratum_b0": float((c.mean() - a.mean()) / sigma), "n_stratum": len(a)}
    return out


def _unstable(out: dict):
    """Section 2: a result whose sign flips across regime terciles, or across >= 3 of the
    years, is reported as unstable whatever the pooled p. Strata with < 5 sessions are
    ignored."""
    e = out.get("effect_sigma")
    if e is None or out["n_sessions"] < MIN_CI_SESSIONS:
        return None
    sign = 1 if e >= 0 else -1
    regimes = [v["effect_vs_stratum_b0"] for v in out["by_regime"].values() if v["n"] >= 5]
    years = [v["effect_vs_stratum_b0"] for v in out["by_year"].values() if v["n"] >= 5]
    flips_regime = any((x >= 0) != (sign > 0) for x in regimes)
    flips_years = sum((x >= 0) != (sign > 0) for x in years) >= 3
    return bool(flips_regime or flips_years)


# ------------------------------------------------------------------ tables

def q1(arm, rows, trades_by_day, sigma, direction, single_only=False):
    """Q1/Q2/Q4 in one table: by category vs B0, with the metric block and the twin."""
    universe = [r for r in rows if r["n_categories"] <= 1] if single_only else rows
    out = {}
    for cat in J.CATEGORIES:
        mask = np.array([cat in r["categories"] for r in universe])
        out[cat] = cell(arm, cat, universe, mask, trades_by_day, sigma, tested_ok=True,
                        never=cat in J.NEVER_TESTED, geo=cat in J.GEO, direction=direction)
    _apply_bh(out)
    return out


def q3(arm, rows, trades_by_day, sigma, detail, single_only=False):
    universe = [r for r in rows if r["n_categories"] <= 1] if single_only else rows
    sched = np.array([r["scheduled_today"] for r in universe])
    surprise = np.array([any(e["surprise"] for e in detail.get(r["date"], [])) for r in universe])
    unsched_excl = surprise & ~sched
    none = np.array([not r["any_event"] for r in universe])
    out = {"scheduled": cell(arm, "scheduled", universe, sched, trades_by_day, sigma, True, False, False),
           "unscheduled_surprise": cell(arm, "unscheduled_surprise", universe, surprise, trades_by_day, sigma,
                                        True, False, True),
           "unscheduled_surprise_no_release_sensitivity": cell(
               arm, "unscheduled_surprise_no_release_sensitivity", universe, unsched_excl, trades_by_day,
               sigma, True, False, True, with_placebo=False),
           "no_event": cell(arm, "no_event", universe, none, trades_by_day, sigma, True, False, False,
                            with_placebo=False)}
    _apply_bh({k: v for k, v in out.items() if k in ("scheduled", "unscheduled_surprise")})
    return out


def _apply_bh(table: dict):
    tested = [k for k, v in table.items() if v["tested"] and v["p_boot"] is not None]
    flags = bh([table[k]["p_boot"] for k in tested])
    for k, f in zip(tested, flags):
        table[k]["bh_q10"] = bool(f)
    for k, v in table.items():
        v.setdefault("bh_q10", None)


def q5(rows, trades_by_day, detail) -> dict:
    """Descriptive, exploratory (13.5). Pre/post by category per 14.4/15.4, and the first-60
    rule-shaped line: P&L of trades entered 09:30-10:30 on scheduled_today sessions vs the
    same window on non-release sessions."""
    pre_post = {}
    for cat in J.CATEGORIES:
        pre, post, excluded = [], [], 0
        for r in rows:
            evs = [e for e in detail.get(r["date"], []) if e["category"] == cat]
            if not evs:
                continue
            if any(e["kind"] == "unknown" for e in evs):
                excluded += 1
                continue
            minute = min((e["minute"] for e in evs if e["minute"]), default=None)
            for t in trades_by_day.get(r["date"], []):
                hhmm = t["entry_ts"][11:16]
                (pre if minute is not None and hhmm < minute else post).append(t["pnl_net"])
        minutes = {e["minute"] for r in rows for e in detail.get(r["date"], [])
                   if e["category"] == cat and e["minute"]}
        excess = None
        if len(minutes) == 1:
            cut = next(iter(minutes))
            per = {r["date"]: sum(t["pnl_net"] for t in trades_by_day.get(r["date"], []) if t["entry_ts"][11:16] < cut)
                   for r in rows}
            cond = [per[r["date"]] for r in rows if any(e["category"] == cat for e in detail.get(r["date"], []))]
            if cond:
                excess = {"cutoff": cut, "condition_pre_net_per_session": float(np.mean(cond)),
                          "all_sessions_pre_net_per_session": float(np.mean(list(per.values()))),
                          "excess_usd_per_session": float(np.mean(cond) - np.mean(list(per.values())))}
        pre_post[cat] = {"pre_n": len(pre), "pre_mean_usd": float(np.mean(pre)) if pre else None,
                         "pre_cutoff_excess": excess,
                         "pre_total_usd": float(np.sum(pre)) if pre else 0.0,
                         "post_n": len(post), "post_mean_usd": float(np.mean(post)) if post else None,
                         "post_total_usd": float(np.sum(post)) if post else 0.0,
                         "sessions_excluded_unknown_time": excluded,
                         "note": "descriptive, exploratory; break times are recollections rounded "
                                 "to 30-60 min; pre-open releases make every trade post"}
    first = {}
    for label, pick in (("scheduled_today", lambda r: r["scheduled_today"]),
                        ("non_release", lambda r: not r["scheduled_today"])):
        per_session = []
        for r in rows:
            if not pick(r):
                continue
            per_session.append(sum(t["pnl_net"] for t in trades_by_day.get(r["date"], [])
                                   if FIRST60[0] <= t["entry_ts"][11:16] < FIRST60[1]))
        first[label] = {"n_sessions": len(per_session),
                        "first60_net_mean_usd": float(np.mean(per_session)) if per_session else None}
    a = np.array([sum(t["pnl_net"] for t in trades_by_day.get(r["date"], [])
                      if FIRST60[0] <= t["entry_ts"][11:16] < FIRST60[1]) for r in rows])
    m = np.array([r["scheduled_today"] for r in rows])
    if m.any() and (~m).any():
        rng = np.random.default_rng(SEED_BOOT)
        idx = rng.integers(0, len(a), size=(N_BOOT, len(a)))
        ab, mb = a[idx], m[idx]
        diff = (ab * mb).sum(1) / np.maximum(mb.sum(1), 1) - (ab * ~mb).sum(1) / np.maximum((~mb).sum(1), 1)
        first["difference_usd"] = float(a[m].mean() - a[~m].mean())
        first["ci95"] = [float(x) for x in np.percentile(diff, [2.5, 97.5])]
    return {"pre_post": pre_post, "first60": first}


def q7(arm, rows, trades_by_day, sigma) -> dict:
    """setup x category: setup z (0 when it did not trade, 15.7) vs the setup's own B0; BH
    q=0.10 over tested cells; shortlist = survivors with |effect| >= 0.25 sigma, top 5."""
    setups = sorted({s for r in rows for s in r["setup_n"]})
    net, year, weekday = _arrays(rows)
    cells = {}
    for s in setups:
        zs = np.array([r["setup_net"].get(s, 0.0) for r in rows]) / sigma
        for cat in J.CATEGORIES:
            if cat in J.NEVER_TESTED:
                continue                                      # 14.7: financial_stress is out of Q7
            mask = np.array([cat in r["categories"] for r in rows])
            n_s = int(mask.sum())
            n_t = sum(r["setup_n"].get(s, 0) for r, m in zip(rows, mask) if m)
            meets = n_s >= J.MIN_SESSIONS and n_t >= J.MIN_TRADES
            stat = bootstrap_effect(zs, mask)
            c = {"setup": s, "category": cat, "n_sessions": n_s, "n_trades": n_t, "tested": meets,
                 "geopolitical": cat in J.GEO,                 # exploratory BH only, never confirmatory
                 "effect_sigma": stat["effect"], "ci95": stat["ci"], "p_boot": stat["p"],
                 "one_sided_rate": ONE_SIDED.get(s), "low_fidelity": s in LOW_FIDELITY}
            if meets and stat["effect"] is not None:
                c["placebo"] = placebo(zs, mask, year, weekday, stat["effect"])
            cells[f"{s}|{cat}"] = c
    tested = [k for k, c in cells.items() if c["tested"] and c["p_boot"] is not None]
    flags = bh([cells[k]["p_boot"] for k in tested])
    for k, f in zip(tested, flags):
        cells[k]["bh_q10"] = bool(f)
    bh_pass = [k for k, f in zip(tested, flags) if f]
    survivors = [k for k in bh_pass if abs(cells[k]["effect_sigma"]) >= MIN_EFFECT]
    shortlist = sorted(survivors, key=lambda k: -abs(cells[k]["effect_sigma"]))[:5]
    return {"m_tested": len(tested), "cells": cells, "bh_pass": bh_pass,
            "bh_pass_and_quarter_sigma": survivors, "shortlist": shortlist,
            "note": "15.7: a setup's z is its net over the ARM's sigma, so one setup's cell is a "
                    "fraction of a day's variation; the 0.25-sigma bar is the pre-registered one"}


def q6(a_opt: dict, a_sh: dict) -> dict:
    """Per category, options effect minus shares effect; divergence only if both arms' own
    CIs exclude zero with opposite signs."""
    out = {}
    for cat in J.CATEGORIES:
        o, s = a_opt["q1"].get(cat), a_sh["q1"].get(cat)
        if not o or not s or o["effect_sigma"] is None or s["effect_sigma"] is None:
            continue
        def excl(c):
            return c["ci95"][0] is not None and (c["ci95"][0] > 0 or c["ci95"][1] < 0)
        div = excl(o) and excl(s) and (o["effect_sigma"] > 0) != (s["effect_sigma"] > 0)
        out[cat] = {"options_effect": o["effect_sigma"], "options_ci": o["ci95"],
                    "shares_effect": s["effect_sigma"], "shares_ci": s["ci95"],
                    "difference": o["effect_sigma"] - s["effect_sigma"], "divergence": bool(div),
                    "note": "different windows and session sets per arm; sigma is each arm's own"}
    return out


# ------------------------------------------------------------------ confirmatory (holdout only)

def confirmatory(rows, sigma_disc) -> dict:
    """C1/C2: standardised daily return on scheduled_today sessions minus B0, discovery sigma.
    PASS needs the 95% CI to exclude 0 and |effect| >= 0.25 sigma; C2 is reported with its MDE
    (13.6). Holm over the two is applied in the combined step."""
    net, _, _ = _arrays(rows)
    z = net / sigma_disc
    m = np.array([r["scheduled_today"] for r in rows])
    stat = bootstrap_effect(z, m)
    n1, n0 = int(m.sum()), int((~m).sum())
    se = float(np.sqrt(1 / max(n1, 1) + 1 / max(n0, 1)))
    mde = 2.8 * se                                           # 80% power, two-sided 5%
    passed = (stat["ci"][0] is not None and (stat["ci"][0] > 0 or stat["ci"][1] < 0)
              and abs(stat["effect"]) >= MIN_EFFECT)
    single = [r for r in rows if r["n_categories"] <= 1]
    zs = np.array([r["net"] for r in single]) / sigma_disc
    ms = np.array([r["scheduled_today"] for r in single])
    return {"n_scheduled": n1, "n_other": n0, "effect_sigma": stat["effect"], "ci95": stat["ci"],
            "p_boot": stat["p"], "undiluted": stat["undiluted"], "se_approx": se, "mde_80pct": mde,
            "pass_rule": bool(passed), "observed_exceeds_mde": abs(stat["effect"] or 0) >= mde,
            "single_category_sensitivity": bootstrap_effect(zs, ms)}


# ------------------------------------------------------------------ run

def exclusions(rows: list[dict], window, chunks: list[dict], no_0dte=()) -> dict:
    """14.1: every open weekday in the window that is not a replayed session, listed and
    classified: early-close half days (dropped by MIN_RTH_BARS by design), days a chunk's
    RUN.json recorded as skipped, and LOST regular sessions, which are a defect until
    re-replayed (gate 2 found 23 in 2022)."""
    from ..live_lab.nyse_calendar import early_close, is_closed
    have = {r["date"] for r in rows}
    recorded = {s["date"]: s["reason"] for c in chunks for s in (c.get("skipped") or [])}
    recorded.update({d: "no_0dte" for d in no_0dte})          # 14.11, excluded by rule, counted
    half, lost, rec = [], [], []
    d = window[0]
    while d <= window[1]:
        iso = d.isoformat()
        if not is_closed(d) and iso not in have:
            if early_close(d):
                half.append(iso)
            elif iso in recorded and recorded[iso] != "half_day":
                rec.append({"date": iso, "reason": recorded[iso]})
            else:                                             # a "half_day" skip on a full day is a loss
                lost.append(iso)
        d += dt.timedelta(days=1)
    return {"open_days_in_window": len(rows) + len(half) + len(lost) + len(rec),
            "replayed": len(rows), "half_days": half, "n_recorded_skips": len(rec),
            "recorded_skips_by_reason": dict(Counter(r["reason"] for r in rec)),
            "recorded_skips": rec if len(rec) <= 40 else rec[:40] + [{"date": "...", "reason": f"{len(rec) - 40} more"}],
            "lost_regular_sessions": lost, "n_lost": len(lost)}


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


def run(arm: str, set_name: str) -> dict:
    rows, trades, meta, cuts = load(arm, set_name)
    disc_rows = rows if set_name == "discovery" else load(arm, "discovery")[0]
    sigma = discovery_sigma(arm, disc_rows)
    sigma_own = float(np.std([r["net"] for r in rows], ddof=1))
    detail = event_detail()
    direction = day_direction(J.NI / ("day_features_QQQ.csv" if arm == "shares" else "day_features_SPY.csv"))
    tb = defaultdict(list)
    for t in trades:
        tb[t["entry_ts"][:10]].append(t)
    sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=J.ROOT, capture_output=True,
                         text=True).stdout.strip()
    window = (J.DISCOVERY if set_name == "discovery" else J.HOLDOUT)[arm]
    res = {"arm": arm, "set": set_name, "window": [window[0].isoformat(), window[1].isoformat()],
           "repo_sha": sha, "config_hashes": meta["hashes"], "live_hash": J.LIVE_HASH[arm],
           "chunks": meta["chunks"], "excluded_no_0dte": meta["excluded_no_0dte"],
           "exclusions": exclusions(rows, window, meta["chunks"], meta["excluded_no_0dte"].get("dates", ())),
           "events_sha": {"scheduled": _sha(J.NI / "events_scheduled.csv"),
                          "geopolitical": _sha(J.NI / "events_geopolitical.csv")},
           "regime_cuts_med20_range_bp": list(cuts), "seeds": {"bootstrap": SEED_BOOT, "placebo": SEED_PLACEBO},
           "n_boot": N_BOOT, "n_placebo": N_PLACEBO, "command": " ".join(sys.argv),
           "generated": dt.datetime.now().isoformat(timespec="seconds"),
           "n_sessions": len(rows), "n_trades": len(trades),
           "sigma_discovery_usd": sigma, "sigma_own_usd_descriptive": sigma_own,
           "b0": {"net_mean": float(np.mean([r["net"] for r in rows])),
                  "net_median": float(np.median([r["net"] for r in rows])),
                  "z_mean_discovery_sigma": float(np.mean([r["net"] for r in rows]) / sigma)},
           "conditioning_columns": list(CONDITIONING),
           "q1": q1(arm, rows, tb, sigma, direction),
           "q1_single_category": q1(arm, rows, tb, sigma, direction, single_only=True),
           "q3": q3(arm, rows, tb, sigma, detail),
           "q3_single_category": q3(arm, rows, tb, sigma, detail, single_only=True),
           "q5": q5(rows, tb, detail),
           "q7": q7(arm, rows, tb, sigma)}
    for cat, c in res["q1"].items():
        s = res["q1_single_category"][cat]
        c["confounded_by_overlap"] = _confounded(c, s)
    if set_name == "holdout":
        res["confirmatory"] = confirmatory(rows, sigma)
    return res


def _confounded(full: dict, single: dict):
    a, b = full.get("effect_sigma"), single.get("effect_sigma")
    if a is None or b is None or single["n_sessions"] < 5:
        return None
    return bool((a > 0) != (b > 0) or abs(b) < 0.5 * abs(a))


GATE2_LINE = "GATE 2 VERDICT: PASS"


def holdout_allowed() -> bool:
    """The holdout is read once, after the discovery report is committed and gate 2 has
    passed. The pass is a whole line in the AUDITOR's file, audit_results.md, never a token
    inside the report (the first version matched a substring of results.md and was open
    while the report described the lock, found at gate 2)."""
    p = OUT / "audit_results.md"
    if not p.exists():
        return False
    lines = [ln.strip() for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]
    return bool(lines) and lines[-1] == GATE2_LINE            # the LAST verdict line, nothing earlier


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--arm", choices=["shares", "options"])
    ap.add_argument("--set", choices=["discovery", "holdout"], default="discovery")
    ap.add_argument("--q6", action="store_true", help="options minus shares from both discovery JSONs")
    a = ap.parse_args(argv)
    if a.q6:
        o = json.loads((OUT / "analysis_options_discovery.json").read_text(encoding="utf-8"))
        s = json.loads((OUT / "analysis_shares_discovery.json").read_text(encoding="utf-8"))
        out = {"generated": dt.datetime.now().isoformat(timespec="seconds"), "q6": q6(o, s)}
        (OUT / "analysis_q6_discovery.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
        for cat, v in out["q6"].items():
            print(f"{cat:<22} options {v['options_effect']:+.3f}  shares {v['shares_effect']:+.3f}  "
                  f"diff {v['difference']:+.3f}  divergence={v['divergence']}")
        return 0
    if not a.arm:
        ap.error("--arm is required")
    if a.set == "holdout" and not holdout_allowed():
        print(f"REFUSING: the holdout is read once, after research/news_arms/audit_results.md carries "
              f"the line '{GATE2_LINE}' for the discovery report (preregistration section 2, 11).")
        return 2
    res = run(a.arm, a.set)
    out = OUT / f"analysis_{a.arm}_{a.set}.json"
    out.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print_summary(res)
    print(f"written: {out}")
    return 0


def print_summary(res: dict) -> None:
    ex = res["exclusions"]
    print(f"{res['arm']} {res['set']}: {res['n_sessions']} sessions, {res['n_trades']} trades, "
          f"sigma(discovery) ${res['sigma_discovery_usd']:.0f}/day, B0 mean ${res['b0']['net_mean']:+.0f}/day; "
          f"half days {len(ex['half_days'])}, LOST regular sessions {ex['n_lost']} {ex['lost_regular_sessions'][:30]}")
    print(f"{'cell':<22}{'n':>5}{'trades':>7}  {'effect σ':>9}  {'95% CI':>17}  {'$/day':>8}  placebo rank  status")
    tables = [("Q1", res["q1"]), ("Q3", {k: v for k, v in res["q3"].items() if k != "no_event"})]
    for name, table in tables:
        for k, c in table.items():
            if c["effect_sigma"] is None:
                print(f"{k:<22}{c['n_sessions']:>5}{c['n_trades']:>7}  (no sessions)")
                continue
            pl = c.get("placebo", {})
            status = ("tested" if c["tested"] else "descriptive+placebo" if c["descriptive_placebo"]
                      else "never tested" if c["never_tested"] else "reported only")
            if c.get("bh_q10"):
                status += ", BH pass"
            if c.get("confounded_by_overlap"):
                status += ", confounded by overlap"
            if c.get("unstable"):
                status += ", UNSTABLE"
            rank = f"{pl['rank']:.3f} {pl['verdict']}" if pl else "-"
            ci = (f"[{c['ci95'][0]:+.3f}, {c['ci95'][1]:+.3f}]" if c['ci95'][0] is not None
                  else f"(n < {MIN_CI_SESSIONS}, no CI)")
            print(f"{k:<22}{c['n_sessions']:>5}{c['n_trades']:>7}  {c['effect_sigma']:>+9.3f}  "
                  f"{ci:>17}  {c['effect_native']:>+8.0f}  {rank:<24}  {status}")
    q = res["q7"]
    print(f"Q7: {q['m_tested']} cells tested, BH q=0.10 passes {len(q['bh_pass'])} {q['bh_pass']}, "
          f"of which |effect| >= 0.25 sigma: {len(q['bh_pass_and_quarter_sigma'])}; shortlist {q['shortlist']}")
    f = res["q5"]["first60"]
    if "difference_usd" in f:
        print(f"Q5 first-60 (descriptive): scheduled {f['scheduled_today']['first60_net_mean_usd']:+.0f}/session vs "
              f"non-release {f['non_release']['first60_net_mean_usd']:+.0f}; diff {f['difference_usd']:+.0f} "
              f"[{f['ci95'][0]:+.0f}, {f['ci95'][1]:+.0f}]")
    if "confirmatory" in res:
        c = res["confirmatory"]
        print(f"C: effect {c['effect_sigma']:+.3f} [{c['ci95'][0]:+.3f}, {c['ci95'][1]:+.3f}] "
              f"MDE {c['mde_80pct']:.3f} pass_rule={c['pass_rule']}")


if __name__ == "__main__":
    raise SystemExit(main())
