"""Join the replayed arms to the event files, and the pre-P&L counts the pre-registration
requires before any result is read (research/news_arms/preregistration.md 13.4, 14.7).

    python -m trade_analysis.news_arms.join --counts            # overlap matrix + m, NO P&L read
    python -m trade_analysis.news_arms.join --write             # join.csv for analyze (P&L included)

Discovery shares = 2018-01-02 .. 2022-12-30 (17.1). Holdout and forward are never touched
here. Replay chunks live under live_lab_backfill/news_arms/<year>/<arm>/[<YYYY-MM>/]; a
session appearing in two chunks is an error (17.2).
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import statistics as st
from collections import Counter, defaultdict
from pathlib import Path

from . import events as E

ROOT = Path(__file__).resolve().parents[2]
BF = ROOT / "live_lab_backfill" / "news_arms"
NI = ROOT / "research" / "news_impact"
OUT = ROOT / "research" / "news_arms"
DISCOVERY = {"shares": (dt.date(2018, 1, 2), dt.date(2022, 12, 30)),
             "options": (dt.date(2020, 1, 2), dt.date(2022, 12, 30))}
HOLDOUT = {"shares": (dt.date(2023, 1, 3), dt.date(2026, 8, 27)),
           "options": (dt.date(2023, 1, 3), dt.date(2023, 12, 29))}
SCHEDULED = ("FOMC", "CPI", "NFP", "GDP_ADV", "PCE")
GEO = ("war_military", "trade_tariff", "financial_stress", "us_fiscal", "pandemic_health",
       "political_shock", "energy_shock", "cyber_infrastructure")
CATEGORIES = SCHEDULED + GEO
NEVER_TESTED = {"financial_stress"}
MIN_SESSIONS, MIN_TRADES = 30, 50          # section 7
LIVE_HASH = {"options": "ccd00ac71f8243e9", "shares": "7c6414fc549315f3"}


def _rows(p: Path) -> list[dict]:
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


# ------------------------------------------------------------------ replay chunks

def chunk_dirs(arm: str, years) -> list[Path]:
    out = []
    for y in years:
        base = BF / str(y) / arm
        if not base.exists():
            continue
        if (base / "RUN.json").exists():
            out.append(base)
        out += sorted(p for p in base.iterdir() if p.is_dir() and (p / "RUN.json").exists())
    return out


def load_replay(arm: str, years, base_dirs=None) -> tuple[dict, list[dict], dict]:
    """-> (sessions: date -> daily dict, trades: list, meta). Raises on a duplicated
    session or on a month marked FAILED.

    Options (14.11): a session with no 0DTE expiry (`sessions.json` written by the replay,
    has_0dte False) is dropped here, before any table, baseline or sigma sees it, and
    counted in meta["excluded_no_0dte"] by year and weekday. Before 2022-11-14 SPY had 0DTE
    on Mon/Wed/Fri only; carrying Tue/Thu as $0 days would shrink sigma and hand GDP_ADV
    (a Thursday release) an effect by construction. A chunk without sessions.json refuses
    to join."""
    sessions, trades, meta = {}, [], {"chunks": [], "hashes": set(),
                                      "excluded_no_0dte": {"n": 0, "by_year": {}, "by_weekday": {}, "dates": []}}
    dropped = set()
    for d in (base_dirs if base_dirs is not None else chunk_dirs(arm, years)):
        if (d / "FAILED.json").exists():
            raise RuntimeError(f"{d} is marked FAILED; rerun it before joining")
        run = json.loads((d / "RUN.json").read_text(encoding="utf-8"))
        rel = str(d.relative_to(ROOT)) if d.is_relative_to(ROOT) else str(d)
        meta["chunks"].append({"dir": rel, **{k: run.get(k) for k in
                                                                    ("start", "end", "repo_sha", "sessions_replayed", "skipped", "dates")}})
        has_0dte = None
        if arm == "options":
            sp = d / "sessions.json"
            if not sp.exists():
                raise RuntimeError(f"{d} has no sessions.json (0DTE map); rerun the chunk before joining")
            has_0dte = json.loads(sp.read_text(encoding="utf-8"))
        for p in sorted((d / "daily").glob("*.json")):
            day = p.stem
            if day in sessions or day in dropped:
                raise RuntimeError(f"session {day} appears in two chunks ({d})")
            if has_0dte is not None and not has_0dte.get(day, False):
                dropped.add(day)
                ex = meta["excluded_no_0dte"]
                ex["n"] += 1
                ex["dates"].append(day)
                ex["by_year"][day[:4]] = ex["by_year"].get(day[:4], 0) + 1
                wd = str(dt.date.fromisoformat(day).weekday())
                ex["by_weekday"][wd] = ex["by_weekday"].get(wd, 0) + 1
                continue
            sessions[day] = json.loads(p.read_text(encoding="utf-8"))
            meta["hashes"].add(sessions[day].get("config_hash"))
        trades += [t for t in _rows(d / "trades.jsonl") if t.get("entry_ts", "")[:10] not in dropped]
    meta["hashes"] = sorted(h for h in meta["hashes"] if h)
    return sessions, trades, meta


# ------------------------------------------------------------------ events and regimes

def load_events() -> dict:
    """date -> {"types": set, "minutes": {type: event_minute or None}}"""
    ev = defaultdict(lambda: {"types": set(), "minutes": {}})
    with open(NI / "events_scheduled.csv", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            ev[r["session"]]["types"].add(r["type"])
            ev[r["session"]]["minutes"][r["type"]] = (r["time_et"] if r["session"] == r["date"]
                                                       and "09:30" <= r["time_et"] < "16:00" else None)
    for r in E.load_geopolitical():
        ev[r["date"]]["types"].add(r["category"])
        ev[r["date"]]["minutes"][r["category"]] = E.event_minute(r)
    return ev


def regime_cuts(features_csv: Path, start: dt.date, end: dt.date) -> tuple[float, float]:
    """Tercile cut-points of med20_range_bp over the DISCOVERY sessions only (section 2)."""
    vals = []
    with open(features_csv, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            d = dt.date.fromisoformat(r["date"])
            if start <= d <= end and r.get("med20_range_bp") and r.get("thin") != "True":
                vals.append(float(r["med20_range_bp"]))
    vals.sort()
    n = len(vals)
    return vals[n // 3], vals[(2 * n) // 3]


def regimes(features_csv: Path, cuts) -> dict:
    out = {}
    lo, hi = cuts
    with open(features_csv, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r.get("med20_range_bp"):
                v = float(r["med20_range_bp"])
                out[r["date"]] = "low" if v < lo else "high" if v >= hi else "mid"
    return out


# ------------------------------------------------------------------ the join

def build_join(arm: str, sessions: dict, trades: list[dict], events: dict, regime: dict,
               window) -> list[dict]:
    """One row per replayed session inside `window`: categories, regime, counts, net,
    per-setup net. Net is carried for the --write step only; --counts never reads it."""
    by_day = defaultdict(list)
    for t in trades:
        if t.get("pnl_net") is None:
            continue
        if arm == "options" and t.get("arm") != "ATM":
            continue
        by_day[t["entry_ts"][:10]].append(t)
    rows = []
    for day, d in sorted(sessions.items()):
        dd = dt.date.fromisoformat(day)
        if not (window[0] <= dd <= window[1]):
            continue
        ts = by_day.get(day, [])
        cats = sorted(events.get(day, {}).get("types", set()) & set(CATEGORIES))
        rows.append({"date": day, "arm": arm, "year": day[:4], "weekday": dd.weekday(),
                     "regime": regime.get(day), "categories": cats,
                     "n_categories": len(cats), "scheduled_today": bool(set(cats) & set(SCHEDULED)),
                     "any_event": bool(cats), "n_trades": len(ts),
                     "net": round(sum(t["pnl_net"] for t in ts), 2),
                     "setup_net": {s: round(sum(t["pnl_net"] for t in ts if t["setup_id"] == s), 2)
                                   for s in sorted({t["setup_id"] for t in ts})},
                     "setup_n": dict(Counter(t["setup_id"] for t in ts)),
                     "config_hash": d.get("config_hash")})
    return rows


# ------------------------------------------------------------------ pre-P&L counts (14.7, 13.4)

def overlap_matrix(rows: list[dict]) -> dict:
    m = {a: {b: 0 for b in CATEGORIES} for a in CATEGORIES}
    for r in rows:
        for a in r["categories"]:
            for b in r["categories"]:
                m[a][b] += 1
    return m


def counts(rows: list[dict]) -> dict:
    """Everything the pre-registration wants committed BEFORE P&L: sessions and trades per
    category, overlap, single-category counts, and m = cells meeting both minimums."""
    per = {}
    for c in CATEGORIES:
        sess = [r for r in rows if c in r["categories"]]
        single = [r for r in sess if r["n_categories"] == 1]
        meets = len(sess) >= MIN_SESSIONS and sum(r["n_trades"] for r in sess) >= MIN_TRADES
        per[c] = {"sessions": len(sess), "trades": sum(r["n_trades"] for r in sess),
                  "single_category_sessions": len(single), "meets_minimums": meets,
                  # A1: geopolitical categories are descriptive with a placebo, never tested
                  "tested": c in SCHEDULED and meets,
                  "descriptive_placebo": c in GEO and c not in NEVER_TESTED and meets,
                  "never_tested": c in NEVER_TESTED}
    setups = sorted({s for r in rows for s in r["setup_n"]})
    q7 = {}
    for c in CATEGORIES:
        if c in NEVER_TESTED:
            continue
        sess = [r for r in rows if c in r["categories"]]
        for s in setups:
            n_tr = sum(r["setup_n"].get(s, 0) for r in sess)
            q7[f"{s}|{c}"] = {"sessions": len(sess), "trades": n_tr,
                              "tested": len(sess) >= MIN_SESSIONS and n_tr >= MIN_TRADES}
    return {
        "n_sessions": len(rows),
        "n_scheduled_sessions": sum(1 for r in rows if r["scheduled_today"]),
        "n_no_scheduled_control": sum(1 for r in rows if not r["scheduled_today"]),
        "n_no_event_any_type": sum(1 for r in rows if not r["any_event"]),
        "n_multi_category_sessions": sum(1 for r in rows if r["n_categories"] > 1),
        "by_year": dict(Counter(r["year"] for r in rows)),
        "by_regime": dict(Counter(r["regime"] or "none" for r in rows)),
        "per_category": per,
        "m_H2_scheduled_types": sum(1 for c in ("CPI", "NFP", "GDP_ADV", "PCE") if per[c]["tested"]),
        "m_H3_scheduled_types": sum(1 for c in SCHEDULED if per[c]["tested"]),
        "m_Q7_setup_x_category": sum(1 for v in q7.values() if v["tested"]),
        "q7_cells": q7,
        "overlap_matrix": overlap_matrix(rows),
        "setups": setups,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--arm", choices=["shares", "options"], default="shares")
    ap.add_argument("--set", choices=["discovery", "holdout"], default="discovery")
    ap.add_argument("--counts", action="store_true", help="write counts/overlap/m only; read no P&L")
    ap.add_argument("--write", action="store_true", help="write join_<arm>_<set>.json with net")
    a = ap.parse_args(argv)
    window = (DISCOVERY if a.set == "discovery" else HOLDOUT)[a.arm]
    years = range(window[0].year, window[1].year + 1)
    sessions, trades, meta = load_replay(a.arm, years)
    events = load_events()
    cuts = regime_cuts(NI / "day_features_QQQ.csv", *DISCOVERY["shares"])
    reg = regimes(NI / "day_features_QQQ.csv", cuts)
    rows = build_join(a.arm, sessions, trades, events, reg, window)
    stamp = {"arm": a.arm, "set": a.set, "window": [window[0].isoformat(), window[1].isoformat()],
             "regime_cuts_med20_range_bp": list(cuts), "chunks": meta["chunks"],
             "config_hashes": meta["hashes"], "live_hash": LIVE_HASH[a.arm],
             "excluded_no_0dte": meta["excluded_no_0dte"],
             "generated": dt.datetime.now().isoformat(timespec="seconds")}
    OUT.mkdir(parents=True, exist_ok=True)
    if a.counts:
        c = counts(rows)
        (OUT / f"counts_{a.arm}_{a.set}.json").write_text(json.dumps({**stamp, **c}, indent=1), encoding="utf-8")
        print(f"{a.arm} {a.set}: {c['n_sessions']} sessions, {c['n_scheduled_sessions']} with a scheduled release, "
              f"{c['n_no_event_any_type']} with no event of any type, {c['n_multi_category_sessions']} multi-category")
        print(f"  regime cuts (med20 range bp): {cuts[0]:.1f} / {cuts[1]:.1f}; by regime {c['by_regime']}")
        print(f"  m: H2 scheduled types tested {c['m_H2_scheduled_types']}/4, H3 {c['m_H3_scheduled_types']}/5, "
              f"Q7 setup x category cells tested {c['m_Q7_setup_x_category']}")
        for cat, v in c["per_category"].items():
            flag = ("never tested" if v["never_tested"] else "tested" if v["tested"]
                    else "descriptive + placebo" if v["descriptive_placebo"] else "reported only")
            print(f"    {cat:<22} sessions {v['sessions']:>4}  trades {v['trades']:>6}  single-cat {v['single_category_sessions']:>4}  {flag}")
        print(f"written: {OUT / f'counts_{a.arm}_{a.set}.json'}  (no P&L in this file)")
    if a.write:
        (OUT / f"join_{a.arm}_{a.set}.json").write_text(json.dumps({**stamp, "rows": rows}, indent=0), encoding="utf-8")
        print(f"written: {OUT / f'join_{a.arm}_{a.set}.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
