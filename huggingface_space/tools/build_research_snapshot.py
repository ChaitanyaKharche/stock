"""Build research_snapshot.json -- the frozen research numbers the Space displays.

    python huggingface_space/tools/build_research_snapshot.py      (from the repo root)

The Space must build with nothing from the parent repository, and it must never import
the live lab (nothing done to make a demo presentable may reach back into a frozen forward
test). So the research tabs read ONE json file, produced here from the repo's own data,
with every number's source file recorded next to it. Rerun and re-upload to refresh.

Needs pandas + pyarrow (for the cost-model parquet); the Space itself needs neither.
"""
from __future__ import annotations

import csv
import datetime as dt
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "huggingface_space" / "research_snapshot.json"
LAB = ROOT / "live_lab_data"


def _jsonl(p: Path):
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


# ----------------------------------------------------------------- fill-price explorer

def fill_grid():
    """Short ATM 0DTE straddle P&L as the fill moves from the MID to the far side.

    net_i(f) = gross_i - f * spread_cost_i. f = 0 is the mid-fill assumption most
    backtests make; f = 1 pays the whole measured spread. t is computed exactly as
    research/vrp_cost_model_results.md does -- on each session's MEAN trade -- which
    reproduces its +7.52 and -5.38 to the cent (checked when this was written).
    """
    import pandas as pd
    d = pd.read_parquet(ROOT / "data" / "cost_model_trades.parquet")
    out = []
    for i in range(0, 21):
        f = i / 20
        x = d["gross"] - f * d["spread_cost"]
        mu = x.groupby(d["date"]).mean()
        t = mu.mean() / (mu.std(ddof=1) / len(mu) ** 0.5)
        out.append({"f": f, "mean": round(float(x.mean()), 5), "t": round(float(t), 2),
                    "win": round(float((x > 0).mean()), 4)})
    return {"n_trades": int(len(d)), "n_sessions": int(d["date"].nunique()),
            "span": [str(d["date"].min()), str(d["date"].max())],
            "grid": out, "source": "data/cost_model_trades.parquet; "
                                   "research/vrp_cost_model_results.md"}


# ----------------------------------------------------------------- forward test

def _per_setup(rows, bar: dict):
    """Status uses THAT ARM's pre-registered bar. The shares arm also needs min_sessions:
    15 correlated names reach 200 raw trades in weeks, and a raw count that crosses 200
    must not read as evidence (shares/FREEZE.json, why_min_sessions_was_added)."""
    by, days = defaultdict(list), defaultdict(set)
    for r in rows:
        by[r["setup_id"]].append(r["pnl_net"])
        days[r["setup_id"]].add(r["entry_ts"][:10])
    min_n, min_s = bar.get("min_n", 200), bar.get("min_sessions", 0)
    out = []
    for k, v in sorted(by.items(), key=lambda kv: -len(kv[1])):
        ok = len(v) >= min_n and len(days[k]) >= min_s
        out.append({"setup": k, "trades": len(v), "sessions": len(days[k]),
                    "total $": round(sum(v), 2), "mean $": round(st.mean(v), 2),
                    "median $": round(st.median(v), 2),
                    "win": f"{100 * sum(x > 0 for x in v) / len(v):.0f}%",
                    "status": "ready for the bar" if ok else "INSUFFICIENT"})
    return out


def forward_test():
    fz = json.loads((LAB / "FREEZE.json").read_text(encoding="utf-8"))
    acc = set(fz["accepted_config_hashes"])
    opt = [r for r in _jsonl(LAB / "trades.jsonl")
           if r.get("arm") == "ATM" and r.get("config_hash") in acc]
    sfz = json.loads((LAB / "shares" / "FREEZE.json").read_text(encoding="utf-8"))
    sacc = set(sfz["accepted_config_hashes"])
    sh = [r for r in _jsonl(LAB / "shares" / "trades.jsonl") if r.get("config_hash") in sacc]
    days = defaultdict(lambda: {"options": 0.0, "shares": 0.0})
    for r in opt:
        days[r["entry_ts"][:10]]["options"] += r["pnl_net"]
    for r in sh:
        days[r["entry_ts"][:10]]["shares"] += r["pnl_net"]
    return {
        "start": fz["start_date"], "through": max(days) if days else None,
        "sessions_options": len({r["entry_ts"][:10] for r in opt}),
        "sessions_shares": len({r["entry_ts"][:10] for r in sh}),
        "options_atm": _per_setup(opt, fz.get("promotion_bar") or {}),
        "shares": _per_setup(sh, sfz.get("promotion_bar") or {}),
        "daily": [{"day": d, "options": round(v["options"], 2),
                   "shares": round(v["shares"], 2)} for d, v in sorted(days.items())],
        "promotion_bar": fz.get("promotion_bar"),
        "note": ("Every session from 2026-08-31 to 2026-09-24 started after the open, so "
                 "opening setups were not tested as defined "
                 "(research/incident_2026-09-24_opening_window.md). Fixed 2026-09-25."),
        "source": "live_lab_data/trades.jsonl, live_lab_data/shares/trades.jsonl, FREEZE.json",
    }


def backfill():
    p = ROOT / "live_lab_backfill" / "options" / "report.json"
    if not p.exists():
        return None
    rep = json.loads(p.read_text(encoding="utf-8"))
    meta = json.loads((p.parent / "BACKFILL.json").read_text(encoding="utf-8"))
    return {"days": rep["days"], "by_setup": rep["by_setup"],
            "six_trades": rep["six_trades"], "config_hash": meta["config_hash"],
            "differences_from_live": meta["differences_from_live"],
            "source": "live_lab_backfill/options/ (trade_analysis/live_lab/backfill.py)"}


# ----------------------------------------------------------------- day clusters

def day_clusters():
    fig = ROOT / "research" / "figures"
    if not (fig / "day_clusters.csv").exists():
        return None
    with open(fig / "day_clusters.csv", encoding="utf-8") as fh:
        pts = [{"day": r["day"], "oc": round(float(r["oc_pct"]), 3),
                "range": round(float(r["range_pct"]), 3), "cluster": int(r["cluster"])}
               for r in csv.DictReader(fh)]
    with open(fig / "day_clusters_journal_summary.csv", encoding="utf-8") as fh:
        summ = [{"cluster": int(r["type"]), "name": r["name"], "looks_like": r["looks_like"],
                 "sessions": int(r["sessions_all"]), "traded_days": int(r["traded_days"]),
                 "total": round(float(r["total"]), 2),
                 "median_per_day": round(float(r["median_per_day"]), 2),
                 "ci": [round(float(r["mean_ci_lo"]), 1), round(float(r["mean_ci_hi"]), 1)]}
                for r in csv.DictReader(fh)]
    return {"points": pts, "summary": summ, "k": 3, "silhouette": 0.236,
            "stability": {"seeds_ari_min": 0.996, "bootstrap_ari": [0.917, 0.960],
                          "fit_2016_20_apply_2021_26_ari": 0.830},
            "caveat": ("Descriptive, not evidence: a day's type is only known at the close, "
                       "every CI crosses zero, and the journal has ~155 traded days."),
            "source": "research/figures/day_clusters*.csv (k-means on QQQ 2016-2026)"}


# ----------------------------------------------------------------- the static record

HEADLINES = [
    {"title": "Predicting intraday direction is closed",
     "plain": "Eight pre-registered tests and a 2.8-million-configuration sweep found no "
              "edge in calling QQQ/SPY direction intraday.",
     "number": "SPA p = 0.82 over 2,822,400 configurations",
     "source": "research/PROJECT_REPORT.md §1; research/setup_sweep_results.md"},
    {"title": "A backtest priced at the mid can have the wrong sign",
     "plain": "Selling 0DTE straddles looks like a strong edge at mid prices and loses "
              "money at the prices you can actually trade at.",
     "number": "t = +7.52 at mid, −5.38 at bid/ask, same 37,517 trades",
     "source": "research/vrp_cost_model_results.md"},
    {"title": "One survivor, and it trades shares, not options",
     "plain": "A published intraday-momentum rule held up on 10 years of QQQ, but most of "
              "its profit comes from a handful of days.",
     "number": "+$3.34/trade, 2,905 trades, Holm p = 0.0043; 74% of P&L from the top 1%",
     "source": "research/PROJECT_REPORT.md §1"},
    {"title": "Capping winners lost money every time it was measured",
     "plain": "A profit target raises the win rate and lowers the money, on three "
              "different datasets.",
     "number": "+25% target: 71.6% win rate, breakeven needs 75.3%; −1.17 bp and −0.72 bp "
               "on two breakout rules",
     "source": "CLAUDE.md; research/six_lines_results.md §3"},
    {"title": "The big model was a constant",
     "plain": "A 398,854-parameter TFT gave the same answer in a crash and in a flat "
              "market. It was replaced by a six-parameter volatility model that responds.",
     "number": "crash vs flat chop: 0.1 points apart on a 0–100 scale",
     "source": "research/PROJECT_REPORT.md §3.3"},
]

SILENT_BUGS = [
    ("backtest", "a 5-min bar's label used as its fill time", "88–95.7% of a measured edge was 1 minute of hindsight"),
    ("backtest", "zero-filled holiday closes -> yesterday_high = 0", "~7% of entries were fabrications"),
    ("demo app", "confidence read a key that layer never sets", "every ticker returned 15%"),
    ("demo app", "signal converter hardcoded 'CALLS'", "PUTS was unreachable; a selloff read as bullish"),
    ("demo app", "gate set above the attainable range", "fired on 0 of 80 observations"),
    ("HPC", "halt bar with all-zero prices -> log(0)", "dropna silently removed 193 of 769 sessions"),
    ("HPC", "exp(X·β) without the exp(s²/2) correction", "turned p = 0.583 into a false p = 0.023"),
    ("live lab", "no stale-bar guard", "21% of trades were signals up to 325 minutes old"),
    ("live lab", "clock sampled before two network calls", "preflight passed future-dated quotes"),
    ("live lab", "runners held until a post-open check passed",
     "both arms blind to the open for 18 sessions; Crabel fired on 11 names in one minute"),
]

TFT_REGIMES = {"columns": ["QQQ", "NVDA", "MSFT", "META"],
               "rows": {"calm uptrend": [68.4, 67.0, 69.7, 63.1],
                        "violent crash": [68.3, 66.9, 69.7, 63.1],
                        "flat chop": [68.4, 67.0, 69.7, 63.1]},
               "source": "research/PROJECT_REPORT.md §3.3"}


def main() -> int:
    snap = {"generated_at": dt.datetime.now().isoformat(timespec="seconds"),
            "headlines": HEADLINES,
            "silent_bugs": [{"where": a, "defect": b, "cost": c} for a, b, c in SILENT_BUGS],
            "tft_regimes": TFT_REGIMES,
            "fill_explorer": fill_grid(), "forward_test": forward_test(),
            "backfill": backfill(), "day_clusters": day_clusters()}
    OUT.write_text(json.dumps(snap, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {OUT} ({OUT.stat().st_size / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
