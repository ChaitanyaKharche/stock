"""Backtest the close-based range-expansion veto over the frozen shares family.

    # once, if you have not already -- produces live_lab_data/sharewf_trades.json
    python -m trade_analysis.live_lab.sharewf

    # then
    python -m trade_analysis.live_lab.orb_veto_backtest
    python -m trade_analysis.live_lab.orb_veto_backtest --classify-only   # base rate only

Frozen by `research/orb_veto_preregistration.md`. This computes §6 and nothing else.

WHAT IS BEING MEASURED
----------------------
The veto is not a fourteenth setup. It adds no parameter to any setup, changes no entry
logic, and cannot invent a trade -- it can only remove trades the frozen thirteen already
took. So no replay is needed: the trades exist in `sharewf_trades.json`, and this joins
them to a per-day classification and takes a contrast.

THE RESTRICTION THAT DEFINES THE DESIGN
---------------------------------------
Only trades entered AFTER T=11:00 may enter the contrast. A veto decided at 11:00 cannot
suppress a 09:36 entry, and crediting it with that P&L would measure a filter nobody can
implement. Days with no post-T trades are RETAINED with D=0 rather than dropped --
dropping them conditions the sample on the family having fired, which is itself correlated
with expansion, and that selection effect would make any veto look good.

THE CHECK MOST LIKELY TO KILL IT
--------------------------------
IntradayMomentumBoundary carries 73.9% of its entire P&L in the top 1% of trades; exclude
them and p=0.1764. Every apparent edge in this project has lived in its tail. So §6.3 is
reported BEFORE any verdict: if the veto removes a material part of the top-1% P&L it is
harmful whatever the mean contrast says, and the finding is that the trader identifies
low-range days correctly and low-range days are not where the losses are.

Requires the feed only for bars, and `levels_test.get_ext` caches those in
live_lab_data/ext_cache -- so the second run is offline.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import random
from collections import defaultdict
from pathlib import Path

from .orb_veto import DECIDE_AT, classify_day, touch_based_for_comparison

LAB = Path(__file__).resolve().parents[2] / "live_lab_data"
TRADES = LAB / "sharewf_trades.json"
OUT = LAB / "orb_veto_results.json"
REPS = 3000
SEED = 20260828


# ------------------------------------------------------------------------ inference

def boot_diff(a_by_day: dict, b_by_day: dict, reps=REPS, seed=SEED):
    """Two-sample day-clustered bootstrap on the DIFFERENCE of means.

    `sharewf.boot` is one-sample; a difference needs the two groups resampled
    independently, each over its own days. Days are the unit because 13 setups on one
    symbol collapse into one cluster -- treating trades as independent would inflate the
    effective sample and manufacture significance, which is the error this repo already
    corrected once at ~300x in the VRP arm.

    Returns (mean_diff, lo, hi, p_two_sided).
    """
    ka, kb = list(a_by_day), list(b_by_day)
    if len(ka) < 8 or len(kb) < 8:
        return (None, None, None, None)
    rng = random.Random(seed)
    obs = (sum(a_by_day.values()) / len(ka)) - (sum(b_by_day.values()) / len(kb))
    diffs = []
    for _ in range(reps):
        sa = sum(a_by_day[ka[rng.randrange(len(ka))]] for _ in ka) / len(ka)
        sb = sum(b_by_day[kb[rng.randrange(len(kb))]] for _ in kb) / len(kb)
        diffs.append(sa - sb)
    diffs.sort()
    neg = sum(1 for x in diffs if x <= 0) / len(diffs)
    pos = sum(1 for x in diffs if x >= 0) / len(diffs)
    return (obs, diffs[int(.025 * len(diffs))], diffs[int(.975 * len(diffs))],
            max(2 * min(neg, pos), 1 / reps))


# -------------------------------------------------------------------------- analysis

def analyse(trades: list[dict], vetoed: dict[str, bool],
            decide_at: dt.time = DECIDE_AT) -> dict:
    """The whole of §6, as a pure function. No I/O, so it is testable without an archive.

    `trades`  rows from sharewf_trades.json: day, setup, entry_ts, pnl
    `vetoed`  {"YYYY-MM-DD": True if V-DECIDE fired}. Days absent are UNCLASSIFIED and
              excluded from everything -- a day we could not classify must not be silently
              counted as tradeable.
    """
    post: dict[str, float] = {d: 0.0 for d in vetoed}          # §4: retained at 0.0
    all_pnl: dict[str, float] = {d: 0.0 for d in vetoed}
    post_rows, top_rows = [], []
    for t in trades:
        d = t["day"]
        if d not in vetoed:
            continue
        all_pnl[d] += t["pnl"]
        top_rows.append(t)
        if dt.time.fromisoformat(t["entry_ts"][11:19]) > decide_at:
            post[d] += t["pnl"]
            post_rows.append(t)

    a = {d: v for d, v in post.items() if vetoed[d]}           # veto days
    b = {d: v for d, v in post.items() if not vetoed[d]}       # tradeable days
    mean_diff, lo, hi, p = boot_diff(a, b)

    # --- 6.2 does suppressing post-T entries on veto days actually help? ---
    unfiltered = sum(all_pnl.values())
    removed = sum(a.values())
    filtered = unfiltered - removed

    by_year = defaultdict(lambda: [0.0, 0.0])                  # [unfiltered, filtered]
    for d, v in all_pnl.items():
        y = d[:4]
        by_year[y][0] += v
        by_year[y][1] += v - (post[d] if vetoed[d] else 0.0)
    # STRICTLY better. A year the filter leaves unchanged (it vetoed nothing, or vetoed
    # only flat days) is not evidence for the filter, and counting ties would let a rule
    # that fires twice a decade claim 11/11.
    years_strict = sum(1 for y in by_year if by_year[y][1] > by_year[y][0])

    # drop the single best day and re-check, so one session cannot carry the result
    best_day = max(all_pnl, key=lambda k: all_pnl[k]) if all_pnl else None
    if best_day:
        u2 = unfiltered - all_pnl[best_day]
        f2 = filtered - (all_pnl[best_day] - (post[best_day] if vetoed[best_day] else 0.0))
    else:
        u2 = f2 = 0.0

    # --- 6.3 the tail check ---
    top_rows.sort(key=lambda r: r["pnl"], reverse=True)
    k = max(1, len(top_rows) // 100)
    top1 = top_rows[:k]
    top1_total = sum(r["pnl"] for r in top1)
    top1_on_veto = sum(r["pnl"] for r in top1 if vetoed[r["day"]])
    top1_n_on_veto = sum(1 for r in top1 if vetoed[r["day"]])
    top10_days = sorted(all_pnl, key=lambda d: all_pnl[d], reverse=True)[:10]
    top10_vetoed = [d for d in top10_days if vetoed[d]]

    return {
        "n_days_classified": len(vetoed),
        "n_veto_days": len(a),
        "n_tradeable_days": len(b),
        "veto_base_rate": round(len(a) / len(vetoed), 4) if vetoed else None,
        "n_trades_total": len(top_rows),
        "n_trades_post_T": len(post_rows),
        "mean_D_veto": round(sum(a.values()) / len(a), 4) if a else None,
        "mean_D_tradeable": round(sum(b.values()) / len(b), 4) if b else None,
        "mean_diff": None if mean_diff is None else round(mean_diff, 4),
        "ci95": [None if lo is None else round(lo, 4),
                 None if hi is None else round(hi, 4)],
        "p_two_sided": p,
        "total_unfiltered": round(unfiltered, 2),
        "total_filtered": round(filtered, 2),
        "pnl_removed_by_veto": round(removed, 2),
        "improves": filtered > unfiltered,
        "improves_without_best_day": f2 > u2,
        "years": {y: [round(v[0], 2), round(v[1], 2)] for y, v in sorted(by_year.items())},
        "n_years": len(by_year),
        "n_years_filtered_better": years_strict,
        "top1pct_n": k,
        "top1pct_total_pnl": round(top1_total, 2),
        "top1pct_pnl_on_veto_days": round(top1_on_veto, 2),
        "top1pct_share_on_veto_days":
            round(top1_on_veto / top1_total, 4) if top1_total else None,
        "top1pct_count_on_veto_days": top1_n_on_veto,
        "top10_days_vetoed": top10_vetoed,
    }


def verdict_text(r: dict) -> list[str]:
    """§6, spelled out. Refuses to conclude when a precondition is not met."""
    out = []
    lo, hi = r["ci95"]
    if r["mean_diff"] is None:
        return ["  INCONCLUSIVE: too few days in one arm for a clustered bootstrap."]

    primary = lo is not None and hi is not None and hi < 0
    out.append(f"  6.1 primary       CI excludes zero and is negative: "
               f"{'YES' if primary else 'NO'}")
    out.append(f"  6.2 filtered > unfiltered:                          "
               f"{'YES' if r['improves'] else 'NO'}")
    out.append(f"      survives dropping the best day:                 "
               f"{'YES' if r['improves_without_best_day'] else 'NO'}")
    out.append(f"      years filtered is better:                       "
               f"{r['n_years_filtered_better']}/{r['n_years']} (need >= 8)")

    share = r["top1pct_share_on_veto_days"]
    tail_ok = share is not None and share < 0.05
    out.append(f"  6.3 share of top-1% P&L falling on veto days:       "
               f"{'n/a' if share is None else f'{share:.1%}'}"
               f"  -> {'OK' if tail_ok else 'HARMFUL'}")
    if r["top10_days_vetoed"]:
        out.append(f"      !! top-10 days the veto would have removed: "
                   f"{', '.join(r['top10_days_vetoed'])}")

    years_ok = r["n_years_filtered_better"] >= 8
    if not tail_ok:
        out.append("\n  VERDICT: HARMFUL. The veto removes days that carry the edge. "
                   "Per §6.3 this\n           stands whatever §6.1 says.")
    elif primary and r["improves"] and r["improves_without_best_day"] and years_ok:
        out.append("\n  VERDICT: the veto SURVIVES §6. Adopting it live is a separate "
                   "decision\n           with its own freeze (§8).")
    else:
        out.append("\n  VERDICT: NULL. Report it with the same care as a positive; per "
                   "§6.4 a powered\n           null on the discretionary no-trade "
                   "decision is the useful finding here.")
    return out


# ------------------------------------------------------------------------------- io

def classify_range(symbol: str, start: dt.date, end: dt.date,
                   compare_touch: bool = True) -> tuple[dict, dict]:
    """Classify every session in the range. Returns (vetoed, detail)."""
    from .feed import ThetaLiveFeed
    from .levels_test import get_ext
    from .walkforward import sessions_between

    feed = ThetaLiveFeed()
    vetoed: dict[str, bool] = {}
    detail: dict[str, dict] = {}
    n = 0
    try:
        for day in sessions_between(start, end):
            try:
                pre, rth = get_ext(feed, symbol, day)
            except Exception as exc:                          # noqa: BLE001
                print(f"  {day}: skipped ({exc!r})", flush=True)
                continue
            if len(rth) < 300:                                # holiday or half day
                continue
            v = classify_day(pre, rth)
            if not v.usable:
                print(f"  {day}: unclassified ({v.reason})", flush=True)
                continue
            key = day.isoformat()
            vetoed[key] = v.gate()
            detail[key] = {
                "decide": v.decide, "post": v.post,
                "band": [v.band_low, v.band_high],
                "premarket": [v.premarket_low, v.premarket_high],
                "or": [v.or_low, v.or_high],
                "first_break": (v.first_break_ts.isoformat()
                                if v.first_break_ts else None),
                "first_break_dir": v.first_break_dir,
                "n_premarket_bars": len(pre),
            }
            if compare_touch:
                detail[key]["touch_decide"] = touch_based_for_comparison(pre, rth)
            n += 1
            if n % 100 == 0:
                print(f"  {n} sessions classified ...", flush=True)
    finally:
        feed.close()
    return vetoed, detail


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--symbol", default="QQQ")
    ap.add_argument("--start", default="2016-01-04")
    ap.add_argument("--end", default="2026-08-27")
    ap.add_argument("--trades", default=str(TRADES))
    ap.add_argument("--classify-only", action="store_true",
                    help="veto base rate and close-vs-touch disagreement; no P&L")
    args = ap.parse_args(argv)

    print(f"ORB VETO BACKTEST  {args.symbol}  {args.start} -> {args.end}")
    print(f"frozen by research/orb_veto_preregistration.md   T={DECIDE_AT:%H:%M} ET\n",
          flush=True)

    vetoed, detail = classify_range(args.symbol,
                                    dt.date.fromisoformat(args.start),
                                    dt.date.fromisoformat(args.end))
    if not vetoed:
        print("no sessions classified; is Theta Terminal running?")
        return 1

    n_v = sum(1 for x in vetoed.values() if x)
    print(f"\n  {len(vetoed)} sessions classified")
    print(f"  V-DECIDE fired on {n_v} ({n_v / len(vetoed):.1%})")
    n_post = sum(1 for d in detail.values() if d["post"])
    print(f"  V-POST  (attribution only) {n_post} ({n_post / len(vetoed):.1%})")

    # Was the close/touch choice material? If these never disagree, §2 was cosmetic.
    dis = [d for d, x in detail.items() if "touch_decide" in x
           and x["touch_decide"] != x["decide"]]
    print(f"  close vs touch disagree on {len(dis)} sessions "
          f"({len(dis) / len(vetoed):.1%})"
          + ("  -- the choice in §2 was material" if dis
             else "  -- §2's distinction was COSMETIC on this sample"))

    payload = {"symbol": args.symbol, "start": args.start, "end": args.end,
               "decide_at": DECIDE_AT.isoformat(), "detail": detail}

    if not args.classify_only:
        p = Path(args.trades)
        if not p.exists():
            print(f"\n{p} not found. Run first:\n"
                  f"  python -m trade_analysis.live_lab.sharewf")
            OUT.write_text(json.dumps(payload, indent=1), encoding="utf-8")
            return 1
        trades = json.loads(p.read_text(encoding="utf-8"))
        res = analyse(trades, vetoed)
        payload["results"] = res

        print("\n" + "=" * 78)
        print(f"  days   veto {res['n_veto_days']}   tradeable {res['n_tradeable_days']}")
        print(f"  trades {res['n_trades_total']} total, "
              f"{res['n_trades_post_T']} entered after {DECIDE_AT:%H:%M}")
        print(f"  mean D per day    veto ${res['mean_D_veto']}   "
              f"tradeable ${res['mean_D_tradeable']}")
        lo, hi = res["ci95"]
        print(f"  difference        ${res['mean_diff']}   "
              f"95% CI [{lo}, {hi}]   p={res['p_two_sided']}")
        print(f"  family total      unfiltered ${res['total_unfiltered']:,.0f}   "
              f"filtered ${res['total_filtered']:,.0f}")
        print("=" * 78)
        for line in verdict_text(res):
            print(line)

    OUT.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    print(f"\n  wrote {OUT}")
    print("\n  V-POST is attribution only. It is never an achievable return (§1).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
