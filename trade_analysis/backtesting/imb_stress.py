"""Stress test IntradayMomentumBoundary on QQQ shares before anything gets built on it.

IMB is the only result in this project that has ever cleared a multiplicity correction on a
large sample and survived honest execution timing: 2,905 trades, +$3.34/trade, Holm 0.0043.
That is exactly the kind of number that has been wrong here twice, so it gets the same
treatment everything else got.

Tests, all decided before looking at any of them:
  1  tail dependence          does a handful of trades carry it?
  2  per-year stability       and what happens if the best year is removed
  3  split-half               first half vs second half, independent windows
  4  direction                long and short separately
  5  equity curve             max drawdown, longest flat stretch, time under water
  6  slippage sensitivity     how much extra cost kills it
  7  time-of-day              is it one checkpoint carrying everything?
  8  exit mix and hold time   what it actually does
  9  day concentration        one huge day, or broad?
 10  peer comparison          is +$3.34 special against the other 12?

Nothing here re-fits or re-tunes anything. It only interrogates trades already recorded by
trade_analysis/live_lab/sharewf.py.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import sys
from collections import defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

TRADES = os.path.join(ROOT, "live_lab_data", "sharewf_trades.json")
SETUP = "IntradayMomentumBoundary"
NOTIONAL = 10_000.0
SEED = 20260829


def boot(vals, days, nboot=10000, seed=SEED):
    """Day-clustered bootstrap, vectorised over per-day sums and counts."""
    vals = np.asarray(vals, float)
    _, inv = np.unique(np.asarray(days), return_inverse=True)
    D = inv.max() + 1
    S = np.bincount(inv, weights=vals, minlength=D)
    N = np.bincount(inv, minlength=D).astype(float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, D, size=(nboot, D))
    ms = np.sort(S[idx].sum(1) / N[idx].sum(1))
    p = max(2 * min((ms <= 0).mean(), (ms >= 0).mean()), 1.0 / nboot)
    return float(ms[int(.025 * nboot)]), float(ms[int(.975 * nboot)]), float(p)


def line(tag, v, days, extra=""):
    v = np.asarray(v, float)
    if len(v) < 5:
        print(f"  {tag:<36} n={len(v):<6} too few")
        return
    lo, hi, p = boot(v, days)
    print(f"  {tag:<36} n={len(v):<6} mean ${v.mean():+7.2f}  total ${v.sum():+10,.0f}  "
          f"win {100*(v>0).mean():5.1f}%  CI [{lo:+6.2f},{hi:+6.2f}]  p {p:.4f} {extra}")


def main():
    rows = json.load(open(TRADES, encoding="utf-8"))
    T = [r for r in rows if r["setup"] == SETUP]
    pnl = np.array([r["pnl"] for r in T])
    days = [r["day"] for r in T]
    tot = pnl.sum()
    print("=" * 118)
    print(f"IMB STRESS TEST -- {len(T)} trades, {len(set(days))} sessions, "
          f"${NOTIONAL:,.0f} notional per trade")
    print("=" * 118)
    line("headline", pnl, days)

    # 1 -------------------------------------------------------------- tail dependence
    print("\n1. TAIL DEPENDENCE")
    s = np.sort(pnl)[::-1]
    for k, lab in ((1, "top 1 trade"), (5, "top 5"), (10, "top 10"),
                   (max(1, len(s) // 100), "top 1%"), (max(1, len(s) // 20), "top 5%")):
        print(f"    {lab:<14} = {100*s[:k].sum()/tot:6.1f}% of total P&L   "
              f"(removing them leaves ${tot - s[:k].sum():+,.0f})")
    ordr = np.argsort(pnl)[::-1]
    for k in (1, 5, 10, max(1, len(s) // 100)):
        keep = ordr[k:]
        line(f"    ex top {k}", pnl[keep], [days[i] for i in keep])
    print(f"    median trade ${np.median(pnl):+.2f}   mean ${pnl.mean():+.2f}   "
          f"skew {float(((pnl-pnl.mean())**3).mean()/pnl.std()**3):.2f}")

    # 2 -------------------------------------------------------------- per year
    print("\n2. PER-YEAR STABILITY")
    yb = defaultdict(list)
    for r, v in zip(T, pnl):
        yb[r["day"][:4]].append(v)
    ys = sorted(yb)
    for y in ys:
        v = np.array(yb[y])
        print(f"    {y}  n={len(v):<5} mean ${v.mean():+6.2f}  total ${v.sum():+9,.0f}  "
              f"win {100*(v>0).mean():5.1f}%")
    print(f"    positive years: {sum(1 for y in ys if np.sum(yb[y]) > 0)} of {len(ys)}")
    best = max(ys, key=lambda y: np.sum(yb[y]))
    keep = [i for i, r in enumerate(T) if r["day"][:4] != best]
    line(f"    ex best year ({best})", pnl[keep], [days[i] for i in keep])
    worst = min(ys, key=lambda y: np.sum(yb[y]))
    keep = [i for i, r in enumerate(T) if r["day"][:4] != worst]
    line(f"    ex worst year ({worst})", pnl[keep], [days[i] for i in keep])

    # 3 -------------------------------------------------------------- split half
    print("\n3. SPLIT-HALF (independent windows)")
    ud = sorted(set(days))
    mid = ud[len(ud) // 2]
    a = [i for i, d in enumerate(days) if d < mid]
    b = [i for i, d in enumerate(days) if d >= mid]
    line(f"    first half  (< {mid})", pnl[a], [days[i] for i in a])
    line(f"    second half (>= {mid})", pnl[b], [days[i] for i in b])

    # 4 -------------------------------------------------------------- direction
    print("\n4. DIRECTION")
    for d_ in ("long", "short"):
        k = [i for i, r in enumerate(T) if r["direction"] == d_]
        line("    " + d_, pnl[k], [days[i] for i in k])

    # 5 -------------------------------------------------------------- equity curve
    print(f"\n5. EQUITY CURVE (one position at a time, ${NOTIONAL:,.0f} capital)")
    order = np.argsort([r["entry_ts"] for r in T])
    eq = np.cumsum(pnl[order])
    peak = np.maximum.accumulate(np.concatenate([[0.0], eq]))[1:]
    dd = eq - peak
    print(f"    total ${eq[-1]:+,.0f}   max drawdown ${-dd.min():,.0f} "
          f"({100*-dd.min()/NOTIONAL:.1f}% of capital)")
    under = dd < -1e-9
    longest, run = 0, 0
    for u in under:
        run = run + 1 if u else 0
        longest = max(longest, run)
    print(f"    {100*under.mean():.1f}% of trades under water; longest run below a prior "
          f"peak: {longest} trades")
    dts = [T[i]["day"] for i in order]
    print(f"    deepest drawdown reached on {dts[int(np.argmin(dd))]}")
    # Annualise on CALENDAR time, not active-trading days. IMB only fires on 1,560 of ~2,657
    # sessions, but the capital has to sit there on the other 1,097 too -- dividing by
    # active-days/252 would have reported 15.7%/yr for a strategy that earns 9.2%.
    d0 = dt.date.fromisoformat(min(days))
    d1 = dt.date.fromisoformat(max(days))
    yrs = (d1 - d0).days / 365.25
    print(f"    calendar span {d0} -> {d1} = {yrs:.2f} years "
          f"(traded {len(set(days))} of them)")
    print(f"    ${eq[-1]/yrs:,.0f}/yr = {100*eq[-1]/yrs/NOTIONAL:.1f}%/yr on "
          f"${NOTIONAL:,.0f}  (${eq[-1]/yrs/52:.0f}/week)")
    mb = defaultdict(float)
    for r, v in zip(T, pnl):
        mb[r["day"][:7]] += v
    mv = np.array([mb[k] for k in sorted(mb)])
    print(f"    {len(mv)} months, {100*(mv>0).mean():.1f}% positive, "
          f"annualised Sharpe on monthly P&L {mv.mean()/mv.std()*np.sqrt(12):.2f}, "
          f"worst month ${mv.min():+,.0f}")
    # longest stretch below a prior peak, in calendar days
    db2 = defaultdict(float)
    for r, v in zip(T, pnl):
        db2[r["day"]] += v
    ad = sorted(db2)
    e2 = np.cumsum([db2[d] for d in ad])
    dd2 = e2 - np.maximum.accumulate(np.concatenate([[0.0], e2]))[1:]
    spans, start = [], None
    for i, f in enumerate(dd2 < -1e-9):
        if f and start is None:
            start = i
        if not f and start is not None:
            spans.append((ad[start], ad[i - 1]))
            start = None
    if start is not None:
        spans.append((ad[start], ad[-1]))
    spans = sorted(((a, b, (dt.date.fromisoformat(b) - dt.date.fromisoformat(a)).days)
                    for a, b in spans), key=lambda x: -x[2])
    print("    longest stretches below a prior equity peak:")
    for a, b, n in spans[:3]:
        print(f"      {a} -> {b}   {n} days ({n/30.4:.1f} months)")

    # 6 -------------------------------------------------------------- slippage
    print("\n6. SLIPPAGE SENSITIVITY (extra cost per side, on top of the real NBBO already"
          " paid)")
    shares = np.array([r["shares"] for r in T])
    for cents in (0.0, 0.005, 0.01, 0.02, 0.03, 0.05):
        adj = pnl - 2 * cents * shares
        lo, hi, p = boot(adj, days)
        flag = "  <- negative" if adj.mean() <= 0 else ""
        print(f"    +{100*cents:4.1f} c/share/side   mean ${adj.mean():+6.2f}  "
              f"total ${adj.sum():+9,.0f}  CI [{lo:+6.2f},{hi:+6.2f}]  p {p:.4f}{flag}")
    print(f"    breakeven extra slippage: {pnl.mean()/(2*shares.mean())*100:.2f} cents "
          f"per share per side")

    # 7 -------------------------------------------------------------- time of day
    print("\n7. TIME OF DAY (IMB evaluates only at :00 and :30)")
    tb = defaultdict(list)
    for r, v in zip(T, pnl):
        tb[r["entry_ts"][11:16]].append(v)
    for t in sorted(tb):
        v = np.array(tb[t])
        if len(v) >= 20:
            print(f"    {t}  n={len(v):<5} mean ${v.mean():+6.2f}  total ${v.sum():+9,.0f}"
                  f"  win {100*(v>0).mean():5.1f}%")
    cands = [t for t in tb if len(tb[t]) >= 20]
    bestt = max(cands, key=lambda t: np.sum(tb[t]))
    keep = [i for i, r in enumerate(T) if r["entry_ts"][11:16] != bestt]
    line(f"    ex best checkpoint ({bestt})", pnl[keep], [days[i] for i in keep])

    # 8 -------------------------------------------------------------- exits, holds
    print("\n8. EXIT MIX AND HOLD TIME")
    eb = defaultdict(list)
    for r, v in zip(T, pnl):
        eb[r["exit_reason"]].append(v)
    for k in sorted(eb, key=lambda k: -len(eb[k])):
        v = np.array(eb[k])
        print(f"    {k:<10} n={len(v):<5} ({100*len(v)/len(T):4.1f}%)  "
              f"mean ${v.mean():+6.2f}  total ${v.sum():+9,.0f}")
    h = np.array([r["hold_min"] for r in T], float)
    print(f"    hold minutes: median {np.median(h):.0f}  mean {h.mean():.1f}  "
          f"p90 {np.percentile(h,90):.0f}  max {h.max():.0f}")
    w = pnl > 0
    print(f"    winners mean ${pnl[w].mean():+.2f} hold {h[w].mean():.0f} min   "
          f"losers mean ${pnl[~w].mean():+.2f} hold {h[~w].mean():.0f} min")

    # 9 -------------------------------------------------------------- day concentration
    print("\n9. DAY CONCENTRATION")
    db = defaultdict(float)
    for r, v in zip(T, pnl):
        db[r["day"]] += v
    dv = np.sort(np.array(list(db.values())))[::-1]
    print(f"    {len(dv)} active days; best day ${dv[0]:+,.0f} = {100*dv[0]/tot:.1f}% of "
          f"total; top 10 days = {100*dv[:10].sum()/tot:.1f}%")
    print(f"    positive days {100*(dv>0).mean():.1f}%   median day ${np.median(dv):+.2f}"
          f"   trades per active day {len(T)/len(dv):.2f}")

    # 10 ------------------------------------------------------------- peers
    print("\n10. PEER COMPARISON (same sweep, same fills)")
    pb, pdd = defaultdict(list), defaultdict(list)
    for r in rows:
        pb[r["setup"]].append(r["pnl"])
        pdd[r["setup"]].append(r["day"])
    print(f"    {'setup':<27} {'n':>6} {'mean$':>9} {'total$':>11} {'p':>9}")
    for k in sorted(pb, key=lambda k: -np.mean(pb[k])):
        v = np.array(pb[k])
        if len(v) < 100:
            continue
        _, _, p = boot(v, pdd[k])
        mark = "  <-- IMB" if k == SETUP else ""
        print(f"    {k:<27} {len(v):>6} {v.mean():>+9.2f} {v.sum():>+11,.0f} "
              f"{p:>9.4f}{mark}")


if __name__ == "__main__":
    main()
