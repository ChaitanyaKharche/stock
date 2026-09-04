"""Is the IMB edge PERSISTENT, or is the pooled 10.6-year number hiding a decay?

Executes research/persistence_preregistration.md. Five tests, Holm m=5, every falsifier
fixed before this ran. Nothing here tunes a parameter or proposes a strategy -- the only
possible outcomes are that the pooled estimate stands, that it must be replaced by a
recent-era estimate, or that the edge is regime-dependent and unusable.

Why it matters here specifically: IMB's parameters are PUBLISHED (SSRN 4824172, ~2024).
That is the main reason the result is credible, and exactly why the edge could have been
arbitraged away since. A forward test sized against +$3.34 is worthless if the current
number is +$1.00.
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

SEED = 20260904
SETUP = "IntradayMomentumBoundary"


def boot_mean(vals, days, nboot=10000, seed=SEED):
    """Day-clustered bootstrap of the mean."""
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


def boot_slope(x, y, days, nboot=4000, seed=SEED):
    """Day-clustered bootstrap of an OLS slope."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    uniq, inv = np.unique(np.asarray(days), return_inverse=True)
    groups = [np.flatnonzero(inv == i) for i in range(len(uniq))]
    rng = np.random.default_rng(seed)
    out = np.empty(nboot)
    for b in range(nboot):
        pick = np.concatenate([groups[i] for i in rng.integers(0, len(groups),
                                                               len(groups))])
        xs, ys = x[pick], y[pick]
        vx = xs.var()
        out[b] = ((xs - xs.mean()) * (ys - ys.mean())).mean() / vx if vx > 0 else 0.0
    out.sort()
    p = max(2 * min((out <= 0).mean(), (out >= 0).mean()), 1.0 / nboot)
    return float(out[int(.025 * nboot)]), float(out[int(.975 * nboot)]), float(p)


def block_boot(daily, nboot=10000, block=5, seed=SEED):
    """Moving-block bootstrap over DAILY totals -- keeps serial structure the
    day-clustered bootstrap destroys by resampling days independently."""
    v = np.asarray(daily, float)
    n = len(v)
    nb = int(np.ceil(n / block))
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, max(n - block + 1, 1), size=(nboot, nb))
    out = np.empty(nboot)
    for b in range(nboot):
        s = np.concatenate([v[i:i + block] for i in starts[b]])[:n]
        out[b] = s.mean()
    out.sort()
    return float(out[int(.025 * nboot)]), float(out[int(.975 * nboot)])


def holm(p):
    m = len(p)
    order = sorted(range(m), key=lambda i: p[i])
    adj, run = [0.0] * m, 0.0
    for r, i in enumerate(order):
        run = max(run, (m - r) * p[i])
        adj[i] = min(run, 1.0)
    return adj


def load(path):
    return [t for t in json.load(open(path, encoding="utf-8")) if t["setup"] == SETUP]


def main():
    T = load(os.path.join(ROOT, "live_lab_data", "sharewf_trades_QQQ.json"))
    T.sort(key=lambda t: (t["day"], t["entry_ts"]))
    pnl = np.array([t["pnl"] for t in T])
    days = [t["day"] for t in T]
    d0 = dt.date.fromisoformat(days[0])
    age = np.array([(dt.date.fromisoformat(d) - d0).days for d in days], float)

    print("=" * 92)
    print(f"PERSISTENCE OF THE IMB EDGE -- {len(T)} trades, {days[0]} .. {days[-1]}")
    print("=" * 92)
    lo, hi, p = boot_mean(pnl, days)
    print(f"  pooled: ${pnl.mean():+.2f}/trade  CI [{lo:+.2f}, {hi:+.2f}]  p {p:.4f}")

    ps, names = [], []

    # ---------------------------------------------------------------- T1 trend
    print("\nT1  LINEAR TREND in the per-trade edge")
    slo, shi, sp = boot_slope(age, pnl, days)
    slope = np.polyfit(age, pnl, 1)[0]
    print(f"  slope {slope*365:+.3f} $/trade per YEAR   CI [{slo*365:+.3f}, {shi*365:+.3f}]"
          f"   p {sp:.4f}")
    if slope < 0:
        start = np.polyval(np.polyfit(age, pnl, 1), 0)
        zero_day = -start / slope if slope else float("inf")
        zdate = d0 + dt.timedelta(days=float(zero_day))
        print(f"  fitted edge at series start ${start:+.2f}, reaches ZERO around {zdate}")
    else:
        print("  slope is positive; no decay to date")
    ps.append(sp); names.append("T1 trend")

    # ---------------------------------------------------------------- T2 rolling
    print("\nT2  ROLLING 250-TRADE MEAN")
    W = 250
    roll = np.array([pnl[i - W:i].mean() for i in range(W, len(pnl) + 1)])
    rdays = days[W - 1:]
    neg = roll < 0
    print(f"  windows {len(roll)}   below zero {100*neg.mean():.1f}% of the time")
    last3 = [i for i, d in enumerate(rdays)
             if dt.date.fromisoformat(d) >= dt.date(2023, 9, 1)]
    if last3:
        r3 = roll[last3[0]:]
        print(f"  last three years: {100*(r3 < 0).mean():.1f}% of windows below zero"
              f"   (falsifier: >40%)")
    for yr in range(2017, 2027):
        sel = [i for i, d in enumerate(rdays) if d[:4] == str(yr)]
        if sel:
            print(f"    {yr}  rolling mean ${roll[sel].mean():+6.2f}")
    ps.append(1.0); names.append("T2 rolling (descriptive)")

    # ---------------------------------------------------------------- T3 eras
    print("\nT3  ERA SPLIT (equal trade counts)")
    k = len(T) // 3
    eras = [("early", 0, k), ("middle", k, 2 * k), ("LATE", 2 * k, len(T))]
    late_p = 1.0
    for nm, a, b in eras:
        lo, hi, p = boot_mean(pnl[a:b], days[a:b])
        print(f"  {nm:<7} {days[a]} .. {days[b-1]}  n={b-a:<5} "
              f"${pnl[a:b].mean():+6.2f}  CI [{lo:+6.2f},{hi:+6.2f}]  p {p:.4f}")
        if nm == "LATE":
            late_p = p
    ps.append(late_p); names.append("T3 late era")

    # ---------------------------------------------------------------- T4 serial
    print("\nT4  SERIAL DEPENDENCE")
    dd = defaultdict(float)
    for t, v in zip(T, pnl):
        dd[t["day"]] += v
    ds = np.array([dd[d] for d in sorted(dd)])
    ac = [float(np.corrcoef(ds[:-l], ds[l:])[0, 1]) for l in range(1, 11)]
    print("  daily-P&L autocorrelation lags 1-10: "
          + " ".join(f"{a:+.2f}" for a in ac))
    dlo, dhi, _ = boot_mean(pnl, days)
    blo, bhi = block_boot(ds, block=5)
    dayw = dhi - dlo
    # convert the daily-total block CI back to a per-trade scale
    per = len(T) / len(ds)
    blkw = (bhi - blo) / per
    print(f"  day-clustered CI width  ${dayw:.2f}/trade")
    print(f"  moving-block  CI width  ${blkw:.2f}/trade   "
          f"({100*(blkw/dayw-1):+.0f}%)   falsifier: >+25%")
    ps.append(1.0); names.append("T4 serial (descriptive)")

    # ---------------------------------------------------------------- T5 regime
    print("\nT5  REGIME (diagnostic only -- cannot become a filter)")
    dr = {}
    prev = None
    for d in sorted(dd):
        dr[d] = dd[d]
    # trailing realised vol proxy: sd of the previous 20 daily totals is circular,
    # so use the day's own |return| proxy from the trade set instead
    absmove = defaultdict(list)
    for t in T:
        absmove[t["day"]].append(abs(t.get("ret", 0.0)))
    vol = np.array([np.mean(absmove[d]) for d in sorted(dd)])
    q = np.quantile(vol, [0.25, 0.5, 0.75])
    lab = np.digitize(vol, q)
    print("  by trailing move quartile (Q1 quietest .. Q4 wildest)")
    for i in range(4):
        sel = [d for d, L in zip(sorted(dd), lab) if L == i]
        tr = [(t["pnl"], t["day"]) for t in T if t["day"] in set(sel)]
        if len(tr) < 30:
            continue
        v = [x for x, _ in tr]
        lo, hi, p = boot_mean(v, [d for _, d in tr])
        print(f"    Q{i+1}  n={len(v):<5} ${np.mean(v):+6.2f}  CI [{lo:+6.2f},{hi:+6.2f}]"
              f"  p {p:.4f}")
    ps.append(1.0); names.append("T5 regime (descriptive)")

    adj = holm(ps)
    print("\n" + "=" * 92)
    print("HOLM ACROSS THE FIVE PRIMARIES")
    for n, r, a in zip(names, ps, adj):
        print(f"  {n:<28} p {r:.4f}   Holm {a:.4f}")


if __name__ == "__main__":
    main()
