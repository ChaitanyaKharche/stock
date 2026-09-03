"""Which symbols could the SHARES arm actually trade?

The shares edge measured on IMB is +3.34 bp per trade GROSS. Round-trip spread comes
straight off that, so the binding constraint on broadening the universe is not sector
coverage or options volume -- it is spread. A name quoting 5 bp wide has no room for a
3 bp signal no matter how liquid its option chain is.

There is a second, opposing constraint. Adding symbols multiplies TRADES quickly but
INFORMATION slowly, because same-day observations across correlated names are not
independent. For observations equicorrelated at rho within a day, the design effect on
the mean is 1 + (m-1)*rho. QQQ and SPY already correlate at 0.717 on daily IMB P&L, and
the 120 days SPY fired while QQQ did not -- the only genuinely new information -- lost
money. So a name is only worth adding if it is BOTH tight enough to trade AND different
enough to inform.

Those two pull against each other: the tightest names are the mega-caps and big ETFs,
which are also the most correlated with what the lab already trades. This measures both
so the choice is made on numbers instead of intuition.

Measured per symbol over the last N completed sessions, 09:45-15:45 ET only (the open and
close are structurally wide and would flatter or punish names unevenly):

    spread_bp     median round-trip (ask-bid)/mid, in basis points
    spread_p90    the tail that actually gets paid on a fast fill
    dollar_vol    median dollar volume per minute -- can $10,000 clear without impact
    corr_qqq      per-minute return correlation with QQQ, the diversification term
    headroom      +3.34 bp signal minus the median spread; negative means untradeable

Read-only. Touches no lab state and changes no frozen definition.
"""
from __future__ import annotations

import datetime as dt
import os
import pickle
import sys
import urllib.request
from csv import DictReader

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

SCRATCH = (r"C:\Users\CHAITA~1\AppData\Local\Temp\claude"
           r"\C--Users-chaitanyakharche-Documents-stock"
           r"\3ad3e562-0324-4d02-bf8b-208ff71f108d\scratchpad\spread_cache")
BASE = "http://127.0.0.1:25503/v3"
GROSS_EDGE_BP = 3.34            # IMB measured mean, QQQ shares 2016-2026
WIN_LO, WIN_HI = "09:45", "15:45"

UNIVERSE = [
    # broad index ETFs -- the incumbents and their nearest neighbours
    "SPY", "QQQ", "IWM", "DIA",
    # sector ETFs
    "XLE", "XLF", "XLK", "XLV", "XLI", "XLY", "XLP", "XLU", "SMH",
    # mega-cap tech
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AVGO", "NFLX",
    # semis and high-beta retail favourites
    "AMD", "MU", "INTC", "PLTR", "COIN",
    # financials
    "JPM", "BAC", "WFC", "GS",
    # energy
    "XOM", "CVX", "OXY",
    # healthcare
    "UNH", "JNJ", "LLY", "PFE",
    # staples and retail
    "WMT", "COST", "HD", "KO", "PG",
    # other very high share volume
    "T", "VZ", "F", "CSCO", "ORCL", "CRM", "DIS", "BA",
]


def _get(path, **params):
    q = "&".join(f"{k}={v}" for k, v in params.items())
    try:
        with urllib.request.urlopen(f"{BASE}{path}?{q}", timeout=30) as r:
            return list(DictReader(r.read().decode().splitlines()))
    except Exception:                                        # noqa: BLE001
        return []


def cached(kind, sym, day):
    os.makedirs(SCRATCH, exist_ok=True)
    p = os.path.join(SCRATCH, f"{kind}_{sym}_{day}.pkl")
    if os.path.exists(p):
        try:
            return pickle.load(open(p, "rb"))
        except Exception:                                    # noqa: BLE001
            pass
    ep = "/stock/history/quote" if kind == "q" else "/stock/history/ohlc"
    rows = _get(ep, symbol=sym, start_date=day, end_date=day, interval="1m")
    out = []
    for r in rows:
        try:
            ts = r["timestamp"][11:16]
            if kind == "q":
                b, a = float(r["bid"]), float(r["ask"])
                if b > 0 and a >= b:
                    out.append((ts, b, a))
            else:
                out.append((ts, float(r["close"]), float(r["volume"])))
        except (KeyError, ValueError):
            continue
    pickle.dump(out, open(p, "wb"))
    return out


def sessions(n=5, end=None):
    end = end or dt.date.today()
    out, d = [], end
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d.isoformat())
        d -= dt.timedelta(days=1)
    return sorted(out)


def main():
    days = sessions(5)
    print(f"spread survey over {len(days)} sessions: {days[0]} .. {days[-1]}")
    print(f"window {WIN_LO}-{WIN_HI} ET, {len(UNIVERSE)} candidates\n", flush=True)

    spreads, closes, dvol = {}, {}, {}
    for i, sym in enumerate(UNIVERSE, 1):
        sp, cl, dv = [], {}, []
        for day in days:
            for ts, b, a in cached("q", sym, day):
                if WIN_LO <= ts <= WIN_HI:
                    mid = (a + b) / 2.0
                    if mid > 0:
                        sp.append(1e4 * (a - b) / mid)
            for ts, c, v in cached("o", sym, day):
                if WIN_LO <= ts <= WIN_HI:
                    cl[(day, ts)] = c
                    dv.append(c * v)
        spreads[sym], closes[sym], dvol[sym] = sp, cl, dv
        if i % 10 == 0:
            print(f"  {i}/{len(UNIVERSE)} fetched", flush=True)

    # per-minute return correlation with QQQ, on the minutes both symbols have
    qkeys = sorted(closes.get("QQQ", {}))
    qmap = closes.get("QQQ", {})

    def corr_with_qqq(sym):
        keys = [k for k in qkeys if k in closes[sym]]
        if len(keys) < 200:
            return float("nan")
        a = np.array([closes[sym][k] for k in keys], float)
        b = np.array([qmap[k] for k in keys], float)
        ra, rb = np.diff(a) / a[:-1], np.diff(b) / b[:-1]
        m = np.isfinite(ra) & np.isfinite(rb)
        if m.sum() < 200 or ra[m].std() == 0 or rb[m].std() == 0:
            return float("nan")
        return float(np.corrcoef(ra[m], rb[m])[0, 1])

    rows = []
    for sym in UNIVERSE:
        sp = spreads[sym]
        if len(sp) < 200:
            rows.append((sym, np.nan, np.nan, np.nan, np.nan, np.nan, len(sp)))
            continue
        med = float(np.median(sp))
        rows.append((sym, med, float(np.percentile(sp, 90)),
                     float(np.median(dvol[sym])) if dvol[sym] else np.nan,
                     corr_with_qqq(sym), GROSS_EDGE_BP - med, len(sp)))

    ok = [r for r in rows if np.isfinite(r[1])]
    ok.sort(key=lambda r: r[1])
    print("\n" + "=" * 96)
    print("ROUND-TRIP SPREAD, cheapest first")
    print("=" * 96)
    print(f"  {'sym':<7}{'spread bp':>10}{'p90':>8}{'$vol/min':>13}"
          f"{'corr QQQ':>10}{'headroom':>10}   verdict")
    for sym, med, p90, dv, c, head, n in ok:
        v = ("UNTRADEABLE" if head <= 0 else
             "thin" if head < 1.5 else
             "ok" if head < 2.5 else "good")
        print(f"  {sym:<7}{med:>10.2f}{p90:>8.2f}{dv:>13,.0f}"
              f"{c:>10.2f}{head:>10.2f}   {v}")
    miss = [r[0] for r in rows if not np.isfinite(r[1])]
    if miss:
        print(f"\n  no usable quotes: {', '.join(miss)}")

    print("\n" + "=" * 96)
    print("THE TRADE-OFF: tight names are the correlated ones")
    print("=" * 96)
    t = [(s, m, c) for s, m, p, d, c, h, n in ok if np.isfinite(c)]
    if len(t) > 5:
        mm = np.array([x[1] for x in t])
        cc = np.array([x[2] for x in t])
        print(f"  correlation between (spread) and (corr with QQQ) across {len(t)} names:"
              f" {np.corrcoef(mm, cc)[0, 1]:+.2f}")
        print("  negative means: the tighter the spread, the MORE correlated with QQQ --")
        print("  i.e. the names you can afford to trade add the least new information.")
    print(f"\n  design effect if m symbols are traded each day at correlation rho:")
    print(f"    DEFF = 1 + (m-1)*rho ;  effective information multiplier = m / DEFF")
    for m in (5, 10, 15, 20):
        for rho in (0.3, 0.5, 0.7):
            deff = 1 + (m - 1) * rho
            print(f"      m={m:<3} rho={rho}  ->  DEFF {deff:5.1f}   "
                  f"information x{m/deff:.2f}")
        print()


if __name__ == "__main__":
    main()
