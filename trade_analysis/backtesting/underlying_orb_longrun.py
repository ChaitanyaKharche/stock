"""The ORB signal tested on the UNDERLYING, 2016-2026 - no options involved.

WHY THIS EXISTS
---------------
Every option-level study is capped by contract availability and by the noise of
a +25%/-68% payoff. Two consequences we hit hard:
  - QQQ yielded only 26 trades in all of 2020, because QQQ ran Friday-only 0DTE
    expirations then. Nothing to do with the signal.
  - Option-payoff SD is ~53 points, so a year of data barely moves a CI.

But the signal lives on the underlying: measured elasticity is 115-140% option
return per 1% of underlying move, so a ~0.2% move IS the 25% option target.
Testing the signal on the stock therefore removes both limits at once, and the
Stock STANDARD line reaches 2016 - roughly 2,500 sessions per symbol instead of
676 option trades, spanning the 2018 vol spike and COVID.

WHAT THIS CAN AND CANNOT TELL YOU
---------------------------------
CAN:  whether the breakout signal itself is stable across a decade and across
      volatility regimes. If it is not, the option strategy cannot be either,
      and that is decisive.
CANNOT: whether the option version is profitable. A positive underlying edge is
      necessary but NOT sufficient - the option pays premium the stock does not.
      Measured breakeven win rates: QQQ needs 75.3% (gets 80.3%), SPY needs
      74.7% (gets 74.5%). The margin is ~5 points and ~0 points respectively,
      so a stable signal still has to clear that bar.

METHOD
------
Entries come from `find_entries`, imported unchanged from the proven code, so
this is the same signal the option studies used - only the measurement differs.
Outcomes are max favourable/adverse excursion and close return from the entry
bar, reported raw and normalised by trailing 20-day realised vol (shifted, so
no look-ahead). Hit rates are reported at several thresholds rather than one
calibrated number, because the option's moneyness drifts with vol regime and
any single calibrated threshold would silently bake in one regime's pricing.

ISOLATION: imports only, edits nothing, writes its own CSV.
"""
import argparse
import math

import numpy as np
import pandas as pd

from ..data_sources.thetadata_client import ThetaDataClient
from .options_premium_backtest import find_entries
from .spy_qqq_0dte_real_backtest import SYMBOLS, regular_hours_5min, daily_from_minute
from .spy_qqq_0dte_quotes_backtest import fetch_underlying_theta
from ..paths import LOGS_DIR

# thresholds on max favourable excursion, in % of spot. 0.20% is roughly the
# move that produced a 25% option gain at measured elasticity; the others
# bracket it so the answer does not hinge on that one calibration.
MFE_THRESHOLDS = [0.10, 0.15, 0.20, 0.25, 0.30]
SIGMA_THRESHOLDS = [0.10, 0.15, 0.20, 0.25]


def measure(symbol, theta, start="2016-01-01"):
    minute = fetch_underlying_theta(theta, symbol, start=start)
    if minute.empty:
        print(f"  {symbol}: no bars returned")
        return pd.DataFrame()

    five = regular_hours_5min(minute)
    daily = daily_from_minute(minute)
    entries = find_entries(symbol, five, daily)
    print(f"\n=== {symbol} === {len(entries)} entries over {len(daily)} sessions "
          f"({daily.index.min().date()} -> {daily.index.max().date()})")

    # 16:00 is a single-print settle stub, often zero-filled - excluding it is
    # required or close/MFE read as -100%.
    bars = minute.between_time("09:30", "15:59")
    bars = bars[bars["Close"] > 0].copy()
    bars["d"] = bars.index.date
    by_day = {d: g for d, g in bars.groupby("d")}

    dclose = bars.groupby("d")["Close"].last()
    trail = dclose.pct_change().rolling(20).std().shift(1) * 100

    rows = []
    for e in entries:
        day = by_day.get(e["date"])
        if day is None or day.empty:
            continue
        sig = trail.get(e["date"], np.nan)
        after = day[day.index >= e["entry_time"]]
        if len(after) < 2:
            continue

        spot = float(after["Open"].iloc[0])
        up = e["direction"] == "UP"
        sign = 1.0 if up else -1.0
        close_px = float(after["Close"].iloc[-1])

        if up:
            mfe = (float(after["High"].max()) / spot - 1) * 100
            mae = (float(after["Low"].min()) / spot - 1) * 100
        else:
            mfe = (1 - float(after["Low"].min()) / spot) * 100
            mae = (1 - float(after["High"].max()) / spot) * 100

        rows.append({"date": e["date"], "symbol": symbol,
                     "direction": e["direction"], "spot": spot,
                     "fwd": sign * (close_px / spot - 1) * 100,
                     "mfe": mfe, "mae": mae, "sigma": sig,
                     "mfe_sigma": mfe / sig if sig and sig > 0 else np.nan,
                     "fwd_sigma": (sign * (close_px / spot - 1) * 100 / sig)
                                  if sig and sig > 0 else np.nan})
    return pd.DataFrame(rows)


def report(d):
    d = d.copy()
    d["year"] = pd.to_datetime(d["date"]).dt.year

    for sym in sorted(d["symbol"].unique()):
        s = d[d["symbol"] == sym]
        print("\n" + "=" * 96)
        print(f"{sym}: HIT RATE BY YEAR - share of entries whose MFE cleared each threshold")
        print("=" * 96)
        hdr = "  ".join(f">{x:.2f}%" for x in MFE_THRESHOLDS)
        print(f"{'year':>6} {'n':>5} {'sigma%':>7} {'fwd%':>7}   {hdr}")
        for y, g in s.groupby("year"):
            hits = "  ".join(f"{100*(g['mfe'] > x).mean():5.1f}" for x in MFE_THRESHOLDS)
            print(f"{y:>6} {len(g):>5} {g['sigma'].mean():>7.2f} "
                  f"{g['fwd'].mean():>+7.3f}   {hits}")
        allhits = "  ".join(f"{100*(s['mfe'] > x).mean():5.1f}" for x in MFE_THRESHOLDS)
        print(f"{'ALL':>6} {len(s):>5} {s['sigma'].mean():>7.2f} "
              f"{s['fwd'].mean():>+7.3f}   {allhits}")

        # stability is the whole question: a 5-point breakeven margin is only
        # usable if the hit rate does not swing more than that between years
        yr = s.assign(hit=s["mfe"] > 0.20).groupby("year")["hit"].mean() * 100
        print(f"\n  at the 0.20% threshold: min {yr.min():.1f}%  max {yr.max():.1f}%  "
              f"spread {yr.max()-yr.min():.1f} pts  SD {yr.std():.1f}")

        print(f"\n  forward return: mean {s['fwd'].mean():+.4f}%  "
              f"SE {s['fwd'].std()/math.sqrt(len(s)):.4f}  "
              f"t={s['fwd'].mean()/(s['fwd'].std()/math.sqrt(len(s))):+.2f}")
        pos = (s.groupby("year")["fwd"].mean() > 0).sum()
        print(f"  forward return positive in {pos}/{s['year'].nunique()} years")

    print("\n" + "=" * 96)
    print("DOES THE SIGNAL DEPEND ON VOLATILITY REGIME? (trailing-vol terciles)")
    print("=" * 96)
    for sym in sorted(d["symbol"].unique()):
        s = d[d["symbol"] == sym].dropna(subset=["sigma"])
        q = s["sigma"].quantile([1 / 3, 2 / 3]).values
        buckets = [("low ", s[s["sigma"] <= q[0]]),
                   ("mid ", s[(s["sigma"] > q[0]) & (s["sigma"] <= q[1])]),
                   ("high", s[s["sigma"] > q[1]])]
        print(f"\n  {sym}   (tercile cuts at sigma {q[0]:.2f}%, {q[1]:.2f}%)")
        for name, g in buckets:
            if g.empty:
                continue
            se = g["fwd"].std() / math.sqrt(len(g))
            print(f"    {name} vol  n={len(g):5d}  sigma {g['sigma'].mean():5.2f}%  "
                  f"MFE>0.20%: {100*(g['mfe']>0.20).mean():5.1f}%  "
                  f"fwd {g['fwd'].mean():+.4f}% (t={g['fwd'].mean()/se:+.2f})  "
                  f"MFE/sigma {g['mfe_sigma'].mean():.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", nargs="*", default=SYMBOLS)
    ap.add_argument("--start", default="2016-01-01")
    args = ap.parse_args()

    theta = ThetaDataClient()
    frames = [measure(s, theta, start=args.start) for s in args.symbols]
    frames = [f for f in frames if not f.empty]
    if frames:
        d = pd.concat(frames, ignore_index=True)
        report(d)
        out = LOGS_DIR / "underlying_orb_longrun.csv"
        d.to_csv(out, index=False)
        print(f"\nSaved -> {out}")
        print(f"theta API: {theta.stats()}")
