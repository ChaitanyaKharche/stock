"""Honest-timing baseline: what the ORB rule is worth when you can only act on
information that exists.

THE BUG THIS CORRECTS
---------------------
`find_entries` (options_premium_backtest.py:152-158) iterates 5-minute bars,
tests `bar['Close']`, and records the bar's **label** as `entry_time`.
`pandas.resample("5min")` labels left, so the bar labelled 09:30 spans
[09:30, 09:35) and its close is not knowable until 09:35:00. Every consumer
then treats that label as an action timestamp -- `spy_qqq_0dte_quotes_backtest`
fills at `chain[timestamp >= entry_time]`, i.e. 09:30.

The backtest therefore bought the option five minutes before the signal that
triggered it existed. 531 of 676 QQQ entries carry the 09:30 label, so ~78% of
trades received the full five minutes. Measured cost of the illusion:

    QQQ 25% target   +6.44% -> -8.34%   (bias +14.79 pts, paired t=+6.67)
    SPY 25% target   -0.26% -> -11.33%  (bias +11.08 pts, paired t=+5.77)
    QQQ underlying   t=3.55 -> t=0.44   (88% of the "signal" was the drift)

The mechanism is exact: the underlying drifts ~0.09% in the signal direction
during those five minutes, and the option's fixed-premium strike sits close
enough that ~0.2% of favourable travel IS the 25% target. The backtest was
handed the profit target before the position opened.

THE 09:45 BAR -- A REAL SPECIFICATION QUESTION, NOT A ROUNDING DETAIL
---------------------------------------------------------------------
`MORNING_END = 09:45` and `between_time` is inclusive, so the bar LABELLED
09:45 is evaluated. Its close is 09:50 -- outside the "confirmed in the first
15 minutes" rule as stated. Under honest timing you must choose:

  strict=True   decision must land at or before 09:45 -> only bars labelled
                09:30, 09:35, 09:40 qualify (three bars, the natural reading)
  strict=False  keep the 09:45-labelled bar, acting at 09:50

This module reports both, because the choice changes the sample and the two
readings are genuinely different rules. Pick one, write it down, never revisit.

WHAT THIS MODULE IS FOR
-----------------------
It is the clean baseline every future variant gets measured against. It does
not try to rescue the strategy. Expect it to confirm the rule sits below its
own breakeven win rate (75.3%) -- the value is a trustworthy floor, not a
reprieve.

ISOLATION: imports only, edits nothing, writes its own CSV.
"""
import argparse
import math
from datetime import time as dt_time

import numpy as np
import pandas as pd

from ..data_sources.thetadata_client import ThetaDataClient, JOINT_HISTORY_START
from .options_premium_backtest import find_entries, MORNING_END
from .spy_qqq_0dte_real_backtest import (
    SYMBOLS, PROFIT_TARGET_GRID, regular_hours_5min, daily_from_minute,
)
from .spy_qqq_0dte_quotes_backtest import (
    fetch_underlying_theta, _chain_for_entry, _pick_contract, simulate,
)
from ..paths import LOGS_DIR

BAR_MINUTES = 5          # width of the bars find_entries iterates
BREAKEVEN_WIN_RATE = 75.3  # measured: avg win +31.5%, avg loss -95.7%

# Modules that source entries from find_entries and therefore inherit the bug.
# The two marked RE-RUN here are backed by the ThetaData cache; the others read
# yfinance/Massive and are listed so nobody assumes they were checked.
AFFECTED = [
    ("spy_qqq_0dte_quotes_backtest.py", "RE-RUN BELOW", "real NBBO, definitive"),
    ("underlying_orb_longrun.py", "RE-RUN BELOW", "decade underlying study"),
    ("spy_qqq_0dte_real_backtest.py", "not re-run", "Massive traded prices"),
    ("retest_vs_gapandgo_backtest.py", "not re-run", "yfinance intraday"),
    ("options_premium_backtest_scalp.py", "not re-run", "yfinance intraday"),
    ("options_premium_backtest_confluence.py", "not re-run", "yfinance intraday"),
]


def honest_entries(symbol, five, daily, bar_minutes=BAR_MINUTES, strict=True):
    """`find_entries`, with entry_time moved to the bar's CLOSE.

    The returned dict keeps the original label under `label_time` so the two
    timings can be paired trade-for-trade. `today_bars` is re-sliced from the
    new decision time so downstream exit logic cannot see the trigger bar.
    """
    out = []
    for e in find_entries(symbol, five, daily):
        close_time = e["entry_time"] + pd.Timedelta(minutes=bar_minutes)
        if strict and close_time.time() > MORNING_END:
            continue
        e2 = dict(e)
        e2["label_time"] = e["entry_time"]
        e2["entry_time"] = close_time
        tb = e.get("today_bars")
        if tb is not None and len(tb):
            e2["today_bars"] = tb[tb.index > close_time]
        out.append(e2)
    return out


def _stat(v):
    v = pd.Series(v).dropna()
    if len(v) < 2:
        return None
    se = v.std() / math.sqrt(len(v))
    return {"n": len(v), "mean": v.mean(), "se": se, "t": v.mean() / se,
            "win": (v > 0).mean() * 100,
            "lo": v.mean() - 1.96 * se, "hi": v.mean() + 1.96 * se}


def audit_options(symbol, theta, start, strict):
    """Price every entry at the label and at the bar close, paired."""
    minute = fetch_underlying_theta(theta, symbol, start=start)
    five = regular_hours_5min(minute)
    daily = daily_from_minute(minute)

    honest = honest_entries(symbol, five, daily, strict=strict)
    rows = []
    for e in honest:
        try:
            chain = _chain_for_entry(theta, symbol, e)
        except Exception:
            continue
        if chain.empty:
            continue
        pick_lbl = _pick_contract(chain, e["label_time"])
        pick_hon = _pick_contract(chain, e["entry_time"])
        if pick_lbl is None or pick_hon is None:
            continue
        row = {"date": e["date"], "symbol": symbol,
               "direction": e["direction"],
               "label_time": e["label_time"].strftime("%H:%M"),
               "year": pd.Timestamp(e["date"]).year}
        for pt in PROFIT_TARGET_GRID + [None]:
            tag = f"t{int(pt*100)}" if pt else "hold"
            row[f"asis_{tag}"] = simulate(e, chain, pick_lbl, pt)["ret_pct_net"]
            row[f"hon_{tag}"] = simulate(e, chain, pick_hon, pt)["ret_pct_net"]
        rows.append(row)
    return pd.DataFrame(rows)


def audit_underlying(symbol, theta, start, strict):
    """Forward return / MFE measured from the label vs from the bar close."""
    minute = fetch_underlying_theta(theta, symbol, start=start)
    five = regular_hours_5min(minute)
    daily = daily_from_minute(minute)

    bars = minute.between_time("09:30", "15:59")
    bars = bars[bars["Close"] > 0].copy()
    bars["d"] = bars.index.date
    by_day = {d: g for d, g in bars.groupby("d")}

    rows = []
    for e in honest_entries(symbol, five, daily, strict=strict):
        day = by_day.get(e["date"])
        if day is None or day.empty:
            continue
        up = e["direction"] == "UP"
        sign = 1.0 if up else -1.0
        rec = {"date": e["date"], "symbol": symbol,
               "year": pd.Timestamp(e["date"]).year}
        ok = True
        for tag, t0 in (("asis", e["label_time"]), ("hon", e["entry_time"])):
            after = day[day.index >= t0]
            if len(after) < 2:
                ok = False
                break
            spot = float(after["Open"].iloc[0])
            rec[f"fwd_{tag}"] = sign * (float(after["Close"].iloc[-1]) / spot - 1) * 100
            rec[f"mfe_{tag}"] = ((float(after["High"].max()) / spot - 1) * 100 if up
                                 else (1 - float(after["Low"].min()) / spot) * 100)
        if ok:
            rows.append(rec)
    return pd.DataFrame(rows)


def report(opt, und, strict):
    line = "=" * 94
    mode = "STRICT (decision by 09:45)" if strict else "LOOSE (09:45 bar acts at 09:50)"
    print(f"\n{line}\nHONEST-TIMING AUDIT  --  {mode}\n{line}")

    print("\nOPTIONS, net of commission. 'honest' = entry at bar close.")
    print(f"breakeven win rate ~{BREAKEVEN_WIN_RATE}%\n")
    hdr = (f"{'sym':4s} {'variant':6s} {'n':>5s} | {'as-is mean':>10s} {'win':>6s} "
           f"| {'HONEST mean':>11s} {'win':>6s} {'t':>6s} {'95% CI':>18s} | {'bias':>7s}")
    print(hdr + "\n" + "-" * len(hdr))
    for sym, g in opt.groupby("symbol"):
        for pt in PROFIT_TARGET_GRID + [None]:
            tag = f"t{int(pt*100)}" if pt else "hold"
            a, h = _stat(g[f"asis_{tag}"]), _stat(g[f"hon_{tag}"])
            if not a or not h:
                continue
            flag = "" if h["win"] >= BREAKEVEN_WIN_RATE else "  <-below BE"
            print(f"{sym:4s} {tag:6s} {h['n']:5d} | {a['mean']:+9.2f}% {a['win']:5.1f}% "
                  f"| {h['mean']:+10.2f}% {h['win']:5.1f}% {h['t']:+6.2f} "
                  f"[{h['lo']:+6.2f},{h['hi']:+6.2f}] | {a['mean']-h['mean']:+6.2f}{flag}")

    print(f"\n{line}\nUNDERLYING forward return (better SNR than option payoff)\n{line}")
    for sym, g in und.groupby("symbol"):
        a, h = _stat(g["fwd_asis"]), _stat(g["fwd_hon"])
        if not a or not h:
            continue
        pa = (g.groupby("year")["fwd_asis"].mean() > 0).sum()
        ph = (g.groupby("year")["fwd_hon"].mean() > 0).sum()
        ny = g["year"].nunique()
        print(f"  {sym} n={h['n']}")
        print(f"     as-is  {a['mean']:+.4f}%  t={a['t']:+5.2f}  {pa}/{ny} yrs  "
              f"MFE>0.20% {100*(g['mfe_asis']>0.20).mean():.1f}%")
        print(f"     HONEST {h['mean']:+.4f}%  t={h['t']:+5.2f}  {ph}/{ny} yrs  "
              f"MFE>0.20% {100*(g['mfe_hon']>0.20).mean():.1f}%")
        d = (g["fwd_asis"] - g["fwd_hon"]).dropna()
        se = d.std() / math.sqrt(len(d))
        share = 100 * d.mean() / a["mean"] if a["mean"] else float("nan")
        print(f"     drift handed over: {d.mean():+.4f}% (paired t={d.mean()/se:+.2f}) "
              f"= {share:.0f}% of the as-is result")

    print(f"\n{line}\nAFFECTED MODULES\n{line}")
    for f, status, note in AFFECTED:
        print(f"  {status:14s} {f:42s} {note}")
    print("\n  All six source entries from find_entries and inherit the bug.")
    print("  The four not re-run here read yfinance/Massive, not the ThetaData cache.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", nargs="*", default=SYMBOLS)
    ap.add_argument("--start", default=JOINT_HISTORY_START)
    ap.add_argument("--underlying-start", default="2016-01-01",
                    help="underlying study can reach further back than options")
    ap.add_argument("--loose", action="store_true",
                    help="keep the 09:45-labelled bar (acts at 09:50)")
    args = ap.parse_args()
    strict = not args.loose

    theta = ThetaDataClient()
    opt = pd.concat([audit_options(s, theta, args.start, strict)
                     for s in args.symbols], ignore_index=True)
    und = pd.concat([audit_underlying(s, theta, args.underlying_start, strict)
                     for s in args.symbols], ignore_index=True)
    report(opt, und, strict)

    suffix = "strict" if strict else "loose"
    opt.to_csv(LOGS_DIR / f"decision_time_audit_options_{suffix}.csv", index=False)
    und.to_csv(LOGS_DIR / f"decision_time_audit_underlying_{suffix}.csv", index=False)
    print(f"\nSaved -> decision_time_audit_*_{suffix}.csv")
    print(f"theta API: {theta.stats()}")
