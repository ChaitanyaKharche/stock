"""Does the variance risk premium survive the spread you would actually pay?

    python -m trade_analysis.hpc.cost_model --symbol SPY --limit 60     # smoke
    python -m trade_analysis.hpc.cost_model --symbol SPY                # full

Section 6 of research/vrp_preregistration.md: *"the improvement surviving a realistic cost
model: the measured ATM straddle bid-ask spread at the forecast origin, not an assumed
one"*, and *"A model that only wins gross of spread is a null."*

This answers that directly and separately from the forecasting question. It reads the raw
option archive rather than the built frame, so nothing here can perturb the frozen
dataset -- and it never touches 2024, which stays sealed per section 6.

THE TRADE
---------
At origin `t`: SELL the ATM 0DTE straddle. Hold `h` minutes. Buy it back.

    gross P&L = mid(t) - mid(t+h)          what a mid-fill fantasy would earn
    net   P&L = bid(t) - ask(t+h)          what a taker actually gets

**Selling at the bid and buying back at the ask is the whole point.** Assuming mid fills
is the single most common way an options backtest invents an edge that does not exist, and
on 0DTE SPY the ATM straddle spread measured in this very archive runs 200-600 bp of mid.
An edge of a few bp cannot survive that, and the only way to know is to subtract it.

WHAT THIS DELIBERATELY DOES NOT DO
----------------------------------
No delta hedge. An unhedged short straddle is a bet on magnitude AND carries directional
risk from drift, so this measures the retail instrument -- sell a straddle, sit on it --
rather than a clean variance swap. A delta-hedged version would isolate variance better
and is the natural refinement; it is not claimed here.

No position sizing, no portfolio, no compounding. Overlapping origins mean six positions
open at once at h=30, which is fine for measuring PER-TRADE economics and meaningless as a
return series. Results are reported per trade and clustered by session, matching the
Diebold-Mariano treatment used for the forecasts.
"""
from __future__ import annotations

import argparse
import datetime as dt

import numpy as np
import pandas as pd

from .build_vrp_dataset import (ORIGIN_END, ORIGIN_START, load_bars, load_option_quotes,
                                sessions_for)

HORIZON_MINUTES = 30
STALE_LIMIT = pd.Timedelta(minutes=5)


def _wide(quotes: pd.DataFrame, right: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(bid, ask) as timestamp x strike frames for one option right.

    Pivoting once per session and using `asof` afterwards turns ~92,000 range-filters into
    a couple of hundred index lookups; done naively this analysis takes hours.
    """
    side = quotes[quotes["right"].str.upper().str.startswith(right)]
    if side.empty:
        return pd.DataFrame(), pd.DataFrame()
    bid = side.pivot_table(index="timestamp", columns="strike", values="bid",
                           aggfunc="last").sort_index()
    ask = side.pivot_table(index="timestamp", columns="strike", values="ask",
                           aggfunc="last").sort_index()
    return bid, ask


def _quote_at(bid: pd.DataFrame, ask: pd.DataFrame, t: pd.Timestamp, strike: float):
    """Last bid/ask at or before `t` for one strike, or None if stale/absent."""
    if bid.empty or strike not in bid.columns:
        return None
    idx = bid.index[bid.index <= t]
    if len(idx) == 0 or (t - idx[-1]) > STALE_LIMIT:
        return None
    b, a = bid.at[idx[-1], strike], ask.at[idx[-1], strike]
    if not (np.isfinite(b) and np.isfinite(a)) or b <= 0 or a < b:
        return None
    return float(b), float(a)


def session_trades(symbol: str, day: dt.date, horizon: int) -> pd.DataFrame | None:
    """One row per origin: what selling the ATM straddle there would have paid."""
    bars = load_bars(symbol, day)
    if bars is None or len(bars) < 20:
        return None
    quotes = load_option_quotes(symbol, day)
    if quotes is None or quotes.empty:
        return None

    cb, ca = _wide(quotes, "C")
    pb, pa = _wide(quotes, "P")
    if cb.empty or pb.empty:
        return None
    strikes = np.asarray(sorted(set(cb.columns) & set(pb.columns)), dtype=float)
    if strikes.size < 3:
        return None

    rows = []
    for t in bars.index:
        if not (ORIGIN_START <= t.time() <= ORIGIN_END):
            continue
        hist = bars.loc[bars.index < t]          # no-lookahead, as in the builder
        if len(hist) < 2:
            continue
        spot = float(hist["close"].iloc[-1])
        k = float(strikes[np.argmin(np.abs(strikes - spot))])

        entry_c = _quote_at(cb, ca, t, k)
        entry_p = _quote_at(pb, pa, t, k)
        if not (entry_c and entry_p):
            continue

        t_exit = t + pd.Timedelta(minutes=horizon)
        # Never hold past the close: a 0DTE straddle at 16:00 is settlement, not a quote.
        if t_exit.time() > dt.time(15, 59):
            continue
        exit_c = _quote_at(cb, ca, t_exit, k)
        exit_p = _quote_at(pb, pa, t_exit, k)
        if not (exit_c and exit_p):
            continue

        entry_mid = (entry_c[0] + entry_c[1]) / 2 + (entry_p[0] + entry_p[1]) / 2
        exit_mid = (exit_c[0] + exit_c[1]) / 2 + (exit_p[0] + exit_p[1]) / 2
        entry_bid = entry_c[0] + entry_p[0]      # sell: you get hit on the bid
        exit_ask = exit_c[1] + exit_p[1]         # buy back: you pay the ask
        if entry_mid <= 0:
            continue

        rows.append({
            "date": day.isoformat(), "t": t, "strike": k, "spot": spot,
            "entry_mid": entry_mid, "exit_mid": exit_mid,
            "entry_bid": entry_bid, "exit_ask": exit_ask,
            "gross": entry_mid - exit_mid,       # the mid-fill fantasy
            "net": entry_bid - exit_ask,         # what a taker actually gets
            "spread_cost": (entry_mid - entry_bid) + (exit_ask - exit_mid),
            "entry_spread_bp": (entry_c[1] - entry_c[0] + entry_p[1] - entry_p[0])
            / entry_mid * 1e4,
        })
    return pd.DataFrame(rows) if rows else None


def report(df: pd.DataFrame, horizon: int) -> None:
    n, sess = len(df), df["date"].nunique()
    print(f"\n  {n:,} trades over {sess} sessions   (sell ATM straddle, hold {horizon}m)")
    print(f"  entry straddle mid: median ${df['entry_mid'].median():.2f}")
    print(f"  ATM spread at entry: median {df['entry_spread_bp'].median():.0f} bp of mid")

    print(f"\n  {'':<10}{'mean/trade':>13}{'median':>10}{'win rate':>11}{'total':>12}")
    for col, label in (("gross", "GROSS"), ("net", "NET")):
        s = df[col]
        print(f"  {label:<10}{s.mean():>13.4f}{s.median():>10.4f}"
              f"{(s > 0).mean() * 100:>10.1f}%{s.sum():>12.2f}")
    print(f"  {'spread':<10}{df['spread_cost'].mean():>13.4f}"
          f"{df['spread_cost'].median():>10.4f}")

    # Session-clustered t-test. Per-trade independence is false -- six positions overlap
    # at h=30 and all of them see the same tape -- so the honest unit is the session.
    for col, label in (("gross", "GROSS"), ("net", "NET")):
        d = df.groupby("date")[col].mean()
        se = d.std(ddof=1) / np.sqrt(len(d))
        t = d.mean() / se if se > 0 else np.nan
        print(f"  {label} per session: mean {d.mean():+.4f}  t={t:+.2f}  n={len(d)}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--symbol", default="SPY")
    ap.add_argument("--horizon", type=int, default=HORIZON_MINUTES)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="")
    args = ap.parse_args(argv)

    days = [d for d in sessions_for(args.symbol) if d.year < 2024]
    if args.limit:
        days = days[:args.limit]
    print(f"{len(days)} sessions (2024 excluded: held out, sealed per section 6)")

    frames = []
    for i, day in enumerate(days, 1):
        f = session_trades(args.symbol, day, args.horizon)
        if f is not None:
            frames.append(f)
        if i % 50 == 0:
            print(f"  {i}/{len(days)}  trades so far {sum(len(x) for x in frames):,}")
    if not frames:
        print("no tradeable origins found")
        return 1

    df = pd.concat(frames, ignore_index=True)
    report(df, args.horizon)
    if args.out:
        df.to_parquet(args.out, index=False)
        print(f"\n  wrote {args.out}")
    print("\n  GROSS is the number a mid-fill backtest would print. NET is the trade.")
    print("  Section 6: a model that only wins gross of spread is a null.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
