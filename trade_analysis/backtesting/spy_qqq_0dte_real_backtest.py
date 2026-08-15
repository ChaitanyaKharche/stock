"""
The live rule, on the live instruments, with real option prices - the first
test in this package with none of the three big substitutions.

Previous attempts each had to fake something:
  options_premium_backtest.py       - synthetic Black-Scholes premium, 60 days
  ..._confluence / ..._scalp        - same synthetic premium, ~30 trades
  vilkov_0dte_conditional_backtest  - REAL premium, 642 trades, but SPX only
                                      and entry forced to 10:00 ET

This file closes both of the Vilkov file's gaps: SPY and QQQ (the symbols
actually traded) at the real 9:30-9:45 retest entry, using minute bars for
already-expired 0DTE contracts from Massive's free tier.

It is also a genuine OUT-OF-SAMPLE window: Vilkov's panel ends 2024-05-01 and
Massive's free tier reaches back ~2 years, so this covers roughly 2024-08
onward - a period the SPX findings never saw.

WHAT IS REUSED VS. NEW
----------------------
Entry detection is `find_entries` imported unchanged from
options_premium_backtest.py - the proven yesterday-H/L + 9:30-9:45 retest
mechanics, fed 5-minute bars resampled from real 1-minute data. No new entry
logic. Stats helpers are imported from vilkov_0dte_conditional_backtest.py so
there is one definition of the confidence intervals (a deliberate coupling
between the two new files; if you delete that one, copy `describe` over).

THE ONE THING STILL MISSING, AND IT MATTERS
-------------------------------------------
The free tier has NO bid/ask - quotes are $199/mo. So option bars here are
TRADED prices. Consequences, stated plainly:

  - Entry/exit fills are modelled from traded prices plus an assumed spread
    (SPREAD_PCT), not from a real quote. The assumption is calibrated off the
    real `bas` column in the Vilkov SPX panel rather than invented, but it is
    still an assumption and it is the weakest link in this file.
  - A minute with no trades has no bar at all. For far-OTM 0DTE strikes that
    is common, so `n_trades` is used to reject illiquid contracts up front.
  - Because a thin strike's printed high/low can be a single odd-lot, exits
    are evaluated on `vwap` where available rather than the bar extreme, which
    is more conservative than assuming a touch of the high filled.

Other limits: 0DTE SPY/QQQ contracts only existed on every weekday from
mid-2022 onward, which is fine for this window. Assignment risk is ignored -
these are American-style but a long holder controls exercise, so for a
buy-only book it is a non-issue.

ISOLATION: imports only; edits nothing. Writes its own CSVs. Delete freely.
"""
import argparse
from datetime import date, timedelta

import numpy as np
import pandas as pd

from ..data_sources.massive_client import MassiveClient, occ_ticker, ET
from ..signals.gamma_exposure import bs_price
from .options_premium_backtest import (
    find_entries, realized_vol, RISK_FREE_RATE, CONTRACT_MULTIPLIER,
    MORNING_START, MORNING_END,
)
from .vilkov_0dte_conditional_backtest import describe
from ..paths import LOGS_DIR

SYMBOLS = ["SPY", "QQQ"]
LOOKBACK_DAYS = 728          # just inside the free tier's 2-year window
CHUNK_DAYS = 40              # see note below - must stay under the 50k page cap
# The vendor serves extended hours (04:00-20:00 ET), so a session is ~960
# minute bars, not 390. 90 calendar days is ~62 sessions ~= 59k rows, which
# silently truncates at the 50,000-row page limit and drops the tail of every
# chunk. 40 days is ~28 sessions ~= 27k rows. The truncation warning in
# massive_client._aggregates is what caught this; leave it in.

TARGET_PREMIUM = 1.25        # midpoint of the user's stated $1.00-1.50 entries
CONTRACTS = 2                # "2-3 lots"
STRIKE_INCREMENT = 1.0       # SPY/QQQ 0DTE list on $1 strikes near the money
MAX_STRIKE_PROBES = 4        # each probe is one API call = 12.6s of rate limit
MIN_ENTRY_TRADES = 25        # reject a contract whose entry minute barely traded

# Calibrated from the Vilkov SPX panel's real `bas` (spread/spot) rather than
# guessed: near-the-money 0DTE spreads there run ~1-2% of premium. 2% is the
# conservative end for SPY/QQQ, which are tighter than SPX in absolute terms.
SPREAD_PCT = 0.02

PROFIT_TARGET_GRID = [0.25, 0.35, 0.50]


# ----------------------------------------------------------------------------
# underlying
# ----------------------------------------------------------------------------
def fetch_underlying(client, symbol, lookback_days=LOOKBACK_DAYS):
    """2 years of 1-minute bars, in ~90-day chunks. Cached, so this is a
    one-time ~8 calls per symbol."""
    end = date.today() - timedelta(days=1)
    start = end - timedelta(days=lookback_days)
    frames, cursor = [], start
    while cursor < end:
        stop = min(cursor + timedelta(days=CHUNK_DAYS), end)
        frames.append(client.stock_minute_bars(symbol, cursor, stop))
        cursor = stop + timedelta(days=1)
    bars = pd.concat([f for f in frames if not f.empty])
    return bars[~bars.index.duplicated(keep="first")].sort_index()


def regular_hours_5min(minute_bars):
    """Regular session only, resampled to 5 minutes.

    Both steps are required for `find_entries` to mean the same thing it means
    live: the vendor serves 04:00-20:00, but the live strategy reads yesterday's
    high/low off regular-hours bars, and an overnight print would corrupt the
    breakout level. The proven code also expects 5-minute bars.
    """
    rth = minute_bars.between_time("09:30", "15:59")
    agg = {"Open": "first", "High": "max", "Low": "min", "Close": "last",
           "Volume": "sum"}
    out = (rth.resample("5min").agg(agg).dropna(subset=["Open"]))
    return out.between_time("09:30", "15:55")


def daily_from_minute(minute_bars):
    """Daily OHLC built from the same minute bars rather than yfinance.

    Deliberate: yfinance daily with auto_adjust back-adjusts SPY/QQQ for
    dividends, which shifts historical closes and would silently disagree with
    the unadjusted intraday bars the breakout levels come from.
    """
    rth = minute_bars.between_time("09:30", "15:59")
    daily = rth.resample("1D").agg({"Open": "first", "High": "max",
                                    "Low": "min", "Close": "last"}).dropna()
    daily.index = pd.to_datetime(daily.index.date)
    return daily


# ----------------------------------------------------------------------------
# strike selection
# ----------------------------------------------------------------------------
def _minutes_to_close(ts):
    close = ts.replace(hour=16, minute=0, second=0, microsecond=0)
    return max((close - ts).total_seconds() / 60.0, 1.0)


def estimate_strike(spot, entry_time, iv, option_type, target_premium=TARGET_PREMIUM):
    """Black-Scholes seed for the strike that costs ~target_premium.

    Only a SEED. Trailing realized vol is a poor stand-in for 0DTE implied vol -
    it is backward-looking, so right after a vol spike (e.g. the week of
    2024-08-05) it stays elevated while forward IV has already mean-reverted.
    Seeding with it alone lands 2-3 strikes too far OTM and buys a $0.40 option
    when the rule calls for $1.25. `build_trade` corrects for that by walking
    the ladder against REAL traded prices; this just picks the starting point.
    """
    t_years = _minutes_to_close(entry_time) / (60 * 6.5 * 252)
    lo, hi = spot * 0.90, spot * 1.10
    for _ in range(50):
        mid = (lo + hi) / 2
        price = bs_price(spot, mid, t_years, iv, RISK_FREE_RATE,
                         "call" if option_type == "C" else "put")
        if price > target_premium:
            if option_type == "C":
                lo = mid
            else:
                hi = mid
        else:
            if option_type == "C":
                hi = mid
            else:
                lo = mid
    return round((lo + hi) / 2 / STRIKE_INCREMENT) * STRIKE_INCREMENT


# ----------------------------------------------------------------------------
# trade construction
# ----------------------------------------------------------------------------
def _probe_strike(client, symbol, entry, option_type, strike):
    """Fetch one contract and read its real premium at the entry bar."""
    ticker = occ_ticker(symbol, entry["date"], option_type, strike)
    bars = client.option_minute_bars(ticker, entry["date"])
    if bars.empty:
        return None
    at_entry = bars[bars.index >= entry["entry_time"]]
    if at_entry.empty:
        return None
    row = at_entry.iloc[0]
    price = float(row["vwap"] if pd.notna(row.get("vwap")) else row["Close"])
    if price <= 0 or float(row.get("n_trades", 0)) < MIN_ENTRY_TRADES:
        return None
    return {"ticker": ticker, "strike": strike, "bars": bars,
            "entry_price_opt": price, "entry_bar": at_entry.index[0],
            "gap": abs(price - TARGET_PREMIUM)}


def build_trade(client, symbol, entry, daily_df, tolerance=0.30,
                max_probes=MAX_STRIKE_PROBES):
    """Walk the strike ladder against REAL prices to land near TARGET_PREMIUM.

    Directional search rather than a fixed ladder, because each probe costs 12
    seconds of rate limit. Start from the Black-Scholes seed; if the real
    premium is too cheap, step toward the money, if too rich step away, and
    stop as soon as it is within `tolerance` of target. Typically 1-2 calls,
    capped at `max_probes`, and the closest probe seen is kept even if nothing
    lands inside tolerance.
    """
    option_type = "C" if entry["direction"] == "UP" else "P"
    spot = float(entry["entry_price"])
    iv = realized_vol(daily_df, entry["date"])
    strike = estimate_strike(spot, entry["entry_time"], iv, option_type)

    # stepping "away from the money" is +1 strike for a call, -1 for a put
    away = STRIKE_INCREMENT if option_type == "C" else -STRIKE_INCREMENT
    best, tried = None, set()

    for _ in range(max_probes):
        if strike in tried:
            break
        tried.add(strike)
        probe = _probe_strike(client, symbol, entry, option_type, strike)
        if probe is None:
            # untraded/illiquid strike: step toward the money, where liquidity is
            strike = round(strike - away, 2)
            continue
        if best is None or probe["gap"] < best["gap"]:
            best = probe
        if probe["gap"] <= tolerance * TARGET_PREMIUM:
            break
        # too cheap -> move toward the money; too rich -> move away
        strike = round(strike + (-away if probe["entry_price_opt"] < TARGET_PREMIUM
                                 else away), 2)
    return best


def simulate_trade(entry, contract, profit_target=None):
    """Replay the contract's own minute bars after entry.

    Exits are evaluated on vwap (falling back to Close) rather than the bar
    high: on a thin 0DTE strike the printed high can be one odd-lot trade, and
    assuming it filled would manufacture profit that wasn't available.
    """
    entry_fill = contract["entry_price_opt"] * (1 + SPREAD_PCT / 2)
    after = contract["bars"][contract["bars"].index > contract["entry_bar"]]
    px = (after["vwap"].fillna(after["Close"]) if "vwap" in after
          else after["Close"])

    exit_fill, exit_time, exit_reason = None, None, None
    if profit_target is not None:
        for ts, raw in px.items():
            candidate = max(float(raw) * (1 - SPREAD_PCT / 2), 0.0)
            if candidate / entry_fill - 1 >= profit_target:
                exit_fill, exit_time, exit_reason = candidate, ts, "TARGET"
                break

    if exit_fill is None:
        # no target, or never reached: mark out at the last traded print.
        # A 0DTE that stops printing is effectively worthless, hence 0.0.
        if len(px):
            exit_fill = max(float(px.iloc[-1]) * (1 - SPREAD_PCT / 2), 0.0)
            exit_time = px.index[-1]
        else:
            exit_fill, exit_time = 0.0, contract["entry_bar"]
        exit_reason = "CLOSE"

    pnl = (exit_fill - entry_fill) * CONTRACT_MULTIPLIER
    return {
        "date": entry["date"], "direction": entry["direction"],
        "gap_regime": entry["gap_regime"], "option_type": contract["ticker"][-9],
        "ticker": contract["ticker"], "strike": contract["strike"],
        "entry_time": str(entry_time := contract["entry_bar"]),
        "entry_premium": round(entry_fill, 3),
        "exit_time": str(exit_time), "exit_premium": round(exit_fill, 3),
        "exit_reason": exit_reason,
        "ret_pct": round((exit_fill / entry_fill - 1) * 100, 2),
        "pnl_per_contract": round(pnl, 2),
        "pnl_position": round(pnl * CONTRACTS, 2),
    }


def run_symbol(client, symbol, max_entries=None):
    print(f"\n=== {symbol} ===")
    minute = fetch_underlying(client, symbol)
    five = regular_hours_5min(minute)
    daily = daily_from_minute(minute)
    print(f"  {len(minute):,} minute bars | {len(five):,} 5-min bars | "
          f"{len(daily):,} sessions | {daily.index.min().date()} -> "
          f"{daily.index.max().date()}")

    entries = find_entries(symbol, five, daily)
    print(f"  {len(entries)} breakout entries at the real "
          f"{MORNING_START}-{MORNING_END} window")
    if max_entries:
        entries = entries[:max_entries]
        print(f"  (limited to {len(entries)} for this run)")

    contracts, skipped = [], 0
    for i, e in enumerate(entries, 1):
        c = build_trade(client, symbol, e, daily)
        if c is None:
            skipped += 1
        else:
            contracts.append((e, c))
        if i % 10 == 0:
            print(f"    {i}/{len(entries)} entries processed "
                  f"({len(contracts)} priced, {skipped} skipped) "
                  f"| {client.stats()}")
    print(f"  priced {len(contracts)}, skipped {skipped} (no liquid contract)")
    return entries, contracts


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--entries-only", action="store_true",
                    help="fetch underlying + detect entries, spend no option calls")
    ap.add_argument("--max-entries", type=int, default=None,
                    help="cap entries per symbol (each costs ~2 API calls)")
    ap.add_argument("--symbols", nargs="*", default=SYMBOLS)
    args = ap.parse_args()

    client = MassiveClient()
    all_trades, summary = [], []

    for sym in args.symbols:
        if args.entries_only:
            minute = fetch_underlying(client, sym)
            five = regular_hours_5min(minute)
            daily = daily_from_minute(minute)
            entries = find_entries(sym, five, daily)
            edf = pd.DataFrame([{k: v for k, v in e.items()
                                 if k != "today_bars"} for e in entries])
            print(f"\n=== {sym} ===")
            print(f"  {len(minute):,} minute bars covering "
                  f"{daily.index.min().date()} -> {daily.index.max().date()} "
                  f"({len(daily):,} sessions)")
            print(f"  {len(entries)} entries "
                  f"({len(entries)/max(len(daily),1)*100:.1f}% of sessions)")
            if not edf.empty:
                print(edf["direction"].value_counts().to_string())
                print(f"  entry times: "
                      f"{edf['entry_time'].dt.strftime('%H:%M').value_counts().to_dict()}")
            continue

        entries, contracts = run_symbol(client, sym, args.max_entries)
        for pt in PROFIT_TARGET_GRID + [None]:
            rows = [simulate_trade(e, c, pt) for e, c in contracts]
            if not rows:
                continue
            df = pd.DataFrame(rows)
            label = f"target {int(pt*100)}%" if pt else "hold to close"
            df["variant"] = label
            df["symbol"] = sym
            all_trades.append(df)
            summary.append({**describe(df["ret_pct"], f"{sym} | {label}")})
            for d in ["UP", "DOWN"]:
                seg = df[df["direction"] == d]
                summary.append({**describe(seg["ret_pct"], f"{sym} | {label} | {d}")})

    if summary:
        sdf = pd.DataFrame(summary)
        print("\n" + "=" * 100)
        print("RESULTS - real SPY/QQQ 0DTE traded prices, real 9:30-9:45 entry")
        print("=" * 100)
        print(sdf.to_string(index=False))
        t_out = LOGS_DIR / "spy_qqq_0dte_real_trades.csv"
        s_out = LOGS_DIR / "spy_qqq_0dte_real_summary.csv"
        pd.concat(all_trades, ignore_index=True).to_csv(t_out, index=False)
        sdf.to_csv(s_out, index=False)
        print(f"\nSaved trades  -> {t_out}")
        print(f"Saved summary -> {s_out}")
        print(f"API usage: {client.stats()}")
        print("\nWEAKEST LINK: fills use an assumed 2% spread (free tier has no "
              "bid/ask).\nTreat magnitudes as indicative; the sign and the "
              "UP-vs-DOWN split are the robust parts.")
