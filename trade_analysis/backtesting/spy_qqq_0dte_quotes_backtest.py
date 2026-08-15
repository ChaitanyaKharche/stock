"""
The same strategy, priced on REAL NBBO quotes instead of traded prices.

This is the run the $40 ThetaData Value subscription was bought for. It removes
the single weakest assumption in spy_qqq_0dte_real_backtest.py: that the
round-trip spread is 2%. Here there is no spread assumption at all - entries
pay the ASK and exits hit the BID, both observed.

WHAT CHANGES VS. THE MASSIVE VERSION
------------------------------------
  entry fill   assumed traded-vwap * (1 + 1%)   ->  the actual ASK
  exit fill    assumed traded-vwap * (1 - 1%)   ->  the actual BID
  target check 1-min traded vwap                ->  1-min BID (what you could sell at)
  strike pick  iterative probing, 2-4 API calls ->  whole chain in ONE call, priced on ask

Everything else is held constant on purpose: identical entries from
`find_entries` (unchanged, imported), identical symbols, identical target grid.
So a difference in results is attributable to the pricing, which is the point.

MEASURED SPREAD REALITY (SPY 532C, 2024-08-07, a contract the Massive run
actually traded): ~1.1% of mid near the money in the opening minutes, median
4.9% across the session, 90th percentile 66.7%. The wide tail is real but
occurs when the contract is quoted in pennies, where the DOLLAR cost is small.
The flat 2% assumption was therefore slightly pessimistic at entry and
optimistic on decayed losers - which is exactly why it needed measuring rather
than arguing about.

SUBSCRIPTION: Options Value + Stock Value are both paid, so underlying minute
bars now come from ThetaData too (regular session only, no 50k page cap, back
to 2021-01-01). Index remains FREE and its intraday endpoints need Standard,
so ThetaDataClient still refuses /index/ outright.

EXPECT LEGITIMATE `chain_empty` SKIPS BEFORE LATE 2022. SPY and QQQ did not
have Tuesday/Thursday expirations until 2022 - before that only Mon/Wed/Fri
had a same-day contract. A Tuesday entry signal in June 2022 has no 0DTE
option to buy, so skipping it is correct, not a data failure. Roughly 40% of
sessions in 2021-mid-2022 are unavailable for this reason, and the skip
itemisation in run_symbol reports them so they can't be mistaken for
selection bias.

ISOLATION: imports only, edits nothing, writes its own CSVs. Delete freely.
"""
import argparse

import numpy as np
import pandas as pd

from ..data_sources.thetadata_client import ThetaDataClient, STOCK_HISTORY_START
from .options_premium_backtest import find_entries, CONTRACT_MULTIPLIER
from .spy_qqq_0dte_real_backtest import (
    SYMBOLS, TARGET_PREMIUM, CONTRACTS, PROFIT_TARGET_GRID,
    regular_hours_5min, daily_from_minute,
)
from .vilkov_0dte_conditional_backtest import describe
from ..paths import LOGS_DIR

STRIKE_RANGE = 15        # strikes above/below spot to pull in the chain request
MIN_ASK_SIZE = 1         # a quote with no size behind it isn't an executable fill
MAX_QUOTE_WAIT_BARS = 10 # minutes to wait for a two-sided market before giving up

# Exits must be executable too, not just non-zero. A bid quoted with zero size
# is not something you can sell into, and accepting one would manufacture
# profit. Set to CONTRACTS so the whole position can actually be lifted.
# Spot-checked before enforcing: on 60 sampled 35%-target exits, 0% sat on a
# zero-size bid and median bid size was 38.5 contracts - so this filter should
# barely bind. It is here so executability is guaranteed by the code rather
# than by a sample I happened to check.
MIN_EXIT_BID_SIZE = CONTRACTS
FEE_PER_CONTRACT = 0.65  # typical retail options commission, per contract per side


def fetch_underlying_theta(theta, symbol, start=STOCK_HISTORY_START, end=None):
    """Minute bars from the Stock Value tier, in 1-month chunks.

    Replaces the Massive free-tier loader for this file. Two reasons that
    matter beyond just reaching further back:
      - Massive free hard-stops ~2 years ("Your plan doesn't include this data
        timeframe"), which capped the study at 500 sessions.
      - ThetaData returns REGULAR SESSION ONLY (391 bars/day), so there is no
        extended-hours contamination of yesterday's high/low and no 50,000-row
        page cap to silently truncate a chunk - both of which bit the Massive
        version.
    The 1-month chunking is not a guess: the API rejects anything larger with
    "Bulk history requests are limited to no more than 1 month".
    """
    # stop at yesterday: today's session may be incomplete, and a partial day
    # would produce a truncated daily high/low for tomorrow's breakout level
    end = pd.Timestamp(end or (pd.Timestamp.today().normalize() - pd.Timedelta(days=1)))
    cursor = pd.Timestamp(start)
    frames = []
    while cursor < end:
        stop = min(cursor + pd.DateOffset(days=27), end)
        df = theta.stock_minute_bars(symbol, cursor.date(), stop.date())
        if not df.empty:
            frames.append(df)
        cursor = stop + pd.Timedelta(days=1)
    if not frames:
        return pd.DataFrame()
    bars = pd.concat(frames)
    return bars[~bars.index.duplicated(keep="first")].sort_index()


def _chain_for_entry(theta, symbol, entry):
    """One request: the full 0DTE chain for this session, one side, 1-minute NBBO."""
    right = "call" if entry["direction"] == "UP" else "put"
    return theta.quotes(symbol, expiration=entry["date"], date=entry["date"],
                        right=right, interval="1m", strike_range=STRIKE_RANGE)


def _pick_contract(chain, entry_time, target_premium=TARGET_PREMIUM):
    """Choose the strike whose ASK at the entry minute is closest to target.

    Priced on the ask because that is what a buyer actually pays - selecting on
    mid would systematically pick a contract you could not buy at that price.
    """
    at_entry = chain[chain["timestamp"] >= entry_time]
    if at_entry.empty:
        return None

    # Walk forward to the first minute that is actually QUOTED. The 09:30 bar
    # very often carries bid=ask=0 with zero size - the exchange has published a
    # bar but no two-sided market yet - and most entries fire at 09:30. Taking
    # only the first timestamp therefore discarded ~5 of every 6 entries.
    # Capped so a dead strike can't silently drag entry far from the signal.
    for ts in at_entry["timestamp"].drop_duplicates().sort_values()[:MAX_QUOTE_WAIT_BARS]:
        snap = at_entry[at_entry["timestamp"] == ts]
        snap = snap[(snap["ask"] > 0) & (snap["bid"] > 0) &
                    (snap["ask_size"] >= MIN_ASK_SIZE)]
        if snap.empty:
            continue
        snap = snap.assign(gap=(snap["ask"] - target_premium).abs())
        best = snap.loc[snap["gap"].idxmin()]
        return {"strike": float(best["strike"]), "entry_ts": ts,
                "entry_ask": float(best["ask"]), "entry_bid": float(best["bid"]),
                "quote_delay_bars": int(
                    (ts - at_entry["timestamp"].min()).total_seconds() // 60)}
    return None


def simulate(entry, chain, pick, profit_target=None):
    """Buy at the ask, sell at the bid. No spread assumption anywhere."""
    leg = chain[chain["strike"] == pick["strike"]].sort_values("timestamp")
    after = leg[leg["timestamp"] > pick["entry_ts"]]
    # require a bid you could actually hit for the full position
    after = after[(after["bid"] > 0) & (after["bid_size"] >= MIN_EXIT_BID_SIZE)]

    entry_fill = pick["entry_ask"]
    exit_fill, exit_ts, reason = None, None, None

    if profit_target is not None:
        for ts, bid in zip(after["timestamp"], after["bid"]):
            if float(bid) / entry_fill - 1 >= profit_target:
                exit_fill, exit_ts, reason = float(bid), ts, "TARGET"
                break

    if exit_fill is None:
        # mark out at the last executable bid; a 0DTE that stops being bid is
        # worthless, so 0.0 rather than carrying the last non-zero print forward
        if len(after):
            exit_fill, exit_ts = float(after["bid"].iloc[-1]), after["timestamp"].iloc[-1]
        else:
            exit_fill, exit_ts = 0.0, pick["entry_ts"]
        reason = "CLOSE"

    gross = (exit_fill - entry_fill) * CONTRACT_MULTIPLIER
    net = gross - 2 * FEE_PER_CONTRACT           # commission both sides
    return {
        "date": entry["date"], "direction": entry["direction"],
        "gap_regime": entry["gap_regime"], "strike": pick["strike"],
        "entry_time": str(pick["entry_ts"]), "entry_ask": round(entry_fill, 3),
        "entry_spread_pct": round((pick["entry_ask"] - pick["entry_bid"])
                                  / ((pick["entry_ask"] + pick["entry_bid"]) / 2) * 100, 2),
        "exit_time": str(exit_ts), "exit_bid": round(exit_fill, 3),
        "exit_reason": reason,
        "ret_pct": round((exit_fill / entry_fill - 1) * 100, 2),
        "ret_pct_net": round(net / (entry_fill * CONTRACT_MULTIPLIER) * 100, 2),
        "pnl_per_contract": round(net, 2),
        "pnl_position": round(net * CONTRACTS, 2),
    }


def run_symbol(symbol, theta, max_entries=None, start=STOCK_HISTORY_START):
    minute = fetch_underlying_theta(theta, symbol, start=start)
    five = regular_hours_5min(minute)
    daily = daily_from_minute(minute)
    entries = find_entries(symbol, five, daily)
    if max_entries:
        entries = entries[:max_entries]
    print(f"\n=== {symbol} === {len(entries)} entries over {len(daily)} sessions "
          f"({daily.index.min().date()} -> {daily.index.max().date()})")

    # Skip reasons are itemised, not just counted. A silent skip count hides
    # selection bias: if the dropped sessions are systematically the illiquid or
    # fast-moving ones, the surviving sample is flattered and the headline mean
    # is not comparable across symbols.
    priced, skips = [], {"fetch_error": [], "chain_empty": [], "no_quoted_strike": []}
    for i, e in enumerate(entries, 1):
        try:
            chain = _chain_for_entry(theta, symbol, e)
        except Exception as exc:
            skips["fetch_error"].append((e["date"], type(exc).__name__))
            continue
        if chain.empty:
            skips["chain_empty"].append((e["date"], None))
            continue
        pick = _pick_contract(chain, e["entry_time"])
        if pick is None:
            quoted = chain[(chain["bid"] > 0) & (chain["ask"] > 0)]
            skips["no_quoted_strike"].append(
                (e["date"], f"{len(chain)} rows, {len(quoted)} two-sided"))
            continue
        priced.append((e, chain, pick))
        if i % 25 == 0:
            n_skip = sum(len(v) for v in skips.values())
            print(f"    {i}/{len(entries)} | priced {len(priced)} skipped {n_skip} "
                  f"| {theta.stats()}")

    n_skip = sum(len(v) for v in skips.values())
    print(f"  priced {len(priced)}, skipped {n_skip}")
    for reason, items in skips.items():
        if items:
            print(f"    {reason}: {len(items)}  e.g. {items[:3]}")
    return priced


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", nargs="*", default=SYMBOLS)
    ap.add_argument("--max-entries", type=int, default=None)
    ap.add_argument("--start", default=STOCK_HISTORY_START,
                    help="earliest session; Stock Value serves from 2021-01-01")
    args = ap.parse_args()

    theta = ThetaDataClient()

    frames, summary = [], []
    for sym in args.symbols:
        priced = run_symbol(sym, theta, args.max_entries, start=args.start)
        if not priced:
            continue
        for pt in PROFIT_TARGET_GRID + [None]:
            rows = [simulate(e, c, p, pt) for e, c, p in priced]
            df = pd.DataFrame(rows)
            label = f"target {int(pt*100)}%" if pt else "hold to close"
            df["variant"], df["symbol"] = label, sym
            frames.append(df)
            summary.append(describe(df["ret_pct_net"], f"{sym} | {label} | net"))
            summary.append(describe(df["ret_pct"], f"{sym} | {label} | gross"))
            for d in ["UP", "DOWN"]:
                seg = df[df["direction"] == d]
                if len(seg) > 1:
                    summary.append(describe(seg["ret_pct_net"], f"{sym} | {label} | {d} net"))

    if frames:
        all_df = pd.concat(frames, ignore_index=True)
        sdf = pd.DataFrame(summary)
        print("\n" + "=" * 108)
        print("REAL NBBO: buy the ask, sell the bid. 'net' includes "
              f"${FEE_PER_CONTRACT}/contract/side commission.")
        print("=" * 108)
        print(sdf.to_string(index=False))

        ent = all_df[all_df["variant"] == "target 25%"]
        print(f"\nobserved entry spread (% of mid): median "
              f"{ent['entry_spread_pct'].median():.2f}%, "
              f"90th pct {ent['entry_spread_pct'].quantile(0.9):.2f}%")
        print(f"observed entry ask: median ${ent['entry_ask'].median():.2f} "
              f"(targeting ${TARGET_PREMIUM})")

        t_out = LOGS_DIR / "spy_qqq_0dte_quotes_trades.csv"
        all_df.to_csv(t_out, index=False)
        sdf.to_csv(LOGS_DIR / "spy_qqq_0dte_quotes_summary.csv", index=False)
        print(f"\nSaved -> {t_out}")
        print(f"theta API: {theta.stats()}")
