"""
Does a REAL retest beat the gap-and-go the code actually trades?

BACKGROUND
----------
`options_premium_backtest.find_entries()` and
`live_trading/swing_breakout_trader.check_morning_confirmation()` both document
a "retest confirmation", but both reduce to a tautology:

    price > yesterday_high  AND  price >= yesterday_high - 0.15*atr
                                 ^^^^^^ implied by the left side, always true

So neither filters anything, and what runs is a gap-and-go breakout: enter on
the first 5-min bar in 9:30-9:45 that closes beyond yesterday's high/low. On
500 sessions of real 1-minute SPY data, 206 of 255 entries fire on the very
first bar at 09:30 - the signature of no retest filter at all.

Two of the live scripts DO implement a genuine retest correctly
(`retest_breakout_websocket.py` and `multi_strategy_trader.py`), using a
bounded zone `zone_low <= low <= zone_high` plus a rejection candle. This file
ports that working pattern into a backtest and measures it head-to-head
against the gap-and-go, so the decision to change live code (or not) rests on
evidence rather than on which one sounds better.

WHAT IS HELD CONSTANT
---------------------
Both arms share: the same 5-minute bars resampled from real 1-minute data, the
same yesterday-high/low levels, the same ATR, and the SAME exit - `simulate_exit`
imported unchanged from options_premium_backtest.py at the default 1.0/2.5 ATR
stop/target. Only the ENTRY RULE differs. Any difference in results is
therefore attributable to the entry, which is the whole question.

RETEST PARAMETERS are ported verbatim from retest_breakout_websocket.py rather
than re-tuned, so this measures that script's actual logic:
  retest_atr_mult 0.15 | break_expiry 30/20/25 min by session phase
  volume filters 0.8 (open) / 1.0 (midday) / 0.9 (power hour)
  rejection: close > prev_close and close > low   (mirrored for shorts)

WHY THIS IS UNDERLYING-ONLY
---------------------------
No option prices here, deliberately. The entry question is about the
underlying's path, and adding synthetic premium would inject the same modelling
noise that made earlier files hard to read. It also costs zero API calls, so it
cannot collide with the rate limit while the real-option run is fetching.
The retest arm produces DIFFERENT entry timestamps, so pricing it properly
needs a separate option fetch once that finishes.

ISOLATION: imports only, edits nothing, writes its own CSV. Delete freely.
"""
import argparse
from datetime import time as dt_time

import numpy as np
import pandas as pd

from ..data_sources.massive_client import MassiveClient
from .options_premium_backtest import (
    find_entries, simulate_exit, compute_atr,
    STOP_ATR_MULT, TARGET_ATR_MULT, GAP_ATR_MULT_THRESHOLD,
    MORNING_START, MORNING_END,
)
from .spy_qqq_0dte_real_backtest import (
    SYMBOLS, fetch_underlying, regular_hours_5min, daily_from_minute,
)
from .vilkov_0dte_conditional_backtest import describe
from ..paths import LOGS_DIR

# --- ported verbatim from retest_breakout_websocket.py ---
RETEST_ATR_MULT = 0.15
BREAK_EXPIRY_MIN = {"opening": 30, "regular": 20, "power_hour": 25}
VOLUME_FILTERS = {"opening": 0.8, "mid_day": 1.0, "power_hour": 0.9}
VOLUME_LOOKBACK_BARS = 12          # 1 hour of 5-min bars, trailing only

SESSION_OPEN = dt_time(9, 30)
MIDDAY_START = dt_time(10, 0)
POWER_HOUR_START = dt_time(14, 0)
LAST_ENTRY = dt_time(15, 30)       # leave room for the exit to play out


def _phase(ts):
    t = ts.time()
    if t < MIDDAY_START:
        return "opening"
    if t >= POWER_HOUR_START:
        return "power_hour"
    return "mid_day"


def _expiry_minutes(ts):
    phase = _phase(ts)
    return BREAK_EXPIRY_MIN["opening" if phase == "opening"
                            else "power_hour" if phase == "power_hour"
                            else "regular"]


def find_retest_entries(symbol, intraday_df, daily_df, morning_break_only=False):
    """Genuine two-phase retest: break the level, pull back INTO a bounded zone
    around it, then reject in the breakout direction on above-average volume.

    Unlike the gap-and-go arm, this can produce NO trade for a session - a break
    that never gets retested inside the expiry window is simply skipped. That
    asymmetry is the point of the comparison, not a defect.

    morning_break_only: require the break itself to occur in the 9:30-9:45
    window, so the signal source matches the gap-and-go arm exactly (the retest
    may still land later, which is unavoidable - a retest takes time).
    """
    entries = []
    atr_all = compute_atr(intraday_df)
    vol_avg = intraday_df["Volume"].rolling(VOLUME_LOOKBACK_BARS).mean().shift(1)
    dates = sorted(set(intraday_df.index.date))

    for i in range(1, len(dates)):
        d, prev_d = dates[i], dates[i - 1]
        today = intraday_df[intraday_df.index.date == d]
        prev_day = intraday_df[intraday_df.index.date == prev_d]
        if today.empty or prev_day.empty:
            continue

        daily_hist = daily_df[daily_df.index.date < d]
        if len(daily_hist) < 2:
            continue
        yesterday_close = float(daily_hist["Close"].iloc[-1])
        yesterday_high = float(prev_day["High"].max())
        yesterday_low = float(prev_day["Low"].min())

        atr = atr_all.reindex(today.index).iloc[0]
        if pd.isna(atr) or atr <= 0:
            continue
        atr = float(atr)

        open_price = float(today["Open"].iloc[0])
        gap = abs(open_price - yesterday_close)
        gap_regime = ("ELEVATED" if gap > GAP_ATR_MULT_THRESHOLD * atr
                      else "NORMAL")

        # ---- phase 1: find the break ----
        break_ts, direction, level = None, None, None
        scan = today.between_time(MORNING_START, MORNING_END) if morning_break_only else today
        for ts, bar in scan.iterrows():
            if bar["Close"] > yesterday_high:
                break_ts, direction, level = ts, "UP", yesterday_high
                break
            if bar["Close"] < yesterday_low:
                break_ts, direction, level = ts, "DOWN", yesterday_low
                break
        if break_ts is None:
            continue

        # ---- phase 2: retest inside the expiry window ----
        expiry = _expiry_minutes(break_ts)
        window = today[(today.index > break_ts) &
                       (today.index <= break_ts + pd.Timedelta(minutes=expiry))]
        window = window[window.index.time <= LAST_ENTRY]

        zone_lo, zone_hi = level - RETEST_ATR_MULT * atr, level + RETEST_ATR_MULT * atr
        prev_close = float(today.loc[break_ts, "Close"])
        entry_ts, entry_price = None, None

        for ts, bar in window.iterrows():
            vt = VOLUME_FILTERS[_phase(ts)]
            avg = vol_avg.get(ts, np.nan)
            vol_ok = True if pd.isna(avg) or avg <= 0 else (bar["Volume"] / avg) >= vt

            if direction == "UP":
                in_zone = zone_lo <= bar["Low"] <= zone_hi
                rejecting = bar["Close"] > prev_close and bar["Close"] > bar["Low"]
            else:
                in_zone = zone_lo <= bar["High"] <= zone_hi
                rejecting = bar["Close"] < prev_close and bar["Close"] < bar["High"]

            if in_zone and rejecting and vol_ok:
                entry_ts, entry_price = ts, float(bar["Close"])
                break
            prev_close = float(bar["Close"])

        if entry_ts is None:
            continue

        entries.append({
            "date": d, "direction": direction, "gap_regime": gap_regime,
            "entry_time": entry_ts, "entry_price": entry_price, "atr": atr,
            "today_bars": today[today.index > entry_ts],
            "break_time": break_ts,
            "minutes_break_to_entry": (entry_ts - break_ts).total_seconds() / 60,
        })
    return entries


def excursions(entry, trade):
    """MFE/MAE in ATR units, plus minutes held.

    For an option buyer the *speed and size* of the favourable move matters more
    than the eventual close, because theta is charging the whole time - so these
    are reported alongside the plain target/stop outcome.
    """
    bars = entry["today_bars"]
    if bars.empty:
        return {"mfe_atr": 0.0, "mae_atr": 0.0, "minutes_held": 0.0}
    ep, atr = entry["entry_price"], entry["atr"]
    if entry["direction"] == "UP":
        mfe = (bars["High"].max() - ep) / atr
        mae = (ep - bars["Low"].min()) / atr
    else:
        mfe = (ep - bars["Low"].min()) / atr
        mae = (bars["High"].max() - ep) / atr
    held = (pd.Timestamp(trade["exit_time"]) - pd.Timestamp(entry["entry_time"])
            ).total_seconds() / 60
    return {"mfe_atr": round(float(mfe), 3), "mae_atr": round(float(mae), 3),
            "minutes_held": round(held, 1)}


def evaluate(entries, arm, symbol):
    """Same exit for both arms: simulate_exit at the proven 1.0/2.5 ATR multiples."""
    rows = []
    for e in entries:
        t = simulate_exit(e, STOP_ATR_MULT, TARGET_ATR_MULT)
        r_multiple = ((t["exit_price"] - e["entry_price"]) / (STOP_ATR_MULT * e["atr"])
                      * (1 if e["direction"] == "UP" else -1))
        rows.append({
            "symbol": symbol, "arm": arm, "date": t["date"],
            "direction": t["direction"], "gap_regime": t["gap_regime"],
            "entry_time": str(t["entry_time"]), "exit_reason": t["exit_reason"],
            "r_multiple": round(r_multiple, 3),
            # np.nan not None: the gap-and-go arm has no break phase, and None
            # makes the column object-dtype and trips a concat FutureWarning
            "minutes_break_to_entry": e.get("minutes_break_to_entry", np.nan),
            **excursions(e, t),
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", nargs="*", default=SYMBOLS)
    ap.add_argument("--morning-break-only", action="store_true",
                    help="require the BREAK in 9:30-9:45, matching the gap-and-go arm")
    args = ap.parse_args()

    client = MassiveClient(verbose=False)   # cached bars only, no live calls
    frames, summary = [], []

    for sym in args.symbols:
        minute = fetch_underlying(client, sym)
        five = regular_hours_5min(minute)
        daily = daily_from_minute(minute)

        gag = find_entries(sym, five, daily)
        rt = find_retest_entries(sym, five, daily,
                                 morning_break_only=args.morning_break_only)

        print(f"\n=== {sym} === ({len(daily)} sessions)")
        print(f"  gap-and-go (what runs today) : {len(gag):4d} entries "
              f"({len(gag)/len(daily)*100:.1f}% of sessions)")
        print(f"  real two-phase retest        : {len(rt):4d} entries "
              f"({len(rt)/len(daily)*100:.1f}% of sessions)")

        for entries, arm in [(gag, "gap_and_go"), (rt, "real_retest")]:
            if not entries:
                continue
            df = evaluate(entries, arm, sym)
            frames.append(df)
            stats = describe(df["r_multiple"], f"{sym} | {arm}")
            stats.update({
                "target_hit_pct": round((df["exit_reason"] == "TARGET").mean() * 100, 1),
                "stop_hit_pct": round((df["exit_reason"] == "STOP").mean() * 100, 1),
                "median_mfe_atr": round(df["mfe_atr"].median(), 2),
                "median_mae_atr": round(df["mae_atr"].median(), 2),
                "median_minutes_held": round(df["minutes_held"].median(), 0),
            })
            summary.append(stats)
            for d in ["UP", "DOWN"]:
                seg = df[df["direction"] == d]
                if len(seg) > 1:
                    summary.append(describe(seg["r_multiple"], f"{sym} | {arm} | {d}"))

        if rt:
            lag = pd.Series([e["minutes_break_to_entry"] for e in rt])
            print(f"  retest lag after break: median {lag.median():.0f} min, "
                  f"max {lag.max():.0f} min")

    if summary:
        sdf = pd.DataFrame(summary)
        print("\n" + "=" * 118)
        print("ENTRY RULE COMPARISON - identical bars, levels, ATR and exit; only "
              "the entry differs.  r_multiple is in units of the 1.0-ATR stop.")
        print("=" * 118)
        cols = [c for c in ["segment", "n", "win_rate", "mean_ret_pct", "mean_ci_lo",
                            "mean_ci_hi", "significant", "target_hit_pct",
                            "stop_hit_pct", "median_mfe_atr", "median_mae_atr",
                            "median_minutes_held"] if c in sdf.columns]
        print(sdf[cols].to_string(index=False))

        out = LOGS_DIR / "retest_vs_gapandgo_trades.csv"
        pd.concat(frames, ignore_index=True).to_csv(out, index=False)
        sdf.to_csv(LOGS_DIR / "retest_vs_gapandgo_summary.csv", index=False)
        print(f"\nSaved -> {out}")
        print(f"API usage (should be all cache): {client.stats()}")
        print("\nNOTE: 'mean_ret_pct' here is the mean R-MULTIPLE, not a percent - "
              "describe() is shared\nwith the option backtests where the input is a "
              "percent return. Read it as R.")
