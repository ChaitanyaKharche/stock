"""
0DTE options-premium backtest for SPY/QQQ ONLY.

Reuses rather than rebuilds:
  - bs_price / bs_delta / solve_strike_for_delta from signals/gamma_exposure.py
  - the entry mechanics (yesterday H/L breakout confirmed inside the 9:30-9:45
    window, ATR-based target/stop, gap-regime detection) and constants from
    live_trading/swing_breakout_trader.py - reimplemented here against a
    plain DataFrame since that script is coupled to a live Alpaca client,
    not a backtest-friendly interface. No new signal logic, same formulas.

What this can and can't tell you (see the earlier assessment in this
conversation - this doesn't change that):
  - The underlying price path is 100% real (free yfinance data) - only the
    option premium is synthetic, priced via Black-Scholes off a realized-
    vol proxy (bumped on ELEVATED gap-regime days) plus a fixed put-skew
    multiplier calibrated from two real QQQ quotes (34.01/24.83 IV). No
    fitted vol-surface, no historical option quotes used or needed.
  - HARD DATA LIMIT: yfinance's free 5-min bars only cover ~60 days, so this
    is a small-sample mechanical check, not a statistically robust backtest.
  - A synthetic backtest looking good is weak evidence (real spreads/skew
    could still erase it); looking bad here is a real "don't bother" signal.
"""
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import time as dt_time
from zoneinfo import ZoneInfo

from ..signals.gamma_exposure import bs_price, solve_strike_for_delta
from ..paths import LOGS_DIR

SYMBOLS = ['SPY', 'QQQ']
TZ = ZoneInfo("America/New_York")

# Mirrors live_trading/swing_breakout_trader.py's tuned parameters exactly
ATR_PERIOD = 14
STOP_ATR_MULT = 1.0
TARGET_ATR_MULT = 2.5
# RETEST_ATR_MULT removed: it only ever appeared in a condition that could not
# fail (see find_entries). A real retest implementation lives in
# retest_vs_gapandgo_backtest.py, which defines its own constant.
GAP_ATR_MULT_THRESHOLD = 2.0
MORNING_START = dt_time(9, 30)
MORNING_END = dt_time(9, 45)
MARKET_CLOSE = dt_time(16, 0)

ELEVATED_VOL_MULT = 1.3   # vol bump applied on ELEVATED gap-regime days
TARGET_DELTA = 0.20       # matches the ~20-delta contracts looked at earlier
PUT_SKEW_MULT = 1.37      # 34.01/24.83 from the real QQQ quotes in this conversation
CALL_SPREAD_PCT = 0.02    # observed ~2% (0.93/0.95 ask)
PUT_SPREAD_PCT = 0.06     # observed ~6% (0.81/0.86 ask)
RISK_FREE_RATE = 0.05
CONTRACT_MULTIPLIER = 100


def _flatten(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    if df.index.tz is None:
        df.index = df.index.tz_localize('UTC')
    df.index = df.index.tz_convert(TZ)
    return df


def fetch_intraday(symbol):
    """5-min bars - yfinance's free tier caps this at ~60 days of history."""
    df = yf.download(symbol, period='59d', interval='5m', progress=False, auto_adjust=True)
    return _flatten(df)


def fetch_daily(symbol, period='1y'):
    df = yf.download(symbol, period=period, interval='1d', progress=False, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


def compute_atr(intraday_df, period=ATR_PERIOD):
    high, low, close = intraday_df['High'], intraday_df['Low'], intraday_df['Close']
    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def realized_vol(daily_df, asof_date, window=20):
    """Annualized realized vol from daily returns up to (not including) asof_date -
    the IV proxy, since no free historical IV data exists."""
    hist = daily_df[daily_df.index.date < asof_date]
    returns = hist['Close'].pct_change().dropna().tail(window)
    if len(returns) < 5:
        return 0.20
    return float(returns.std() * np.sqrt(252))


def find_entries(symbol, intraday_df, daily_df):
    """Same entry mechanics as swing_breakout_trader.py: enter on the first
    5-min bar in the 9:30-9:45 window that CLOSES beyond yesterday's high/low.
    This is a gap-and-go breakout, not a retest.

    HISTORY - why the name changed: this used to carry a second clause,
    `price >= yesterday_high - RETEST_ATR_MULT * atr`, described as a retest
    confirmation. It could never fail: if price is already above
    yesterday_high, it is necessarily above yesterday_high minus a positive
    buffer. The mirror clause for shorts had the same defect, so neither
    filtered anything.

    It was measured before being removed. A genuine two-phase retest (break,
    pull back into a bounded zone around the level, then a rejection candle -
    the logic live_trading/retest_breakout_websocket.py implements correctly)
    was run head-to-head on 500 sessions of real 1-minute SPY/QQQ data in
    retest_vs_gapandgo_backtest.py. It cut trade count 5-6x and was worse on
    win rate, mean R and target-hit rate on BOTH symbols. So the behaviour here
    is intentionally unchanged and only the misleading name is gone.

    Entry detection doesn't depend on stop/target multiples, so this runs once
    and gets reused across a parameter sweep - only the exit changes per combo.
    """
    entries = []
    atr_all = compute_atr(intraday_df)
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
        yesterday_close = float(daily_hist['Close'].iloc[-1])

        yesterday_high = float(prev_day['High'].max())
        yesterday_low = float(prev_day['Low'].min())

        morning = today.between_time(MORNING_START, MORNING_END)
        if morning.empty:
            continue

        open_price = float(today['Open'].iloc[0])
        atr = atr_all.reindex(today.index).iloc[0]
        if pd.isna(atr) or atr <= 0:
            continue

        gap = abs(open_price - yesterday_close)
        gap_regime = "ELEVATED" if gap > GAP_ATR_MULT_THRESHOLD * atr else "NORMAL"

        direction = None
        entry_price, entry_time = None, None
        for ts, bar in morning.iterrows():
            price = bar['Close']
            if price > yesterday_high:
                direction, entry_price, entry_time = 'UP', price, ts
                break
            if price < yesterday_low:
                direction, entry_price, entry_time = 'DOWN', price, ts
                break

        if direction is None:
            continue

        entries.append({
            'date': d, 'direction': direction, 'gap_regime': gap_regime,
            'entry_time': entry_time, 'entry_price': entry_price, 'atr': float(atr),
            'today_bars': today[today.index > entry_time],
        })

    return entries


def simulate_exit(entry, stop_mult, target_mult):
    """Given an entry (from find_entries) and a stop/target ATR multiple
    combo, replays the rest of that day's bars to find the exit."""
    direction, atr, entry_price = entry['direction'], entry['atr'], entry['entry_price']
    stop_dist = stop_mult * atr
    target_dist = target_mult * atr
    stop_price = entry_price - stop_dist if direction == 'UP' else entry_price + stop_dist
    target_price = entry_price + target_dist if direction == 'UP' else entry_price - target_dist

    exit_price, exit_time, exit_reason = None, None, None
    for ts, bar in entry['today_bars'].iterrows():
        if direction == 'UP':
            if bar['High'] >= target_price:
                exit_price, exit_time, exit_reason = target_price, ts, 'TARGET'
                break
            if bar['Low'] <= stop_price:
                exit_price, exit_time, exit_reason = stop_price, ts, 'STOP'
                break
        else:
            if bar['Low'] <= target_price:
                exit_price, exit_time, exit_reason = target_price, ts, 'TARGET'
                break
            if bar['High'] >= stop_price:
                exit_price, exit_time, exit_reason = stop_price, ts, 'STOP'
                break

    if exit_price is None:
        if entry['today_bars'].empty:
            exit_price, exit_time = entry_price, entry['entry_time']
        else:
            exit_price = float(entry['today_bars']['Close'].iloc[-1])
            exit_time = entry['today_bars'].index[-1]
        exit_reason = 'CLOSE'

    return {
        'date': entry['date'], 'direction': direction, 'gap_regime': entry['gap_regime'],
        'entry_time': entry['entry_time'], 'entry_price': entry_price,
        'exit_time': exit_time, 'exit_price': exit_price, 'exit_reason': exit_reason,
    }


def find_trades(symbol, intraday_df, daily_df, stop_mult=STOP_ATR_MULT, target_mult=TARGET_ATR_MULT):
    """Convenience wrapper: entries + default-multiple exits in one call."""
    entries = find_entries(symbol, intraday_df, daily_df)
    return [simulate_exit(e, stop_mult, target_mult) for e in entries]


def _minutes_to_close(ts):
    close_dt = ts.replace(hour=MARKET_CLOSE.hour, minute=MARKET_CLOSE.minute, second=0, microsecond=0)
    return max((close_dt - ts).total_seconds() / 60, 1)


def simulate_option_pnl(trade, daily_df):
    option_type = 'call' if trade['direction'] == 'UP' else 'put'
    spot_entry = trade['entry_price']

    base_iv = realized_vol(daily_df, trade['date'])
    iv = base_iv * (ELEVATED_VOL_MULT if trade['gap_regime'] == 'ELEVATED' else 1.0)
    if option_type == 'put':
        iv *= PUT_SKEW_MULT

    t_entry = _minutes_to_close(trade['entry_time']) / (60 * 6.5 * 252)
    t_exit = _minutes_to_close(trade['exit_time']) / (60 * 6.5 * 252)

    strike = solve_strike_for_delta(spot_entry, t_entry, iv, TARGET_DELTA, option_type, RISK_FREE_RATE)

    entry_premium = bs_price(spot_entry, strike, t_entry, iv, RISK_FREE_RATE, option_type)
    exit_premium = bs_price(trade['exit_price'], strike, t_exit, iv, RISK_FREE_RATE, option_type)

    spread_pct = CALL_SPREAD_PCT if option_type == 'call' else PUT_SPREAD_PCT
    entry_fill = entry_premium * (1 + spread_pct / 2)   # buy at ~mid+half-spread
    exit_fill = max(exit_premium * (1 - spread_pct / 2), 0.0)  # sell at ~mid-half-spread

    pnl_per_contract = (exit_fill - entry_fill) * CONTRACT_MULTIPLIER
    pnl_pct = (exit_fill / entry_fill - 1) * 100 if entry_fill > 0 else 0.0

    return {
        **trade, 'option_type': option_type, 'strike': round(strike, 2),
        'iv_used': round(iv, 4), 'entry_premium': round(entry_fill, 3),
        'exit_premium': round(exit_fill, 3), 'pnl_per_contract': round(pnl_per_contract, 2),
        'pnl_pct': round(pnl_pct, 2),
    }


def run(symbol):
    intraday = fetch_intraday(symbol)
    daily = fetch_daily(symbol)
    trades = find_trades(symbol, intraday, daily)
    results = [simulate_option_pnl(t, daily) for t in trades]
    return pd.DataFrame(results)


# Grid for the retune pass. Kept modest on purpose - with ~30 trades/symbol,
# a wide grid just guarantees finding noise that looks like an edge.
STOP_MULT_GRID = [0.5, 0.75, 1.0, 1.5]
TARGET_MULT_GRID = [1.5, 2.0, 2.5, 3.0, 4.0]


def sweep(symbol, intraday_df, daily_df):
    """Reuses the SAME real entries across every (stop_mult, target_mult)
    combo - only the exit/premium simulation changes - so this is a fair
    like-for-like comparison, not 20 separate re-detections."""
    entries = find_entries(symbol, intraday_df, daily_df)
    if not entries:
        return pd.DataFrame()

    rows = []
    for stop_mult in STOP_MULT_GRID:
        for target_mult in TARGET_MULT_GRID:
            trades = [simulate_exit(e, stop_mult, target_mult) for e in entries]
            results = [simulate_option_pnl(t, daily_df) for t in trades]
            df = pd.DataFrame(results)
            win_rate = (df['pnl_pct'] > 0).mean() * 100
            rows.append({
                'symbol': symbol, 'stop_mult': stop_mult, 'target_mult': target_mult,
                'n_trades': len(df), 'win_rate': round(win_rate, 1),
                'avg_pnl_pct': round(df['pnl_pct'].mean(), 2),
                'total_pnl_per_contract': round(df['pnl_per_contract'].sum(), 2),
            })
    return pd.DataFrame(rows)


if __name__ == '__main__':
    import sys

    if '--sweep' in sys.argv:
        print(f"Retune sweep - {len(STOP_MULT_GRID)}x{len(TARGET_MULT_GRID)} combos per symbol, "
              f"SAME entries reused across each combo (only exit multiples change).\n"
              f"CAVEAT: ~30 trades/symbol means picking 'the best' combo here is likely just\n"
              f"curve-fitting to this specific 60-day window, not a discovered edge - use this\n"
              f"to see whether ANY broad region beats the default, not to lock in one winner.\n")
        all_sweeps = []
        for sym in SYMBOLS:
            intraday = fetch_intraday(sym)
            daily = fetch_daily(sym)
            sweep_df = sweep(sym, intraday, daily)
            if sweep_df.empty:
                print(f"{sym}: no entries found in this window")
                continue
            all_sweeps.append(sweep_df)
            print(f"=== {sym} ===")
            print(sweep_df.sort_values('total_pnl_per_contract', ascending=False).to_string(index=False))
            print()

        if all_sweeps:
            combined = pd.concat(all_sweeps, ignore_index=True)
            out_path = LOGS_DIR / "options_premium_sweep.csv"
            combined.to_csv(out_path, index=False)
            print(f"Saved full sweep ({len(combined)} rows) to {out_path}")
        sys.exit(0)

    all_results = []
    for sym in SYMBOLS:
        print(f"Running 0DTE options-premium backtest for {sym}...")
        df = run(sym)
        if df.empty:
            print(f"  No trades found (small ~60-day sample, may just be no signals)")
            continue
        df['symbol'] = sym
        all_results.append(df)

        win_rate = (df['pnl_pct'] > 0).mean() * 100
        print(f"  {len(df)} trades | win rate {win_rate:.1f}% | avg P&L/contract ${df['pnl_per_contract'].mean():+.2f} "
              f"| total P&L/contract ${df['pnl_per_contract'].sum():+.2f}")

    if all_results:
        combined = pd.concat(all_results, ignore_index=True)
        out_path = LOGS_DIR / "options_premium_backtest.csv"
        combined.to_csv(out_path, index=False)
        print(f"\nSaved {len(combined)} simulated trades to {out_path}")
