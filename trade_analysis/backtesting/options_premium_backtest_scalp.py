"""
Scalp-exit variant matching the user's actual real-world profitable rule:
buy long calls/puts (same directional breakout signal) at ~$1-1.50 premium,
2-3 contracts, exit at +25% to +50% of premium paid - NOT the ATR-based
underlying target/stop used in options_premium_backtest.py, which the sweep
already showed rewards patience the premium doesn't have time for.

Isolation: only imports from options_premium_backtest.py and
signals/gamma_exposure.py, never edits either. Delete this file if it
doesn't pan out; nothing else is affected.

Still buy-only: a single long call or long put, no short legs anywhere -
matches the "only option buying" constraint.

Approximation flagged: the profit-target/stop check re-prices the option at
each 5-min bar's CLOSE (not intrabar high/low), since inverting Black-
Scholes for an intrabar extreme isn't well-defined the way it is for the
underlying's own high/low. At 5-min granularity for a same-day scalp this
is a real but minor source of imprecision, not swept under the rug.
"""
import pandas as pd

from .options_premium_backtest import (
    SYMBOLS, find_entries, fetch_intraday, fetch_daily, realized_vol,
    ELEVATED_VOL_MULT, PUT_SKEW_MULT, CALL_SPREAD_PCT, PUT_SPREAD_PCT,
    RISK_FREE_RATE, CONTRACT_MULTIPLIER, MARKET_CLOSE,
)
from ..signals.gamma_exposure import bs_price
from ..paths import LOGS_DIR

TARGET_PREMIUM = 1.25     # midpoint of the user's stated $1-1.50 entries
CONTRACTS = 2              # matches "2-3 lots"
PROFIT_TARGET_GRID = [0.25, 0.35, 0.50]   # the user's stated 25-50% range
STOP_PCT = -0.50            # symmetric loss cap - NOT something the user stated, an assumption


def solve_strike_for_premium(spot, t_years, iv, target_premium, option_type, r=RISK_FREE_RATE):
    """Same bisection pattern as gamma_exposure.solve_strike_for_delta, but
    targets a premium level instead of a delta - price is monotonic in
    strike-distance-from-spot for OTM options, so this is well-behaved."""
    lo, hi = spot * 0.5, spot * 1.5
    for _ in range(60):
        mid = (lo + hi) / 2
        price = bs_price(spot, mid, t_years, iv, r, option_type)
        if price > target_premium:
            if option_type == 'call':
                lo = mid
            else:
                hi = mid
        else:
            if option_type == 'call':
                hi = mid
            else:
                lo = mid
    return (lo + hi) / 2


def _minutes_to_close(ts):
    close_dt = ts.replace(hour=MARKET_CLOSE.hour, minute=MARKET_CLOSE.minute, second=0, microsecond=0)
    return max((close_dt - ts).total_seconds() / 60, 1)


def simulate_scalp_trade(entry, daily_df, profit_target_pct, stop_pct=STOP_PCT):
    option_type = 'call' if entry['direction'] == 'UP' else 'put'
    spot_entry = entry['entry_price']

    base_iv = realized_vol(daily_df, entry['date'])
    iv = base_iv * (ELEVATED_VOL_MULT if entry['gap_regime'] == 'ELEVATED' else 1.0)
    if option_type == 'put':
        iv *= PUT_SKEW_MULT

    t_entry = _minutes_to_close(entry['entry_time']) / (60 * 6.5 * 252)
    strike = solve_strike_for_premium(spot_entry, t_entry, iv, TARGET_PREMIUM, option_type)

    entry_mid = bs_price(spot_entry, strike, t_entry, iv, RISK_FREE_RATE, option_type)
    spread_pct = CALL_SPREAD_PCT if option_type == 'call' else PUT_SPREAD_PCT
    entry_fill = entry_mid * (1 + spread_pct / 2)

    exit_fill, exit_time, exit_reason = None, None, None
    for ts, bar in entry['today_bars'].iterrows():
        t_now = _minutes_to_close(ts) / (60 * 6.5 * 252)
        mid_price = bs_price(bar['Close'], strike, t_now, iv, RISK_FREE_RATE, option_type)
        candidate = max(mid_price * (1 - spread_pct / 2), 0.0)
        change_pct = (candidate / entry_fill - 1) if entry_fill > 0 else 0.0

        if change_pct >= profit_target_pct:
            exit_fill, exit_time, exit_reason = candidate, ts, 'TARGET'
            break
        if change_pct <= stop_pct:
            exit_fill, exit_time, exit_reason = candidate, ts, 'STOP'
            break

    if exit_fill is None:
        if entry['today_bars'].empty:
            exit_fill, exit_time = entry_fill, entry['entry_time']
        else:
            last_ts = entry['today_bars'].index[-1]
            last_bar = entry['today_bars'].iloc[-1]
            t_last = _minutes_to_close(last_ts) / (60 * 6.5 * 252)
            mid_price = bs_price(last_bar['Close'], strike, t_last, iv, RISK_FREE_RATE, option_type)
            exit_fill = max(mid_price * (1 - spread_pct / 2), 0.0)
            exit_time = last_ts
        exit_reason = 'CLOSE'

    pnl_per_contract = (exit_fill - entry_fill) * CONTRACT_MULTIPLIER
    pnl_pct = (exit_fill / entry_fill - 1) * 100 if entry_fill > 0 else 0.0

    return {
        'date': entry['date'], 'direction': entry['direction'], 'option_type': option_type,
        'strike': round(strike, 2), 'entry_premium': round(entry_fill, 3),
        'exit_premium': round(exit_fill, 3), 'exit_reason': exit_reason,
        'profit_target_pct': profit_target_pct,
        'pnl_per_contract': round(pnl_per_contract, 2),
        'pnl_position': round(pnl_per_contract * CONTRACTS, 2),
        'pnl_pct': round(pnl_pct, 2),
    }


def sweep_symbol(symbol, intraday_df, daily_df):
    entries = find_entries(symbol, intraday_df, daily_df)
    if not entries:
        return pd.DataFrame()

    rows = []
    for target_pct in PROFIT_TARGET_GRID:
        trades = [simulate_scalp_trade(e, daily_df, target_pct) for e in entries]
        df = pd.DataFrame(trades)
        win_rate = (df['pnl_pct'] > 0).mean() * 100
        rows.append({
            'symbol': symbol, 'profit_target_pct': target_pct, 'n_trades': len(df),
            'win_rate': round(win_rate, 1), 'avg_pnl_pct': round(df['pnl_pct'].mean(), 2),
            'total_pnl_per_contract': round(df['pnl_per_contract'].sum(), 2),
            'total_pnl_position': round(df['pnl_position'].sum(), 2),
        })
    return pd.DataFrame(rows)


if __name__ == '__main__':
    print(f"Scalp-exit backtest: ~${TARGET_PREMIUM} entries, {CONTRACTS} contracts, "
          f"exit at {[f'{p*100:.0f}%' for p in PROFIT_TARGET_GRID]} profit "
          f"(stop at {STOP_PCT*100:.0f}%, an assumption - not stated by user).\n")

    all_rows = []
    for sym in SYMBOLS:
        intraday = fetch_intraday(sym)
        daily = fetch_daily(sym)
        sweep_df = sweep_symbol(sym, intraday, daily)
        if sweep_df.empty:
            print(f"{sym}: no entries found")
            continue
        all_rows.append(sweep_df)
        print(f"=== {sym} ===")
        print(sweep_df.to_string(index=False))
        print()

    if all_rows:
        combined = pd.concat(all_rows, ignore_index=True)
        out_path = LOGS_DIR / "options_premium_scalp_backtest.csv"
        combined.to_csv(out_path, index=False)
        print(f"Saved sweep to {out_path}")
