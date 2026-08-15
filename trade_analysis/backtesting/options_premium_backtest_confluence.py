"""
Confluence-filtered variant of the 0DTE options-premium backtest — kept in
its OWN file, fully isolated from options_premium_backtest.py, on purpose.

Isolation contract:
  - This file only ever IMPORTS from options_premium_backtest.py (entry
    detection, exit simulation, option pricing, fetch helpers). It never
    edits that file and never redefines its logic.
  - If this confluence experiment doesn't pan out: delete this file. The
    working yesterday-H/L backtest in options_premium_backtest.py is
    completely untouched and keeps working exactly as it did before.
  - Output goes to a separate CSV (options_premium_confluence_backtest.csv),
    never overwriting the original's output file.

What this actually tests: does a breakout whose broken level (yesterday's
high/low) lines up with one of the prior 3 sessions' highs/lows (a "magnet"
level touched more than once recently) perform better than a breakout with
no such confluence? Splits the SAME entries/exits from find_entries() by a
post-hoc confluence label - no new entry logic, no re-simulation needed for
the base signal.

IMPORTANT LIMIT - why GEX regime isn't in this backtest:
gamma_exposure.compute_gex() reads the LIVE option chain each call - yfinance
has no historical option-chain endpoint, so there is no way to know what
today's GEX regime would have been on a day 40 sessions ago. GEX-regime
filtering (already wired into live_trading/*.py) can only be validated
forward, day by day, in paper trading - not retroactively backtested with
free data. Don't let that quietly get faked here; it's a real data gap.
"""
import pandas as pd

from .options_premium_backtest import (
    SYMBOLS, STOP_ATR_MULT, TARGET_ATR_MULT,
    find_entries, simulate_exit, simulate_option_pnl,
    fetch_intraday, fetch_daily,
)
from ..paths import LOGS_DIR

CONFLUENCE_LOOKBACK_DAYS = 3
CONFLUENCE_TOLERANCE_PCT = 0.15  # % - matches live_trading's confluence_zone_pct


def _breakout_level(entry, daily_df):
    """The level that was actually broken: yesterday's high for an UP entry,
    yesterday's low for a DOWN entry - recomputed independently here rather
    than trusting a stored value, since find_entries() doesn't return it."""
    hist = daily_df[daily_df.index.date < entry['date']]
    yesterday = hist.iloc[-1]
    return float(yesterday['High']) if entry['direction'] == 'UP' else float(yesterday['Low'])


def _prior_session_levels(entry, daily_df, lookback_days=CONFLUENCE_LOOKBACK_DAYS):
    """Highs and lows from the `lookback_days` sessions BEFORE yesterday
    (yesterday itself is the breakout level, not a confluence candidate)."""
    hist = daily_df[daily_df.index.date < entry['date']]
    prior = hist.iloc[-(lookback_days + 1):-1]
    return list(prior['High']) + list(prior['Low'])


def label_confluence(entries, daily_df):
    """Adds 'breakout_level' and 'has_confluence' to each entry dict."""
    labeled = []
    for entry in entries:
        level = _breakout_level(entry, daily_df)
        prior_levels = _prior_session_levels(entry, daily_df)
        has_confluence = any(
            abs(p - level) / level <= CONFLUENCE_TOLERANCE_PCT / 100
            for p in prior_levels
        )
        labeled.append({**entry, 'breakout_level': level, 'has_confluence': has_confluence})
    return labeled


def run_confluence_split(symbol):
    intraday = fetch_intraday(symbol)
    daily = fetch_daily(symbol)

    entries = find_entries(symbol, intraday, daily)
    if not entries:
        return pd.DataFrame()

    labeled_entries = label_confluence(entries, daily)
    trades = [simulate_exit(e, STOP_ATR_MULT, TARGET_ATR_MULT) for e in labeled_entries]
    results = [
        {**simulate_option_pnl(t, daily), 'has_confluence': e['has_confluence']}
        for t, e in zip(trades, labeled_entries)
    ]
    df = pd.DataFrame(results)
    df['symbol'] = symbol
    return df


def _segment_stats(df, label):
    if df.empty:
        return {'segment': label, 'n_trades': 0}
    win_rate = (df['pnl_pct'] > 0).mean() * 100
    return {
        'segment': label,
        'n_trades': len(df),
        'win_rate': round(win_rate, 1),
        'avg_pnl_pct': round(df['pnl_pct'].mean(), 2),
        'total_pnl_per_contract': round(df['pnl_per_contract'].sum(), 2),
    }


if __name__ == '__main__':
    all_results = []
    summary_rows = []

    for sym in SYMBOLS:
        print(f"Running confluence-labeled 0DTE options-premium backtest for {sym}...")
        df = run_confluence_split(sym)
        if df.empty:
            print(f"  No trades found")
            continue
        all_results.append(df)

        confluence_df = df[df['has_confluence']]
        no_confluence_df = df[~df['has_confluence']]

        for seg_df, label in [(df, f'{sym} - ALL'), (confluence_df, f'{sym} - confluence'),
                               (no_confluence_df, f'{sym} - no confluence')]:
            stats = _segment_stats(seg_df, label)
            summary_rows.append(stats)
            print(f"  {stats}")

    if all_results:
        combined = pd.concat(all_results, ignore_index=True)
        out_path = LOGS_DIR / "options_premium_confluence_backtest.csv"
        combined.to_csv(out_path, index=False)
        print(f"\nSaved {len(combined)} labeled trades to {out_path}")

        print("\n" + "=" * 100)
        print("SUMMARY: confluence vs no-confluence subsets (same entries/exits, split post-hoc)")
        print(pd.DataFrame(summary_rows).to_string(index=False))
        print(
            "\nCAVEAT: confluence subsets here are ~10-15 trades each (a fraction of an already "
            "small ~30-trade sample) - treat any split difference as a hint to investigate further, "
            "not a proven edge."
        )
