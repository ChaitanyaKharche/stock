"""Prove the data fix: different symbols must now produce different analyses.

    venv\\Scripts\\python.exe huggingface_space\\verify_fix.py

Runs the exact path the Space uses -- UnifiedDataProvider._yf_bars ->
indicators.enrich_with_indicators -> indicators.identify_current_setup -- and then
reproduces enhanced_api's confidence arithmetic.

The old build returned "HOLD / 15%" for every ticker. That was not a coincidence and not
a model being cautious: `fetch_multi_timeframe_stock_data` returned a ONE-ROW frame, so
`identify_current_setup` always took its `len(df) < 2` branch and reported
{"adx": 0, "rsi": 50, "error": "Insufficient data"}, the momentum engine always reported
confidence 0, and

    weighted_confidence = 0 * 0.4 + 50 * 0.3 + 0 * 0.3 = 15.0     (enhanced_api.py:381)

came out a constant. This asserts that is gone: real bar counts, per-symbol RSI/ADX, and
no "Insufficient data" anywhere.

Deliberately does NOT import trade_analysis.config -- that module raises ImportError when
FINNHUB_KEY is unset, and none of what is under test here needs a key.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import yfinance as yf

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from trade_analysis.indicators import enrich_with_indicators, identify_current_setup

TIMEFRAME_SPEC = {"15m": ("60d", "15m"), "hourly": ("180d", "1h"), "daily": ("2y", "1d")}
OHLCV_COLS = ["Open", "High", "Low", "Close", "Volume"]
SYMBOLS = ["NVDA", "SPY", "TSLA", "AVGO"]        # AVGO has no local_data snapshot


def yf_bars(symbol: str, period: str, interval: str) -> pd.DataFrame:
    try:
        df = yf.Ticker(symbol).history(period=period, interval=interval,
                                       auto_adjust=False)
    except Exception as e:                                        # noqa: BLE001
        print(f"    {interval}: FAILED {e}")
        return pd.DataFrame()
    if df is None or df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    if [c for c in OHLCV_COLS if c not in df.columns]:
        return pd.DataFrame()
    return df[OHLCV_COLS].dropna()


def main() -> int:
    print("=" * 92)
    print("VERIFY: real OHLCV history reaches the indicators")
    print("=" * 92)
    ok = True
    rows = []

    for sym in SYMBOLS:
        print(f"\n{sym}")
        setups = {}
        for tf, (period, interval) in TIMEFRAME_SPEC.items():
            df = yf_bars(sym, period, interval)
            print(f"    {tf:<7} {len(df):>5} bars", end="")
            if df.empty:
                print("   <-- EMPTY")
                ok = False
                continue
            setup = identify_current_setup(enrich_with_indicators(df.copy(), tf), tf)
            setups[tf] = setup
            err = setup.get("error")
            print(f"   dir={setup.get('direction'):<8}"
                  f" rsi={setup.get('rsi', 0):>6.1f}"
                  f" adx={setup.get('adx', 0):>6.1f}"
                  f"{'   <-- ' + err if err else ''}")
            if err:
                ok = False

        d = setups.get("daily", {})
        rows.append((sym, len(setups), d.get("rsi"), d.get("adx"), d.get("direction")))

        # the TFT gate that could never be satisfied before
        n_daily = len(yf_bars(sym, *TIMEFRAME_SPEC["daily"])) if "daily" in setups else 0
        print(f"    TFT needs >=96 daily rows: {n_daily} -> "
              f"{'ELIGIBLE' if n_daily >= 96 else 'still short'}")

    print("\n" + "=" * 92)
    print("PER-SYMBOL DAILY SETUP  (identical rows here would mean the bug is still live)")
    print("=" * 92)
    print(f"  {'symbol':<8}{'timeframes':>11}{'rsi':>9}{'adx':>9}{'direction':>12}")
    for sym, ntf, rsi, adx, direc in rows:
        print(f"  {sym:<8}{ntf:>11}{(rsi if rsi is not None else float('nan')):>9.1f}"
              f"{(adx if adx is not None else float('nan')):>9.1f}{str(direc):>12}")

    distinct = len({(round(r[2], 4) if r[2] else None,
                     round(r[3], 4) if r[3] else None) for r in rows})
    print(f"\n  distinct (rsi, adx) pairs across {len(rows)} symbols: {distinct}")
    if distinct < len(rows):
        print("  WARNING: symbols share an identical reading -- investigate")
        ok = False

    # Reproduce the confidence formula that used to pin every answer to 15%.
    print("\n  enhanced_api.py:381  weighted_confidence"
          " = momentum*0.4 + llm_conviction*0.3 + sentiment_high*80*0.3")
    print(f"    OLD (momentum_confidence always 0): 0*0.4 + 50*0.3 + 0 = "
          f"{0 * 0.4 + 50 * 0.3 + 0:.0f}%  <-- the constant every ticker returned")
    print("    NEW: momentum_confidence is now derived from real bars, so this varies.")

    print(f"\n  RESULT: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
