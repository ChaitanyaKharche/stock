"""Measure the attainable range of weighted_confidence over real bars, so that
enhanced_api's min_confidence is a percentile of something real rather than a guess.

    venv\\Scripts\\python.exe huggingface_space\\tools\\calibrate_confidence_gate.py

min_confidence was originally 70/65/60/55 against a quantity whose measured maximum was
47: it fired on 0 of 80 observations, which is why the app answered HOLD for every symbol
on every timeframe. Re-run this after changing any confidence component, or when the
volatility regime shifts -- the sample it was set from was a quiet tape (VIX ~14.5).
"""
from __future__ import annotations

import os
import statistics as st
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
for _k in ("FINNHUB_KEY", "REDDIT_CLIENT_ID", "REDDIT_CLIENT_SECRET", "REDDIT_USER_AGENT"):
    os.environ.setdefault(_k, "unused")

import pandas as pd
import yfinance as yf

from trade_analysis.data import TIMEFRAME_SPEC
from trade_analysis.enhanced_api import (TIMEFRAME_CONFIGS,
                                         _generate_master_signal)
from trade_analysis.momentum_trading_engine import IntegratedMomentumEngine

COLS = ["Open", "High", "Low", "Close", "Volume"]
TFS = ["15m", "1h", "4h", "1d"]
# Derived from production, never copied. The previous hardcoded GATE drifted to
# 36/34/33/32 while enhanced_api ran 36/36/37/38, so this tool reported a fire rate for a
# gate that did not exist -- the measurement instrument itself had the bug it was built to
# catch. Import, do not transcribe.
THRESH = {tf: c["threshold"] for tf, c in TIMEFRAME_CONFIGS.items()}
GATE = {tf: c["min_confidence"] for tf, c in TIMEFRAME_CONFIGS.items()}
assert set(TFS) == set(TIMEFRAME_CONFIGS), (
    f"timeframes drifted: tool {sorted(TFS)} vs production "
    f"{sorted(TIMEFRAME_CONFIGS)}")

BASKET = ["NVDA", "TSLA", "SPY", "QQQ", "AAPL", "MSFT", "AMD", "META", "AMZN", "GOOGL",
          "NFLX", "COIN", "XLE", "JPM", "WMT", "BA", "DIS", "UBER", "PLTR", "SMCI"]
SENT = {"composite_score": 0.05, "confidence": "LOW"}
ALT = {"vix_level": 14.5, "implied_vol_pct": 60.0}
LLM = {"signal": "HOLD", "confidence": 40}


def bars(symbol: str) -> dict:
    out = {}
    for tf, (period, interval) in TIMEFRAME_SPEC.items():
        try:
            df = yf.Ticker(symbol).history(period=period, interval=interval,
                                           auto_adjust=False)
        except Exception:
            continue
        if df is None or df.empty:
            continue
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        out[tf] = df[COLS].dropna()
    return out


def main() -> int:
    eng = IntegratedMomentumEngine()
    rows = []
    for sym in BASKET:
        o = bars(sym)
        if not o:
            print(f"  {sym}: no data, skipped")
            continue
        gap = None
        d = o.get("daily")
        if d is not None and len(d) >= 2:
            prev, opn = float(d["Close"].iloc[-2]), float(d["Open"].iloc[-1])
            gap = (opn - prev) / prev * 100 if prev else None
        for tf in TFS:
            mom = eng.generate_enhanced_signal(o, SENT, ALT, tf)
            r = _generate_master_signal(mom, LLM, SENT, None, tf, "momentum", gap)
            rows.append((tf, r["confidence"], r["weighted_score"], r["signal"]))
        print(f"  {sym}: " + " ".join(f"{t}={c}" for t, c, _, _ in rows[-4:]))

    if not rows:
        print("no data at all -- market data unreachable?")
        return 1
    confs = sorted(c for _, c, _, _ in rows)

    def p(q):
        return confs[min(len(confs) - 1, int(len(confs) * q))]

    print("\n" + "=" * 64)
    print(f"n = {len(rows)} observations")
    print(f"weighted_confidence  min {min(confs)}  p50 {p(.50)}  p75 {p(.75)}  "
          f"p90 {p(.90)}  max {max(confs)}  (mean {st.mean(confs):.1f})")
    fires = sum(1 for tf, c, s, _ in rows if c > GATE[tf] and abs(s) > THRESH[tf])
    print(f"current gate {GATE}")
    print(f"fires on {fires}/{len(rows)} ({fires / len(rows) * 100:.1f}%)")
    print("\nA gate ABOVE the observed max is not strict, it is broken. Keep the fire "
          "rate low but\nnon-zero: HOLD should be the common answer, not the only one.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
