"""Calibrate the direction gates by PERMUTATION on real bars.

    venv\\Scripts\\python.exe huggingface_space\\tools\\calibrate_direction_gate.py

The first question a trading signal must answer is not "how often is it right" but "how
often does it fire when there is nothing there". Shuffling the returns of a real series
destroys any persistent trend while preserving the volume profile, volatility level and
bar geometry that the momentum MAGNITUDE depends on -- so anything the engine emits on a
permuted series is a false positive, and the ratio of its real firing rate to its permuted
firing rate is its lift over noise.

Two earlier attempts at this were wrong and are recorded here so they are not repeated:

  * Independent random walks per timeframe. In production 5m/15m/hourly/daily are nested
    aggregations of ONE price path, so they agree far more often than independent draws
    do. Treating them as independent flatters the multi-timeframe agreement test.
  * Synthetic nested walks with uniform random volume. The momentum magnitude gate then
    blocks every case before the direction logic is ever consulted, so the measurement
    reported 0% for everything and was insensitive to the knobs being swept.

Permutation avoids both: the geometry and the volume are real, only the path is destroyed.
"""
from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
for _k in ("FINNHUB_KEY", "REDDIT_CLIENT_ID", "REDDIT_CLIENT_SECRET", "REDDIT_USER_AGENT"):
    os.environ.setdefault(_k, "unused")

import numpy as np
import pandas as pd
import yfinance as yf

import trade_analysis.momentum_trading_engine as MTE
from trade_analysis.data import TIMEFRAME_SPEC, fold_to_4h
from trade_analysis.enhanced_api import _generate_master_signal

COLS = ["Open", "High", "Low", "Close", "Volume"]
TFS = ["15m", "1h", "4h", "1d"]
BASKET = ["NVDA", "TSLA", "SPY", "QQQ", "AAPL", "MSFT", "AMD", "META", "AMZN", "GOOGL",
          "NFLX", "COIN", "XLE", "JPM", "WMT", "BA", "DIS", "UBER", "PLTR", "SMCI"]
GRID = [(0.15, 0.60), (0.25, 0.60), (0.35, 0.60), (0.35, 0.75),
        (0.40, 0.60), (0.40, 0.75), (0.45, 0.75), (0.50, 0.75)]
IN_USE = (0.40, 0.75)
REPS = 6

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
    # 4h is derived in the provider rather than fetched, so it is absent from
    # TIMEFRAME_SPEC. Without this the tool would calibrate against a bar set the
    # running app does not use.
    if "hourly" in out:
        four = fold_to_4h(out["hourly"])
        if four is not None and not four.empty:
            out["4h"] = four
    return out


def permute(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Shuffle the return sequence, rescaling OHLC together so bars stay coherent."""
    close = df["Close"].to_numpy(float)
    rets = np.diff(close) / close[:-1]
    rng.shuffle(rets)
    rebuilt = np.empty_like(close)
    rebuilt[0] = close[0]
    rebuilt[1:] = close[0] * np.cumprod(1.0 + rets)
    scale = rebuilt / close
    out = df.copy()
    for col in ("Open", "High", "Low", "Close"):
        out[col] = df[col].to_numpy(float) * scale
    return out


def main() -> int:
    eng = MTE.IntegratedMomentumEngine()

    def sig(bar_set, tf):
        m = eng.generate_enhanced_signal(bar_set, SENT, ALT, tf)
        return _generate_master_signal(m, LLM, SENT, None, tf, "momentum", None)["signal"]

    print("fetching...")
    data = {s: b for s in BASKET if (b := bars(s))}
    if not data:
        print("no market data -- cannot calibrate")
        return 1
    n_real = len(data) * len(TFS)
    print(f"{len(data)} symbols x {len(TFS)} timeframes; "
          f"{n_real * REPS} permuted observations, {n_real} real\n")

    print(f"{'dead':>6}{'agr':>6}{'FPR(perm)':>11}{'fire(real)':>12}{'lift':>7}")
    print("-" * 42)
    for dead, agr in GRID:
        MTE.DIRECTION_DEADBAND, MTE.DIRECTION_AGREEMENT = dead, agr
        fp = n = 0
        for i, (sym, o) in enumerate(data.items()):
            rng = np.random.default_rng(11 + i)
            for _ in range(REPS):
                pb = {k: permute(v, rng) for k, v in o.items()}
                for tf in TFS:
                    n += 1
                    if sig(pb, tf) != "HOLD":
                        fp += 1
        real = sum(1 for o in data.values() for tf in TFS if sig(o, tf) != "HOLD")
        fpr, rr = fp / n * 100, real / n_real * 100
        lift = rr / fpr if fpr else float("inf")
        mark = "  <-- in use" if (dead, agr) == IN_USE else ""
        print(f"{dead:>6.2f}{agr:>6.2f}{fpr:>10.1f}%{rr:>11.1f}%{lift:>7.2f}{mark}")

    print("\nLIFT IS THE COLUMN THAT MATTERS. Below 1.0 the engine fires more often on")
    print("shuffled data than on real data, which is worse than useless. Restore the")
    print("gates from this table -- never by eye -- and re-run after touching the")
    print("direction score, the momentum magnitude, or the confidence weights.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
