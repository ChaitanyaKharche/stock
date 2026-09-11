"""Does the displayed confidence actually respond to the symbol?

    venv\\Scripts\\python.exe huggingface_space\\confidence_spread_test.py

The app was reported as showing near-identical numbers for every ticker. It was not a
constant -- but it was close enough to be a fair complaint, and the reason was that two of
the three terms in `_generate_master_signal`'s confidence formula could never move:

    weighted_confidence = momentum_conf*0.4 + llm.get('conviction',50)*0.3
                                            + (sentiment=='HIGH')*80*0.3

  * the CPU LLM fallback returns "confidence", not "conviction", so the second term was a
    constant 15;
  * sentiment HIGH requires std_dev < 0.2 AND mean_abs > 0.3, which headline scores
    clustered near 0 and 1 effectively never satisfy, so the third was a constant 0.

60% of the weight was pinned. This runs the REAL `_generate_master_signal` over real bars
and asserts the output responds to its inputs, so the complaint cannot silently return.
"""
from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

# config.py raises at import time when FINNHUB_KEY / REDDIT_CLIENT_ID are unset, and
# importing enhanced_api pulls it in. This test never calls Finnhub or Reddit -- bars come
# from yfinance and the sentiment/LLM inputs are supplied directly -- so it only needs the
# guard to pass. Placeholders, set only when the real ones are absent, so a developer with
# real keys configured is unaffected.
for _k in ("FINNHUB_KEY", "REDDIT_CLIENT_ID", "REDDIT_CLIENT_SECRET", "REDDIT_USER_AGENT"):
    os.environ.setdefault(_k, "unused-by-this-test")

import pandas as pd
import yfinance as yf

from trade_analysis.enhanced_api import _generate_master_signal
from trade_analysis.momentum_trading_engine import IntegratedMomentumEngine

SPEC = {"15m": ("60d", "15m"), "hourly": ("180d", "1h"), "daily": ("2y", "1d")}
COLS = ["Open", "High", "Low", "Close", "Volume"]
SYMS = ["SPY", "QQQ", "NVDA", "GOOG", "TSLA", "AVGO", "XLE", "KO"]

# A minimum spread the number must show across a mixed basket. Not a quality bar -- purely
# a regression guard: anything at or below this means a term has gone constant again.
MIN_SPREAD = 5.0


def bars(sym: str) -> dict:
    out = {}
    for tf, (period, interval) in SPEC.items():
        df = yf.Ticker(sym).history(period=period, interval=interval, auto_adjust=False)
        if df is None or df.empty:
            continue
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        out[tf] = df[COLS].dropna()
    return out


def main() -> int:
    eng = IntegratedMomentumEngine()
    sentiment = {"composite_score": 0.05, "confidence": "LOW"}
    alt = {"vix_level": 14.5, "implied_vol_pct": 60.0}
    tft = {"expected_direction": "FLAT", "gap_probability": 50}

    print("=" * 78)
    print("CONFIDENCE SPREAD -- real bars through the real master-signal function")
    print("=" * 78)
    print(f"  {'sym':<7}{'conviction':>12}{'mom_conf':>10}{'shown %':>10}  signal")
    rows = []
    for sym in SYMS:
        try:
            momentum = eng.generate_enhanced_signal(bars(sym), sentiment, alt)
        except Exception as exc:                                  # noqa: BLE001
            print(f"  {sym:<7}  ERROR {type(exc).__name__}: {str(exc)[:44]}")
            continue
        master = _generate_master_signal(momentum, {"signal": "HOLD", "confidence": 40},
                                         sentiment, tft, "15m", "momentum")
        conv = momentum["momentum_analysis"]["master_signal"]["conviction"]
        rows.append((sym, master["confidence"]))
        print(f"  {sym:<7}{conv:>12.3f}{momentum['confidence']:>10}"
              f"{master['confidence']:>10}  {master['signal']}")

    vals = [c for _, c in rows]
    spread = max(vals) - min(vals)
    distinct = len(set(vals))
    print(f"\n  min {min(vals)}   max {max(vals)}   spread {spread}   "
          f"distinct values {distinct}/{len(vals)}")

    ok = spread >= MIN_SPREAD
    print(f"  [{'PASS' if ok else 'FAIL'}] spread >= {MIN_SPREAD:.0f} points "
          f"(guards against a term going constant again)")

    # The specific regression: the llm term must not be the old hardcoded default.
    from trade_analysis.enhanced_api import _generate_master_signal as f
    a = f({"signal": "HOLD", "confidence": 0}, {"signal": "HOLD", "confidence": 20},
          sentiment, tft, "15m", "momentum")["confidence"]
    b = f({"signal": "HOLD", "confidence": 0}, {"signal": "HOLD", "confidence": 90},
          sentiment, tft, "15m", "momentum")["confidence"]
    moved = a != b
    print(f"  [{'PASS' if moved else 'FAIL'}] llm confidence 20 vs 90 -> {a} vs {b} "
          f"({'reads the real key' if moved else 'STILL constant'})")

    c = f({"signal": "HOLD", "confidence": 0}, {"signal": "HOLD", "confidence": 40},
          {"composite_score": 0.0, "confidence": "MEDIUM"}, tft, "15m", "momentum")["confidence"]
    d = f({"signal": "HOLD", "confidence": 0}, {"signal": "HOLD", "confidence": 40},
          {"composite_score": 0.0, "confidence": "LOW"}, tft, "15m", "momentum")["confidence"]
    graded = c != d
    print(f"  [{'PASS' if graded else 'FAIL'}] sentiment MEDIUM vs LOW -> {c} vs {d} "
          f"({'graded' if graded else 'MEDIUM still discarded'})")

    # The TFT was removed from the weighting on 2026-09-06 after being measured as
    # degenerate: fed six different symbols' full histories it moved gap_probability by
    # 0.10 on a 0-100 scale, and out of sample it emitted ONE constant direction per symbol
    # at exactly the majority-class base rate. A constant direction times a +-0.7 score was
    # a fixed per-symbol bias, not information. It must no longer move the decision.
    mom = {"signal": "HOLD", "confidence": 30,
           "momentum_analysis": {"master_signal": {"strategy": "CAUTIOUS_ENTRY",
                                                   "conviction": 0.3}}}
    outs = {}
    for direction in ("UP", "DOWN", "FLAT"):
        r = f(mom, {"signal": "HOLD", "confidence": 40}, sentiment,
              {"expected_direction": direction, "gap_probability": 50}, "15m", "momentum")
        outs[direction] = (r["signal"], r["confidence"], round(r["weighted_score"], 6))
    inert = len(set(outs.values())) == 1
    print(f"  [{'PASS' if inert else 'FAIL'}] TFT UP/DOWN/FLAT -> "
          f"{outs['UP']} / {outs['DOWN']} / {outs['FLAT']}")
    print(f"        {'inert, as intended' if inert else 'STILL VOTING -- it was removed'}")

    # -- the production gate must survive being read -----------------------------------
    # TIMEFRAME_CONFIGS moved to module scope on 2026-09-11 so the calibrator could import
    # it instead of keeping a copy that had silently drifted to 36/34/33/32. That fixed
    # one hazard and created another: the strategy-mode branches MUTATE the config
    # (gap -10, confirmed reversal -8), and against a shared dict that would ratchet the
    # live gate down on every request -- a drift with no bad commit to point at. The
    # dict() copy in _generate_master_signal is the only thing preventing it, so it gets
    # a test rather than a comment.
    from trade_analysis.enhanced_api import TIMEFRAME_CONFIGS
    gate_before = {k: v["min_confidence"] for k, v in TIMEFRAME_CONFIGS.items()}
    big_gap_mom = {"signal": "HOLD", "confidence": 30,
                   "momentum_analysis": {"master_signal": {"strategy": "STANDARD_MOMENTUM",
                                                           "conviction": 0.5}}}
    for _ in range(5):
        f(big_gap_mom, {"signal": "HOLD", "confidence": 40}, sentiment, alt,
          "1d", "gap", gap_pct=3.0)
    gate_after = {k: v["min_confidence"] for k, v in TIMEFRAME_CONFIGS.items()}
    unmutated = gate_before == gate_after
    print(f"  [{'PASS' if unmutated else 'FAIL'}] gap mode x5 leaves the live gate alone "
          f"-> {gate_after}")
    if not unmutated:
        print(f"        RATCHETED from {gate_before} -- the dict() copy is gone")

    ok = ok and moved and graded and inert and unmutated
    print(f"\n  RESULT: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
