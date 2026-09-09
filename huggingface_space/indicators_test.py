"""Check the Space's vectorised Wilder indicators against the live lab's proven ones.

    venv\\Scripts\\python.exe huggingface_space\\indicators_test.py

The Space needs pandas Series (thousands of bars, one value per row). The live lab needs
point-in-time scalars from plain lists and is deliberately dependency-free. Those are two
implementations of the same maths, so one of them has to be checked against the other --
otherwise "I ported it" is just a claim.

lab_indicators.py is the oracle: a verbatim copy of trade_analysis/live_lab/indicators.py,
which is exercised every session by the live forward test. This asserts the vectorised
versions in indicators.py agree with it at the final bar, which is the only bar the Space
ever reads.

Why this file exists at all: the fallback it guards used to be
`df_enriched['ADX_9'] = 25.0`, a constant, reached whenever pandas_ta fails to import --
which it does on numpy 2.x, because pandas_ta 0.3.14b0 still does `from numpy import NaN`.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from trade_analysis import lab_indicators as lab
from trade_analysis.indicators import wilder_adx, wilder_atr, wilder_rsi

TOL = 1e-6


def synthetic(n: int = 600, seed: int = 7) -> pd.DataFrame:
    """A trending-then-chopping series, so ADX has something to actually move through."""
    rng = np.random.default_rng(seed)
    drift = np.concatenate([np.full(n // 3, 0.0009), np.full(n // 3, -0.0011),
                            np.zeros(n - 2 * (n // 3))])
    ret = drift + rng.normal(0, 0.004, n)
    close = 100.0 * np.exp(np.cumsum(ret))
    high = close * (1 + np.abs(rng.normal(0, 0.0025, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.0025, n)))
    open_ = np.concatenate([[close[0]], close[:-1]])
    vol = rng.integers(1_000, 90_000, n).astype(float)
    return pd.DataFrame({"Open": open_, "High": high, "Low": low,
                         "Close": close, "Volume": vol})


def main() -> int:
    ok = True

    def check(name, got, want, tol=TOL):
        nonlocal ok
        if want is None:
            print(f"  [SKIP] {name}: lab returned None")
            return
        if got is None or (isinstance(got, float) and np.isnan(got)):
            ok = False
            print(f"  [FAIL] {name}: vectorised produced {got}, lab {want:.10f}")
            return
        d = abs(float(got) - float(want))
        good = d <= tol
        ok = ok and good
        print(f"  [{'PASS' if good else 'FAIL'}] {name:<22} "
              f"vectorised {float(got):>12.8f}   lab {float(want):>12.8f}   d={d:.2e}")

    print("=" * 88)
    print("SPACE (pandas, vectorised)  vs  LIVE LAB (dependency-free, proven)")
    print("=" * 88)

    for seed in (7, 21, 99):
        df = synthetic(seed=seed)
        h = df["High"].tolist()
        lo = df["Low"].tolist()
        c = df["Close"].tolist()
        print(f"\nseed {seed}  ({len(df)} bars)")

        for period in (9, 14):
            a_lab, p_lab, m_lab = lab.adx_dmi(h, lo, c, period)
            a_vec, p_vec, m_vec = wilder_adx(df, period)
            check(f"ADX({period})", a_vec.iloc[-1], a_lab)
            check(f"+DI({period})", p_vec.iloc[-1], p_lab)
            check(f"-DI({period})", m_vec.iloc[-1], m_lab)

        check("ATR(14)", wilder_atr(df, 14).iloc[-1], lab.atr(h, lo, c, 14))

    # The specific defect this replaced: a constant that made every gate degenerate.
    print("\n" + "=" * 88)
    print("REGRESSION: ADX must not be constant")
    print("=" * 88)
    vals = []
    for seed in (1, 2, 3, 4, 5):
        a, _, _ = wilder_adx(synthetic(seed=seed), 9)
        vals.append(round(float(a.iloc[-1]), 6))
    print(f"  ADX(9) across 5 series: {vals}")
    distinct = len(set(vals))
    if distinct < len(vals):
        ok = False
        print(f"  [FAIL] only {distinct} distinct values -- looks constant again")
    else:
        print(f"  [PASS] {distinct}/{len(vals)} distinct")

    a, _, _ = wilder_adx(synthetic(seed=1), 9)
    exactly_25 = bool(np.isclose(float(a.iloc[-1]), 25.0))
    print(f"  [{'FAIL' if exactly_25 else 'PASS'}] final ADX is not exactly 25.0 "
          f"(the old hardcoded default, which made `adx > 25` permanently False)")
    ok = ok and not exactly_25

    print(f"\n  RESULT: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
