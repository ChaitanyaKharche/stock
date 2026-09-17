"""Does the pandas_ta replacement agree with the indicators the live lab trades on?

    python -m trade_analysis.indicators_pandas_test
    pytest trade_analysis/indicators_pandas_test.py

`indicators_pandas.py` replaced `pandas_ta`, which was deleted from PyPI. A replacement
for a numerical dependency is worthless unless something pins the numbers, because the
failure mode is not a crash -- it is an ADX that is 3% different, which moves a gate and
changes which trades a backtest takes, silently.

The oracle is `trade_analysis/live_lab/indicators.py`: dependency-free, pure-Python,
one-value-at-a-time, and what the live lab actually computes on. `replay.py` already
proves live and batch agree with it byte-for-byte. So if the vectorised pandas version
matches the oracle, it matches what is traded.

The comparison deliberately runs on a RANDOM WALK, not a synthetic ramp. A monotone
series makes minus_dm identically zero, so DX is 100 everywhere and an ADX bug cannot
show up. Wilder smoothing also needs a long burn-in before the seed stops dominating,
which is why 400 bars are generated and only the tail is compared.
"""
from __future__ import annotations

import random
import sys

import numpy as np
import pandas as pd

from . import indicators_pandas as ip
from .live_lab import indicators as oracle

N = 400
SEED = 20260917
# Wilder's seed is an SMA, so early values legitimately differ from the recursion's
# steady state. Compare only once smoothing has converged; 3x the longest period.
BURN = 60
# Round-off, not "close enough". These are the SAME recursion, vectorised -- a correct
# port differs only by float summation order, which is ~1e-15 relative here.
#
# The first version of this file used 1e-9 relative, and that was nearly useless: the
# wrong EMA (first-value-seeded `ewm`, the trap this module exists to avoid) produced a
# max error of 4.712e-07 against a tolerance of 4.806e-07. It passed by 2%. A tolerance
# calibrated by accident to just admit the bug is worse than no test, because it reads
# green. Measured errors at this tolerance: ema exactly 0.0, atr/adx/macd ~1e-13.
TOL = 1e-12
REL = 1e-12


def _walk() -> pd.DataFrame:
    rng = random.Random(SEED)
    px, rows = 480.0, []
    for _ in range(N):
        o = px
        px = o * (1 + rng.gauss(0, 0.0012))
        hi = max(o, px) * (1 + abs(rng.gauss(0, 0.0004)))
        lo = min(o, px) * (1 - abs(rng.gauss(0, 0.0004)))
        rows.append({"Open": o, "High": hi, "Low": lo, "Close": px,
                     "Volume": float(rng.randint(1000, 90000))})
    idx = pd.date_range("2026-09-01 09:30", periods=N, freq="1min")
    return pd.DataFrame(rows, index=idx)


DF = _walk()
H, L, C, V = DF["High"], DF["Low"], DF["Close"], DF["Volume"]


def _close_enough(got, want, what):
    assert want is not None, f"{what}: oracle returned None"
    assert not np.isnan(got), f"{what}: vectorised returned NaN"
    assert abs(got - want) <= max(TOL, abs(want) * REL), \
        f"{what}: vectorised {got!r} vs oracle {want!r} (diff {abs(got - want):.3e})"


def test_atr_matches_the_live_lab():
    """The oracle returns the value AT the last bar, so compare bar by bar."""
    series = ip.atr(H, L, C, length=14)
    for i in range(BURN, N, 37):
        want = oracle.atr(H.iloc[:i + 1].tolist(), L.iloc[:i + 1].tolist(),
                          C.iloc[:i + 1].tolist(), period=14)
        _close_enough(series.iloc[i], want, f"atr@{i}")


def test_adx_and_both_dis_match_the_live_lab():
    """ADX is the one that matters most: three gates in this repo compare against it."""
    frame = ip.adx(H, L, C, length=14)
    for i in range(BURN, N, 37):
        hi, lo, cl = (H.iloc[:i + 1].tolist(), L.iloc[:i + 1].tolist(),
                      C.iloc[:i + 1].tolist())
        w_adx, w_plus, w_minus = oracle.adx_dmi(hi, lo, cl, period=14)
        _close_enough(frame["ADX_14"].iloc[i], w_adx, f"adx@{i}")
        _close_enough(frame["DMP_14"].iloc[i], w_plus, f"plus_di@{i}")
        _close_enough(frame["DMN_14"].iloc[i], w_minus, f"minus_di@{i}")


def test_ema_matches_the_live_lab():
    got = ip.ema(C, length=9)
    want = oracle.ema_series(C.tolist(), 9)
    for i in range(BURN, N, 37):
        _close_enough(got.iloc[i], want[i], f"ema@{i}")


def test_macd_histogram_matches_the_live_lab():
    """The oracle's MACD defaults are 9/17/9, not 12/26/9. Pass them explicitly."""
    frame = ip.macd(C, fast=9, slow=17, signal=9)
    for i in range(BURN, N, 37):
        want = oracle.macd_hist(C.iloc[:i + 1].tolist(), fast=9, slow=17, signal=9)
        want_hist = want[-1] if isinstance(want, (tuple, list)) else want
        _close_enough(frame["MACDh_9_17_9"].iloc[i], want_hist, f"macdh@{i}")


def test_vwap_is_session_anchored_not_cumulative():
    """A VWAP that runs across days is not a VWAP. Two sessions, checked separately."""
    two = pd.concat([DF, DF.set_index(DF.index + pd.Timedelta(days=1))])
    got = ip.vwap(two["High"], two["Low"], two["Close"], two["Volume"])
    # The last bar of day 1 and the last bar of day 2 see identical bars, so an
    # anchored VWAP gives identical values; a cumulative one cannot.
    assert abs(got.iloc[N - 1] - got.iloc[-1]) < 1e-9, "vwap leaked across the day break"

    bars = [{"high": h, "low": l, "close": c, "volume": v}
            for h, l, c, v in zip(H, L, C, V)]
    want = oracle.session_vwap(bars)
    _close_enough(got.iloc[N - 1], want, "vwap")


def test_rsi_is_bounded_and_responds():
    """No pure-Python RSI oracle exists, so assert the properties instead of a value."""
    got = ip.rsi(C, length=14).iloc[BURN:]
    assert got.notna().all(), "RSI went NaN after burn-in"
    assert got.between(0.0, 100.0).all(), "RSI left [0, 100]"
    assert got.std() > 1.0, "RSI is nearly constant; it is not reading the input"

    rising = pd.Series(np.linspace(100.0, 200.0, 100))
    assert ip.rsi(rising, length=14).iloc[-1] > 99.0, "RSI not ~100 on a pure uptrend"


def test_bbands_use_the_population_sigma():
    """pandas' rolling default is ddof=1; pandas_ta uses ddof=0.

    At length=20 that is a 2.6% difference in band width -- small enough to look like
    noise, large enough to move a gate. Pinned so a future edit cannot drop the ddof.
    """
    frame = ip.bbands(C, length=20, std=2.0)
    mid = C.rolling(20).mean()
    pop = C.rolling(20).std(ddof=0)
    assert np.allclose(frame["BBM_20_2.0"].iloc[20:], mid.iloc[20:])
    assert np.allclose(frame["BBU_20_2.0"].iloc[20:], (mid + 2 * pop).iloc[20:])
    sample = C.rolling(20).std(ddof=1)
    assert not np.allclose(frame["BBU_20_2.0"].iloc[20:], (mid + 2 * sample).iloc[20:]), \
        "bbands used the sample sigma; pandas_ta used the population sigma"


def test_wilder_rma_is_not_a_plain_ewm():
    """The trap a naive port falls into, pinned.

    `ewm(alpha=1/period).mean()` seeds from the first value alone and drifts for
    hundreds of bars on intraday data. If someone 'simplifies' wilder_rma to that, every
    ADX and RSI in the repo changes quietly. This asserts the two are NOT equal.
    """
    got = ip.wilder_rma(C, 14)
    naive = C.ewm(alpha=1 / 14, adjust=False).mean()
    assert not np.allclose(got.iloc[BURN:], naive.iloc[BURN:]), \
        "wilder_rma collapsed into a plain ewm"
    assert pd.isna(got.iloc[12]) and not pd.isna(got.iloc[13]), \
        "Wilder seeding must start at index period-1"


CHECKS = [(n, f) for n, f in sorted(globals().items())
          if n.startswith("test_") and callable(f)]


def main() -> int:
    ok = True
    print("=" * 78)
    print("pandas_ta REPLACEMENT vs THE LIVE LAB'S OWN INDICATORS")
    print("=" * 78)
    for name, fn in CHECKS:
        try:
            fn()
            print(f"  [PASS] {name[5:].replace('_', ' ')}")
        except AssertionError as exc:
            ok = False
            print(f"  [FAIL] {name[5:].replace('_', ' ')}\n         {exc}")
        except Exception as exc:                              # noqa: BLE001
            ok = False
            print(f"  [FAIL] {name[5:].replace('_', ' ')}\n         raised {exc!r}")
    print(f"\n  {len(CHECKS)} checks on {N} bars of a seeded random walk")
    print(f"  RESULT: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
