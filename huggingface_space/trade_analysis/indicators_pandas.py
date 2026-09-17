"""pandas-based indicators with pandas_ta-compatible signatures.

WHY THIS EXISTS
---------------
`pandas_ta==0.3.14b0` was pinned in three requirements files and imported by three
modules, and it has been **deleted from PyPI** -- the whole 0.3.x release history was
removed and the package changed maintainer. `pip install -r requirements.txt` therefore
fails outright:

    ERROR: Could not find a version that satisfies the requirement
           pandas_ta==0.3.14b0 (from versions: none)

Only 0.4.67b0 / 0.4.71b0 remain, and both require Python >= 3.12 and numpy >= 2.2.6 --
which conflicts with this repo's own `numpy==1.26.4`. Even with the old wheel in hand, 20
of its modules do `from numpy import NaN`, an alias numpy 2.0 expired, so it fails at
import on any numpy 2.x. There is no version of "keep pandas_ta" that works.

So the dependency is dropped, and this is what replaces it.

WHERE THE MATH COMES FROM
-------------------------
Nothing here is newly invented. Every function is ported from
`huggingface_space/trade_analysis/indicators.py`, whose Wilder implementations already
exist for exactly this reason and are checked by `indicators_test.py` against
`trade_analysis/live_lab/indicators.py` -- the dependency-free pure-Python
implementations the live lab actually trades on. That file is the oracle, and
`indicators_pandas_test.py` re-checks this module against it directly.

Wilder smoothing is seeded with the SMA of the first `period` values and then recursed.
That is NOT `ewm(alpha=1/period).mean()`, which seeds from the first value alone and
drifts for hundreds of bars on intraday data. pandas_ta got this right; a naive `ewm`
replacement would silently change every ADX and RSI in the repo.

WHAT IS AND IS NOT HERE
-----------------------
Only the seven functions the three call sites actually used: ema, rsi, atr, adx, macd,
bbands, vwap. This is not a pandas_ta replacement and should not grow into one -- if you
need a new indicator, the live lab's `indicators.py` is the place it belongs, because
that one is proven against replay.

Column names match pandas_ta's output names (`ADX_14`, `MACDh_12_26_9`, `BBL_20_2.0`,
...) so the call sites did not need their downstream column references rewritten.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["wilder_rma", "true_range", "ema", "rsi", "atr", "adx", "macd", "bbands",
           "vwap"]


def wilder_rma(series: pd.Series, period: int) -> pd.Series:
    """Wilder's smoothing: seed with the SMA of the first `period`, then recurse."""
    s = series.astype(float)
    out = pd.Series(np.nan, index=s.index, dtype=float)
    if len(s) < period:
        return out
    seed = s.iloc[:period].mean()
    out.iloc[period - 1] = seed
    prev = seed
    vals = s.to_numpy()
    o = out.to_numpy()
    for i in range(period, len(s)):
        prev = (prev * (period - 1) + vals[i]) / period
        o[i] = prev
    return pd.Series(o, index=s.index)


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    return pd.concat([high - low,
                      (high - prev_close).abs(),
                      (low - prev_close).abs()], axis=1).max(axis=1)


def ema(close: pd.Series, length: int = 9) -> pd.Series:
    """Standard EMA with k = 2/(length+1), seeded with an SMA of the first `length`.

    DELIBERATELY NOT `ewm(span=length, adjust=False)`, and this is the one place this
    module knowingly departs from pandas_ta. `ewm(adjust=False)` seeds from the FIRST
    VALUE; `live_lab/indicators.py:ema_series` seeds from the SMA of the first `length`,
    and leaves everything before index `length-1` undefined.

    The live lab's definition wins because it is the one `replay.py` proves live and
    batch agree on, and the one every trade in `live_lab_data/` was taken under. Matching
    a library that no longer exists, at the cost of having two EMAs in one repo, is the
    worse trade. `indicators_pandas_test.py` pins this against the oracle.

    The practical difference is a first-bar seed error decaying as (1-k)^n -- invisible
    after a few hundred bars, and about 0.8% on a MACD histogram at bar 60, which is
    exactly the size of error that reads as noise and moves a gate.
    """
    s = close.astype(float)
    out = pd.Series(np.nan, index=s.index, dtype=float)
    if len(s) < length:
        return out
    k = 2.0 / (length + 1.0)
    vals = s.to_numpy()
    o = out.to_numpy()
    prev = float(vals[:length].mean())
    o[length - 1] = prev
    for i in range(length, len(s)):
        prev = vals[i] * k + prev * (1.0 - k)
        o[i] = prev
    return pd.Series(o, index=s.index)


def rsi(close: pd.Series, length: int = 14) -> pd.Series:
    delta = close.astype(float).diff()
    avg_gain = wilder_rma(delta.clip(lower=0.0), length)
    avg_loss = wilder_rma((-delta).clip(lower=0.0), length)
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out = 100.0 - (100.0 / (1.0 + rs))
    return out.fillna(100.0).where(avg_loss.notna(), np.nan)


def atr(high: pd.Series, low: pd.Series, close: pd.Series,
        length: int = 14) -> pd.Series:
    return wilder_rma(true_range(high, low, close), length)


def adx(high: pd.Series, low: pd.Series, close: pd.Series,
        length: int = 14) -> pd.DataFrame:
    """Returns a frame with ADX_n / DMP_n / DMN_n, matching pandas_ta's column names."""
    up = high.astype(float).diff()
    dn = -low.astype(float).diff()
    plus_dm = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=high.index)
    minus_dm = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=high.index)

    atr_ = wilder_rma(true_range(high, low, close), length)
    plus_di = 100.0 * wilder_rma(plus_dm, length) / atr_.replace(0.0, np.nan)
    minus_di = 100.0 * wilder_rma(minus_dm, length) / atr_.replace(0.0, np.nan)

    denom = (plus_di + minus_di).replace(0.0, np.nan)
    dx = 100.0 * (plus_di - minus_di).abs() / denom
    # dropna before smoothing, then reindex: DX is undefined until both DIs exist, and
    # smoothing across those NaNs would poison every later value.
    adx_ = wilder_rma(dx.dropna(), length).reindex(high.index)
    return pd.DataFrame({f"ADX_{length}": adx_,
                         f"DMP_{length}": plus_di,
                         f"DMN_{length}": minus_di})


def macd(close: pd.Series, fast: int = 12, slow: int = 26,
         signal: int = 9) -> pd.DataFrame:
    """MACD line / histogram / signal, matching `live_lab/indicators.py:macd_hist`.

    The subtlety is the signal line. The MACD line is undefined until BOTH EMAs exist,
    i.e. from index `slow-1`. The signal EMA must be seeded from that point, over the
    defined values only -- not run across the leading NaNs, and not started where the
    fast EMA alone became available. Getting this wrong shifts the whole histogram by
    under a percent, which is far too small to notice and far too large to ignore.

    Note `fast`/`slow` default to the textbook 12/26 here, while the live lab's
    `macd_hist` defaults to the trader's declared 9/17. Pass them explicitly.
    """
    line = ema(close, fast) - ema(close, slow)
    defined = line.dropna()
    sig = ema(defined, signal).reindex(line.index)
    tag = f"{fast}_{slow}_{signal}"
    return pd.DataFrame({f"MACD_{tag}": line,
                         f"MACDh_{tag}": line - sig,
                         f"MACDs_{tag}": sig})


def bbands(close: pd.Series, length: int = 20, std: float = 2.0) -> pd.DataFrame:
    c = close.astype(float)
    mid = c.rolling(length).mean()
    # ddof=0 -- pandas_ta uses the population standard deviation here, and pandas'
    # rolling default is ddof=1. At length=20 that is a 2.6% difference in band width,
    # which is small enough to look like noise and large enough to move a gate.
    dev = c.rolling(length).std(ddof=0)
    tag = f"{length}_{float(std)}"
    return pd.DataFrame({f"BBL_{tag}": mid - std * dev,
                         f"BBM_{tag}": mid,
                         f"BBU_{tag}": mid + std * dev})


def vwap(high: pd.Series, low: pd.Series, close: pd.Series,
         volume: pd.Series) -> pd.Series:
    """Session-anchored VWAP on typical price.

    pandas_ta anchors per calendar day off a DatetimeIndex. Replicated: grouping by date
    is the whole point, because a VWAP that runs cumulatively across days is not a VWAP,
    it is a slowly rising line that never means anything intraday.
    """
    tp = (high.astype(float) + low.astype(float) + close.astype(float)) / 3.0
    vol = volume.astype(float)
    idx = high.index
    if isinstance(idx, pd.DatetimeIndex):
        key = idx.normalize()
        num = (tp * vol).groupby(key).cumsum()
        den = vol.groupby(key).cumsum()
    else:
        num = (tp * vol).cumsum()
        den = vol.cumsum()
    return num / den.replace(0.0, np.nan)
