"""Pure indicator functions for the live lab.

Every function here takes a sequence ending at the bar being evaluated and returns the
value AT that bar. Nothing reads forward. Nothing mutates its input.

Deliberately dependency-free (no `ta`, no pandas) so the replay gate in replay.py can
prove that live and batch computation produce byte-identical values.
"""
from __future__ import annotations

import math
from typing import Sequence

# --------------------------------------------------------------------------- moving averages


def ema_series(values: Sequence[float], period: int) -> list[float]:
    """Wilder-free standard EMA, seeded with an SMA of the first `period` values.

    Returns a list the same length as `values`; entries before the seed are None.
    """
    out: list[float | None] = [None] * len(values)
    if len(values) < period:
        return out  # type: ignore[return-value]
    seed = sum(values[:period]) / period
    out[period - 1] = seed
    k = 2.0 / (period + 1.0)
    prev = seed
    for i in range(period, len(values)):
        prev = values[i] * k + prev * (1.0 - k)
        out[i] = prev
    return out  # type: ignore[return-value]


def sma(values: Sequence[float], period: int) -> float | None:
    if len(values) < period:
        return None
    return sum(values[-period:]) / period


def stdev(values: Sequence[float], period: int) -> float | None:
    """Population SD over the trailing `period` values (matches Bollinger convention)."""
    if len(values) < period:
        return None
    w = values[-period:]
    m = sum(w) / period
    return math.sqrt(sum((x - m) ** 2 for x in w) / period)


# --------------------------------------------------------------------------- MACD


def macd_hist(closes: Sequence[float], fast: int = 9, slow: int = 17, signal: int = 9):
    """MACD histogram at the last bar. Returns None until fully seeded.

    Defaults are the trader's declared 9/17/9, not the textbook 12/26/9.
    """
    if len(closes) < slow + signal:
        return None
    ef = ema_series(closes, fast)
    es = ema_series(closes, slow)
    line = [
        (a - b) if (a is not None and b is not None) else None
        for a, b in zip(ef, es)
    ]
    valid = [x for x in line if x is not None]
    if len(valid) < signal:
        return None
    sig = ema_series(valid, signal)
    if sig[-1] is None:
        return None
    return valid[-1] - sig[-1]


# --------------------------------------------------------------------------- Wilder ATR / ADX / DMI


def _true_ranges(highs, lows, closes) -> list[float]:
    tr = [highs[0] - lows[0]]
    for i in range(1, len(highs)):
        tr.append(max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1]),
        ))
    return tr


def _wilder_smooth(values: Sequence[float], period: int) -> list[float | None]:
    out: list[float | None] = [None] * len(values)
    if len(values) < period:
        return out
    acc = sum(values[:period])
    out[period - 1] = acc
    for i in range(period, len(values)):
        acc = acc - (acc / period) + values[i]
        out[i] = acc
    return out


def atr(highs, lows, closes, period: int = 14) -> float | None:
    """Wilder ATR at the last bar."""
    if len(highs) < period + 1:
        return None
    tr = _true_ranges(highs, lows, closes)
    sm = _wilder_smooth(tr, period)
    return None if sm[-1] is None else sm[-1] / period


def adx_dmi(highs, lows, closes, period: int = 14):
    """Returns (adx, plus_di, minus_di) at the last bar, or (None, None, None)."""
    n = len(highs)
    if n < 2 * period + 1:
        return (None, None, None)
    tr = _true_ranges(highs, lows, closes)
    plus_dm, minus_dm = [0.0], [0.0]
    for i in range(1, n):
        up = highs[i] - highs[i - 1]
        dn = lows[i - 1] - lows[i]
        plus_dm.append(up if (up > dn and up > 0) else 0.0)
        minus_dm.append(dn if (dn > up and dn > 0) else 0.0)

    str_ = _wilder_smooth(tr, period)
    spd = _wilder_smooth(plus_dm, period)
    smd = _wilder_smooth(minus_dm, period)

    dxs: list[float] = []
    for i in range(len(tr)):
        if str_[i] is None or not str_[i]:
            continue
        p = 100.0 * spd[i] / str_[i]
        m = 100.0 * smd[i] / str_[i]
        denom = p + m
        dxs.append(0.0 if denom == 0 else 100.0 * abs(p - m) / denom)
    if len(dxs) < period:
        return (None, None, None)
    # Wilder's ADX: SMA of the first `period` DX values, then smoothed
    a = sum(dxs[:period]) / period
    for i in range(period, len(dxs)):
        a = (a * (period - 1) + dxs[i]) / period
    p_last = 100.0 * spd[-1] / str_[-1] if str_[-1] else None
    m_last = 100.0 * smd[-1] / str_[-1] if str_[-1] else None
    return (a, p_last, m_last)


# --------------------------------------------------------------------------- VWAP


def session_vwap(bars: Sequence[dict]) -> float | None:
    """Cumulative session VWAP anchored at the FIRST bar supplied.

    The caller is responsible for passing only bars from 09:30 onward. Passing a frame
    that spans sessions is the classic anchoring bug and is not defended against here --
    session.py owns that guarantee.
    """
    num = den = 0.0
    for b in bars:
        typ = (b["high"] + b["low"] + b["close"]) / 3.0
        num += typ * b["volume"]
        den += b["volume"]
    return None if den <= 0 else num / den


def vwap_sigma(bars: Sequence[dict], vwap: float) -> float | None:
    """Volume-weighted SD of typical price about the session VWAP (for 2-sigma bands)."""
    num = den = 0.0
    for b in bars:
        typ = (b["high"] + b["low"] + b["close"]) / 3.0
        num += b["volume"] * (typ - vwap) ** 2
        den += b["volume"]
    return None if den <= 0 else math.sqrt(num / den)


# --------------------------------------------------------------------------- RVOL


def rvol_time_of_day(cum_volume_today: float, baseline_cum_by_minute: dict,
                     minute_key: str) -> float | None:
    """Time-of-day normalised RVOL.

    cumulative volume to minute m today  /  mean cumulative volume to the SAME minute m
    over the prior N sessions.

    Dividing by a whole-day average instead makes every morning bar look high-volume.
    That is a normalisation bug, not a signal, and it is why this takes a per-minute map.
    """
    base = baseline_cum_by_minute.get(minute_key)
    if not base:
        return None
    return cum_volume_today / base


def rvol_bar(volumes: Sequence[float], period: int) -> float | None:
    """This bar's volume vs the mean of the `period` bars BEFORE it (excludes itself)."""
    if len(volumes) < period + 1:
        return None
    prior = volumes[-(period + 1):-1]
    m = sum(prior) / len(prior)
    return None if m <= 0 else volumes[-1] / m


# --------------------------------------------------------------------------- misc


def linreg_endpoint(values: Sequence[float], period: int) -> float | None:
    """Fitted value at the LAST point of an OLS line over the trailing `period` points.

    Endpoint, never centred. A centred or refit regression would peek at later bars and
    is the specific lookahead trap flagged for TTM_Squeeze.
    """
    if len(values) < period:
        return None
    y = list(values[-period:])
    n = period
    sx = n * (n - 1) / 2.0
    sxx = (n - 1) * n * (2 * n - 1) / 6.0
    sy = sum(y)
    sxy = sum(i * y[i] for i in range(n))
    denom = n * sxx - sx * sx
    if denom == 0:
        return None
    slope = (n * sxy - sx * sy) / denom
    intercept = (sy - slope * sx) / n
    return intercept + slope * (n - 1)


def realised_sigma(closes: Sequence[float]) -> float | None:
    """SD of 1-bar close-to-close returns over the supplied window."""
    if len(closes) < 3:
        return None
    rets = [(closes[i] / closes[i - 1] - 1.0) for i in range(1, len(closes)) if closes[i - 1]]
    if len(rets) < 2:
        return None
    m = sum(rets) / len(rets)
    var = sum((r - m) ** 2 for r in rets) / (len(rets) - 1)
    s = math.sqrt(var)
    return s if s > 0 else None
