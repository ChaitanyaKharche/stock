"""
Dealer gamma exposure (GEX) for a symbol's option chain, using free yfinance
option-chain data (strike, open interest, implied vol) + Black-Scholes gamma.

Convention (matches the widely-used SqueezeMetrics/SpotGamma-style formula):
dealers are assumed net long the calls and net short the puts customers hold,
so dealer GEX per strike = OI_call * gamma_call - OI_put * gamma_put, scaled
to dollars-of-delta-hedging-per-1%-move. This is an industry heuristic, not
an observed fact (true dealer positioning isn't public) - the sign convention
is the standard one, but treat it as a modeling assumption.

Positive total GEX -> dealers net long gamma -> they hedge by buying dips /
selling rallies -> price gets pinned/dampened, breakouts tend to fade.
Negative total GEX -> dealers net short gamma -> they hedge by selling dips /
buying rallies -> moves amplify, breakouts tend to run.
"""
import math
import time
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yfinance as yf

from ..utils import cache

CONTRACT_MULTIPLIER = 100
RISK_FREE_RATE = 0.05
MIN_T_YEARS = 1 / (365 * 24 * 60)  # 1 minute floor, avoids div-by-zero as expiry approaches

# In-process cache so repeated calls within a short window (e.g. if a caller
# ends up polling more often than once/day) don't re-hit yfinance every time.
MEMORY_CACHE_TTL_SECONDS = 900  # 15 min
_memory_cache = {}  # {cache_key: (fetched_at_epoch, result_dict)}

MAX_RETRIES = 3
RETRY_BASE_DELAY = 1.5  # seconds, doubles each attempt


def _with_retry(fn, *args, **kwargs):
    """Retries a flaky network call (yfinance is prone to transient
    rate-limit/connection errors) with exponential backoff."""
    last_exc = None
    for attempt in range(MAX_RETRIES):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            last_exc = e
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_BASE_DELAY * (2 ** attempt))
    raise last_exc


def _norm_pdf(x):
    return np.exp(-0.5 * x ** 2) / np.sqrt(2 * math.pi)


def bs_gamma(spot, strike, t_years, iv, r=RISK_FREE_RATE):
    """Black-Scholes gamma - same formula for calls and puts."""
    t_years = max(t_years, MIN_T_YEARS)
    iv = np.maximum(iv, 1e-4)
    d1 = (np.log(spot / strike) + (r + 0.5 * iv ** 2) * t_years) / (iv * np.sqrt(t_years))
    return _norm_pdf(d1) / (spot * iv * np.sqrt(t_years))


def _norm_cdf(x):
    return 0.5 * (1 + np.vectorize(math.erf)(x / math.sqrt(2)))


def _d1_d2(spot, strike, t_years, iv, r):
    t_years = max(t_years, MIN_T_YEARS)
    iv = np.maximum(iv, 1e-4)
    d1 = (np.log(spot / strike) + (r + 0.5 * iv ** 2) * t_years) / (iv * np.sqrt(t_years))
    d2 = d1 - iv * np.sqrt(t_years)
    return d1, d2


def bs_price(spot, strike, t_years, iv, r=RISK_FREE_RATE, option_type='call'):
    """Black-Scholes premium for a European call/put (0DTE American-style
    index/ETF options are conventionally priced this way at this level of
    approximation - early-exercise premium is negligible for SPY/QQQ)."""
    t_years = max(t_years, MIN_T_YEARS)
    d1, d2 = _d1_d2(spot, strike, t_years, iv, r)
    if option_type == 'call':
        return spot * _norm_cdf(d1) - strike * np.exp(-r * t_years) * _norm_cdf(d2)
    return strike * np.exp(-r * t_years) * _norm_cdf(-d2) - spot * _norm_cdf(-d1)


def bs_delta(spot, strike, t_years, iv, r=RISK_FREE_RATE, option_type='call'):
    """Black-Scholes delta for a call/put."""
    d1, _ = _d1_d2(spot, strike, t_years, iv, r)
    return _norm_cdf(d1) if option_type == 'call' else _norm_cdf(d1) - 1


def solve_strike_for_delta(spot, t_years, iv, target_delta, option_type='call', r=RISK_FREE_RATE):
    """Finds the strike whose BS delta matches target_delta (e.g. 0.20-0.30
    for the kind of OTM 0DTE contracts this project actually trades), via
    bisection - bs_delta is monotonic in strike so this is well-behaved."""
    target_delta = abs(target_delta)
    lo, hi = spot * 0.5, spot * 1.5
    for _ in range(60):
        mid = (lo + hi) / 2
        d = abs(bs_delta(spot, mid, t_years, iv, r, option_type))
        if d > target_delta:
            # higher strike -> lower call delta / less-negative put delta magnitude moves away from ATM
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


def _time_to_expiry_years(expiry_str: str, tz=ZoneInfo("America/New_York")) -> float:
    now = datetime.now(tz)
    expiry_close = datetime.strptime(expiry_str, "%Y-%m-%d").replace(
        hour=16, minute=0, second=0, tzinfo=tz
    )
    seconds_left = max((expiry_close - now).total_seconds(), 60)
    return seconds_left / (365 * 24 * 3600)


def fetch_spot(symbol: str) -> float:
    ticker = yf.Ticker(symbol)
    try:
        return float(_with_retry(lambda: ticker.fast_info['lastPrice']))
    except Exception:
        hist = _with_retry(ticker.history, period='1d')
        return float(hist['Close'].iloc[-1])


def _compute_gex_uncached(symbol: str, expiry: str, include_next_n_expiries: int) -> dict:
    ticker = yf.Ticker(symbol)
    try:
        all_expiries = _with_retry(lambda: ticker.options)
    except Exception as e:
        return {'symbol': symbol, 'error': f'failed to fetch expiries after retries: {e}'}

    if not all_expiries:
        return {'symbol': symbol, 'error': 'no option chain available'}

    if expiry:
        expiries = [expiry]
    else:
        expiries = list(all_expiries[:max(include_next_n_expiries, 1)])

    spot = fetch_spot(symbol)
    per_strike = {}

    for exp in expiries:
        t_years = _time_to_expiry_years(exp)
        try:
            chain = _with_retry(ticker.option_chain, exp)
        except Exception as e:
            print(f"[GEX] {symbol} {exp}: failed to fetch chain after retries ({e}), skipping this expiry")
            continue

        for opt_type, df in (('call', chain.calls), ('put', chain.puts)):
            if df.empty:
                continue
            gamma = bs_gamma(spot, df['strike'].values, t_years, df['impliedVolatility'].values)
            dollar_gamma = gamma * (spot ** 2) * 0.01 * CONTRACT_MULTIPLIER
            signed_gex = dollar_gamma * df['openInterest'].fillna(0).values * (1 if opt_type == 'call' else -1)

            for strike, g in zip(df['strike'].values, signed_gex):
                per_strike[strike] = per_strike.get(strike, 0.0) + g

    if not per_strike:
        return {'symbol': symbol, 'error': 'no contracts with open interest found'}

    strikes = np.array(sorted(per_strike.keys()))
    gex_by_strike = np.array([per_strike[k] for k in strikes])
    total_gex = gex_by_strike.sum()

    # Gamma flip: nearest strike where cumulative GEX (strikes below spot vs above) changes sign
    below = strikes <= spot
    cum_below = gex_by_strike[below].sum()
    cum_above = gex_by_strike[~below].sum()

    top_idx = np.argsort(-np.abs(gex_by_strike))[:5]
    top_strikes = [(float(strikes[i]), float(gex_by_strike[i])) for i in top_idx]
    top_strikes.sort(key=lambda x: x[0])

    regime = "NET_LONG_GAMMA" if total_gex > 0 else "NET_SHORT_GAMMA"

    return {
        'symbol': symbol,
        'spot': spot,
        'expiries_used': expiries,
        'total_gex': float(total_gex),
        'regime': regime,
        'cum_gex_below_spot': float(cum_below),
        'cum_gex_above_spot': float(cum_above),
        'top_gamma_strikes': top_strikes,   # [(strike, signed_gex), ...] - magnet/pin levels
        'strikes': strikes.tolist(),
        'gex_by_strike': gex_by_strike.tolist(),
    }


def compute_gex(symbol: str, expiry: str = None, include_next_n_expiries: int = 1,
                 force_refresh: bool = False) -> dict:
    """Computes per-strike and total dealer GEX for the given symbol, cached
    two ways so calling this more than once/day doesn't hammer yfinance:
      - in-memory, for MEMORY_CACHE_TTL_SECONDS (protects repeated calls
        within the same process, e.g. accidental tight polling loops)
      - on disk via trade_analysis.utils.cache, keyed by today's date (protects
        against a same-day process restart re-fetching everything)

    expiry: specific 'YYYY-MM-DD' expiry, or None to use the nearest
    (today's 0DTE, if listed) plus `include_next_n_expiries - 1` more.
    """
    today = datetime.now(ZoneInfo("America/New_York")).date().isoformat()
    expiry_key = expiry or f"auto{include_next_n_expiries}"
    cache_key = f"gex_{symbol}_{expiry_key}_{today}"

    if not force_refresh:
        cached = _memory_cache.get(cache_key)
        if cached and (time.time() - cached[0]) < MEMORY_CACHE_TTL_SECONDS:
            return cached[1]

        disk_cached = cache.get(cache_key)
        if disk_cached is not None:
            _memory_cache[cache_key] = (time.time(), disk_cached)
            return disk_cached

    result = _compute_gex_uncached(symbol, expiry, include_next_n_expiries)

    if 'error' not in result:
        _memory_cache[cache_key] = (time.time(), result)
        cache.put(cache_key, result)

    return result


if __name__ == '__main__':
    for sym in ['SPY', 'QQQ']:
        print(f"\n{'='*70}\n{sym} GEX (today's 0DTE expiry)\n{'='*70}")
        result = compute_gex(sym)
        if 'error' in result:
            print(' ERROR:', result['error'])
            continue
        print(f" Spot: {result['spot']:.2f}")
        print(f" Expiries used: {result['expiries_used']}")
        print(f" Regime: {result['regime']}  (total GEX: ${result['total_gex']:,.0f} per 1% move)")
        print(f" Cumulative GEX below spot: ${result['cum_gex_below_spot']:,.0f}")
        print(f" Cumulative GEX above spot: ${result['cum_gex_above_spot']:,.0f}")
        print(" Top 5 gamma concentration strikes (magnet/pin levels):")
        for strike, gex in result['top_gamma_strikes']:
            print(f"   ${strike:.2f}: ${gex:,.0f}")
