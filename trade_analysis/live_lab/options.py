"""Contract selection and derived Greeks.

ATM is defined ONCE here and never varies:

    ATM = the listed strike with the smallest absolute distance to the underlying's NBBO
    midpoint at the signal bar's close timestamp. Ties break to the LOWER strike.
    ATM-1 / ATM+1 are the adjacent strikes by LADDER POSITION, not by dollars.

All three arms are recorded on every signal. The primary arm is ATM, fixed before any
data existed. The +/-1 arms exist to be measured and can never be promoted to primary --
that is what stops a post-hoc "actually ATM+1 was best all along".

The subscription tier exposes no Greeks endpoint (every route 404s), so implied vol and
delta are solved here from the mid price and stored as `iv_derived` / `delta_derived`.
They are never presented as vendor data.
"""
from __future__ import annotations

import datetime as dt
import math

MINUTES_PER_YEAR = 525_600.0
RISK_FREE = 0.04          # flat; only used for the derived Greeks, never for P&L


# --------------------------------------------------------------------------- Black-Scholes


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def bs_price(spot, strike, t_years, vol, right, r=RISK_FREE):
    if t_years <= 0 or vol <= 0 or spot <= 0 or strike <= 0:
        intrinsic = (spot - strike) if right == "call" else (strike - spot)
        return max(intrinsic, 0.0)
    sd = vol * math.sqrt(t_years)
    d1 = (math.log(spot / strike) + (r + 0.5 * vol * vol) * t_years) / sd
    d2 = d1 - sd
    disc = math.exp(-r * t_years)
    if right == "call":
        return spot * _norm_cdf(d1) - strike * disc * _norm_cdf(d2)
    return strike * disc * _norm_cdf(-d2) - spot * _norm_cdf(-d1)


def bs_delta(spot, strike, t_years, vol, right, r=RISK_FREE):
    if t_years <= 0 or vol <= 0 or spot <= 0 or strike <= 0:
        if right == "call":
            return 1.0 if spot > strike else 0.0
        return -1.0 if spot < strike else 0.0
    sd = vol * math.sqrt(t_years)
    d1 = (math.log(spot / strike) + (r + 0.5 * vol * vol) * t_years) / sd
    return _norm_cdf(d1) if right == "call" else _norm_cdf(d1) - 1.0


def implied_vol(price, spot, strike, t_years, right, lo=0.01, hi=5.0, tol=1e-5):
    """Bisection IV. Returns None when the price is outside the no-arbitrage envelope."""
    if price <= 0 or t_years <= 0 or spot <= 0 or strike <= 0:
        return None
    intrinsic = max((spot - strike) if right == "call" else (strike - spot), 0.0)
    if price < intrinsic - 1e-9:
        return None
    if bs_price(spot, strike, t_years, hi, right) < price:
        return None
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if bs_price(spot, strike, t_years, mid, right) < price:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return 0.5 * (lo + hi)


def minutes_to_expiry(now: dt.datetime, expiration: dt.date) -> float:
    """0DTE contracts settle at 16:00 ET on their expiration date."""
    exp_dt = dt.datetime.combine(expiration, dt.time(16, 0))
    return max((exp_dt - now).total_seconds() / 60.0, 0.0)


# --------------------------------------------------------------------------- selection


class UnusableQuote(Exception):
    pass


def select_arms(chain: list[dict], spot_mid: float, right: str,
                max_spread_pct_of_mid: float = 0.25) -> dict:
    """Pick ATM / ATM-1 / ATM+1 from a chain snapshot.

    `chain` is the output of ThetaLiveFeed.chain_quotes. `right` is "call" or "put".
    Raises UnusableQuote if no ATM arm is tradeable; individual wing arms may be absent
    and are simply recorded as None rather than substituted.
    """
    side = sorted((q for q in chain if q["right"] == right), key=lambda q: q["strike"])
    if not side:
        raise UnusableQuote("no contracts for right")

    strikes = [q["strike"] for q in side]
    # nearest strike to spot; ties -> LOWER strike (stable, declared in the spec)
    best_i, best_d = 0, float("inf")
    for i, k in enumerate(strikes):
        d = abs(k - spot_mid)
        if d < best_d - 1e-12:
            best_i, best_d = i, d
    atm = side[best_i]
    if not _usable(atm, max_spread_pct_of_mid):
        raise UnusableQuote(
            f"ATM unusable: bid={atm['bid']} ask={atm['ask']} "
            f"spread_pct={_spread_pct(atm):.3f}")

    def wing(idx):
        if 0 <= idx < len(side):
            q = side[idx]
            return q if _usable(q, max_spread_pct_of_mid) else None
        return None

    # ladder position, not dollars: -1 is one strike toward lower, +1 one toward higher
    return {"ATM": atm, "ATM-1": wing(best_i - 1), "ATM+1": wing(best_i + 1),
            "atm_strike": atm["strike"], "spot_mid": spot_mid,
            "strike_distance_pct": 100.0 * abs(atm["strike"] - spot_mid) / spot_mid}


def _spread_pct(q: dict) -> float:
    return (q["ask"] - q["bid"]) / q["mid"] if q["mid"] > 0 else 9.99


def _usable(q: dict, max_spread_pct_of_mid: float) -> bool:
    return q["bid"] > 0 and q["ask"] > q["bid"] and _spread_pct(q) <= max_spread_pct_of_mid


def enrich(q: dict, spot_mid: float, expiration: dt.date, now: dt.datetime) -> dict:
    """Attach derived IV/delta. Always suffixed _derived -- these are not vendor values."""
    mins = minutes_to_expiry(now, expiration)
    t = mins / MINUTES_PER_YEAR
    iv = implied_vol(q["mid"], spot_mid, q["strike"], t, q["right"])
    return {
        **q,
        "spread_pct_of_mid": _spread_pct(q),
        "minutes_to_expiry": mins,
        "iv_derived": iv,
        "delta_derived": bs_delta(spot_mid, q["strike"], t, iv, q["right"]) if iv else None,
    }
